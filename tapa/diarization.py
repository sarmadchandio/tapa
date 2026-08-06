"""Speaker diarization using Silero VAD + Resemblyzer embeddings."""

import csv

import numpy as np
import torch
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import pdist, squareform

from .audio import load_audio_16k
from .config import TAPAConfig


def load_silero_vad():
    """Load Silero VAD model and return (model, get_speech_timestamps)."""
    model, utils = torch.hub.load(
        repo_or_dir="snakers4/silero-vad", model="silero_vad",
        force_reload=False, onnx=False, trust_repo=True,
    )
    return model, utils[0]


def get_speech_segments(audio, vad_model, get_speech_timestamps, cfg=None):
    """Detect speech segments using Silero VAD.

    ``audio`` is either a path or a float32 mono 16 kHz numpy array (as
    returned by load_audio_16k) — passing the array lets callers decode the
    file once and share the buffer across pipeline stages.

    Returns (segments, wav_1d_tensor, sample_rate). The tensor shares memory
    with the input array when one is given.
    """
    if cfg is None:
        cfg = TAPAConfig()
    sr = 16000
    if isinstance(audio, np.ndarray):
        wav_1d = torch.from_numpy(audio)
    else:
        wav_1d = torch.from_numpy(load_audio_16k(audio, sr))
    speech_timestamps = get_speech_timestamps(
        wav_1d, vad_model, sampling_rate=sr,
        min_speech_duration_ms=100, min_silence_duration_ms=300, speech_pad_ms=30,
    )
    segments = []
    for ts in speech_timestamps:
        s, e = ts["start"] / sr, ts["end"] / sr
        if (e - s) >= cfg.min_segment_duration:
            segments.append({"start": round(s, 4), "end": round(e, 4)})
    return segments, wav_1d, sr


def assign_speakers(segments, audio_np_16k, sr, voice_encoder, cfg=None):
    """Assign speaker labels to segments using Resemblyzer embeddings + clustering."""
    if cfg is None:
        cfg = TAPAConfig()
    if not segments:
        return []
    embeddings, valid_idx = [], []
    for i, seg in enumerate(segments):
        s_samp = int(seg["start"] * sr)
        e_samp = int(seg["end"] * sr)
        chunk = audio_np_16k[s_samp:e_samp]
        if len(chunk) < sr * 0.5:
            continue
        embeddings.append(voice_encoder.embed_utterance(chunk.astype(np.float32)))
        valid_idx.append(i)
    if len(embeddings) < 2:
        return [{"speaker": "SPEAKER_00", **s} for s in segments]
    embs = np.array(embeddings)
    # Ward is defined for Euclidean distances; Resemblyzer embeddings are
    # unit-norm, where squared Euclidean is just 2x cosine distance, so this
    # keeps the geometry and makes the linkage valid.
    dists = pdist(embs, metric="euclidean")
    Z = linkage(dists, method="ward")
    if cfg.num_speakers:
        labels = fcluster(Z, t=cfg.num_speakers, criterion="maxclust")
    else:
        k, score = _estimate_num_speakers(Z, squareform(dists), cfg)
        labels = (fcluster(Z, t=k, criterion="maxclust") if k > 1
                  else np.ones(len(embs), dtype=int))
        print(f"[TAPA] Estimated {k} speaker(s) (silhouette {score:+.2f}). "
              f"Pass num_speakers=N if you know the true count.", flush=True)
    if cfg.min_speaker_share > 0:
        labels = _absorb_small_clusters(labels, embs, segments, valid_idx, cfg)

    lmap = {}
    labeled = []
    for idx, vi in enumerate(valid_idx):
        cid = int(labels[idx])
        if cid not in lmap:
            lmap[cid] = f"SPEAKER_{len(lmap):02d}"
        labeled.append({"speaker": lmap[cid], "start": segments[vi]["start"], "end": segments[vi]["end"]})
    labeled_set = set(valid_idx)
    for i, seg in enumerate(segments):
        if i not in labeled_set:
            nearest = min(labeled, key=lambda s: abs(s["start"] - seg["start"]))
            labeled.append({"speaker": nearest["speaker"], **seg})
    labeled.sort(key=lambda s: s["start"])
    return _merge_segments(labeled, cfg)


def _silhouette(D, labels):
    """Mean silhouette coefficient for a labelling, from a distance matrix."""
    uniq = np.unique(labels)
    if len(uniq) < 2:
        return -1.0
    n = len(labels)
    sums = np.empty((n, len(uniq)))
    counts = np.empty(len(uniq))
    for j, lab in enumerate(uniq):
        mask = labels == lab
        counts[j] = mask.sum()
        sums[:, j] = D[:, mask].sum(axis=1)
    own = np.searchsorted(uniq, labels)
    rows = np.arange(n)
    own_cnt = counts[own] - 1                       # exclude the point itself
    a = np.where(own_cnt > 0, sums[rows, own] / np.maximum(own_cnt, 1), 0.0)
    means = sums / counts[None, :]
    means[rows, own] = np.inf                       # nearest *other* cluster
    b = means.min(axis=1)
    denom = np.maximum(a, b)
    return float(np.where(denom > 0, (b - a) / denom, 0.0).mean())


def _estimate_num_speakers(Z, D, cfg):
    """Pick the speaker count by silhouette score; returns (k, score).

    A fixed distance cut cannot be used here: Ward merge heights grow with
    cluster size, so the same threshold yields more and more clusters as a
    recording lengthens — a 2 h recording was split into 8 speakers where the
    same audio at 20 min gave 2. Silhouette compares labellings on a scale-free
    basis, so its answer does not drift with recording length. Below
    min_speaker_silhouette no split is convincing and we report one speaker.
    """
    best_k, best_score = 1, -1.0
    kmax = min(cfg.max_speakers, len(D) - 1)
    for k in range(2, kmax + 1):
        labels = fcluster(Z, t=k, criterion="maxclust")
        if len(np.unique(labels)) < 2:
            continue
        score = _silhouette(D, labels)
        if score > best_score:
            best_score, best_k = score, k
    if best_score < cfg.min_speaker_silhouette:
        return 1, best_score
    return best_k, best_score


def _absorb_small_clusters(labels, embs, segments, valid_idx, cfg):
    """Reassign clusters holding less than cfg.min_speaker_share of the speech.

    Automatic speaker-count estimation can split one talker into several
    clusters when their voice varies (loudness, laughter, channel changes).
    Such spurious clusters are typically tiny; each of their segments is moved
    to the nearest surviving cluster by cosine distance between the segment's
    embedding and the cluster centroid. Off by default (share = 0).
    """
    labels = np.asarray(labels).copy()
    durations = {}
    for idx, vi in enumerate(valid_idx):
        seg = segments[vi]
        durations[int(labels[idx])] = durations.get(int(labels[idx]), 0.0) + (seg["end"] - seg["start"])
    total = sum(durations.values())
    if total <= 0:
        return labels
    small = {c for c, d in durations.items() if d / total < cfg.min_speaker_share}
    keep = [c for c in durations if c not in small]
    if not small or not keep:
        return labels

    centroids = {}
    for c in keep:
        rows = embs[labels == c]
        v = rows.mean(axis=0)
        n = np.linalg.norm(v)
        centroids[c] = v / n if n else v
    for idx in range(len(labels)):
        if int(labels[idx]) not in small:
            continue
        e = embs[idx]
        n = np.linalg.norm(e)
        e = e / n if n else e
        labels[idx] = max(keep, key=lambda c: float(np.dot(e, centroids[c])))
    print(f"[TAPA] Merged {len(small)} speaker cluster(s) holding less than "
          f"{cfg.min_speaker_share:.0%} of speech into the nearest speaker.", flush=True)
    return labels


def speaker_summary(segments):
    """Per-speaker speech time and segment count, longest first."""
    stats = {}
    for s in segments:
        d = stats.setdefault(s["speaker"], {"seconds": 0.0, "segments": 0})
        d["seconds"] += s["end"] - s["start"]
        d["segments"] += 1
    total = sum(v["seconds"] for v in stats.values()) or 1.0
    for v in stats.values():
        v["share"] = v["seconds"] / total
    return dict(sorted(stats.items(), key=lambda kv: -kv[1]["seconds"]))


def _merge_segments(segments, cfg):
    if not segments:
        return segments
    segments.sort(key=lambda s: s["start"])
    merged = [segments[0].copy()]
    for seg in segments[1:]:
        prev = merged[-1]
        if seg["speaker"] == prev["speaker"] and (seg["start"] - prev["end"]) <= cfg.merge_gap:
            prev["end"] = max(prev["end"], seg["end"])
        else:
            merged.append(seg.copy())
    return merged


def save_diarization_csv(segments, path):
    """Save diarization segments to CSV."""
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["speaker", "start", "end"])
        for seg in segments:
            w.writerow([seg["speaker"], seg["start"], seg["end"]])
