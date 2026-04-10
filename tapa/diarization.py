"""Speaker diarization using Silero VAD + Resemblyzer embeddings."""

import csv

import numpy as np
import torch
import torchaudio
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import pdist

from .config import TAPAConfig


def load_silero_vad():
    """Load Silero VAD model and return (model, get_speech_timestamps)."""
    model, utils = torch.hub.load(
        repo_or_dir="snakers4/silero-vad", model="silero_vad",
        force_reload=False, onnx=False,
    )
    return model, utils[0]


def get_speech_segments(audio_path, vad_model, get_speech_timestamps, cfg=None):
    """Detect speech segments using Silero VAD.

    Returns (segments, wav_1d_tensor, sample_rate).
    """
    if cfg is None:
        cfg = TAPAConfig()
    wav, sr = torchaudio.load(audio_path)
    if sr != 16000:
        wav = torchaudio.functional.resample(wav, sr, 16000)
        sr = 16000
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    wav_1d = wav.squeeze(0)
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
    dists = pdist(embs, metric="cosine")
    Z = linkage(dists, method="ward")
    labels = (fcluster(Z, t=cfg.num_speakers, criterion="maxclust") if cfg.num_speakers
              else fcluster(Z, t=1.5, criterion="distance"))
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
