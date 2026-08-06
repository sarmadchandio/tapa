"""Speaker clustering: silhouette scoring and speaker-count estimation.

Regression cover for two shipped bugs:
  * a fixed Ward distance cut made the speaker count grow with recording
    length (20 min -> 2 speakers, the same audio at 2 h -> 8);
  * singleton clusters scored a perfect silhouette, so 11 segments were split
    into 8 "speakers".
"""
import numpy as np
import pytest
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import pdist, squareform

from tapa.config import TAPAConfig
from tapa.diarization import (
    _absorb_small_clusters,
    _estimate_num_speakers,
    _silhouette,
    speaker_summary,
)

DIM = 64


def embeddings(n_speakers, n_segments, within=0.32, between=0.72, seed=0,
               bimodal=False):
    """Unit-norm embeddings with controlled within/between-speaker distance."""
    rng = np.random.default_rng(seed)
    rho = 1.0 - between
    shared = rng.normal(size=DIM); shared /= np.linalg.norm(shared)
    centres = []
    for _ in range(n_speakers):
        ind = rng.normal(size=DIM); ind /= np.linalg.norm(ind)
        c = np.sqrt(max(rho, 0)) * shared + np.sqrt(1 - max(rho, 0)) * ind
        centres.append(c / np.linalg.norm(c))
    noise = np.sqrt(1.0 / (1.0 - within) - 1.0) / np.sqrt(DIM)
    per = [n_segments // n_speakers] * n_speakers
    for i in range(n_segments - sum(per)):
        per[i] += 1
    X, y = [], []
    for k in range(n_speakers):
        base = np.repeat(centres[k][None, :], per[k], axis=0)
        if bimodal and k == 0:                      # one talker, two registers
            shift = rng.normal(size=DIM); shift /= np.linalg.norm(shift)
            half = per[k] // 2
            base[half:] = centres[k] + 0.55 * shift
            base[half:] /= np.linalg.norm(base[half:], axis=1, keepdims=True)
        v = base + rng.normal(0, noise, (per[k], DIM))
        X.append(v / np.linalg.norm(v, axis=1, keepdims=True))
        y += [k] * per[k]
    return np.vstack(X), np.array(y)


def estimate(X, cfg=None):
    cfg = cfg or TAPAConfig()
    d = pdist(X, metric="euclidean")
    return _estimate_num_speakers(linkage(d, method="ward"), squareform(d), cfg)


# --------------------------------------------------------------- silhouette

def _reference_silhouette(D, labels):
    """Straightforward definition, used to check the vectorised version."""
    labs = np.unique(labels)
    if len(labs) < 2:
        return -1.0
    out = []
    for i in range(len(labels)):
        same = labels == labels[i]
        same[i] = False
        if not same.any():                       # singleton: 0 by convention
            out.append(0.0)
            continue
        a = D[i, same].mean()
        b = min(D[i, labels == L].mean() for L in labs if L != labels[i])
        out.append(0.0 if max(a, b) == 0 else (b - a) / max(a, b))
    return float(np.mean(out))


@pytest.mark.parametrize("seed", range(5))
def test_silhouette_matches_reference(seed):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(60, 8))
    D = squareform(pdist(X))
    labels = rng.integers(0, 4, 60)
    assert _silhouette(D, labels) == pytest.approx(_reference_silhouette(D, labels))


def test_singleton_clusters_score_zero():
    """A cluster of one has no within-cluster distance; it must not score 1.0."""
    X = np.array([[0.0, 0.0], [0.1, 0.0], [5.0, 5.0]])
    D = squareform(pdist(X))
    labels = np.array([0, 0, 1])
    # the singleton contributes 0, so the mean is (s0 + s1 + 0) / 3 < 1
    assert _silhouette(D, labels) < 0.9
    all_singletons = _silhouette(squareform(pdist(np.eye(4))), np.arange(4))
    assert all_singletons == pytest.approx(0.0)


def test_silhouette_undefined_for_single_cluster():
    D = squareform(pdist(np.random.default_rng(0).normal(size=(10, 4))))
    assert _silhouette(D, np.zeros(10, dtype=int)) == -1.0


# ------------------------------------------------------- count estimation

@pytest.mark.parametrize("n_speakers", [1, 2, 3, 4])
@pytest.mark.parametrize("n_segments", [40, 160, 600])
def test_estimates_correct_count(n_speakers, n_segments):
    X, _ = embeddings(n_speakers, n_segments, seed=n_speakers * 100 + n_segments)
    assert estimate(X)[0] == n_speakers


@pytest.mark.parametrize("n_segments", [40, 80, 160, 300, 600, 1000])
def test_count_does_not_drift_with_recording_length(n_segments):
    """The shipped bug: more segments of the same speakers gave more speakers."""
    X, _ = embeddings(2, n_segments, seed=n_segments)
    assert estimate(X)[0] == 2


@pytest.mark.parametrize("n_segments", [8, 11, 16, 24])
def test_short_recordings_are_not_shredded(n_segments):
    """The shipped bug: 11 segments were split into 8 'speakers'."""
    X, _ = embeddings(2, n_segments, seed=n_segments)
    k, _ = estimate(X)
    assert k == 2, f"{n_segments} segments -> {k} speakers"


def test_candidate_count_capped_by_segments():
    cfg = TAPAConfig(min_segments_per_speaker=4)
    X, _ = embeddings(2, 9, seed=1)
    k, _ = estimate(X, cfg)
    assert k <= 9 // cfg.min_segments_per_speaker


def test_single_speaker_not_forced_into_two():
    X, _ = embeddings(1, 120, seed=3)
    k, score = estimate(X)
    assert k == 1
    assert score < TAPAConfig().min_speaker_silhouette


def test_bimodal_talker_stays_one_speaker():
    """A talker who changes register must not become two speakers."""
    X, _ = embeddings(2, 200, seed=5, bimodal=True)
    assert estimate(X)[0] == 2


def test_num_speakers_override_wins():
    X, _ = embeddings(3, 200, seed=7)
    Z = linkage(pdist(X, metric="euclidean"), method="ward")
    assert len(set(fcluster(Z, t=2, criterion="maxclust"))) == 2


# ---------------------------------------------------- summary and absorption

def test_speaker_summary_shares_sum_to_one():
    segs = [{"speaker": "SPEAKER_00", "start": 0, "end": 10},
            {"speaker": "SPEAKER_01", "start": 10, "end": 12},
            {"speaker": "SPEAKER_00", "start": 12, "end": 18}]
    summary = speaker_summary(segs)
    assert list(summary) == ["SPEAKER_00", "SPEAKER_01"]      # longest first
    assert sum(v["share"] for v in summary.values()) == pytest.approx(1.0)
    assert summary["SPEAKER_00"]["segments"] == 2
    assert summary["SPEAKER_00"]["seconds"] == pytest.approx(16.0)


def test_absorb_small_clusters_merges_slivers():
    X, y = embeddings(2, 60, seed=11)
    labels = y.copy() + 1
    labels[:2] = 99                                   # a sliver split off
    segs = [{"start": i * 3.0, "end": i * 3.0 + 2.5} for i in range(len(labels))]
    cfg = TAPAConfig(min_speaker_share=0.05)
    out = _absorb_small_clusters(labels, X, segs, list(range(len(labels))), cfg)
    assert 99 not in set(out.tolist())
    assert len(set(out.tolist())) == 2


def test_absorb_disabled_by_default():
    X, y = embeddings(2, 40, seed=13)
    labels = y.copy() + 1
    labels[:1] = 99
    segs = [{"start": i * 2.0, "end": i * 2.0 + 1.5} for i in range(len(labels))]
    cfg = TAPAConfig()
    assert cfg.min_speaker_share == 0
    out = _absorb_small_clusters(labels, X, segs, list(range(len(labels))), cfg)
    assert np.array_equal(out, labels)
