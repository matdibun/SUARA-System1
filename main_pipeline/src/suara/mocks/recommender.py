"""Deterministic pitch-range compatibility ranking.

This rule is a transparent baseline, not an evaluated measure of comfort or
song suitability. It scores only songs with a verified melody range (or an
explicit synthetic mock range) and a user's observed pitch range.
"""
from math import log2
from typing import Sequence

from suara.contracts import Recommendation, SongFeatures, VoiceProfile
from suara.errors import validate_top_k

SCORING_VERSION = "pitch-range-overlap-v1"


def _range_overlap_score(voice_min_hz: float, voice_max_hz: float,
                        song_min_hz: float, song_max_hz: float) -> float | None:
    """Return the fraction of the song's semitone span inside the observed range."""
    voice_low, voice_high = log2(voice_min_hz), log2(voice_max_hz)
    song_low, song_high = log2(song_min_hz), log2(song_max_hz)
    song_span = song_high - song_low
    if song_span == 0:
        return 1.0 if voice_low <= song_low <= voice_high else None
    overlap = min(voice_high, song_high) - max(voice_low, song_low)
    if overlap <= 0:
        return None
    return min(1.0, max(0.0, overlap / song_span))


def recommend_songs(voice_profile: VoiceProfile, song_catalog: Sequence[SongFeatures],
                    top_k: int = 5) -> list[Recommendation]:
    """Rank eligible songs by melody-span overlap; skip unsupported comparisons.

    Score = overlapping semitone span / the song's full semitone span.
    Ties sort by song_id. Null pitch, null ranges, unverified song ranges, and
    ranges with no overlap are skipped. Scores are normalized heuristic values,
    not probabilities or evidence of vocal comfort.
    """
    validate_top_k(top_k)
    voice_profile.validate()
    for song in song_catalog:
        song.validate()

    pitch = voice_profile.pitch
    if (pitch.voiced_ratio == 0 or pitch.min_pitch_hz is None or
            pitch.max_pitch_hz is None):
        return []

    scored: list[tuple[SongFeatures, float]] = []
    for song in song_catalog:
        if song.provenance not in ("verified_melody", "mock"):
            continue
        if song.melody_min_pitch_hz is None or song.melody_max_pitch_hz is None:
            continue
        score = _range_overlap_score(
            pitch.min_pitch_hz, pitch.max_pitch_hz,
            song.melody_min_pitch_hz, song.melody_max_pitch_hz,
        )
        if score is not None:
            scored.append((song, score))

    scored.sort(key=lambda item: (-item[1], item[0].song_id))
    recommendations = []
    for song, score in scored[:top_k]:
        mode = "mock" if (voice_profile.analysis_mode == "mock" or
                          song.provenance == "mock") else "real"
        explanation = (
            f"{score * 100:.1f}% of this song excerpt's documented melody pitch span "
            "overlaps the observed vocal pitch range. This range-only heuristic "
            "is not a probability or evidence of comfort or suitability."
        )
        if voice_profile.analysis_mode == "mock":
            explanation += " The input pitch profile is synthetic; this result is a demonstration only."
        if song.provenance == "mock":
            explanation += " The song range is synthetic."
        recommendations.append(Recommendation(
            song_id=song.song_id, title=song.title, artist=song.artist,
            score=float(score), scoring_version=SCORING_VERSION, mode=mode,
            explanation=explanation,
        ))
    return recommendations
