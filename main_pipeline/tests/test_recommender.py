"""Rule-based recommendation tests use synthetic inputs and the committed E3 export."""
import json
from pathlib import Path
import unittest

from suara.contracts import PitchFeatures, SongFeatures, TimbreFeatures, VoiceProfile
from suara.mocks.recommender import SCORING_VERSION, recommend_songs


def profile(low=110.0, high=330.0, *, mode="real"):
    warnings = ["Synthetic test profile; not measured."] if mode == "mock" else []
    pitch = PitchFeatures(
        min_pitch_hz=low, max_pitch_hz=high, median_pitch_hz=(low + high) / 2,
        voiced_ratio=1.0, confidence=None, mode=mode, model_version="test-pitch-v1",
        warnings=warnings,
    )
    timbre = TimbreFeatures(
        mfcc_mean=[0.0] * 20, spectral_centroid_mean_hz=1000.0, mode=mode,
        model_version="test-timbre-v1", processing_version="test-mfcc-v1", warnings=warnings,
    )
    return VoiceProfile(
        pitch=pitch, timbre=timbre, analysis_mode=mode,
        status="warning" if mode == "mock" else "ok", warnings=warnings,
    )


def song(song_id, low, high, *, provenance="verified_melody"):
    return SongFeatures(
        song_id=song_id, title=f"Title {song_id}", artist="Test artist",
        melody_min_pitch_hz=low, melody_max_pitch_hz=high,
        feature_version="test-range-v1", provenance=provenance,
        source_description="Synthetic test melody range.",
    )


class RecommendationTests(unittest.TestCase):
    def test_overlap_formula_and_primary_order(self):
        catalog = [
            song("contained", 160.0, 290.0),
            song("partial", 250.0, 500.0),
            song("wider", 200.0, 400.0),
            song("outside", 400.0, 500.0),
        ]
        result = recommend_songs(profile(150.0, 300.0), catalog)
        self.assertEqual([rec.song_id for rec in result],
                         ["contained", "wider", "partial"])
        self.assertAlmostEqual(result[0].score, 1.0)
        self.assertGreater(result[1].score, result[2].score)
        self.assertEqual(result[0].scoring_version, SCORING_VERSION)
        self.assertIn("not a probability", result[0].explanation)

    def test_equal_ranges_and_deterministic_ties(self):
        catalog = [song("z", 110.0, 220.0), song("a", 110.0, 220.0)]
        result = recommend_songs(profile(110.0, 220.0), catalog)
        self.assertEqual([rec.song_id for rec in result], ["a", "z"])
        self.assertEqual([rec.score for rec in result], [1.0, 1.0])

    def test_top_k_empty_catalog_and_input_order(self):
        catalog = [song("b", 110.0, 220.0), song("a", 120.0, 220.0)]
        expected = [rec.song_id for rec in recommend_songs(profile(), catalog)]
        self.assertEqual([rec.song_id for rec in recommend_songs(profile(), catalog[::-1])], expected)
        self.assertEqual(len(recommend_songs(profile(), catalog, top_k=1)), 1)
        self.assertEqual(recommend_songs(profile(), [], top_k=5), [])
        self.assertEqual(recommend_songs(profile(), catalog, top_k=5), recommend_songs(profile(), catalog))

    def test_missing_and_unverified_data_are_skipped(self):
        catalog = [
            song("missing", None, None, provenance="unverified"),
            song("unverified", 120.0, 220.0, provenance="unverified"),
            song("zero-overlap", 400.0, 500.0),
        ]
        self.assertEqual(recommend_songs(profile(), catalog), [])

    def test_missing_pitch_returns_no_suitability_results(self):
        pitch = PitchFeatures(
            min_pitch_hz=None, max_pitch_hz=None, median_pitch_hz=None, voiced_ratio=0.0,
            confidence=None, mode="real", model_version="test-pitch-v1",
            warnings=["No accepted voiced frames."],
        )
        timbre = profile().timbre
        no_pitch = VoiceProfile(
            pitch=pitch, timbre=timbre, analysis_mode="real",
            status="no_voiced_pitch", warnings=["No accepted voiced frames."],
        )
        self.assertEqual(recommend_songs(no_pitch, [song("eligible", 110.0, 220.0)]), [])

    def test_mock_provenance_stays_explicit(self):
        result = recommend_songs(profile(mode="mock"), [song("verified", 110.0, 220.0)])
        self.assertEqual(result[0].mode, "mock")
        self.assertIn("synthetic", result[0].explanation.lower())
        mock_song = song("mock-range", 110.0, 220.0, provenance="mock")
        self.assertEqual(recommend_songs(profile(), [mock_song])[0].mode, "mock")

    def test_e3_catalog_differs_from_naive_song_id_baseline(self):
        root = Path(__file__).resolve().parents[2]
        rows = json.loads((root / "experiments/E3/handoff/song_features.json").read_text(encoding="utf-8"))
        catalog = [SongFeatures(**row) for row in rows]
        naive = [item.song_id for item in sorted(catalog, key=lambda item: item.song_id)]
        ranked = [rec.song_id for rec in recommend_songs(profile(150.0, 260.0), catalog)]
        self.assertEqual(naive, ["E3-S01", "E3-S02", "E3-S03", "E3-S04", "E3-S05"])
        self.assertEqual(ranked[0], "E3-S01")
        self.assertNotEqual(ranked, naive)
        self.assertEqual(len(ranked), 5)


if __name__ == "__main__":
    unittest.main()
