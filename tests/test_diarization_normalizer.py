import unittest

from server.diarization_normalizer import (
    normalize_punctuation,
    normalize_result,
)


class DiarizationNormalizerTest(unittest.TestCase):
    """Verify stable anonymous labels and speaker intervals."""

    def test_normalizes_millisecond_sentence_info(self):
        """FunASR sentence_info becomes Speaker_00-style timed segments."""
        result = normalize_result(
            [
                {
                    "text": "你好今天继续测试",
                    "sentence_info": [
                        {"spk": 1, "start": 0, "end": 1000, "text": "你好"},
                        {"spk": 1, "start": 1100, "end": 2000, "text": "今天"},
                        {"spk": 2, "start": 3000, "end": 4200, "text": "继续测试"},
                    ],
                }
            ]
        )

        self.assertEqual(result["speaker_count"], 2)
        self.assertEqual(result["speakers"], ["Speaker_00", "Speaker_01"])
        self.assertEqual(result["segments"][0]["start"], 0.0)
        self.assertEqual(result["segments"][1]["end"], 2.0)
        self.assertEqual(len(result["speaker_segments"]), 2)
        self.assertEqual(result["speaker_segments"][0]["speaker"], "Speaker_00")
        self.assertEqual(result["speaker_segments"][0]["end"], 2.0)

    def test_preserves_explicit_second_unit(self):
        """A result explicitly marked in seconds is not divided by 1000."""
        result = normalize_result(
            {
                "time_unit": "seconds",
                "sentence_info": [
                    {
                        "spk": "alice",
                        "start": 1.25,
                        "end": 2.5,
                        "text": "hello",
                    }
                ],
            }
        )

        self.assertEqual(result["segments"][0]["start"], 1.25)
        self.assertEqual(result["segments"][0]["end"], 2.5)

    def test_requires_speaker_label(self):
        """Missing CAM++ speaker output fails loudly instead of guessing."""
        with self.assertRaises(ValueError):
            normalize_result(
                {"sentence_info": [{"start": 0, "end": 1000, "text": "hello"}]}
            )

    def test_removes_duplicate_chinese_punctuation(self):
        """SenseVoice and CT-Punc punctuation is emitted only once."""
        self.assertEqual(
            normalize_punctuation("上一刻，， 只是我的侥幸。。"),
            "上一刻，只是我的侥幸。",
        )
        self.assertEqual(normalize_punctuation("keep... waiting"), "keep... waiting")

    def test_normalizes_postprocessed_and_merged_segments(self):
        """Cleanup also applies after postprocessing and same-speaker merging."""
        result = normalize_result(
            {
                "text": "你好。。继续，，",
                "sentence_info": [
                    {"spk": 0, "start": 0, "end": 1000, "text": "你好。。"},
                    {"spk": 0, "start": 1100, "end": 1800, "text": "继续，，"},
                ],
            },
            postprocess_text=lambda value: value,
        )

        self.assertEqual(result["text"], "你好。继续，")
        self.assertEqual(result["segments"][0]["text"], "你好。")
        self.assertEqual(result["speaker_segments"][0]["text"], "你好。继续，")


if __name__ == "__main__":
    unittest.main()
