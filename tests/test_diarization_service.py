import importlib
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


def load_service(api_key=""):
    """Load the in-process service with deterministic environment settings."""
    os.environ["FUNASR_API_KEY"] = api_key
    os.environ["LIVETALKING_SPEAKER_DIARIZATION"] = "0"
    os.environ["LIVETALKING_SPEAKER_DIARIZATION_PRELOAD"] = "0"
    sys.modules.pop("server.diarization_service", None)
    return importlib.import_module("server.diarization_service")


class DiarizationServiceTest(unittest.TestCase):
    """Validate local FunASR inference and public API authentication helpers."""

    def test_request_flag(self):
        """Request-level true enables diarization without global enablement."""
        service = load_service()
        self.assertFalse(service.requested())
        self.assertTrue(service.requested("true"))
        self.assertFalse(service.requested("0"))

    def test_diarize_runs_local_model(self):
        """The service invokes the in-process model and normalizes its output."""
        service = load_service()
        model = mock.Mock()
        model.generate.return_value = [
            {
                "text": "你好。。",
                "sentence_info": [
                    {
                        "spk": 1,
                        "start": 0,
                        "end": 1000,
                        "text": "上一刻，， 只是我的侥幸。。",
                    }
                ],
            }
        ]
        service.MODEL = model

        with tempfile.TemporaryDirectory() as temp_dir:
            audio_path = Path(temp_dir) / "sample.wav"
            audio_path.write_bytes(b"RIFF")
            with mock.patch.object(
                service,
                "_postprocess",
                side_effect=lambda value: value,
            ):
                result = service.diarize(str(audio_path), "zh")

        self.assertEqual(result["speaker_count"], 1)
        self.assertEqual(result["text"], "你好。")
        self.assertEqual(
            result["speaker_segments"][0]["text"],
            "上一刻，只是我的侥幸。",
        )
        self.assertEqual(model.generate.call_args.kwargs["language"], "zh")

    def test_api_key_accepts_header_and_bearer(self):
        """Configured API keys work through both documented HTTP schemes."""
        service = load_service("test-secret")

        class Request:
            def __init__(self, headers):
                self.headers = headers

        self.assertFalse(service.authorize_request(Request({})))
        self.assertTrue(
            service.authorize_request(Request({"X-API-Key": "test-secret"}))
        )
        self.assertTrue(
            service.authorize_request(
                Request({"Authorization": "Bearer test-secret"})
            )
        )

    def test_missing_api_key_keeps_development_mode(self):
        """Without a configured key local development calls remain available."""
        service = load_service("")

        class Request:
            headers = {}

        self.assertTrue(service.authorize_request(Request()))


if __name__ == "__main__":
    unittest.main()
