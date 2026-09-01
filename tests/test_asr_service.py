import importlib
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


def load_whisperx_adapter():
    """Reload the adapter with WhisperX selected for an isolated test.

    Returns:
        The freshly imported `server.asr_service` module.
    """
    os.environ["LIVETALKING_ASR_BACKEND"] = "whisperx"
    os.environ["WHISPERX_URL"] = "http://whisperx.test"
    sys.modules.pop("server.asr_service", None)
    return importlib.import_module("server.asr_service")


class ASRServiceTest(unittest.TestCase):
    """Validate the LiveTalking ASR adapter without requiring GPU models."""

    def test_whisperx_transcribe_normalizes_response(self):
        """The adapter sends the expected multipart request and returns text."""
        adapter = load_whisperx_adapter()
        captured = {}

        class Response:
            ok = True
            status_code = 200

            def json(self):
                return {"text": "  hello world  "}

        def fake_post(url, files, data, timeout):
            captured.update(url=url, files=files, data=data, timeout=timeout)
            return Response()

        with tempfile.TemporaryDirectory() as temp_dir:
            audio_path = Path(temp_dir) / "sample.wav"
            audio_path.write_bytes(b"RIFF")
            with mock.patch.object(adapter.requests, "post", side_effect=fake_post):
                self.assertEqual(adapter.transcribe(str(audio_path)), "hello world")

        self.assertEqual(captured["url"], "http://whisperx.test/v1/transcribe")
        self.assertEqual(captured["data"]["mode"], "fast")
        self.assertEqual(captured["data"]["language"], "auto")
        self.assertEqual(captured["files"]["file"][0], "sample.wav")

    def test_whisperx_transcribe_reports_upstream_error(self):
        """HTTP errors include the sidecar status and diagnostic message."""
        adapter = load_whisperx_adapter()

        class Response:
            ok = False
            status_code = 503

            def json(self):
                return {"detail": "模型仍在加载"}

        with tempfile.TemporaryDirectory() as temp_dir:
            audio_path = Path(temp_dir) / "sample.wav"
            audio_path.write_bytes(b"RIFF")
            with mock.patch.object(adapter.requests, "post", return_value=Response()):
                with self.assertRaises(RuntimeError) as raised:
                    adapter.transcribe(str(audio_path))

        message = str(raised.exception)
        self.assertIn("HTTP 503", message)
        self.assertIn("模型仍在加载", message)


if __name__ == "__main__":
    unittest.main()
