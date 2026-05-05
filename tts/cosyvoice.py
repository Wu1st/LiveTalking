import time
import numpy as np
import resampy
import requests
from typing import Iterator

from utils.logger import logger
from .base_tts import BaseTTS, State
from registry import register

@register("tts", "cosyvoice")
class CosyVoiceTTS(BaseTTS):
    def txt_to_audio(self, msg: tuple[str, dict]):
        text, textevent = msg
        tts_cfg  = textevent.get('tts', {})
        ref_file = tts_cfg.get('ref_file', self.opt.REF_FILE)
        ref_text = tts_cfg.get('ref_text', self.opt.REF_TEXT)
        dialect  = tts_cfg.get('dialect', '')
        language = tts_cfg.get('language', getattr(self.opt, 'TTS_LANG', 'zh'))

        if dialect == 'minnan':
            self.stream_tts(
                self.cosy_voice_minnan(text, ref_file, self.opt.TTS_SERVER),
                msg
            )
        else:
            self.stream_tts(
                self.cosy_voice(text, ref_file, ref_text, language, self.opt.TTS_SERVER),
                msg
            )

    def cosy_voice(self, text, reffile, reftext, language, server_url) -> Iterator[bytes]:
        start = time.perf_counter()
        # 前缀由 server.py 统一添加，这里只传原始文字
        payload = {
            'tts_text':    text,
            'prompt_text': reftext,
        }
        try:
            with open(reffile, 'rb') as f:
                files = [('prompt_wav', ('prompt_wav', f, 'application/octet-stream'))]
                res = requests.post(
                    f"{server_url}/inference_zero_shot",
                    data=payload,
                    files=files,
                    stream=True,
                    timeout=60
                )

            end = time.perf_counter()
            logger.info(f"cosy_voice Time to make POST: {end-start:.2f}s")

            if res.status_code != 200:
                logger.error(f"CosyVoice error {res.status_code}: {res.text}")
                return

            first = True
            for chunk in res.iter_content(chunk_size=9600):
                if first:
                    end = time.perf_counter()
                    logger.info(f"cosy_voice Time to first chunk: {end-start:.2f}s")
                    first = False
                if chunk and self.state == State.RUNNING:
                    yield chunk

        except Exception as e:
            logger.exception('cosyvoice error:')

    def cosy_voice_minnan(self, text, reffile, server_url) -> Iterator[bytes]:
        """闽南语合成，调用 /inference_minnan 接口"""
        start = time.perf_counter()
        payload = {'tts_text': text}
        try:
            with open(reffile, 'rb') as f:
                files = [('prompt_wav', ('prompt_wav', f, 'application/octet-stream'))]
                res = requests.post(
                    f"{server_url}/inference_minnan",
                    data=payload,
                    files=files,
                    stream=True,
                    timeout=60
                )

            end = time.perf_counter()
            logger.info(f"cosy_voice_minnan Time to make POST: {end-start:.2f}s")

            if res.status_code != 200:
                logger.error(f"CosyVoice minnan error {res.status_code}: {res.text}")
                return

            first = True
            for chunk in res.iter_content(chunk_size=9600):
                if first:
                    end = time.perf_counter()
                    logger.info(f"cosy_voice_minnan Time to first chunk: {end-start:.2f}s")
                    first = False
                if chunk and self.state == State.RUNNING:
                    yield chunk

        except Exception as e:
            logger.exception('cosyvoice minnan error:')

    def stream_tts(self, audio_stream, msg: tuple[str, dict]):
        text, textevent = msg
        first = True
        for chunk in audio_stream:
            if chunk is not None and len(chunk) > 0:
                stream = np.frombuffer(chunk, dtype=np.int16).astype(np.float32) / 32767
                stream = resampy.resample(x=stream, sr_orig=24000, sr_new=self.sample_rate)
                streamlen = stream.shape[0]
                idx = 0
                while streamlen >= self.chunk:
                    eventpoint = {}
                    if first:
                        eventpoint = {'status': 'start', 'text': text}
                        first = False
                    eventpoint.update(**textevent)
                    self.parent.put_audio_frame(stream[idx:idx+self.chunk], eventpoint)
                    streamlen -= self.chunk
                    idx += self.chunk
        eventpoint = {'status': 'end', 'text': text}
        eventpoint.update(**textevent)
        self.parent.put_audio_frame(np.zeros(self.chunk, np.float32), eventpoint)