import time
import re
import queue
import threading
from typing import Iterator, Callable

import numpy as np
import requests
from av import AudioFrame
from av.audio.resampler import AudioResampler

from utils.logger import logger
from utils.latency import emit_latency, get_trace
from .base_tts import BaseTTS, State
from registry import register


# ============================================================
# CosyVoice 本地调试开关
# ============================================================

# 是否默认开启切分。
# True：长文本会在 cosyvoice.py 内部自动切成多段。
# False：所有文本整段送给 CosyVoice。
COSYVOICE_SPLIT_ENABLED = False

# 常规段落最大字符数。
# 建议 14~20。比 24 更激进，首包一般更低，但语气会更碎。
COSYVOICE_SPLIT_MAX_CHARS = 24

# 第一段最大字符数。
# 这是降低体感首包延迟的重点。
# 第一段越短，越快开始说话。
COSYVOICE_FIRST_SEGMENT_MAX_CHARS = 12

# 是否默认开启下一段预取。
# True：当前段播放时，后台提前合成下一段。
# 风险：同一个 CosyVoice 服务端可能并发处理多个请求，GPU 压力会变高。
COSYVOICE_SPLIT_PREFETCH = True

# 是否允许 /human 请求里的 tts 字段覆盖上面的本地开关。
# 调试阶段建议 False，避免请求参数干扰本地配置。
COSYVOICE_ALLOW_REQUEST_OVERRIDE = False

# 是否打印切分配置和切分结果。
COSYVOICE_LOG_SPLIT_RESULT = True

# ============================================================
# 缓存开场音频 filler
# ============================================================

# 是否启用缓存开场音频。
# True：如果 filler 已经缓存好，会先播放一句“我看看。”之类的短音频，
# 同时后台合成真实回答，从而掩盖一部分首包等待。
COSYVOICE_FILLER_ENABLED = False

# 开场音频文本。建议短，不要太抢戏。
COSYVOICE_FILLER_TEXT = "嗯嗯"

# 文本长度达到多少才插入 filler。
# 太短的回复不插入，否则会显得多余。
COSYVOICE_FILLER_MIN_TEXT_CHARS = 10

# 是否打印 filler 缓存状态。
COSYVOICE_LOG_FILLER = True

# ============================================================
# 音频参数
# ============================================================

# 服务端 PCM 采样率。CosyVoice 服务端返回的是 24kHz int16 mono PCM。
COSYVOICE_SERVER_SAMPLE_RATE = 24000

# HTTP 读取 chunk 大小。
# 4800 bytes = 24kHz int16 mono 约 0.1 秒音频。
COSYVOICE_HTTP_CHUNK_SIZE = 4800


_QUEUE_END = object()


def _iter_resampled_pcm(
    audio_stream: Iterator[bytes],
    source_rate: int,
    target_rate: int,
) -> Iterator[np.ndarray]:
    """连续地把流式 int16 单声道 PCM 重采样为 float32。

    HTTP 分块边界不一定与 int16 样本边界一致，也不是音频时间边界。
    因此这里先保留不完整的单字节，再让同一个 PyAV resampler 处理整个
    响应流，避免逐 HTTP 块重置滤波器造成接缝、噪音和电音。
    """
    resampler = AudioResampler(
        format="fltp",
        layout="mono",
        rate=target_rate,
    )
    pending_bytes = bytearray()

    for chunk in audio_stream:
        if not chunk:
            continue

        pending_bytes.extend(chunk)
        complete_byte_count = len(pending_bytes) & ~1
        if complete_byte_count == 0:
            continue

        pcm_bytes = bytes(pending_bytes[:complete_byte_count])
        del pending_bytes[:complete_byte_count]

        samples = np.frombuffer(pcm_bytes, dtype="<i2").astype(np.float32)
        samples /= 32768.0

        input_frame = AudioFrame.from_ndarray(
            samples.reshape(1, -1),
            format="fltp",
            layout="mono",
        )
        input_frame.sample_rate = source_rate

        for output_frame in resampler.resample(input_frame):
            output = output_frame.to_ndarray().reshape(-1)
            if output.size > 0:
                yield output.astype(np.float32, copy=False)

    if pending_bytes:
        logger.warning(
            "CosyVoice PCM stream ended with one incomplete int16 byte; dropping it"
        )

    # 刷出重采样器内部为保证滤波连续性而暂存的尾部样本。
    for output_frame in resampler.resample(None):
        output = output_frame.to_ndarray().reshape(-1)
        if output.size > 0:
            yield output.astype(np.float32, copy=False)


def _to_bool(value, default: bool = True) -> bool:
    if value is None:
        return default

    if isinstance(value, bool):
        return value

    if isinstance(value, str):
        return value.strip().lower() not in ("0", "false", "no", "off", "none", "")

    return bool(value)


def _safe_int(value, default: int) -> int:
    try:
        return int(value)
    except Exception:
        return default


def split_tts_text(text: str, max_chars: int = 18, min_chars: int = 6) -> list[str]:
    """
    面向 TTS 的中文/中英混合切句。

    目标：
    - 避免把长文本一次性送给 CosyVoice，降低单次首包压力；
    - 优先在句号、问号、感叹号、分号处分段；
    - 对特别长的句子，再尝试在逗号、顿号、冒号处分段；
    - 避免过短碎片，减少请求次数和语气不自然。
    """
    text = (text or "").strip()

    if not text:
        return []

    if len(text) <= max_chars:
        return [text]

    # 第一轮：强标点切分
    strong_parts = re.split(r"([。！？!?；;])", text)
    strong_units: list[str] = []

    for i in range(0, len(strong_parts), 2):
        body = strong_parts[i]
        punct = strong_parts[i + 1] if i + 1 < len(strong_parts) else ""
        unit = (body + punct).strip()
        if unit:
            strong_units.append(unit)

    # 第二轮：弱标点切分
    weak_units: list[str] = []

    for unit in strong_units:
        if len(unit) <= max_chars:
            weak_units.append(unit)
            continue

        weak_parts = re.split(r"([，,、：:])", unit)

        for i in range(0, len(weak_parts), 2):
            body = weak_parts[i]
            punct = weak_parts[i + 1] if i + 1 < len(weak_parts) else ""
            piece = (body + punct).strip()

            if not piece:
                continue

            if len(piece) <= max_chars:
                weak_units.append(piece)
            else:
                # 仍然太长，硬切。
                start = 0
                while start < len(piece):
                    sub = piece[start:start + max_chars].strip()
                    if sub:
                        weak_units.append(sub)
                    start += max_chars

    # 合并过短片段
    segments: list[str] = []
    buf = ""

    for unit in weak_units:
        if not buf:
            buf = unit
            continue

        if len(buf) < min_chars or len(buf) + len(unit) <= max_chars:
            buf += unit
        else:
            segments.append(buf.strip())
            buf = unit

    if buf.strip():
        segments.append(buf.strip())

    return [s for s in segments if s]


def split_first_segment_short(
    segments: list[str],
    first_max_chars: int,
    normal_max_chars: int,
) -> list[str]:
    """
    把第一段进一步切短。

    目的：
    - 降低第一段 TTS 首包时间；
    - 让数字人尽快开始说第一句话；
    - 后续内容仍按 normal_max_chars 切，避免全程太碎。
    """
    if not segments:
        return []

    if first_max_chars <= 0:
        return segments

    first = segments[0]

    if len(first) <= first_max_chars:
        return segments

    # 优先在第一段前 first_max_chars 范围内找标点。
    # 例如 “好的，我来看看。” 希望切成 “好的，”
    candidate_punct = "，,。！？!?；;、：:"
    cut_pos = -1

    for i, ch in enumerate(first):
        if i + 1 > first_max_chars:
            break
        if ch in candidate_punct and i + 1 >= 2:
            cut_pos = i + 1

    if cut_pos <= 0:
        cut_pos = first_max_chars

    first_piece = first[:cut_pos].strip()
    rest = first[cut_pos:].strip()

    new_segments: list[str] = []

    if first_piece:
        new_segments.append(first_piece)

    if rest:
        rest_segments = split_tts_text(
            rest,
            max_chars=normal_max_chars,
            min_chars=6,
        )
        new_segments.extend(rest_segments)

    new_segments.extend(segments[1:])

    return [s for s in new_segments if s]


@register("tts", "cosyvoice")
class CosyVoiceTTS(BaseTTS):
    def _ensure_spk_cache(self):
        """
        不重写 __init__，避免和不同版本 LiveTalking 的 BaseTTS 构造函数冲突。
        """
        if not hasattr(self, "_spk_id_cache"):
            self._spk_id_cache = {}

    def _ensure_filler_cache(self):
        """
        filler 缓存结构：
        key -> {
            "status": "loading" | "ready" | "error",
            "chunks": list[bytes],
            "text": str,
        }
        """
        if not hasattr(self, "_filler_cache"):
            self._filler_cache = {}

    def txt_to_audio(self, msg: tuple[str, dict]):
        """
        LiveTalking 的文本最终会进入这里。

        切句放在这里：
        - /human 路由不改；
        - BaseAvatar 不改；
        - 只影响 cosyvoice 这个 TTS 适配器；
        - echo、chat、外部 API 直接传入 text 都能覆盖。
        """
        text, textevent = msg
        text = text or ""
        textevent = textevent or {}
        trace = get_trace(textevent)
        emit_latency(
            "tts_processing_started",
            trace,
            text_chars=len(text),
        )

        tts_cfg = textevent.get("tts", {}) or {}

        ref_file = tts_cfg.get("ref_file", self.opt.REF_FILE)
        ref_text = tts_cfg.get("ref_text", self.opt.REF_TEXT)
        dialect = tts_cfg.get("dialect", "")
        language = tts_cfg.get("language", getattr(self.opt, "TTS_LANG", "zh"))

        # 默认由 cosyvoice.py 文件顶部的本地开关控制
        split_enabled = COSYVOICE_SPLIT_ENABLED
        split_max_chars = COSYVOICE_SPLIT_MAX_CHARS
        first_segment_max_chars = COSYVOICE_FIRST_SEGMENT_MAX_CHARS
        split_prefetch = COSYVOICE_SPLIT_PREFETCH
        filler_enabled = COSYVOICE_FILLER_ENABLED

        # 可选：允许请求里的 tts 参数覆盖本地默认值
        if COSYVOICE_ALLOW_REQUEST_OVERRIDE:
            split_enabled = _to_bool(
                tts_cfg.get("split", split_enabled),
                default=split_enabled,
            )
            split_max_chars = _safe_int(
                tts_cfg.get("split_max_chars", split_max_chars),
                default=split_max_chars,
            )
            first_segment_max_chars = _safe_int(
                tts_cfg.get("first_segment_max_chars", first_segment_max_chars),
                default=first_segment_max_chars,
            )
            split_prefetch = _to_bool(
                tts_cfg.get("split_prefetch", split_prefetch),
                default=split_prefetch,
            )
            filler_enabled = _to_bool(
                tts_cfg.get("filler", filler_enabled),
                default=filler_enabled,
            )

        if split_enabled:
            segments = split_tts_text(
                text,
                max_chars=split_max_chars,
                min_chars=6,
            )
            segments = split_first_segment_short(
                segments,
                first_max_chars=first_segment_max_chars,
                normal_max_chars=split_max_chars,
            )
        else:
            segments = [text.strip()] if text and text.strip() else []

        if not segments:
            return

        emit_latency(
            "tts_split_decided",
            trace,
            enabled=split_enabled,
            segment_count=len(segments),
            first_segment_chars=len(segments[0]),
            prefetch=split_prefetch,
            filler=filler_enabled,
        )

        if COSYVOICE_LOG_SPLIT_RESULT:
            logger.info(
                "cosy_voice split config: enabled=%s, max_chars=%d, first_max_chars=%d, prefetch=%s, override=%s",
                split_enabled,
                split_max_chars,
                first_segment_max_chars,
                split_prefetch,
                COSYVOICE_ALLOW_REQUEST_OVERRIDE,
            )
            logger.info(
                "cosy_voice split result: %d segments: %s",
                len(segments),
                segments,
            )

        def make_audio_stream(segment_text: str) -> Iterator[bytes]:
            if dialect == "minnan":
                return self.cosy_voice_minnan(
                    segment_text,
                    ref_file,
                    self.opt.TTS_SERVER,
                    trace=trace,
                )

            return self.cosy_voice(
                segment_text,
                ref_file,
                ref_text,
                language,
                self.opt.TTS_SERVER,
                trace=trace,
            )

        filler_chunks = None
        filler_text = COSYVOICE_FILLER_TEXT

        if filler_enabled and len(text.strip()) >= COSYVOICE_FILLER_MIN_TEXT_CHARS:
            filler_chunks = self.get_ready_filler_chunks(
                ref_file=ref_file,
                ref_text=ref_text,
                dialect=dialect,
                language=language,
                server_url=self.opt.TTS_SERVER,
            )

        try:
            if filler_chunks:
                self.stream_segments_with_optional_filler(
                    segments=segments,
                    make_audio_stream=make_audio_stream,
                    textevent=textevent,
                    full_text=text,
                    filler_text=filler_text,
                    filler_chunks=filler_chunks,
                    prefetch=split_prefetch,
                )
                return

            if len(segments) == 1 or not split_prefetch:
                for idx, segment in enumerate(segments):
                    self.stream_tts(
                        make_audio_stream(segment),
                        msg=(segment, textevent),
                        start_status=(idx == 0),
                        end_status=(idx == len(segments) - 1),
                        full_text=text,
                        segment_index=idx,
                        segment_count=len(segments),
                    )
                return

            self.stream_segmented_tts(
                segments=segments,
                make_audio_stream=make_audio_stream,
                textevent=textevent,
                full_text=text,
            )

        finally:
            # 当前回复结束后，后台预生成 filler。
            # 这样不会抢当前回答的首包，下一轮对话才能立刻使用。
            if filler_enabled:
                self.start_filler_preload_if_needed(
                    ref_file=ref_file,
                    ref_text=ref_text,
                    dialect=dialect,
                    language=language,
                    server_url=self.opt.TTS_SERVER,
                )

    def get_or_register_spk(
        self,
        reffile: str,
        reftext: str,
        server_url: str,
        trace: dict | None = None,
    ) -> str | None:
        """
        尝试把参考音频注册为 zero-shot speaker。
        成功后，后续请求只传 spk_id + text，不再重复上传 prompt_wav。

        如果 server.py 或当前 CosyVoice 模型不支持，则返回 None，
        调用方自动回退到 /inference_zero_shot。
        """
        self._ensure_spk_cache()

        cache_key = f"{reffile}|{reftext}"

        if cache_key in self._spk_id_cache:
            return self._spk_id_cache[cache_key]

        start = time.perf_counter()

        payload = {
            "prompt_text": reftext,
        }
        if trace and trace.get("trace_id"):
            payload["trace_id"] = trace["trace_id"]

        try:
            with open(reffile, "rb") as f:
                files = [
                    ("prompt_wav", ("prompt_wav.wav", f, "audio/wav")),
                ]

                res = requests.post(
                    f"{server_url}/register_zero_shot_spk",
                    data=payload,
                    files=files,
                    timeout=(5, 60),
                )

            end = time.perf_counter()
            logger.info("cosy_voice register speaker time: %.2fs", end - start)

            if res.status_code != 200:
                logger.warning(
                    "register speaker unavailable, fallback to upload mode. "
                    "status=%s, body=%s",
                    res.status_code,
                    res.text,
                )
                return None

            data = res.json()
            spk_id = data.get("spk_id")

            if not spk_id:
                logger.warning("register speaker response missing spk_id: %s", res.text)
                return None

            self._spk_id_cache[cache_key] = spk_id
            logger.info("cosy_voice speaker registered: %s", spk_id)

            return spk_id

        except Exception:
            logger.exception("register speaker error, fallback to upload mode:")
            return None

    def cosy_voice(
        self,
        text: str,
        reffile: str,
        reftext: str,
        language: str,
        server_url: str,
        trace: dict | None = None,
    ) -> Iterator[bytes]:
        """
        普通 zero-shot TTS。

        优先路径：
        - 第一次注册 speaker；
        - 后续只传 spk_id + text。

        回退路径：
        - 如果注册 speaker 不可用，仍然每次上传 prompt_wav；
        - 服务端有 wav path 缓存，且 inference stream=True。
        """
        start = time.perf_counter()

        try:
            spk_id = self.get_or_register_spk(
                reffile,
                reftext,
                server_url,
                trace=trace,
            )

            if spk_id:
                logger.info("cosy_voice using registered spk_id mode: %s", spk_id)

                payload = {
                    "tts_text": text,
                    "spk_id": spk_id,
                }
                if trace and trace.get("trace_id"):
                    payload["trace_id"] = trace["trace_id"]

                res = requests.post(
                    f"{server_url}/inference_zero_shot_by_spk",
                    data=payload,
                    stream=True,
                    timeout=(5, 120),
                )
            else:
                logger.info("cosy_voice using upload prompt_wav fallback mode")

                payload = {
                    "tts_text": text,
                    "prompt_text": reftext,
                }
                if trace and trace.get("trace_id"):
                    payload["trace_id"] = trace["trace_id"]

                with open(reffile, "rb") as f:
                    files = [
                        ("prompt_wav", ("prompt_wav.wav", f, "audio/wav")),
                    ]

                    res = requests.post(
                        f"{server_url}/inference_zero_shot",
                        data=payload,
                        files=files,
                        stream=True,
                        timeout=(5, 120),
                    )

            post_end = time.perf_counter()
            logger.info("cosy_voice Time to make POST: %.2fs, text=%s", post_end - start, text)
            emit_latency(
                "tts_http_posted",
                trace,
                duration_ms=(post_end - start) * 1000,
                text_chars=len(text),
                speaker_mode="registered" if spk_id else "upload",
                status_code=res.status_code,
            )

            if res.status_code != 200:
                logger.error("CosyVoice error %s: %s", res.status_code, res.text)
                emit_latency(
                    "tts_http_error",
                    trace,
                    status_code=res.status_code,
                    text_chars=len(text),
                )
                return

            first = True
            chunk_count = 0
            byte_count = 0

            for chunk in res.iter_content(chunk_size=COSYVOICE_HTTP_CHUNK_SIZE):
                if not chunk:
                    continue

                chunk_count += 1
                byte_count += len(chunk)

                if first:
                    first_end = time.perf_counter()
                    logger.info("cosy_voice Time to first chunk: %.2fs, text=%s", first_end - start, text)
                    logger.info("cosy_voice POST to first chunk: %.2fs, text=%s", first_end - post_end, text)
                    emit_latency(
                        "tts_http_first_chunk",
                        trace,
                        duration_ms=(first_end - start) * 1000,
                        post_to_first_ms=(first_end - post_end) * 1000,
                        text_chars=len(text),
                        speaker_mode="registered" if spk_id else "upload",
                    )
                    first = False

                if self.state == State.RUNNING:
                    yield chunk

            emit_latency(
                "tts_http_finished",
                trace,
                duration_ms=(time.perf_counter() - start) * 1000,
                chunks=chunk_count,
                bytes=byte_count,
                text_chars=len(text),
            )

        except Exception as exc:
            emit_latency("tts_http_error", trace, error=repr(exc), text_chars=len(text))
            logger.exception("cosyvoice error:")

    def cosy_voice_minnan(
        self,
        text: str,
        reffile: str,
        server_url: str,
        trace: dict | None = None,
    ) -> Iterator[bytes]:
        """
        闽南语合成，调用 /inference_minnan。
        这里仍然上传参考音频；服务端会缓存 wav path，并使用 stream=True。
        """
        start = time.perf_counter()

        payload = {
            "tts_text": text,
        }
        if trace and trace.get("trace_id"):
            payload["trace_id"] = trace["trace_id"]

        try:
            with open(reffile, "rb") as f:
                files = [
                    ("prompt_wav", ("prompt_wav.wav", f, "audio/wav")),
                ]

                res = requests.post(
                    f"{server_url}/inference_minnan",
                    data=payload,
                    files=files,
                    stream=True,
                    timeout=(5, 120),
                )

            post_end = time.perf_counter()
            logger.info("cosy_voice_minnan Time to make POST: %.2fs, text=%s", post_end - start, text)
            emit_latency(
                "tts_http_posted",
                trace,
                duration_ms=(post_end - start) * 1000,
                text_chars=len(text),
                speaker_mode="minnan_upload",
                status_code=res.status_code,
            )

            if res.status_code != 200:
                logger.error("CosyVoice minnan error %s: %s", res.status_code, res.text)
                return

            first = True
            chunk_count = 0
            byte_count = 0

            for chunk in res.iter_content(chunk_size=COSYVOICE_HTTP_CHUNK_SIZE):
                if not chunk:
                    continue

                chunk_count += 1
                byte_count += len(chunk)

                if first:
                    first_end = time.perf_counter()
                    logger.info(
                        "cosy_voice_minnan Time to first chunk: %.2fs, text=%s",
                        first_end - start,
                        text,
                    )
                    logger.info(
                        "cosy_voice_minnan POST to first chunk: %.2fs, text=%s",
                        first_end - post_end,
                        text,
                    )
                    emit_latency(
                        "tts_http_first_chunk",
                        trace,
                        duration_ms=(first_end - start) * 1000,
                        post_to_first_ms=(first_end - post_end) * 1000,
                        text_chars=len(text),
                        speaker_mode="minnan_upload",
                    )
                    first = False

                if self.state == State.RUNNING:
                    yield chunk

            emit_latency(
                "tts_http_finished",
                trace,
                duration_ms=(time.perf_counter() - start) * 1000,
                chunks=chunk_count,
                bytes=byte_count,
                text_chars=len(text),
                speaker_mode="minnan_upload",
            )

        except Exception as exc:
            emit_latency("tts_http_error", trace, error=repr(exc), text_chars=len(text))
            logger.exception("cosyvoice minnan error:")

    def filler_cache_key(
        self,
        ref_file: str,
        ref_text: str,
        dialect: str,
        language: str,
    ) -> str:
        return f"{ref_file}|{ref_text}|{dialect}|{language}|{COSYVOICE_FILLER_TEXT}"

    def get_ready_filler_chunks(
        self,
        ref_file: str,
        ref_text: str,
        dialect: str,
        language: str,
        server_url: str,
    ) -> list[bytes] | None:
        self._ensure_filler_cache()

        key = self.filler_cache_key(
            ref_file=ref_file,
            ref_text=ref_text,
            dialect=dialect,
            language=language,
        )

        item = self._filler_cache.get(key)

        if not item:
            if COSYVOICE_LOG_FILLER:
                logger.info("cosy_voice filler not ready yet")
            return None

        if item.get("status") != "ready":
            if COSYVOICE_LOG_FILLER:
                logger.info("cosy_voice filler status=%s", item.get("status"))
            return None

        chunks = item.get("chunks") or []

        if not chunks:
            return None

        if COSYVOICE_LOG_FILLER:
            logger.info(
                "cosy_voice using cached filler: text=%s, chunks=%d",
                item.get("text"),
                len(chunks),
            )

        return list(chunks)

    def start_filler_preload_if_needed(
        self,
        ref_file: str,
        ref_text: str,
        dialect: str,
        language: str,
        server_url: str,
    ):
        """
        后台预生成 filler 音频。

        设计：
        - 不在当前请求开始时生成，避免和当前正式回答抢首包；
        - 当前回答结束后启动后台预生成；
        - 下一轮对话可立即播放缓存 filler。
        """
        self._ensure_filler_cache()

        key = self.filler_cache_key(
            ref_file=ref_file,
            ref_text=ref_text,
            dialect=dialect,
            language=language,
        )

        item = self._filler_cache.get(key)

        if item and item.get("status") in ("loading", "ready"):
            return

        self._filler_cache[key] = {
            "status": "loading",
            "chunks": [],
            "text": COSYVOICE_FILLER_TEXT,
        }

        def worker():
            chunks: list[bytes] = []

            try:
                if COSYVOICE_LOG_FILLER:
                    logger.info("cosy_voice preload filler start: %s", COSYVOICE_FILLER_TEXT)

                if dialect == "minnan":
                    audio_iter = self.cosy_voice_minnan(
                        COSYVOICE_FILLER_TEXT,
                        ref_file,
                        server_url,
                    )
                else:
                    audio_iter = self.cosy_voice(
                        COSYVOICE_FILLER_TEXT,
                        ref_file,
                        ref_text,
                        language,
                        server_url,
                    )

                for chunk in audio_iter:
                    if chunk:
                        chunks.append(chunk)

                self._filler_cache[key] = {
                    "status": "ready",
                    "chunks": chunks,
                    "text": COSYVOICE_FILLER_TEXT,
                }

                if COSYVOICE_LOG_FILLER:
                    logger.info(
                        "cosy_voice preload filler done: chunks=%d",
                        len(chunks),
                    )

            except Exception:
                logger.exception("cosy_voice preload filler failed")
                self._filler_cache[key] = {
                    "status": "error",
                    "chunks": [],
                    "text": COSYVOICE_FILLER_TEXT,
                }

        thread = threading.Thread(
            target=worker,
            name="cosyvoice-filler-preload",
            daemon=True,
        )
        thread.start()

    def _start_audio_worker(
        self,
        audio_stream_factory: Callable[[], Iterator[bytes]],
        name: str,
    ) -> queue.Queue:
        """
        后台拉取某一段 TTS 的 HTTP 流，放入队列。
        用于“当前段播放时，下一段提前合成”。
        """
        q: queue.Queue = queue.Queue(maxsize=60)

        def worker():
            try:
                for chunk in audio_stream_factory():
                    if self.state != State.RUNNING:
                        break
                    q.put(chunk)
            except Exception as e:
                q.put(e)
            finally:
                q.put(_QUEUE_END)

        thread = threading.Thread(
            target=worker,
            name=name,
            daemon=True,
        )
        thread.start()

        return q

    def _queue_to_audio_iter(self, q: queue.Queue) -> Iterator[bytes]:
        while True:
            item = q.get()

            if item is _QUEUE_END:
                break

            if isinstance(item, Exception):
                raise item

            yield item

    def _iter_with_first_chunk_callback(
        self,
        audio_iter: Iterator[bytes],
        callback: Callable[[], None],
    ) -> Iterator[bytes]:
        first = True

        for chunk in audio_iter:
            if first:
                first = False
                callback()
            yield chunk

    def stream_segmented_tts(
        self,
        segments: list[str],
        make_audio_stream: Callable[[str], Iterator[bytes]],
        textevent: dict,
        full_text: str,
    ):
        """
        多段 TTS 播放。

        当前段收到第一个 chunk 后，后台启动下一段。
        """
        workers: dict[int, queue.Queue] = {}

        def start_worker(index: int):
            if index >= len(segments):
                return

            if index in workers:
                return

            segment_text = segments[index]

            logger.info(
                "cosy_voice prefetch start segment %d/%d: %s",
                index + 1,
                len(segments),
                segment_text,
            )

            workers[index] = self._start_audio_worker(
                audio_stream_factory=lambda s=segment_text: make_audio_stream(s),
                name=f"cosyvoice-segment-{index}",
            )

        start_worker(0)

        for idx, segment in enumerate(segments):
            start_worker(idx)

            audio_iter = self._queue_to_audio_iter(workers[idx])

            def on_first_chunk(next_index=idx + 1):
                start_worker(next_index)

            audio_iter = self._iter_with_first_chunk_callback(
                audio_iter,
                callback=on_first_chunk,
            )

            self.stream_tts(
                audio_iter,
                msg=(segment, textevent),
                start_status=(idx == 0),
                end_status=(idx == len(segments) - 1),
                full_text=full_text,
                segment_index=idx,
                segment_count=len(segments),
            )

            start_worker(idx + 1)

    def stream_segments_with_optional_filler(
        self,
        segments: list[str],
        make_audio_stream: Callable[[str], Iterator[bytes]],
        textevent: dict,
        full_text: str,
        filler_text: str,
        filler_chunks: list[bytes],
        prefetch: bool,
    ):
        """
        播放缓存 filler，同时后台开始合成真实第一段。

        这样可以把一部分“真实回答首包等待”隐藏在 filler 播放期间。
        """
        workers: dict[int, queue.Queue] = {}

        def start_worker(index: int):
            if index >= len(segments):
                return

            if index in workers:
                return

            segment_text = segments[index]

            logger.info(
                "cosy_voice real segment worker start %d/%d: %s",
                index + 1,
                len(segments),
                segment_text,
            )

            workers[index] = self._start_audio_worker(
                audio_stream_factory=lambda s=segment_text: make_audio_stream(s),
                name=f"cosyvoice-real-segment-{index}",
            )

        # 先启动真实第一段合成
        start_worker(0)

        # 立即播放缓存 filler
        self.stream_tts(
            iter(filler_chunks),
            msg=(filler_text, textevent),
            start_status=True,
            end_status=False,
            full_text=full_text,
            segment_index=-1,
            segment_count=len(segments),
        )

        # filler 播放完后，继续播放真实回答
        for idx, segment in enumerate(segments):
            start_worker(idx)

            audio_iter = self._queue_to_audio_iter(workers[idx])

            if prefetch:
                def on_first_chunk(next_index=idx + 1):
                    start_worker(next_index)

                audio_iter = self._iter_with_first_chunk_callback(
                    audio_iter,
                    callback=on_first_chunk,
                )

            self.stream_tts(
                audio_iter,
                msg=(segment, textevent),
                start_status=False,
                end_status=(idx == len(segments) - 1),
                full_text=full_text,
                segment_index=idx,
                segment_count=len(segments),
            )

            if prefetch:
                start_worker(idx + 1)

    def stream_tts(
        self,
        audio_stream: Iterator[bytes],
        msg: tuple[str, dict],
        start_status: bool = True,
        end_status: bool = True,
        full_text: str | None = None,
        segment_index: int = 0,
        segment_count: int = 1,
    ):
        """
        将服务端返回的 24kHz int16 PCM bytes 转为 LiveTalking 需要的 float32 audio frame。

        改动：
        - 增加 audio_buffer，避免小 chunk 被丢弃；
        - 使用一个有状态重采样器贯穿整个 HTTP 响应，避免分块接缝；
        - 缓存落在 HTTP 分块末尾的单个字节，避免 int16 样本错位；
        - 分段时只在第一段发 start，最后一段发 end；
        - eventpoint 里附带 segment 信息，方便日志或字幕调试。
        """
        segment_text, textevent = msg
        textevent = textevent or {}
        trace = get_trace(textevent)
        stream_started = time.perf_counter()
        dequeued_at = textevent.get("_tts_dequeued_monotonic")

        if full_text is None:
            full_text = segment_text

        first_frame_of_this_segment = True
        audio_buffer = np.zeros(0, dtype=np.float32)
        emitted_frames = 0

        def log_first_pcm():
            nonlocal first_frame_of_this_segment
            if not first_frame_of_this_segment:
                return
            now = time.perf_counter()
            duration_ms = None
            if isinstance(dequeued_at, (int, float)):
                duration_ms = (now - dequeued_at) * 1000
            emit_latency(
                "tts_first_pcm_enqueued",
                trace,
                duration_ms=duration_ms,
                stream_ms=(now - stream_started) * 1000,
                segment_index=segment_index,
                segment_count=segment_count,
                text_chars=len(segment_text),
            )

        for stream in _iter_resampled_pcm(
            audio_stream,
            source_rate=COSYVOICE_SERVER_SAMPLE_RATE,
            target_rate=self.sample_rate,
        ):
            audio_buffer = np.concatenate([audio_buffer, stream])

            while audio_buffer.shape[0] >= self.chunk:
                frame = audio_buffer[: self.chunk]
                audio_buffer = audio_buffer[self.chunk :]

                eventpoint = {}

                if start_status and first_frame_of_this_segment:
                    eventpoint = {
                        "status": "start",
                        "text": full_text,
                        "segment_text": segment_text,
                        "segment_index": segment_index,
                        "segment_count": segment_count,
                    }

                log_first_pcm()
                first_frame_of_this_segment = False
                eventpoint.update(**textevent)

                self.parent.put_audio_frame(
                    frame,
                    eventpoint,
                )
                emitted_frames += 1

        # 处理最后不足一帧的尾巴。
        if audio_buffer.shape[0] > 0:
            pad_len = self.chunk - audio_buffer.shape[0]
            frame = np.pad(audio_buffer, (0, pad_len)).astype(np.float32)

            eventpoint = {}

            if start_status and first_frame_of_this_segment:
                eventpoint = {
                    "status": "start",
                    "text": full_text,
                    "segment_text": segment_text,
                    "segment_index": segment_index,
                    "segment_count": segment_count,
                }

            log_first_pcm()
            first_frame_of_this_segment = False
            eventpoint.update(**textevent)

            self.parent.put_audio_frame(
                frame,
                eventpoint,
            )
            emitted_frames += 1

        if end_status:
            eventpoint = {
                "status": "end",
                "text": full_text,
                "segment_text": segment_text,
                "segment_index": segment_index,
                "segment_count": segment_count,
            }
            eventpoint.update(**textevent)

            self.parent.put_audio_frame(
                np.zeros(self.chunk, np.float32),
                eventpoint,
            )

        emit_latency(
            "tts_pcm_segment_finished",
            trace,
            duration_ms=(time.perf_counter() - stream_started) * 1000,
            segment_index=segment_index,
            segment_count=segment_count,
            emitted_frames=emitted_frames,
            audio_duration_ms=emitted_frames * 20,
            text_chars=len(segment_text),
        )
