import time
import os
import threading
from typing import TYPE_CHECKING, List, Dict
if TYPE_CHECKING:
    from avatars.base_avatar import BaseAvatar
from utils.logger import logger
from utils.latency import emit_latency, ensure_trace

# ── 全局对话历史存储 ──────────────────────────────────────────
# key: sessionid (str), value: list of {"role": ..., "content": ...}
_conversation_history: Dict[str, List[dict]] = {}

# 存储每个 session 最新的完整 LLM 回复，供前端轮询
_pending_replies: Dict[str, str] = {}

def get_pending_reply(sessionid: str) -> str:
    """取出并清空该 session 的待显示回复，取完即删"""
    return _pending_replies.pop(sessionid, '')

# ── 可配置参数 ────────────────────────────────────────────────
MAX_HISTORY_TURNS   = 10    # 最多保留几轮对话（1轮 = user + assistant）
MAX_HISTORY_CHARS   = 3000  # 历史内容字符总数上限，超出则从最早一轮开始删除
DEFAULT_SYSTEM_PROMPT = "你是一个知识助手，尽量以简短、口语化的方式回答问题。"

# ── Ollama OpenAI 兼容客户端 ──────────────────────────────────
# 客户端内部维护 HTTP 连接池，应在进程内复用，避免每轮对话都重新创建。
# 使用懒加载可保持模块导入轻量；加锁避免多个会话首次请求时重复初始化。
_ollama_client = None
_ollama_client_lock = threading.Lock()


def _get_ollama_client():
    global _ollama_client

    if _ollama_client is None:
        with _ollama_client_lock:
            if _ollama_client is None:
                from openai import OpenAI

                _ollama_client = OpenAI(
                    api_key='ollama',
                    base_url='http://127.0.0.1:11434/v1',
                )
                logger.info("Ollama OpenAI client initialized")

    return _ollama_client


def get_history(sessionid: str) -> List[dict]:
    """获取指定 session 的对话历史，不存在则初始化"""
    if sessionid not in _conversation_history:
        _conversation_history[sessionid] = []
    return _conversation_history[sessionid]


def trim_history(history: List[dict]):
    """
    裁剪历史，保证不超过 MAX_HISTORY_TURNS 轮和 MAX_HISTORY_CHARS 字符。
    策略：从最早的一轮（user+assistant 各一条）开始删除。
    """
    # 按轮数裁剪（每轮2条：user + assistant）
    while len(history) > MAX_HISTORY_TURNS * 2:
        history.pop(0)  # 删最早的 user
        if history and history[0]['role'] == 'assistant':
            history.pop(0)  # 删对应的 assistant

    # 按字符数裁剪
    while history:
        total = sum(len(m['content']) for m in history)
        if total <= MAX_HISTORY_CHARS:
            break
        history.pop(0)
        if history and history[0]['role'] == 'assistant':
            history.pop(0)


def clear_history(sessionid: str):
    """清空指定 session 的对话历史（可在路由里暴露为接口）"""
    _conversation_history.pop(sessionid, None)
    logger.info(f"History cleared for session {sessionid}")


def llm_response(message, avatar_session: 'BaseAvatar', datainfo: dict = {}):
    trace = None
    try:
        opt = avatar_session.opt
        sessionid = str(getattr(opt, 'sessionid', '0'))
        datainfo, trace = ensure_trace(datainfo, sessionid, "llm_direct")

        # 取出该 session 的历史
        history = get_history(sessionid)

        # 追加本轮用户消息
        history.append({'role': 'user', 'content': message})

        start = time.perf_counter()
        dispatched_at = datainfo.get("_llm_dispatched_monotonic")
        dispatch_wait_ms = None
        if isinstance(dispatched_at, (int, float)):
            dispatch_wait_ms = (start - dispatched_at) * 1000
        emit_latency(
            "llm_started",
            trace,
            dispatch_wait_ms=dispatch_wait_ms,
            history_messages=len(history) - 1,
            input_chars=len(message),
            model="qwen2.5:7b",
        )
        client = _get_ollama_client()

        # 读取可自定义的 system prompt（opt 上有就用，否则用默认值）
        system_prompt = getattr(opt, 'system_prompt', DEFAULT_SYSTEM_PROMPT)

        # 构建完整 messages：system + 历史 + 本轮
        messages = [{'role': 'system', 'content': system_prompt}] + history

        logger.info(f"LLM call | session={sessionid} | history={len(history)-1}条 | msg={message}")

        completion = client.chat.completions.create(
            model="qwen2.5:7b",                                             # 服务器模型名
            messages=messages,
            stream=True,
            stream_options={"include_usage": True}
        )

        # 收集完整的 assistant 回复，用于写入历史
        full_reply = ""
        result = ""
        first = True
        tts_chunk_count = 0
        first_tts_enqueued = False
        prompt_tokens = None
        completion_tokens = None

        for chunk in completion:
            usage = getattr(chunk, "usage", None)
            if usage is not None:
                prompt_tokens = getattr(usage, "prompt_tokens", prompt_tokens)
                completion_tokens = getattr(usage, "completion_tokens", completion_tokens)
            if len(chunk.choices) > 0:
                if first:
                    end = time.perf_counter()
                    logger.info(f"LLM Time to first chunk: {end-start:.2f}s")
                    emit_latency(
                        "llm_first_token",
                        trace,
                        duration_ms=(end - start) * 1000,
                    )
                    first = False

                msg = chunk.choices[0].delta.content
                if msg is None:
                    continue

                full_reply += msg  # 累积完整回复

                # 原有的标点切片逻辑，保证 TTS 低延迟
                lastpos = 0
                for i, char in enumerate(msg):
                    if char in ",.!;:，。！？：；":
                        result = result + msg[lastpos:i+1]
                        lastpos = i+1
                        if len(result) > 10:
                            logger.info(f"TTS chunk: {result}")
                            tts_chunk_count += 1
                            enqueue_time = time.perf_counter()
                            emit_latency(
                                "llm_tts_chunk_enqueued",
                                trace,
                                duration_ms=(enqueue_time - start) * 1000,
                                chunk_index=tts_chunk_count,
                                text_chars=len(result),
                            )
                            if not first_tts_enqueued:
                                emit_latency(
                                    "llm_tts_first_enqueue",
                                    trace,
                                    duration_ms=(enqueue_time - start) * 1000,
                                    text_chars=len(result),
                                )
                                first_tts_enqueued = True
                            avatar_session.put_msg_txt(result, datainfo)
                            result = ""
                result = result + msg[lastpos:]

        end = time.perf_counter()
        logger.info(f"LLM Time to last chunk: {end-start:.2f}s")

        if result:
            tts_chunk_count += 1
            enqueue_time = time.perf_counter()
            emit_latency(
                "llm_tts_chunk_enqueued",
                trace,
                duration_ms=(enqueue_time - start) * 1000,
                chunk_index=tts_chunk_count,
                text_chars=len(result),
            )
            if not first_tts_enqueued:
                emit_latency(
                    "llm_tts_first_enqueue",
                    trace,
                    duration_ms=(enqueue_time - start) * 1000,
                    text_chars=len(result),
                )
            avatar_session.put_msg_txt(result, datainfo)

        emit_latency(
            "llm_finished",
            trace,
            duration_ms=(end - start) * 1000,
            output_chars=len(full_reply),
            tts_chunks=tts_chunk_count,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
        )

        # ── 将 assistant 完整回复写入历史，然后裁剪 ──
        if full_reply:
            history.append({'role': 'assistant', 'content': full_reply})
            trim_history(history)
            logger.info(f"History updated | session={sessionid} | 现有{len(history)//2}轮")
            _pending_replies[sessionid] = full_reply  # 存给前端轮询

    except Exception as e:
        emit_latency("llm_error", trace, error=repr(e))
        logger.exception('llm exception:')
        # 出错时把刚才加入的 user 消息从历史里移除，避免历史污染
        history = get_history(str(getattr(avatar_session.opt, 'sessionid', '0')))
        if history and history[-1]['role'] == 'user':
            history.pop()
