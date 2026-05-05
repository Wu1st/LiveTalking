import time
import os
from typing import TYPE_CHECKING, List, Dict
if TYPE_CHECKING:
    from avatars.base_avatar import BaseAvatar
from utils.logger import logger

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
    try:
        opt = avatar_session.opt
        sessionid = str(getattr(opt, 'sessionid', '0'))

        # 取出该 session 的历史
        history = get_history(sessionid)

        # 追加本轮用户消息
        history.append({'role': 'user', 'content': message})

        start = time.perf_counter()
        from openai import OpenAI
        client = OpenAI(

            api_key='ollama', 
                             
            base_url='http://113.98.61.52:11434/v1',                        #服务器
            #base_url="https://0f5a89e0fdf0.ngrok-free.app/v1",

            #base_url='http://localhost:11434/v1',                          #本地ollama  
        )

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

        for chunk in completion:
            if len(chunk.choices) > 0:
                if first:
                    end = time.perf_counter()
                    logger.info(f"LLM Time to first chunk: {end-start:.2f}s")
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
                            avatar_session.put_msg_txt(result, datainfo)
                            result = ""
                result = result + msg[lastpos:]

        end = time.perf_counter()
        logger.info(f"LLM Time to last chunk: {end-start:.2f}s")

        if result:
            avatar_session.put_msg_txt(result, datainfo)

        # ── 将 assistant 完整回复写入历史，然后裁剪 ──
        if full_reply:
            history.append({'role': 'assistant', 'content': full_reply})
            trim_history(history)
            logger.info(f"History updated | session={sessionid} | 现有{len(history)//2}轮")
            _pending_replies[sessionid] = full_reply  # 存给前端轮询

    except Exception as e:
        logger.exception('llm exception:')
        # 出错时把刚才加入的 user 消息从历史里移除，避免历史污染
        history = get_history(str(getattr(avatar_session.opt, 'sessionid', '0')))
        if history and history[-1]['role'] == 'user':
            history.pop()