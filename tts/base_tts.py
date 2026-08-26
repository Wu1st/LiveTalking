from threading import Thread
import queue
import time
from queue import Queue
from io import BytesIO
from enum import Enum

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from avatars.base_avatar import BaseAvatar

from utils.logger import logger
from utils.latency import emit_latency, ensure_trace, get_trace

class State(Enum):
    RUNNING = 0
    PAUSE = 1

class BaseTTS:
    def __init__(self, opt, parent: "BaseAvatar"):
        self.opt = opt
        self.parent = parent

        #self.fps = opt.fps # 20 ms per frame
        self.sample_rate = 16000
        self.chunk = self.sample_rate // (opt.fps*2) # 320 samples per chunk (20ms * 16000 / 1000)
        self.input_stream = BytesIO()

        self.msgqueue = Queue()
        self.state = State.RUNNING
        self.current_text_chars = 0
        self.current_request_started = None

    def flush_talk(self):
        self.msgqueue.queue.clear()
        self.state = State.PAUSE

    def put_msg_txt(self, msg: str, datainfo: dict = {}): 
        if len(msg) > 0:
            sessionid = str(getattr(getattr(self.parent, "opt", None), "sessionid", "0"))
            item_datainfo, trace = ensure_trace(datainfo, sessionid, "tts_direct")
            item_datainfo["_tts_enqueued_monotonic"] = time.perf_counter()
            self.msgqueue.put((msg, item_datainfo))
            emit_latency(
                "tts_enqueued",
                trace,
                text_chars=len(msg),
                queue_size=self.msgqueue.qsize(),
            )

    def render(self, quit_event):
        process_thread = Thread(target=self.process_tts, args=(quit_event,))
        process_thread.start()
    
    def process_tts(self, quit_event):        
        while not quit_event.is_set():
            try:
                msg: tuple[str, dict] = self.msgqueue.get(block=True, timeout=1)
                self.state = State.RUNNING
            except queue.Empty:
                continue
            trace = get_trace(msg[1])
            enqueued_at = msg[1].get("_tts_enqueued_monotonic")
            queue_ms = None
            if isinstance(enqueued_at, (int, float)):
                queue_ms = (time.perf_counter() - enqueued_at) * 1000
            msg[1]["_tts_dequeued_monotonic"] = time.perf_counter()
            emit_latency(
                "tts_dequeued",
                trace,
                queue_ms=queue_ms,
                text_chars=len(msg[0]),
                queue_size=self.msgqueue.qsize(),
            )
            self.current_text_chars = len(msg[0])
            self.current_request_started = time.perf_counter()
            try:
                self.txt_to_audio(msg)
            finally:
                self.current_text_chars = 0
                self.current_request_started = None
        self.stop_tts()
        logger.info('ttsreal thread stop')
    
    def txt_to_audio(self, msg: tuple[str, dict]):
        pass

    def stop_tts(self):
        pass
