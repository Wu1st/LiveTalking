(function () {
  'use strict';

  const TARGET_PATHS = new Set(['/humanaudio', '/humanaudio_monitor', '/transcribe_audio']);
  const nativeFetch = window.fetch.bind(window);
  let statusPill = null;

  function requestPath(input) {
    try {
      const url = typeof input === 'string' ? input : input.url;
      return new URL(url, window.location.href).pathname;
    } catch (_) {
      return '';
    }
  }

  function frontendFile(form) {
    const value = form && form.get('file');
    return value instanceof Blob ? value : null;
  }

  function copyForm(source) {
    const target = new FormData();
    for (const [key, value] of source.entries()) {
      if (value instanceof File) target.append(key, value, value.name);
      else target.append(key, value);
    }
    return target;
  }

  function setStatus(state, text) {
    if (!statusPill) return;
    const dot = statusPill.querySelector('.status-dot');
    const label = statusPill.querySelector('span:last-child');
    dot.classList.remove('connected', 'connecting');
    if (state === 'ready') dot.classList.add('connected');
    else if (state === 'busy') dot.classList.add('connecting');
    label.textContent = text;
    statusPill.title = text;
  }

  async function prepareAudio(form) {
    const preparedForm = copyForm(form);
    preparedForm.set('output', 'speech');
    preparedForm.set('gap_ms', '160');
    const response = await nativeFetch('/api/speech_frontend/prepare', {
      method: 'POST',
      body: preparedForm
    });
    if (response.status === 204) return null;
    const contentType = (response.headers.get('Content-Type') || '').toLowerCase();
    if (!response.ok || !contentType.startsWith('audio/wav')) {
      let message = '语音前端处理失败';
      try {
        const payload = await response.json();
        message = payload.msg || message;
      } catch (_) {}
      throw new Error(message);
    }

    const buffer = await response.arrayBuffer();
    const bytes = new Uint8Array(buffer);
    const isWav = bytes.length >= 12
      && bytes[0] === 0x52 && bytes[1] === 0x49
      && bytes[2] === 0x46 && bytes[3] === 0x46
      && bytes[8] === 0x57 && bytes[9] === 0x41
      && bytes[10] === 0x56 && bytes[11] === 0x45;
    if (!isWav) throw new Error('语音前端返回的不是有效 WAV 音频');
    return new Blob([buffer], { type: 'audio/wav' });
  }

  function emptySpeechResponse() {
    return new Response(JSON.stringify({ code: -2, msg: '未检测到有效语音', data: null }), {
      status: 200,
      headers: { 'Content-Type': 'application/json; charset=utf-8' }
    });
  }

  window.fetch = async function speechFrontendFetch(input, options) {
    const path = requestPath(input);
    const body = options && options.body;
    if (!TARGET_PATHS.has(path) || !(body instanceof FormData) || !frontendFile(body)) {
      return nativeFetch(input, options);
    }

    setStatus('busy', '语音前端处理中');
    try {
      const prepared = await prepareAudio(body);
      if (!prepared) {
        setStatus('ready', '未检测到有效语音');
        return emptySpeechResponse();
      }
      const forwarded = copyForm(body);
      forwarded.set('file', prepared, 'speech_frontend.wav');
      const nextOptions = Object.assign({}, options, { body: forwarded });
      const response = await nativeFetch(input, nextOptions);
      setStatus('ready', '语音前端已就绪');
      return response;
    } catch (error) {
      console.warn('语音前端暂不可用，使用原始音频继续处理:', error);
      setStatus('error', '语音前端已降级');
      return nativeFetch(input, options);
    }
  };

  async function checkHealth() {
    try {
      const response = await nativeFetch('/api/speech_frontend/health');
      const payload = await response.json();
      if (response.ok && payload.code === 0) setStatus('ready', '语音前端已就绪');
      else setStatus('error', '语音前端不可用');
    } catch (_) {
      setStatus('error', '语音前端不可用');
    }
  }

  function mountStatus() {
    const controls = document.querySelector('.topbar-controls');
    if (!controls || document.getElementById('speech-frontend-status')) return;
    statusPill = document.createElement('div');
    statusPill.id = 'speech-frontend-status';
    statusPill.className = 'connection-pill';
    statusPill.innerHTML = '<span class="status-dot connecting"></span><span>语音前端连接中</span>';
    controls.prepend(statusPill);
    checkHealth();
  }

  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', mountStatus);
  else mountStatus();
})();
