# OCR 与翻译统一 API

LiveTalking 在 `8010` 端口提供 OCR—翻译文档 API；实际 OCR、翻译模型和任务持久化由独立文档服务负责。这样数字人、语音和文档使用同一个公网入口，同时不会把 GPU 模型或 600 页任务状态复制进 LiveTalking 进程。

## 配置

启动 LiveTalking 前设置文档服务地址：

```bash
export OCR_DOCUMENT_API_URL=http://127.0.0.1:8000
export OCR_DOCUMENT_PROXY_TIMEOUT_SECONDS=3600
export LIVETALKING_MAX_UPLOAD_MB=1024
```

文档服务应将 `LIVETALKING_BASE_URL` 设置为 `http://127.0.0.1:8010`，这样完成的 OCR 或译文可发布给当前 LiveTalking 会话。

## 一次上传，先 OCR，再按需翻译

只做 OCR：

```bash
curl -F file=@document.pdf \
  'http://127.0.0.1:8010/api/document/jobs?translate=false'
```

OCR 后英译中：

```bash
curl -F file=@document.pdf \
  'http://127.0.0.1:8010/api/document/jobs?translate=true&source_language=en&target_language=zh'
```

日译中使用 `source_language=ja&target_language=zh`。接口立即返回 `202` 和任务 ID，客户端随后轮询，而不是保持 600 页上传请求一直阻塞。

## 状态、结果与控制

```text
GET  /api/document/health
POST /api/document/image
POST /api/document/jobs
GET  /api/document/jobs/{job_id}
GET  /api/document/jobs/{job_id}/result
GET  /api/document/jobs/{job_id}/download?format=json|txt|bilingual_json|source_txt|translated_txt|bilingual_txt|audit_json|manifest_json
POST /api/document/jobs/{job_id}/cancel
POST /api/document/jobs/{job_id}/resume
```

已有 OCR 任务无需重新识别，可直接创建翻译任务：

```text
POST /api/document/translation-jobs
GET  /api/document/translation-jobs/{job_id}
GET  /api/document/translation-jobs/{job_id}/result
GET  /api/document/translation-jobs/{job_id}/download
POST /api/document/translation-jobs/{job_id}/cancel
POST /api/document/translation-jobs/{job_id}/resume
```

上述接口也以原始 `/v1/document/*` 和 `/v1/translation/document-*` 路径暴露，现有 OCR SDK 只需把 base URL 改成 LiveTalking 的 `8010` 地址。

## 交给数字人播报

任务完成后调用：

```bash
curl -X POST -H 'Content-Type: application/json' \
  -d '{"sessionid":"SESSION_ID","content":"translated","mode":"echo"}' \
  http://127.0.0.1:8010/api/document/jobs/JOB_ID/livetalking
```

`content` 可取 `auto`、`source`、`translated` 或 `bilingual`。`mode=echo` 会忠实播报文本；`mode=chat` 会把文本交给对话模型处理。

## 部署检查

```bash
curl http://127.0.0.1:8000/health
curl http://127.0.0.1:8010/api/document/health
```

两个请求都成功后，前端只需访问 `8010`。代理会流式转发上传与下载，不会在 LiveTalking 内存中缓存整份 600 页文档。
