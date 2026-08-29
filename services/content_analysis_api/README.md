# Multimodal Conversation Content Analysis API

This service completes work item 4 by orchestrating already-deployed services:

- LiveTalking `/transcribe_audio`: Qwen3-ASR, BEATs and optional FunASR/CAM++ diarization;
- emotion2vec `/predict`;
- Ollama `qwen2.5:7b` for evidence-grounded content analysis.

It does not load another GPU model.

## Endpoints

- `GET /health/live`
- `GET /health/ready`
- `GET /v1/model/info`
- `POST /v1/conversation/analyze`: analyze supplied structured observations.
- `POST /v1/audio/analyze`: run the full audio-to-structured-to-Ollama chain.

Protected endpoints accept `X-API-Key` or `Authorization: Bearer ...`. The
deployed service generates a mode-0600 key file on first start when no key is
supplied by the environment.

The deployment listens on port `18081`. If the vendor firewall does not expose
that port, use an SSH tunnel until network access is approved:

```bash
ssh -p SSH_PORT -L 18081:127.0.0.1:18081 USER@DEPLOY_HOST
```

Then call `http://127.0.0.1:18081`. On the deployment host the generated key is
stored at `/home/hf/.config/content-analysis-api.key` with mode `0600`; it is
not committed to Git.

## Audio example

```bash
curl -X POST http://SERVER:18081/v1/audio/analyze \
  -H "X-API-Key: $CONTENT_ANALYSIS_API_KEY" \
  -F "file=@conversation.wav" \
  -F "question=梳理内容如何推进，并分析Speaker之间的观点和关系" \
  -F "tasks=digest,summary,qa,relationships,speakers,emotion,scene" \
  -F "content_type_hint=auto" \
  -F 'speaker_aliases={"Speaker_00":"张三","Speaker_01":"李四"}'
```

`content_digest` is not a fixed meeting template. It classifies the content as
meeting, interview, customer service, lecture, daily conversation or other,
then returns an evidence-linked overview, topic progression and speaker
contributions. Decisions and action items are kept empty for non-meeting
content unless the transcript explicitly contains them.

`emotion.scope=whole_audio` means the emotion result must not be attributed to
an individual speaker. When Qwen3-ASR text differs from the diarization
pipeline's segmented text, both values and `transcript_conflict=true` are sent
to Ollama.
