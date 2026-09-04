import asyncio
import os
from types import SimpleNamespace

os.environ.setdefault("CONTENT_ANALYSIS_REQUIRE_API_KEY", "0")

from fastapi.testclient import TestClient
import httpx

import app as app_module
from app import (
    StructuredAnalysisRequest,
    _build_analysis_plan,
    _ollama_analyze,
    app,
    apply_grounding_guardrails,
    build_structured_input,
)
import dialogue_understanding as dialogue_module
from dialogue_understanding import (
    _fallback,
    _minimum_fact_count,
    _normalize_ledger,
    _prose_safety_errors,
    _requested_fact_count,
    _resolve_facts,
    _valid_ledger,
)


class _FakeResponse:
    def __init__(self, payload, status_code=200):
        self._payload = payload
        self.status_code = status_code

    def raise_for_status(self):
        if self.status_code >= 400:
            raise httpx.HTTPStatusError(
                "fake upstream error",
                request=httpx.Request("POST", "http://test"),
                response=httpx.Response(self.status_code),
            )

    def json(self):
        return self._payload


class _FakeAudioClient:
    def __init__(self, fail_task=None):
        self.calls = []
        self.fail_task = fail_task

    async def post(self, url, **kwargs):
        if url.endswith("/transcribe_audio"):
            task = "asr"
            payload = {"code": 0, "data": {"recognized": "你好", "duration": 2.0}}
        elif url.endswith("/diarize_audio"):
            task = "diarization"
            payload = {
                "code": 0,
                "data": {
                    "speaker_count": 1,
                    "segments": [
                        {"speaker": "Speaker_00", "start": 0.0, "end": 2.0, "text": "你好"}
                    ],
                },
            }
        elif url.endswith("/analyze_environment"):
            task = "acoustic"
            payload = {
                "code": 0,
                "data": {
                    "audio_duration_seconds": 62.0,
                    "analyzed_duration_seconds": 62.0,
                    "window_count": 12,
                    "event_segments": [
                        {
                            "class_id": 1,
                            "label": "Speech",
                            "label_zh": "人声",
                            "confidence": 0.9,
                            "start_seconds": 50.0,
                            "end_seconds": 62.0,
                        }
                    ],
                },
            }
        elif url.endswith("/predict"):
            task = "emotion"
            payload = {"emotion": "中立", "confidence": 0.8}
        else:
            raise AssertionError(url)
        self.calls.append(task)
        if task == self.fail_task:
            raise httpx.ConnectError(f"{task} unavailable")
        return _FakeResponse(payload)

    async def aclose(self):
        return None


def _route_result(text: str):
    return (
        {
            "text": text,
            "used_fact_ids": [],
            "evidence_segment_ids": ["seg-1"],
            "fact_count": 0,
            "uncertainties": [],
            "fallback_used": False,
        },
        {"mode": "test"},
    )


def test_dialogue_prefers_training_primary(monkeypatch):
    calls = []

    async def fake_route(client, *, ollama_url, model, timeout_seconds, segments):
        calls.append((ollama_url, model))
        return _route_result("训练服务器结果")

    monkeypatch.setattr(
        dialogue_module, "_generate_dialogue_understanding_on_route", fake_route
    )
    result, metadata = asyncio.run(
        dialogue_module.generate_dialogue_understanding(
            object(),
            ollama_url="http://local",
            model="qwen2.5:7b",
            timeout_seconds=180,
            segments=[SimpleNamespace(id="seg-1", text="测试")],
            primary_ollama_url="http://training",
            primary_model="qwen3:8b",
            primary_timeout_seconds=75,
        )
    )

    assert result["text"] == "训练服务器结果"
    assert calls == [("http://training", "qwen3:8b")]
    assert metadata["routing"] == {
        "selected": "training-primary",
        "model": "qwen3:8b",
        "fallback_used": False,
    }


def test_dialogue_falls_back_to_local_when_training_is_unreachable(monkeypatch):
    calls = []

    async def fake_route(client, *, ollama_url, model, timeout_seconds, segments):
        calls.append((ollama_url, model))
        if ollama_url == "http://training":
            raise httpx.ConnectError("training route unavailable")
        return _route_result("部署服务器结果")

    monkeypatch.setattr(
        dialogue_module, "_generate_dialogue_understanding_on_route", fake_route
    )
    result, metadata = asyncio.run(
        dialogue_module.generate_dialogue_understanding(
            object(),
            ollama_url="http://local",
            model="qwen2.5:7b",
            timeout_seconds=180,
            segments=[SimpleNamespace(id="seg-1", text="测试")],
            primary_ollama_url="http://training",
            primary_model="qwen3:8b",
            primary_timeout_seconds=75,
        )
    )

    assert result["text"] == "部署服务器结果"
    assert calls == [
        ("http://training", "qwen3:8b"),
        ("http://local", "qwen2.5:7b"),
    ]
    assert metadata["routing"]["selected"] == "deployment-fallback"
    assert metadata["routing"]["fallback_used"] is True


def test_digest_only_request_bypasses_general_local_analysis(monkeypatch):
    calls = []

    async def fake_dialogue(client, **kwargs):
        calls.append(kwargs)
        result, metadata = _route_result("训练服务器直接结果")
        metadata["routing"] = {
            "selected": "training-primary",
            "model": "qwen3:8b",
            "fallback_used": False,
        }
        return result, metadata

    class NoGeneralPostClient:
        async def post(self, *args, **kwargs):
            raise AssertionError("digest-only path must not call the general analyzer")

    monkeypatch.setattr(app_module, "generate_dialogue_understanding", fake_dialogue)
    settings = SimpleNamespace(
        max_structured_bytes=512 * 1024,
        max_transcript_chars=24000,
        ollama_url="http://local",
        ollama_model="qwen2.5:7b",
        ollama_timeout_seconds=180,
        dialogue_primary_url="http://training",
        dialogue_primary_model="qwen3:8b",
        dialogue_primary_timeout_seconds=75,
    )
    request = StructuredAnalysisRequest.model_validate(
        {
            "conversation": {
                "segments": [
                    {"id": "seg-1", "speaker": "Speaker_00", "text": "测试内容"}
                ]
            },
            "tasks": ["digest", "summary"],
        }
    )

    analysis, metadata = asyncio.run(
        _ollama_analyze(NoGeneralPostClient(), settings, request)
    )

    assert len(calls) == 1
    assert analysis["dialogue_understanding"] == "训练服务器直接结果"
    assert analysis["dialogue_understanding_meta"]["route"] == "training-primary"
    assert metadata["model"] == "qwen3:8b"


def test_dialogue_ledger_normalizes_only_known_boolean_drift():
    ledger = {
        "facts": [
            {
                "fact_id": "fact-001",
                "fact_type": "topic",
                "statement": "会议围绕项目后续安排展开讨论。",
                "critical": None,
                "status": "confirmed",
            },
            {
                "fact_id": "fact-002",
                "fact_type": "decision",
                "statement": "讨论决定先修改报告再进入实施阶段。",
                "critical": "true",
                "status": "confirmed",
            },
        ],
        "uncertainties": [],
    }

    normalized = _normalize_ledger(ledger)

    assert normalized["facts"][0]["critical"] is False
    assert normalized["facts"][1]["critical"] is True
    assert _valid_ledger(normalized, minimum_fact_count=2) == []


def test_dialogue_ledger_requires_more_coverage_for_long_inputs():
    assert _minimum_fact_count([{"text": "短对话"}]) == 1
    assert _minimum_fact_count([{"text": "中" * 200}]) == 3
    assert _minimum_fact_count([{"text": "长" * 700}]) == 5
    assert _requested_fact_count([{"text": "短对话"}]) == 1
    assert _requested_fact_count([{"text": "中" * 200}]) == 5
    assert _requested_fact_count([{"text": "长" * 700}]) == 10
    one_fact = {
        "facts": [{
            "fact_id": "fact-001",
            "fact_type": "topic",
            "statement": "这是一条满足长度要求的主题事实。",
            "critical": False,
            "status": "confirmed",
        }],
        "uncertainties": [],
    }
    assert _valid_ledger(one_fact, minimum_fact_count=5) == ["facts数量必须为5到16"]


def test_dialogue_evidence_can_cross_adjacent_asr_windows():
    ledger = {
        "facts": [{
            "fact_id": "fact-001",
            "fact_type": "requirement",
            "statement": "顾问需要对行业有更深入的了解，以减少评价上的失真。",
            "critical": True,
            "status": "confirmed",
        }],
        "uncertainties": [],
    }
    segments = [
        SimpleNamespace(
            id="seg-0001",
            text="客户的一些背景可能需要再深入了解。我们安排的一些顾问是什么性质也要考虑，顾问可以对这个行业更了解一点，这样在整个评价上不会有太多的一些失，",
            start=0.0,
        ),
        SimpleNamespace(
            id="seg-0002",
            text="真或者是低点，对吧。后面改一改就可以实施。",
            start=30.0,
        ),
    ]

    facts, uncertainties = _resolve_facts(ledger, segments)

    assert [fact["fact_id"] for fact in facts] == ["fact-001"]
    assert facts[0]["evidence_segment_ids"] == ["seg-0001", "seg-0002"]
    assert uncertainties == []


def test_dialogue_fallback_keeps_all_confirmed_facts_in_natural_order():
    facts = [
        {"fact_id": "fact-001", "fact_type": "topic", "statement": "团队讨论部署方案。", "critical": True},
        {"fact_id": "fact-002", "fact_type": "requirement", "statement": "需要核查公网访问条件。", "critical": True},
        {"fact_id": "fact-003", "fact_type": "risk", "statement": "当前权限仍存在风险。", "critical": True},
    ]
    assert _fallback(facts) == "团队讨论部署方案。对话中提到，需要核查公网访问条件。当前权限仍存在风险。"


def test_dialogue_safety_rejects_an_unsupported_outlook():
    facts = [
        {"fact_id": "fact-001", "fact_type": "number", "statement": "集团收入有所增长。", "critical": True},
    ]
    draft = {"dialogue_understanding": "集团收入有所增长。集团希望保持积极向上，迎接新业绩。"}
    assert _prose_safety_errors(draft, facts)


def test_build_structured_input_preserves_speaker_evidence_and_conflict():
    value = build_structured_input(
        {
            "code": 0,
            "data": {
                "recognized": "and.",
                "duration": 1.2,
                "acoustic": {
                    "events": [
                        {"class_id": 0, "label": "Speech", "confidence": 0.91}
                    ]
                },
                "speaker_diarization": {
                    "speaker_count": 1,
                    "segments": [
                        {
                            "speaker": "Speaker_00",
                            "start": 0.1,
                            "end": 1.1,
                            "text": "Yeah.",
                        }
                    ],
                },
            },
        },
        {"emotion": "其他/other", "confidence": 0.8, "scores": {}},
        filename="sample.wav",
        language="auto",
        question="说了什么？",
        tasks=["qa"],
    )

    assert value.conversation.segments[0].id == "seg-0001"
    assert value.conversation.segments[0].speaker == "Speaker_00"
    assert value.conversation.transcript_conflict is True
    assert value.acoustic_context.events[0].confidence == 0.91
    assert value.emotion.scope == "whole_audio"


def test_api_key_is_required_when_configured(monkeypatch):
    monkeypatch.setenv("CONTENT_ANALYSIS_API_KEY", "test-secret")
    monkeypatch.setenv("CONTENT_ANALYSIS_REQUIRE_API_KEY", "1")
    with TestClient(app) as client:
        response = client.get("/v1/model/info")
        assert response.status_code == 401
        accepted = client.get(
            "/v1/model/info", headers={"X-API-Key": "test-secret"}
        )
        assert accepted.status_code == 200


def test_segment_interval_validation():
    with TestClient(app) as client:
        response = client.post(
            "/v1/conversation/analyze",
            json={
                "conversation": {
                    "segments": [
                        {
                            "id": "seg-1",
                            "speaker": "Speaker_00",
                            "start": 2,
                            "end": 1,
                            "text": "hello",
                        }
                    ]
                }
            },
        )
        assert response.status_code in {401, 422}


def test_grounding_guardrails_preserve_upstream_observations():
    value = build_structured_input(
        {
            "code": 0,
            "data": {
                "recognized": "and.",
                "duration": 1.2,
                "acoustic": {
                    "events": [
                        {"class_id": 0, "label": "Speech", "confidence": 0.91}
                    ]
                },
                "speaker_diarization": {
                    "speaker_count": 1,
                    "segments": [
                        {
                            "speaker": "Speaker_00",
                            "start": 0.1,
                            "end": 1.1,
                            "text": "Yeah.",
                        }
                    ],
                },
            },
        },
        {"emotion": "其他/other", "confidence": 0.8, "scores": {}},
        filename="sample.wav",
        language="auto",
        question="说了什么？",
        tasks=["qa"],
    )
    guarded = apply_grounding_guardrails(
        {
            "relationships": [
                {
                    "speaker_a": "Speaker_00",
                    "speaker_b": "Speaker_99",
                    "relation": "兄弟",
                    "confidence": 0.99,
                }
            ],
            "emotion_analysis": {"emotion": "中立"},
            "scene_analysis": {"scene": "办公室"},
            "evidence_segment_ids": ["seg-0001", "invented"],
        },
        value,
    )
    assert guarded["relationships"] == []
    assert guarded["emotion_analysis"]["observed_label"] == "其他/other"
    assert guarded["emotion_analysis"]["speaker_attribution"] is None
    assert guarded["scene_analysis"]["scene"] == "无法判断具体环境"
    assert guarded["evidence_segment_ids"] == ["seg-0001"]


def test_ambiguous_greeting_does_not_become_relationship_or_today_fact():
    from app import StructuredAnalysisRequest

    value = StructuredAnalysisRequest.model_validate(
        {
            "conversation": {
                "speaker_count": 2,
                "segments": [
                    {
                        "id": "seg-1",
                        "speaker": "Speaker_01",
                        "text": "大哥你好，吃了吗？",
                    },
                    {
                        "id": "seg-2",
                        "speaker": "Speaker_02",
                        "text": "我昨天吃饭了。",
                    },
                ],
            },
            "question": "Speaker_02今天吃饭了吗？两人是什么关系？",
            "tasks": ["qa", "relationships"],
        }
    )
    guarded = apply_grounding_guardrails(
        {
            "answer": "今天吃了",
            "relationships": [
                {
                    "speaker_a": "Speaker_01",
                    "speaker_b": "Speaker_02",
                    "relation": "朋友",
                    "confidence": 0.8,
                    "evidence_segment_ids": ["seg-1", "seg-2"],
                }
            ],
        },
        value,
    )
    assert "无法根据" in guarded["answer"]
    assert "不足以确定" in guarded["answer"]
    assert guarded["relationships"][0]["relation"] == "无法判断"
    assert guarded["relationships"][0]["confidence"] <= 0.3


def test_content_digest_adapts_to_non_meeting_content_and_preserves_alias():
    from app import StructuredAnalysisRequest

    value = StructuredAnalysisRequest.model_validate(
        {
            "content_type_hint": "interview",
            "conversation": {
                "speaker_count": 2,
                "segments": [
                    {
                        "id": "seg-1",
                        "speaker": "Speaker_00",
                        "display_name": "采访者",
                        "start": 0,
                        "end": 3,
                        "text": "你为什么选择这个研究方向？",
                    },
                    {
                        "id": "seg-2",
                        "speaker": "Speaker_01",
                        "display_name": "受访者",
                        "start": 3,
                        "end": 8,
                        "text": "因为它能解决真实场景里的效率问题。",
                    },
                ],
            },
            "tasks": ["digest"],
        }
    )
    guarded = apply_grounding_guardrails(
        {
            "summary": "受访者介绍了研究动机。",
            "content_digest": {
                "content_type": "meeting",
                "overview": "围绕研究动机展开访谈。",
                "topic_progression": [
                    {
                        "speaker": "Speaker_01",
                        "title": "研究动机",
                        "narrative": "受访者把研究价值落到真实场景效率。",
                        "evidence_segment_ids": ["seg-2"],
                    }
                ],
                "decisions": [
                    {
                        "text": "决定启动项目",
                        "evidence_segment_ids": ["seg-2"],
                    }
                ],
                "action_items": [
                    {
                        "owner": "Speaker_01",
                        "task": "启动项目",
                        "evidence_segment_ids": ["seg-2"],
                    }
                ],
            },
        },
        value,
    )
    digest = guarded["content_digest"]
    assert digest["content_type"] == "interview"
    assert digest["decisions"] == []
    assert digest["action_items"] == []
    assert digest["topic_progression"][0]["display_name"] == "受访者"


def test_content_digest_recovers_misplaced_object_and_meeting_context():
    from app import StructuredAnalysisRequest

    value = StructuredAnalysisRequest.model_validate(
        {
            "content_type_hint": "auto",
            "background_context": "团队围绕部署方案进行项目讨论。",
            "conversation": {
                "speaker_count": 2,
                "segments": [
                    {
                        "id": "seg-1",
                        "speaker": "speaker-1",
                        "display_name": "甲",
                        "start": 0,
                        "end": 4,
                        "text": "下一步先确认部署方案。",
                    },
                    {
                        "id": "seg-2",
                        "speaker": "speaker-2",
                        "display_name": "乙",
                        "start": 4,
                        "end": 8,
                        "text": "我来负责确认服务器。",
                    },
                ],
            },
            "tasks": ["digest"],
        }
    )
    guarded = apply_grounding_guardrails(
        {
            "summary": {
                "content_type": "other",
                "overview": "团队讨论了部署方案。",
                "topic_progression": [
                    {
                        "speaker": "wrong-example-speaker",
                        "title": "部署确认",
                        "narrative": "乙承接前文并提出负责确认服务器。",
                    }
                ],
                "speaker_contributions": [
                    {
                        "speaker": "wrong-example-speaker",
                        "summary": "乙负责确认服务器。",
                    }
                ],
            }
        },
        value,
    )
    digest = guarded["content_digest"]
    assert guarded["summary"] == "团队讨论了部署方案。"
    assert digest["content_type"] == "meeting"
    assert digest["topic_progression"][0]["speaker"] == "speaker-2"
    assert digest["topic_progression"][0]["display_name"] == "乙"
    assert digest["speaker_contributions"][0]["speaker"] == "speaker-2"


def test_auto_content_type_does_not_force_greeting_into_meeting():
    from app import StructuredAnalysisRequest

    value = StructuredAnalysisRequest.model_validate(
        {
            "conversation": {
                "speaker_count": 2,
                "segments": [
                    {"id": "seg-1", "speaker": "a", "text": "今天天气不错。"},
                    {"id": "seg-2", "speaker": "b", "text": "是啊，出去走走。"},
                ],
            },
            "tasks": ["digest"],
        }
    )
    guarded = apply_grounding_guardrails(
        {
            "summary": "两人闲聊天气。",
            "content_digest": {
                "content_type": "other",
                "overview": "两人闲聊天气。",
            },
        },
        value,
    )
    assert guarded["content_digest"]["content_type"] == "other"


def test_content_digest_recovers_when_model_nests_it_under_answer():
    from app import StructuredAnalysisRequest

    value = StructuredAnalysisRequest.model_validate(
        {
            "background_context": "团队项目讨论。",
            "question": "梳理讨论内容。",
            "conversation": {
                "speaker_count": 2,
                "segments": [
                    {
                        "id": "seg-1",
                        "speaker": "speaker-1",
                        "display_name": "甲",
                        "text": "先确认部署条件。",
                    },
                    {
                        "id": "seg-2",
                        "speaker": "speaker-2",
                        "display_name": "乙",
                        "text": "公网访问还没有确定。",
                    },
                ],
            },
            "tasks": ["digest", "summary"],
        }
    )
    guarded = apply_grounding_guardrails(
        {
            "summary": "团队讨论了部署条件。",
            "answer": {
                "content_digest": {
                    "content_type": "other",
                    "overview": "讨论从部署条件推进到公网访问问题。",
                    "key_points": ["确认了服务器支持公网访问。"],
                    "topic_progression": [
                        {
                            "speaker": "甲",
                            "title": "部署前提",
                            "narrative": "甲提出先确认部署条件。",
                            "evidence_segment_ids": ["seg-1"],
                        },
                        {
                            "speaker": "乙",
                            "title": "待确认项",
                            "narrative": "乙确认了公网访问可用。",
                            "evidence_segment_ids": ["seg-2"],
                        },
                    ],
                    "open_questions": [
                        {
                            "text": "公网访问还没有确定。",
                            "evidence_segment_ids": ["seg-2"],
                        },
                        {
                            "text": "核心模型是否需要更换？",
                            "evidence_segment_ids": ["seg-1"],
                        }
                    ],
                }
            },
        },
        value,
    )
    digest = guarded["content_digest"]
    assert guarded["answer"] == "团队讨论了部署条件。"
    assert digest["content_type"] == "meeting"
    assert "确认了服务器支持公网访问。" not in digest["key_points"]
    assert "待确认：公网访问还没有确定。" in digest["key_points"]
    assert len(digest["open_questions"]) == 1
    assert [item["speaker"] for item in digest["topic_progression"]] == [
        "speaker-1",
        "speaker-2",
    ]
    assert "确认了" not in digest["topic_progression"][1]["narrative"]


def test_meeting_digest_supplements_uncovered_topics_and_scopes_contributions():
    from app import StructuredAnalysisRequest

    value = StructuredAnalysisRequest.model_validate(
        {
            "content_type_hint": "meeting",
            "question": "梳理讨论。",
            "conversation": {
                "speaker_count": 2,
                "segments": [
                    {"id": "s1", "speaker": "a", "start": 0, "text": "先讨论部署。"},
                    {"id": "s2", "speaker": "b", "start": 2, "text": "公网还没确定。"},
                    {"id": "s3", "speaker": "b", "start": 4, "text": "需要继续核查。"},
                    {"id": "s4", "speaker": "a", "start": 6, "text": "可以再安排分工。"},
                ],
            },
            "tasks": ["digest"],
        }
    )
    guarded = apply_grounding_guardrails(
        {
            "content_digest": {
                "content_type": "meeting",
                "overview": "讨论部署和分工。",
                "topic_progression": [
                    {
                        "speaker": "a",
                        "title": "部署",
                        "narrative": "先讨论部署。",
                        "evidence_segment_ids": ["s1"],
                    }
                ],
                "speaker_contributions": [
                    {
                        "speaker": "a",
                        "summary": "讨论部署和分工。",
                        "evidence_segment_ids": ["s1", "s2", "s4"],
                    }
                ],
            }
        },
        value,
    )
    digest = guarded["content_digest"]
    covered = {
        evidence_id
        for item in digest["topic_progression"]
        for evidence_id in item["evidence_segment_ids"]
    }
    assert covered == {"s1", "s2", "s3", "s4"}
    assert digest["speaker_contributions"][0]["evidence_segment_ids"] == [
        "s1",
        "s4",
    ]


def test_auto_interview_fallback_builds_question_answer_progression():
    from app import StructuredAnalysisRequest

    value = StructuredAnalysisRequest.model_validate(
        {
            "background_context": "一段人物访谈。",
            "question": "总结访谈。",
            "conversation": {
                "speaker_count": 2,
                "segments": [
                    {
                        "id": "q1",
                        "speaker": "host",
                        "display_name": "主持人",
                        "start": 0,
                        "text": "为什么转向做产品？",
                    },
                    {
                        "id": "a1",
                        "speaker": "guest",
                        "display_name": "受访者",
                        "start": 2,
                        "text": "因为完整链路更能解决用户问题。",
                    },
                    {
                        "id": "q2",
                        "speaker": "host",
                        "display_name": "主持人",
                        "start": 5,
                        "text": "最大的困难是什么？",
                    },
                    {
                        "id": "a2",
                        "speaker": "guest",
                        "display_name": "受访者",
                        "start": 7,
                        "text": "困难是数据权限和部署资源。",
                    },
                ],
            },
            "tasks": ["digest", "summary"],
        }
    )
    guarded = apply_grounding_guardrails(
        {"summary": "受访者介绍了产品转向和落地困难。"},
        value,
    )
    digest = guarded["content_digest"]
    assert digest["content_type"] == "interview"
    assert len(digest["topic_progression"]) == 2
    assert digest["topic_progression"][0]["evidence_segment_ids"] == ["q1", "a1"]
    assert digest["decisions"] == []
    assert digest["action_items"] == []


def test_analysis_plan_separates_quick_and_complete_modes():
    quick = _build_analysis_plan("quick_summary", ["digest", "summary"])
    assert quick.mode == "quick_summary"
    assert quick.run_asr is True
    assert quick.run_diarization is False
    assert quick.run_acoustic is False
    assert quick.run_emotion is False

    complete = _build_analysis_plan("multimodal_analysis", ["summary"])
    assert complete.mode == "multimodal_analysis"
    assert complete.run_asr is True
    assert complete.run_diarization is True
    assert complete.run_acoustic is True
    assert complete.run_emotion is True
    assert set(complete.tasks) == {
        "digest", "summary", "qa", "relationships", "speakers", "emotion", "scene"
    }


def test_quick_audio_mode_does_not_call_multimodal_services(monkeypatch):
    fake_client = _FakeAudioClient()

    async def fake_analyze(client, settings, structured):
        return (
            {
                "summary": "快速总结",
                "dialogue_understanding": "快速总结",
                "content_digest": {"overview": "快速总结"},
            },
            {"model": "test-model"},
        )

    monkeypatch.setattr(app_module, "_ollama_analyze", fake_analyze)
    with TestClient(app) as client:
        app.state.client = fake_client
        response = client.post(
            "/v1/audio/analyze",
            files={"file": ("sample.wav", b"audio", "audio/wav")},
            data={
                "analysis_mode": "quick_summary",
                "tasks": "digest,summary",
            },
        )
        payload = response.json()

    assert response.status_code == 200
    assert payload["status"] == "completed"
    assert payload["analysis_mode"] == "quick_summary"
    assert fake_client.calls == ["asr"]
    assert payload["executed_tasks"] == ["asr", "llm"]
    assert {item["task"] for item in payload["skipped_tasks"]} == {
        "diarization", "acoustic", "emotion"
    }


def test_complete_audio_mode_preserves_results_when_emotion_fails(monkeypatch):
    fake_client = _FakeAudioClient(fail_task="emotion")

    async def fake_analyze(client, settings, structured):
        assert structured.conversation.diarization_available is True
        assert structured.acoustic_context.window_count == 12
        assert structured.acoustic_context.events[0].start_seconds == 50.0
        assert structured.emotion is None
        return (
            apply_grounding_guardrails(
                {
                    "summary": "完整分析",
                    "content_digest": {"overview": "完整分析"},
                    "relationships": [],
                },
                structured,
            ),
            {"model": "test-model"},
        )

    monkeypatch.setattr(app_module, "_ollama_analyze", fake_analyze)
    with TestClient(app) as client:
        app.state.client = fake_client
        response = client.post(
            "/v1/audio/analyze",
            files={"file": ("sample.wav", b"audio", "audio/wav")},
            data={
                "analysis_mode": "multimodal_analysis",
                "tasks": "digest,summary",
            },
        )
        payload = response.json()

    assert response.status_code == 200
    assert payload["status"] == "partial"
    assert payload["degraded"] is True
    assert fake_client.calls == ["asr", "diarization", "acoustic", "emotion"]
    assert set(payload["executed_tasks"]) == {"asr", "diarization", "acoustic", "llm"}
    assert [item["task"] for item in payload["failed_tasks"]] == ["emotion"]
    assert payload["analysis"]["speaker_analysis"]["speaker_count"] == 1
    assert payload["analysis"]["scene_analysis"]["window_count"] == 12
