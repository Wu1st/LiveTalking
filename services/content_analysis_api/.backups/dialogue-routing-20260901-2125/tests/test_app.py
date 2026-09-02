import os

os.environ.setdefault("CONTENT_ANALYSIS_REQUIRE_API_KEY", "0")

from fastapi.testclient import TestClient

from app import app, apply_grounding_guardrails, build_structured_input


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
