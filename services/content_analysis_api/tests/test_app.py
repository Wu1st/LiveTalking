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
