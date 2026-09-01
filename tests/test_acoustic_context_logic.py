import asyncio

from server import acoustic_context as module


def event(label_id, label, label_zh, confidence):
    return {
        "label_id": label_id,
        "label": label,
        "label_zh": label_zh,
        "confidence": confidence,
    }


def add(manager, session, events):
    manager.add_window(session, {"events": events})


def test_noise_filter_and_coarse_level():
    manager = module.AcousticContextManager()
    for index in range(12):
        events = []
        if index < 10:
            events.append(event("speech", "Speech", "人声", 0.62))
        if index < 8:
            events.append(event("keyboard", "Computer keyboard", "电脑键盘声", 0.28))
        if index < 2:
            events.append(event("car", "Car", "汽车声", 0.22))
        add(manager, "noise-filter", events)
    evidence = manager.build_evidence("noise-filter")
    names = {item["event_en"] for item in evidence["observations"]}
    assert "Speech" in names
    assert "Computer keyboard" in names
    assert "Car" not in names
    assert evidence["inference_level_hint"] == "coarse"


def test_direct_scene_is_a_coarse_inference():
    manager = module.AcousticContextManager()
    for _ in range(4):
        add(manager, "scene", [
            event("inside", "Inside, public space", "室内公共空间", 0.48)
        ])
    evidence = manager.build_evidence("scene")
    assert evidence["inference_level_hint"] == "coarse"
    fallback = module.AcousticReportService._rule_fallback(evidence)
    assert fallback["can_infer"] is True
    assert fallback["inference_level"] == "coarse"
    assert "室内公共空间" in fallback["inference"]


def test_one_high_confidence_event_stays_transient():
    manager = module.AcousticContextManager()
    for index in range(8):
        events = [event("speech", "Speech", "人声", 0.58)]
        if index == 1:
            events.append(event("siren", "Siren", "警报器声", 0.82))
        add(manager, "transient", events)
    evidence = manager.build_evidence("transient")
    siren = next(
        item for item in evidence["observations"] if item["event_en"] == "Siren"
    )
    assert siren["evidence_level"] == "transient"
    assert evidence["inference_level_hint"] == "none"


def test_repeated_cooccurrence_supports_specific_level():
    manager = module.AcousticContextManager()
    for _ in range(5):
        add(manager, "cooccurrence", [
            event("keyboard", "Computer keyboard", "电脑键盘声", 0.36),
            event("click", "Clicking", "点击声", 0.31),
        ])
    evidence = manager.build_evidence("cooccurrence")
    assert evidence["cooccurrences"]
    assert evidence["inference_level_hint"] == "specific"


def test_speech_subtypes_are_not_independent_scene_cues():
    manager = module.AcousticContextManager()
    for _ in range(5):
        add(manager, "speech-subtypes", [
            event("/m/0brhx", "Speech synthesizer", "语音合成", 0.61),
            event("/m/02qldy", "Narration, monologue", "叙述/独白", 0.35),
        ])
    evidence = manager.build_evidence("speech-subtypes")
    assert evidence["cooccurrences"]
    assert evidence["inference_level_hint"] == "coarse"
    assert not any(
        item["independent_scene_cue"] for item in evidence["observations"]
    )


def test_scene_change_lowers_the_maximum_level():
    manager = module.AcousticContextManager()
    for _ in range(4):
        add(manager, "change", [
            event("vehicle", "Vehicle", "车辆声", 0.42),
            event("traffic", "Traffic noise, roadway noise", "道路交通声", 0.35),
        ])
    for _ in range(4):
        add(manager, "change", [
            event("bird", "Bird", "鸟鸣声", 0.46),
            event("wind", "Wind", "风声", 0.34),
        ])
    evidence = manager.build_evidence("change")
    assert evidence["scene_change_suspected"] is True
    assert evidence["inference_level_hint"] == "coarse"


class FakeService:
    def __init__(self):
        self.calls = 0

    async def generate(self, evidence, transcript):
        self.calls += 1
        return {
            "facts": evidence["facts"],
            "can_infer": False,
            "inference_level": "none",
            "inference": module.FIXED_UNCERTAIN_TEXT,
            "source": "fake",
        }


async def test_cache_only_when_evidence_is_unchanged():
    manager = module.AcousticContextManager()
    service = FakeService()
    add(manager, "cache", [event("speech", "Speech", "人声", 0.7)])
    await manager.report("cache", service)
    cached = await manager.report("cache", service)
    assert service.calls == 1
    assert cached["cache_hit"] is True
    add(manager, "cache", [event("speech", "Speech", "人声", 0.72)])
    changed = await manager.report("cache", service)
    assert service.calls == 2
    assert changed["cache_hit"] is False


async def test_llm_level_guardrails_and_coarse_fallback():
    manager = module.AcousticContextManager()
    for _ in range(4):
        add(manager, "llm-level", [
            event("inside", "Inside, public space", "室内公共空间", 0.48)
        ])
    evidence = manager.build_evidence("llm-level")
    service = module.AcousticReportService()
    service._infer = lambda current, transcript: {
        "inference_level": "none",
        "inference": "",
        "basis": [],
        "uncertainty": "",
    }
    fallback = await service.generate(evidence, "")
    assert fallback["inference_level"] == "coarse"
    assert fallback["source"] == "rule_coarse_after_ollama_uncertain"
    assert "【声音事实】" in fallback["display_text"]
    assert "【场景推断 · 粗略推断】" in fallback["display_text"]
    assert "【推断依据】" not in fallback["display_text"]
    assert "【不确定性】" in fallback["display_text"]
    assert "\n- " not in fallback["display_text"]

    service._infer = lambda current, transcript: {
        "inference_level": "specific",
        "inference": "可能是某个确定地点。",
        "basis": ["单一场景标签"],
        "uncertainty": "",
    }
    guarded = await service.generate(evidence, "")
    assert guarded["inference_level"] == "coarse"
    assert guarded["source"] == "rule_guardrail"


if __name__ == "__main__":
    test_noise_filter_and_coarse_level()
    print("noise filter ok", flush=True)
    test_direct_scene_is_a_coarse_inference()
    print("direct scene ok", flush=True)
    test_one_high_confidence_event_stays_transient()
    print("transient ok", flush=True)
    test_repeated_cooccurrence_supports_specific_level()
    print("cooccurrence ok", flush=True)
    test_speech_subtypes_are_not_independent_scene_cues()
    print("speech subtype guard ok", flush=True)
    test_scene_change_lowers_the_maximum_level()
    print("scene change ok", flush=True)
    asyncio.run(test_cache_only_when_evidence_is_unchanged())
    print("cache ok", flush=True)
    asyncio.run(test_llm_level_guardrails_and_coarse_fallback())
    print("all acoustic context tests passed")
