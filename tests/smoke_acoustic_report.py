"""Optional live smoke test for the configured Ollama acoustic-report model."""

import asyncio
import json

from server.acoustic_context import AcousticContextManager, AcousticReportService


async def main():
    manager = AcousticContextManager()
    for _ in range(4):
        manager.add_window("smoke", {
            "events": [{
                "label_id": "/m/03v3yw",
                "label": "Inside, public space",
                "label_zh": "室内公共空间",
                "confidence": 0.48,
            }]
        })
    evidence = manager.build_evidence("smoke")
    report = await AcousticReportService().generate(
        evidence,
        "现场有人正在交流。",
    )
    assert report["inference_level"] == "coarse", report
    assert report["can_infer"] is True, report
    print(json.dumps({
        "inference_level": report["inference_level"],
        "source": report["source"],
        "inference": report["inference"],
        "basis": report.get("basis"),
        "uncertainty": report.get("uncertainty"),
        "elapsed": report.get("elapsed"),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
