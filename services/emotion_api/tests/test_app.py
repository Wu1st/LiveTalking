from app import _parse_result


def test_parse_result_preserves_upstream_label():
    result = _parse_result(
        [{"labels": ["中立/neutral", "其他/other"], "scores": [0.2, 0.8]}]
    )
    assert result["emotion"] == "其他/other"
    assert result["confidence"] == 0.8
    assert result["scope"] == "whole_audio"

