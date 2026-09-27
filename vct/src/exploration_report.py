"""Shared output semantics for the four historical exploration scripts."""
import json
from pathlib import Path

EXPLORATION_NOTICE = (
    "探索报告：计算完成不代表科学验证通过；相关性、编码器位移及历史 pass 字段"
    "不代表真实细胞响应、因果关系或治疗效果。"
)


def write_exploration_report(path, results):
    if not isinstance(results, dict) or not results:
        raise ValueError("探索结果必须是非空字典；不能将空结果标为完成")
    report = dict(results, report_type="exploration", completion_status="completed",
                  interpretation=EXPLORATION_NOTICE)
    # Validate before opening: undefined/non-finite metrics must not replace an
    # existing report or be silently serialized as non-standard JSON NaN/Infinity.
    payload = json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False)
    Path(path).write_text(payload + "\n", encoding="utf-8")
    print(f"探索计算完成，报告已保存：{path}\n{EXPLORATION_NOTICE}")
