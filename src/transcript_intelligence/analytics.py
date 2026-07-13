from pathlib import Path

import pandas as pd
import plotly.express as px

from transcript_intelligence.io_utils import write_json
from transcript_intelligence.logging_setup import get_logger
from transcript_intelligence.models import Metric

log = get_logger(__name__)


def _frame(metrics: list[Metric], metric_type: str) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "source_set": metric.source_set,
                "time_window": metric.time_window,
                "category": metric.category,
                "value": metric.value,
                "numerator": metric.numerator,
                "denominator": metric.denominator,
                "chart_point_id": metric.chart_point_id,
            }
            for metric in metrics
            if metric.metric_type == metric_type
        ]
    )


def write_charts(metrics: list[Metric], stage_dir: Path) -> None:
    html_dir = stage_dir / "html"
    html_dir.mkdir(parents=True, exist_ok=True)
    specs = (
        (
            "topic_prevalence_monthly",
            "Topic prevalence by month",
            "time_window",
            True,
        ),
        (
            "topic_prevalence_all_time",
            "Topic prevalence all time",
            "source_set",
            False,
        ),
        (
            "sentiment_distribution",
            "Sentiment distribution",
            "time_window",
            True,
        ),
        (
            "finding_prevalence",
            "Finding prevalence",
            "time_window",
            True,
        ),
    )
    manifest = {}
    for metric_type, title, x_column, facet in specs:
        frame = _frame(metrics, metric_type)
        path = html_dir / f"{metric_type}.html"
        if frame.empty:
            path.write_text(
                f"<html><body><p>No data for {title}</p></body></html>\n",
                encoding="utf-8",
            )
        else:
            kwargs = dict(
                data_frame=frame,
                x=x_column,
                y="value",
                color="category",
                hover_data=[
                    "numerator",
                    "denominator",
                    "chart_point_id",
                ],
                title=title,
            )
            if facet:
                kwargs["facet_row"] = "source_set"
            px.bar(**kwargs).write_html(str(path), include_plotlyjs="cdn")
        manifest[metric_type] = f"html/{metric_type}.html"
    write_json(stage_dir / "chart_manifest.json", manifest)
    log.info("charts written", charts=len(manifest))
