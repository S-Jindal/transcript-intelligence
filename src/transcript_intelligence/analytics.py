from collections import defaultdict
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.cluster.hierarchy import dendrogram, fcluster, linkage
from scipy.spatial.distance import pdist
from sentence_transformers import SentenceTransformer

from transcript_intelligence.io_utils import read_json, write_json
from transcript_intelligence.logging_setup import get_logger
from transcript_intelligence.models import Metric, TopicLabel

log = get_logger(__name__)

TOPIC_TOP_N = 5
FINDING_TOP_N = 7

_TOPIC_COLORS = px.colors.qualitative.Safe
_POSITIVE_SHADES = (
    "#BBF7D0",
    "#86EFAC",
    "#4ADE80",
    "#22C55E",
    "#16A34A",
    "#15803D",
    "#166534",
)
_NEGATIVE_SHADES = (
    "#FECACA",
    "#FCA5A5",
    "#F87171",
    "#EF4444",
    "#DC2626",
    "#B91C1C",
    "#7F1D1D",
)
_NEUTRAL_SHADES = (
    "#E2E8F0",
    "#CBD5E1",
    "#94A3B8",
    "#64748B",
    "#475569",
)
_SOURCE_SET_COLORS = {
    "customer-support": ("#1D4ED8", "blue"),
    "account-manager": ("#047857", "green"),
    "internal-discuss": ("#7E22CE", "purple"),
}
_CROSS_SOURCE_COLOR = "#0F172A"


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
                "distinct_transcripts": metric.distinct_transcripts,
                "chart_point_id": metric.chart_point_id,
                "polarity": metric.polarity,
            }
            for metric in metrics
            if metric.metric_type == metric_type
        ]
    )


def _shade_for(
    category: str,
    mode: str,
    polarity: str | None,
    palette: dict[str, str],
    counters: dict[str, int],
) -> str:
    if category in palette:
        return palette[category]
    if mode == "topic":
        color = _TOPIC_COLORS[len(palette) % len(_TOPIC_COLORS)]
    else:
        resolved = polarity if polarity in {"positive", "negative"} else "neutral"
        shades = {
            "positive": _POSITIVE_SHADES,
            "negative": _NEGATIVE_SHADES,
            "neutral": _NEUTRAL_SHADES,
        }[resolved]
        index = counters[resolved] % len(shades)
        counters[resolved] += 1
        color = shades[index]
    palette[category] = color
    return color


def _empty_chart(path: Path, title: str) -> None:
    path.write_text(
        f"<html><body><p>No data for {title}</p></body></html>\n",
        encoding="utf-8",
    )


def _add_stack_segment(
    figure: go.Figure,
    row: pd.Series,
    x_value: str,
    color_mode: str,
    color_palette: dict[str, str],
    shade_counters: dict[str, int],
    seen_legend: set[str],
    subplot_row: int | None = None,
) -> None:
    color = _shade_for(
        row.category,
        color_mode,
        getattr(row, "polarity", None),
        color_palette,
        shade_counters,
    )
    # Dark greens/reds need white text; light shades need dark text.
    text_color = (
        "#0F172A"
        if color.lower() in {"#bbf7d0", "#86efac", "#fecaca", "#fca5a5", "#e2e8f0", "#cbd5e1"}
        else "#FFFFFF"
    )
    trace = go.Bar(
        name=row.category,
        x=[x_value],
        y=[row.value],
        text=[str(int(row.distinct_transcripts))],
        textposition="inside",
        insidetextanchor="middle",
        textfont={"size": 12, "color": text_color},
        marker_color=color,
        legendgroup=row.category,
        showlegend=row.category not in seen_legend,
        customdata=[
            [
                int(row.numerator),
                int(row.denominator),
                int(row.distinct_transcripts),
                row.chart_point_id,
            ]
        ],
        hovertemplate=(
            "%{x}<br>%{fullData.name}<br>"
            "share=%{y:.1%}<br>"
            "segments=%{customdata[0]} / %{customdata[1]}<br>"
            "transcripts=%{customdata[2]}<br>"
            "%{customdata[3]}<extra></extra>"
        ),
    )
    seen_legend.add(row.category)
    if subplot_row is None:
        figure.add_trace(trace)
    else:
        figure.add_trace(trace, row=subplot_row, col=1)


def _finalize_figure(
    figure: go.Figure,
    path: Path,
    title: str,
    legend_title: str,
    subplot_rows: int = 1,
) -> None:
    chart_height = max(780, 420 * subplot_rows)
    figure.update_layout(
        barmode="stack",
        title={"text": title, "x": 0.02, "xanchor": "left"},
        legend={
            "orientation": "h",
            "yanchor": "top",
            "y": -0.18,
            "xanchor": "left",
            "x": 0,
            "title_text": legend_title,
            "font": {"size": 11},
        },
        margin={"l": 70, "r": 40, "t": 70, "b": 220},
        height=chart_height,
        autosize=True,
        uniformtext_minsize=10,
        uniformtext_mode="hide",
        plot_bgcolor="#F8FAFC",
        paper_bgcolor="#FFFFFF",
    )
    figure.write_html(
        str(path),
        include_plotlyjs="cdn",
        full_html=True,
        default_width="100%",
        default_height="100%",
        config={"responsive": True, "displayModeBar": True},
    )
    html = path.read_text(encoding="utf-8")
    head_inject = """
<style>
  html, body {
    margin: 0;
    padding: 0;
    width: 100%;
    min-height: 100%;
    background: #ffffff;
    font-family: "IBM Plex Sans", "Segoe UI", sans-serif;
  }
  .plotly-graph-div, .js-plotly-plot {
    width: 100vw !important;
    min-height: calc(100vh - 72px) !important;
  }
  #chart-point-panel {
    position: sticky;
    top: 0;
    z-index: 20;
    display: flex;
    gap: 8px;
    align-items: center;
    padding: 10px 14px;
    background: #0f172a;
    color: #e2e8f0;
    border-bottom: 1px solid #334155;
  }
  #chart-point-panel label {
    font-size: 12px;
    white-space: nowrap;
    color: #94a3b8;
  }
  #chart-point-id {
    flex: 1;
    min-width: 0;
    font: 12px/1.4 ui-monospace, SFMono-Regular, Menlo, monospace;
    padding: 8px 10px;
    border: 1px solid #334155;
    border-radius: 6px;
    background: #1e293b;
    color: #f8fafc;
  }
  #chart-point-copy {
    border: 0;
    border-radius: 6px;
    padding: 8px 12px;
    background: #38bdf8;
    color: #0f172a;
    font-weight: 600;
    cursor: pointer;
  }
  #chart-point-status {
    font-size: 12px;
    color: #86efac;
    min-width: 140px;
  }
</style>
"""
    body_inject = """
<div id="chart-point-panel">
  <label for="chart-point-id">chart_point_id</label>
  <input id="chart-point-id" readonly placeholder="Hover a bar, then copy" />
  <button id="chart-point-copy" type="button">Copy</button>
  <span id="chart-point-status"></span>
</div>
<script>
(function () {
  function bind(plot) {
    var input = document.getElementById("chart-point-id");
    var status = document.getElementById("chart-point-status");
    var button = document.getElementById("chart-point-copy");

    function chartPointId(point) {
      var data = point && point.customdata;
      if (Array.isArray(data)) return data[3];
      return null;
    }

    function setId(value) {
      if (!value) return;
      input.value = value;
      status.textContent = "Ready to copy";
    }

    function copyId() {
      if (!input.value) return;
      input.select();
      navigator.clipboard.writeText(input.value).then(function () {
        status.textContent = "Copied";
      }).catch(function () {
        document.execCommand("copy");
        status.textContent = "Copied";
      });
    }

    button.addEventListener("click", copyId);
    input.addEventListener("focus", function () { input.select(); });
    plot.on("plotly_hover", function (event) {
      if (!event || !event.points || !event.points.length) return;
      setId(chartPointId(event.points[0]));
    });
    plot.on("plotly_click", function (event) {
      if (!event || !event.points || !event.points.length) return;
      setId(chartPointId(event.points[0]));
      copyId();
    });
  }

  function start() {
    var plot = document.querySelector(".js-plotly-plot");
    if (plot) {
      bind(plot);
      return;
    }
    var observer = new MutationObserver(function () {
      var found = document.querySelector(".js-plotly-plot");
      if (found) {
        observer.disconnect();
        bind(found);
      }
    });
    observer.observe(document.body, { childList: true, subtree: true });
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", start);
  } else {
    start();
  }
})();
</script>
"""
    html = html.replace("</head>", head_inject + "</head>", 1)
    html = html.replace("<body>", "<body>" + body_inject, 1)
    path.write_text(html, encoding="utf-8")


def _write_stacked_by_source(
    frame: pd.DataFrame,
    path: Path,
    title: str,
    top_n: int,
    legend_title: str,
    color_mode: str,
) -> None:
    if frame.empty:
        _empty_chart(path, title)
        return

    figure = go.Figure()
    color_palette: dict[str, str] = {}
    shade_counters = {"positive": 0, "negative": 0, "neutral": 0}
    seen_legend: set[str] = set()
    for source in sorted(frame["source_set"].unique()):
        top = (
            frame[frame["source_set"] == source]
            .nlargest(top_n, "numerator")
            .sort_values("numerator", ascending=False)
        )
        for _, row in top.iterrows():
            _add_stack_segment(
                figure,
                row,
                source,
                color_mode,
                color_palette,
                shade_counters,
                seen_legend,
            )
    figure.update_yaxes(title_text="segment share", tickformat=".0%")
    _finalize_figure(figure, path, title, legend_title)


def _write_stacked_monthly(
    frame: pd.DataFrame,
    path: Path,
    title: str,
    top_n: int,
    legend_title: str,
    color_mode: str,
) -> None:
    if frame.empty:
        _empty_chart(path, title)
        return

    sources = sorted(frame["source_set"].unique())
    figure = make_subplots(
        rows=len(sources),
        cols=1,
        shared_xaxes=False,
        subplot_titles=sources,
        vertical_spacing=0.10,
    )
    color_palette: dict[str, str] = {}
    shade_counters = {"positive": 0, "negative": 0, "neutral": 0}
    seen_legend: set[str] = set()
    for row_index, source in enumerate(sources, start=1):
        source_frame = frame[frame["source_set"] == source]
        for month in sorted(source_frame["time_window"].unique()):
            top = (
                source_frame[source_frame["time_window"] == month]
                .nlargest(top_n, "numerator")
                .sort_values("numerator", ascending=False)
            )
            for _, row in top.iterrows():
                _add_stack_segment(
                    figure,
                    row,
                    month,
                    color_mode,
                    color_palette,
                    shade_counters,
                    seen_legend,
                    subplot_row=row_index,
                )
        figure.update_yaxes(
            title_text="segment share",
            tickformat=".0%",
            row=row_index,
            col=1,
        )
        # NOTE: YYYY-MM strings are otherwise parsed as dates (e.g. Apr → Mar 29).
        figure.update_xaxes(
            type="category",
            categoryorder="category ascending",
            row=row_index,
            col=1,
        )
    _finalize_figure(
        figure,
        path,
        title,
        legend_title,
        subplot_rows=len(sources),
    )


def write_charts(metrics: list[Metric], stage_dir: Path) -> None:
    html_dir = stage_dir / "html"
    html_dir.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, str] = {}

    charts = (
        (
            "topic_prevalence_monthly",
            "Topic prevalence by month (top 5 per source / month)",
            TOPIC_TOP_N,
            "topic",
            True,
            "topic",
        ),
        (
            "topic_prevalence_all_time",
            "Topic prevalence all time (top 5 per source set)",
            TOPIC_TOP_N,
            "topic",
            False,
            "topic",
        ),
        (
            "sentiment_distribution",
            "Sentiment distribution (top 7 per source / month)",
            FINDING_TOP_N,
            "sentiment",
            True,
            "sentiment",
        ),
        (
            "finding_prevalence",
            "Finding prevalence (top 7 per source / month)",
            FINDING_TOP_N,
            "finding",
            True,
            "finding",
        ),
    )
    for metric_type, title, top_n, legend_title, monthly, color_mode in charts:
        frame = _frame(metrics, metric_type)
        path = html_dir / f"{metric_type}.html"
        if monthly:
            _write_stacked_monthly(
                frame,
                path,
                title,
                top_n,
                legend_title,
                color_mode,
            )
        else:
            _write_stacked_by_source(
                frame,
                path,
                title,
                top_n,
                legend_title,
                color_mode,
            )
        manifest[metric_type] = f"html/{metric_type}.html"

    write_json(stage_dir / "chart_manifest.json", manifest)
    log.info("charts written", charts=len(manifest))


def _dominant_source_set(topic: TopicLabel) -> str:
    return max(topic.source_distribution.items(), key=lambda item: item[1])[0]


def _source_short_name(source: str) -> str:
    return {
        "customer-support": "support",
        "account-manager": "am",
        "internal-discuss": "internal",
    }.get(source, source)


def _branch_colors(
    merge_tree,
    leaf_sources: list[str],
) -> dict[int, str]:
    leaf_count = len(leaf_sources)
    colors = {
        index: _SOURCE_SET_COLORS[source][0]
        for index, source in enumerate(leaf_sources)
    }
    for merge_index, (left, right, _, _) in enumerate(merge_tree):
        left_color = colors[int(left)]
        right_color = colors[int(right)]
        colors[leaf_count + merge_index] = (
            left_color
            if left_color == right_color
            else _CROSS_SOURCE_COLOR
        )
    return colors


def write_topic_hierarchy(
    topics: list[TopicLabel],
    embedding_model: SentenceTransformer,
    similarity_threshold: float,
    stage_dir: Path,
) -> None:
    html_dir = stage_dir / "html"
    html_dir.mkdir(parents=True, exist_ok=True)
    path = html_dir / "topic_hierarchy.html"
    if len(topics) < 2:
        _empty_chart(path, "topic hierarchy")
        return

    matrix = embedding_model.encode(
        [f"{topic.label}: {topic.description}" for topic in topics],
        normalize_embeddings=True,
        show_progress_bar=False,
        convert_to_numpy=True,
    )
    distance_threshold = 1.0 - similarity_threshold
    leaf_sources = [_dominant_source_set(topic) for topic in topics]
    leaf_labels = [
        (
            f"{topic.label} "
            f"[{_source_short_name(source)}:{topic.cluster_id}]"
        )
        for topic, source in zip(topics, leaf_sources)
    ]
    merge_tree = linkage(pdist(matrix, metric="cosine"), method="average")
    branch_colors = _branch_colors(merge_tree, leaf_sources)
    layout = dendrogram(merge_tree, no_plot=True, labels=leaf_labels)

    # Match each scipy U to its merge by height, then color arms by child
    # and the vertical join by parent — so same-set leaves stay blue/green/
    # purple until a cross-set merge, after which the join is black.
    merge_by_height = {
        float(row[2]): (
            int(row[0]),
            int(row[1]),
            len(topics) + index,
        )
        for index, row in enumerate(merge_tree)
    }
    figure = go.Figure()
    for xs, ys in zip(layout["dcoord"], layout["icoord"]):
        height = float(xs[1])
        left_id, right_id, parent_id = merge_by_height[min(
            merge_by_height,
            key=lambda value: abs(value - height),
        )]
        segments = (
            (xs[:2], ys[:2], branch_colors[left_id]),
            (xs[1:3], ys[1:3], branch_colors[parent_id]),
            (xs[2:], ys[2:], branch_colors[right_id]),
        )
        for segment_x, segment_y, color in segments:
            figure.add_trace(
                go.Scatter(
                    x=segment_x,
                    y=segment_y,
                    mode="lines",
                    line={"color": color, "width": 2},
                    hoverinfo="skip",
                    showlegend=False,
                )
            )

    figure.add_vline(
        x=distance_threshold,
        line_dash="dash",
        line_color="#0F172A",
        annotation_text=(
            f"merges stop below similarity {similarity_threshold:.2f}"
        ),
        annotation_position="top",
    )

    leaf_position_by_name = {
        name: (index + 0.5) * 10
        for index, name in enumerate(layout["ivl"])
    }
    source_by_label = dict(zip(leaf_labels, leaf_sources))
    for source, (color, color_name) in _SOURCE_SET_COLORS.items():
        members = [
            (topic, label)
            for topic, label, leaf_source in zip(
                topics, leaf_labels, leaf_sources
            )
            if leaf_source == source
        ]
        if not members:
            continue
        figure.add_trace(
            go.Scatter(
                x=[0] * len(members),
                y=[leaf_position_by_name[label] for _, label in members],
                mode="markers",
                marker={"color": color, "size": 11},
                name=f"{source} ({color_name})",
                showlegend=True,
                hovertext=[
                    f"{topic.label}<br>{topic.topic_id}"
                    f"<br>segments={topic.cluster_size}"
                    for topic, _ in members
                ],
                hoverinfo="text",
            )
        )
    figure.add_trace(
        go.Scatter(
            x=[None],
            y=[None],
            mode="lines",
            line={"color": _CROSS_SOURCE_COLOR, "width": 2},
            name="cross-source merge (black)",
            showlegend=True,
            hoverinfo="skip",
        )
    )

    label_annotations = [
        {
            "x": 0,
            "y": leaf_position_by_name[name],
            "xref": "x",
            "yref": "y",
            "text": name,
            "showarrow": False,
            "xanchor": "right",
            "xshift": -12,
            "font": {
                "size": 12,
                "color": _SOURCE_SET_COLORS[source_by_label[name]][0],
            },
        }
        for name in layout["ivl"]
    ]

    figure.update_layout(
        title={
            "text": (
                "Topic hierarchy across source sets "
                "(label + description embeddings, average linkage)"
            ),
            "x": 0.02,
            "xanchor": "left",
        },
        legend={
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.02,
            "xanchor": "right",
            "x": 1,
        },
        xaxis={
            "title": "cosine distance (1 − similarity)",
            "range": [0, max(distance_threshold * 1.15, 0.5)],
        },
        yaxis={
            "tickmode": "array",
            "tickvals": [
                leaf_position_by_name[name] for name in layout["ivl"]
            ],
            "showticklabels": False,
            "showgrid": False,
        },
        annotations=label_annotations,
        height=max(760, 30 * len(topics) + 220),
        margin={"l": 360, "r": 60, "t": 110, "b": 70},
        plot_bgcolor="#F8FAFC",
        paper_bgcolor="#FFFFFF",
        autosize=True,
        showlegend=True,
    )
    figure.write_html(
        str(path),
        include_plotlyjs="cdn",
        full_html=True,
        default_width="100%",
        config={"responsive": True, "displayModeBar": True},
    )

    grouped: dict[int, list[TopicLabel]] = defaultdict(list)
    for topic, group_id in zip(
        topics,
        fcluster(merge_tree, t=distance_threshold, criterion="distance"),
    ):
        grouped[int(group_id)].append(topic)
    write_json(
        stage_dir / "topic_hierarchy_groups.json",
        {
            "similarity_threshold": similarity_threshold,
            "groups": [
                {
                    "group_id": group_id,
                    "source_sets": sorted(
                        {_dominant_source_set(topic) for topic in members}
                    ),
                    "topics": [
                        {
                            "topic_id": topic.topic_id,
                            "label": topic.label,
                            "source_set": _dominant_source_set(topic),
                        }
                        for topic in members
                    ],
                }
                for group_id, members in sorted(
                    grouped.items(),
                    key=lambda item: -len(item[1]),
                )
            ],
        },
    )

    manifest_path = stage_dir / "chart_manifest.json"
    manifest = read_json(manifest_path) if manifest_path.exists() else {}
    manifest["topic_hierarchy"] = "html/topic_hierarchy.html"
    write_json(manifest_path, manifest)
    log.info(
        "topic hierarchy written",
        topics=len(topics),
        groups=len(grouped),
        similarity_threshold=similarity_threshold,
    )
