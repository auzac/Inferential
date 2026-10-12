"""Model card / validation report as one self-contained HTML file (inline CSS and SVG).

The report is generated from a :class:`~inferential_studio.core.artifacts.ModelArtifact` (the
same content that is saved as the machine-readable model), the project configuration (data
sources, cleaning, alignment) and the per-sample predictions. It contains no scripts and no
external references, so it opens offline in any browser and prints to PDF from there.
"""

from __future__ import annotations

from html import escape
from typing import Any

import numpy as np
import pandas as pd

from inferential_studio.core import colours
from inferential_studio.core.artifacts import ModelArtifact
from inferential_studio.core.models import METRIC_COLUMNS, METRIC_LABELS
from inferential_studio.core.schemas import Project
from inferential_studio.core.splits import TEST, TRAIN, VALIDATION
from inferential_studio.core.svg import Series, bar_chart, xy_chart

SPLIT_NAMES = {TRAIN: "Training", VALIDATION: "Validation", TEST: "Final test"}
SPLIT_COLOURS = {TRAIN: colours.TRAIN, VALIDATION: colours.VALIDATION, TEST: colours.TEST}

CSS = """
body { font-family: "Segoe UI", Arial, sans-serif; color: #222; margin: 28px auto; max-width: 980px;
       line-height: 1.45; font-size: 14px; }
h1 { font-size: 24px; margin-bottom: 2px; } h2 { font-size: 18px; border-bottom: 1px solid #ccc;
     padding-bottom: 3px; margin-top: 30px; } h3 { font-size: 15px; margin-bottom: 4px; }
.sub { color: #555; } .warn { background: #FFF4E5; border: 1px solid #F0C080; padding: 8px 10px;
       color: #7A4100; margin: 10px 0; } .ok { color: #1b6e2f; }
table { border-collapse: collapse; margin: 6px 0 12px 0; font-size: 12.5px; }
th, td { border: 1px solid #d0d0d0; padding: 3px 7px; text-align: left; vertical-align: top; }
th { background: #f2f4f7; } td.num { text-align: right; font-variant-numeric: tabular-nums; }
code, .eq { font-family: Consolas, "Courier New", monospace; font-size: 12.5px; }
.eq { background: #f6f6f6; padding: 8px; white-space: pre-wrap; word-break: break-word; }
.small { font-size: 12px; color: #555; } figure { margin: 8px 0 16px 0; }
@media print { body { margin: 0; max-width: none; } h2 { page-break-after: avoid; }
               figure, table { page-break-inside: avoid; } }
"""


def _num(v: Any) -> str:
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return ""
    if isinstance(v, (int, np.integer)) and not isinstance(v, bool):
        return f"{v:,}"
    v = float(v)
    if v == 0:
        return "0"
    return f"{v:.4g}" if abs(v) >= 1e5 or abs(v) < 1e-3 else f"{v:.6g}"


def _table(headers: list[str], rows: list[list[Any]], numeric: set[int] | None = None) -> str:
    numeric = numeric or set()
    head = "".join(f"<th>{escape(h)}</th>" for h in headers)
    body = []
    for r in rows:
        cells = []
        for i, v in enumerate(r):
            if isinstance(v, (bool, np.bool_)):
                cells.append(f"<td>{'yes' if v else 'no'}</td>")
            elif i in numeric:
                cells.append(f'<td class="num">{_num(v)}</td>')
            else:
                cells.append(f"<td>{escape('' if v is None else str(v))}</td>")
        body.append("<tr>" + "".join(cells) + "</tr>")
    return f"<table><thead><tr>{head}</tr></thead><tbody>{''.join(body)}</tbody></table>"


def _local(iso: str | None, tz: str) -> str:
    if not iso:
        return ""
    return pd.Timestamp(iso).tz_convert(tz).strftime("%Y-%m-%d %H:%M")


def build_report(
    artifact: ModelArtifact,
    project: Project,
    *,
    predictions: pd.DataFrame | None = None,
    cleaning_summary: pd.DataFrame | None = None,
    title: str = "",
    limitations: str = "",
) -> str:
    """Return the HTML text of the model card."""
    d = artifact.data
    tz = d.get("display_timezone", "UTC")
    target = d["target"]
    units = target.get("units") or ""
    model = d["model"]
    revealed = d["split"]["test_revealed"]
    title = title or f"{target['name']} inferential — {model['label']}"
    parts: list[str] = [
        "<!DOCTYPE html><html lang='en'><head><meta charset='utf-8'>",
        f"<title>{escape(title)}</title><style>{CSS}</style></head><body>",
        f"<h1>{escape(title)}</h1>",
        f"<div class='sub'>Model version {escape(str(d.get('model_version', '')))} · "
        f"owner {escape(d.get('owner') or '—')} · generated {escape(_local(d['created_at'], tz))} "
        f"({escape(tz)}) by Inferential Model Studio {escape(d['app_version'])}</div>",
        "<div class='warn'>Engineering document. Fit statistics alone do not establish that a "
        "model is ready for online or closed-loop use; independent engineering review is "
        "required.</div>",
    ]

    # -- summary
    periods = d["split"]["periods"]
    parts.append("<h2>1. Summary</h2>")
    if d.get("purpose"):
        parts.append(f"<p><b>Purpose.</b> {escape(d['purpose'])}</p>")
    metrics = d["metrics"]
    rank = VALIDATION if periods[VALIDATION]["samples"] else TRAIN
    m_rank = metrics.get(rank, {})
    summary_rows = [
        ["Target", f"{target['name']} [{units}]" if units else target["name"]],
        ["Model", f"{model['label']} ({_hyper(model)}; scaling: {model['scaling']})"],
        ["Predictors", ", ".join(p["name"] for p in d["predictors"])],
        ["Training period", _period(periods[TRAIN], tz)],
        ["Validation period", _period(periods[VALIDATION], tz)],
        [
            "Final test period",
            _period(periods[TEST], tz) + ("" if revealed else " — not evaluated in this report"),
        ],
        [
            f"{SPLIT_NAMES[rank]} RMSE / R²",
            f"{_num(m_rank.get('rmse'))} {units} / {_num(m_rank.get('r2'))}",
        ],
    ]
    if revealed and TEST in metrics:
        mt = metrics[TEST]
        summary_rows.append(
            ["Final test RMSE / R²", f"{_num(mt.get('rmse'))} {units} / {_num(mt.get('r2'))}"]
        )
    parts.append(_table(["Item", "Value"], summary_rows))

    # -- data
    parts.append("<h2>2. Data</h2>")
    parts.append(
        _table(
            ["Source", "Kind", "File", "SHA-256 at import"],
            [
                [s["source_id"], s["kind"], s["path"], (s.get("sha256") or "")[:16] + "…"]
                for s in d["sources"]
            ],
        )
    )
    parts.append(
        f"<p class='small'>All times are stored in UTC and shown in {escape(tz)}. One supervised "
        "observation = one lab sample at its sample (collection) time; the target is never "
        "forward-filled.</p>"
    )

    # -- cleaning
    parts.append("<h2>3. Cleaning and exclusions</h2>")
    ex_rows = [
        [_local(iv.start.isoformat(), tz), _local(iv.end.isoformat(), tz), iv.reason.value,
         ", ".join(iv.signals) or "all signals", iv.note, iv.author, "yes" if iv.enabled else "no"]
        for iv in project.exclusions
    ]  # fmt: skip
    parts.append("<h3>Manual exclusions</h3>")
    parts.append(
        _table(["Start", "End", "Reason", "Scope", "Note", "Author", "Enabled"], ex_rows)
        if ex_rows
        else "<p>None.</p>"
    )
    parts.append("<h3>Rules</h3>")
    rule_rows = [
        [r.name, r.describe(), r.action.value, ", ".join(r.signals) or "whole rows", "yes" if r.enabled else "no"]
        for r in project.rules
    ]  # fmt: skip
    parts.append(
        _table(["Rule", "Condition", "Action", "Applies to", "Enabled"], rule_rows)
        if rule_rows
        else "<p>None.</p>"
    )
    parts.append("<h3>Automatic detectors and reviews</h3>")
    det_rows = []
    for det in project.detectors:
        decisions = [f for f in project.flag_decisions if f.detector_id == det.detector_id]
        acc = sum(f.decision.value == "accepted" for f in decisions)
        rej = len(decisions) - acc
        det_rows.append(
            [det.kind.value, ", ".join(det.variables), det.describe(),
             "auto-exclude" if det.auto_exclude else "review", acc, rej, "yes" if det.enabled else "no"]
        )  # fmt: skip
    parts.append(
        _table(
            ["Detector", "Variables", "Criterion", "Mode", "Accepted", "Rejected", "Enabled"],
            det_rows,
            {4, 5},
        )
        if det_rows
        else "<p>None.</p>"
    )
    if cleaning_summary is not None and not cleaning_summary.empty:
        parts.append("<h3>Exclusion summary by reason</h3>")
        parts.append(
            _table(
                ["Reason", "Kind", "Excludes", "Rows", "≈ hours"],
                cleaning_summary[
                    ["reason", "kind", "excludes", "rows", "approx_hours"]
                ].values.tolist(),
                {3, 4},
            )
        )

    # -- alignment
    parts.append("<h2>4. Time alignment and features</h2>")
    al = d["alignment"]
    parts.append(
        "<p>Each feature is computed per lab sample time <i>t</i> from the cleaned process tag: "
        "optional resampling (bins labelled at their end), optional trailing filter, then "
        "aggregation over a left-open, right-closed window. Windows that fail the coverage "
        "rules make the sample ineligible"
        + ("; windows overlapping operating-regime exclusions are rejected." if al.get("reject_regime_overlap") else ".")
        + "</p>"
    )  # fmt: skip
    feat_rows = []
    for p in d["predictors"]:
        f = p["feature"]
        feat_rows.append(
            [p["name"], p.get("units") or "", f["deadtime_s"] / 60, f["shift_s"] / 60, f["window_s"] / 60,
             f["placement"], f["aggregation"], _filter_text(f["filter"]),
             f["resample_s"] / 60 if f["resample_s"] else "original", f["min_coverage"] * 100,
             f["max_gap_s"] / 60 if f["max_gap_s"] is not None else "—", "yes" if p["causal"] else "NO"]
        )  # fmt: skip
    parts.append(
        _table(
            ["Tag", "Units", "Deadtime (min)", "Shift (min)", "Window (min)", "Placement", "Aggregation",
             "Filter", "Resample (min)", "Min coverage %", "Max gap (min)", "Causal"],
            feat_rows, {2, 3, 4, 9},
        )
    )  # fmt: skip

    # -- model
    parts.append("<h2>5. Model</h2>")
    parts.append(
        f"<p>{escape(model['label'])}, {escape(_hyper(model))}. Scaling ({escape(model['scaling'])}) "
        "was fitted on training samples only; hyperparameters were chosen by time-aware "
        "cross-validation inside the training period"
        + (f" ({escape(model['cv']['rule'])})" if model.get("cv") else "")
        + ".</p>"
    )
    parts.append("<h3>Equation on aligned features</h3>")
    parts.append(f"<div class='eq'>{escape(d['equation'])}</div>")
    parts.append(f"<p class='small'>{escape(d['equation_caveat'])}</p>")
    vip = model.get("details", {}).get("vip") if model.get("details") else None
    coef_rows = [["(intercept)", units, model["intercept"], None] + ([None] if vip else [])]
    for p in d["predictors"]:
        row = [p["name"], p.get("units") or "", p["coefficient"], p["standardized"]]
        if vip:
            row.append(vip.get(p["name"]))
        coef_rows.append(row)
    headers = ["Feature", "X units", f"Coefficient ({units or 'Y'} per X unit)", "Standardised"]
    if vip:
        headers.append("VIP")
    parts.append(_table(headers, coef_rows, {2, 3, 4}))
    order = sorted(d["predictors"], key=lambda p: -abs(p["standardized"] or 0))
    parts.append(
        "<figure>"
        + bar_chart([p["name"] for p in order], [p["standardized"] or 0 for p in order],
                    title="Standardised coefficients (SD of Y per SD of X)")
        + "</figure>"
    )  # fmt: skip
    det = model.get("details") or {}
    if det.get("type") in ("pls", "pcr"):
        xv = det.get("x_variance_explained", [])
        yv = det.get("y_variance_explained", [])
        rows = [[i + 1, 100 * xv[i], 100 * yv[i] if yv else None] for i in range(len(xv))]
        hdr = ["Component", "X variance %", "Y variance % (training)" if yv else "—"]
        parts.append("<h3>Components</h3>" + _table(hdr, rows, {1, 2}))
    parts.append(
        "<p class='small'>Coefficients show how the model uses its inputs, not physical cause and "
        "effect. No confidence intervals are given: lab samples are autocorrelated and predictors "
        "correlated, so textbook standard errors would overstate certainty.</p>"
    )

    # -- validation
    parts.append("<h2>6. Validation</h2>")
    split_cfg = d["split"]["config"]
    parts.append(
        f"<p>Chronological split ({escape(split_cfg['mode'])}); purge gap "
        f"{d['split']['purge_s'] / 60:g} min; {split_cfg['cv_folds']}-fold "
        f"{escape(split_cfg['cv_kind'])} cross-validation inside the training period. "
        f"About {d.get('effective_training_samples', 0):.0f} effectively independent training "
        "samples after autocorrelation.</p>"
    )
    met_rows = []
    for s in (TRAIN, VALIDATION, TEST):
        if s in metrics:
            met_rows.append([SPLIT_NAMES[s]] + [metrics[s].get(k) for k in METRIC_COLUMNS])
    parts.append(
        _table(["Period"] + [METRIC_LABELS[k] for k in METRIC_COLUMNS], met_rows, set(range(1, 9)))
    )
    if not revealed:
        parts.append(
            "<div class='warn'>The final test period has not been evaluated. Reveal it on the "
            "Validation page once the model choice is final, then regenerate this report.</div>"
        )
    if predictions is not None and len(predictions):
        shown = [s for s in (TRAIN, VALIDATION, TEST) if s != TEST or revealed]
        p = predictions[predictions["split"].isin(shown)]
        series = [
            Series(p.loc[p["split"] == s, "actual"].to_numpy(float), p.loc[p["split"] == s, "predicted"].to_numpy(float),
                   SPLIT_COLOURS[s], SPLIT_NAMES[s])
            for s in shown if (p["split"] == s).any()
        ]  # fmt: skip
        parts.append(
            "<figure>"
            + xy_chart(
                series, identity=True, x_label=f"Actual {units}", y_label=f"Predicted {units}"
            )
            + "</figure>"
        )
        t = pd.DatetimeIndex(p["sample_time"]).as_unit("ns").asi8 / 1e9
        trend = [Series(t, p["predicted"].to_numpy(float), "#555555", "Predicted", kind="line")]
        trend += [
            Series(t[(p["split"] == s).to_numpy()], p.loc[p["split"] == s, "actual"].to_numpy(float),
                   SPLIT_COLOURS[s], f"Lab — {SPLIT_NAMES[s]}")
            for s in shown if (p["split"] == s).any()
        ]  # fmt: skip
        parts.append(
            "<figure>"
            + xy_chart(trend, time_zone=tz, y_label=f"{target['name']} {units}")
            + "</figure>"
        )
        resid = [
            Series(t[(p["split"] == s).to_numpy()], p.loc[p["split"] == s, "residual"].to_numpy(float),
                   SPLIT_COLOURS[s], SPLIT_NAMES[s])
            for s in shown if (p["split"] == s).any()
        ]  # fmt: skip
        parts.append(
            "<figure>"
            + xy_chart(
                resid, time_zone=tz, y_label="Residual (actual − predicted)", hline=0.0, height=240
            )
            + "</figure>"
        )

    # -- applicability
    parts.append("<h2>7. Operating limits (training ranges)</h2>")
    parts.append(
        "<p>Linear models extrapolate silently. Predictions with any input outside these ranges "
        "need engineering judgement; batch predictions flag them.</p>"
    )
    parts.append(
        _table(
            ["Feature", "Units", "Min", "Max", "Mean", "SD"],
            [[p["name"], p.get("units") or "", *(p["training_range"][k] for k in ("min", "max", "mean", "std"))]
             for p in d["predictors"]],
            {2, 3, 4, 5},
        )
    )  # fmt: skip

    # -- limitations
    parts.append("<h2>8. Known limitations</h2><ul>")
    auto = list(model.get("warnings", [])) + list(d.get("notes", []))
    if not all(p["causal"] for p in d["predictors"]):
        auto.append(
            "Some features are non-causal (they read data after the sample time); the model cannot run online as configured."
        )
    if not revealed:
        auto.append("Final-test performance is not included.")
    auto.append(
        "Valid only for the operating range and conditions represented in the training period."
    )
    for line in auto:
        parts.append(f"<li>{escape(line)}</li>")
    for line in (limitations or "").splitlines():
        if line.strip():
            parts.append(f"<li>{escape(line.strip())}</li>")
    parts.append("</ul>")

    # -- provenance
    parts.append("<h2>9. Provenance</h2>")
    lib = d.get("libraries", {})
    prov = [["Project", d["project"].get("name", "")], ["Project file", d["project"].get("file") or "(unsaved)"],
            ["Model created", _local(d["created_at"], tz)], ["Application", d["app_version"]]]  # fmt: skip
    prov += [[k, v] for k, v in lib.items()]
    parts.append(_table(["Item", "Value"], prov))
    if d["project"].get("notes"):
        parts.append(f"<p><b>Project notes.</b> {escape(d['project']['notes'])}</p>")
    parts.append("</body></html>")
    return "\n".join(parts)


def _hyper(model: dict[str, Any]) -> str:
    h = model.get("hyperparameters") or {}
    if "components" in h:
        return f"{int(h['components'])} component(s)"
    if "alpha" in h:
        return f"alpha = {h['alpha']:.4g}"
    return "no hyperparameters"


def _period(p: dict[str, Any], tz: str) -> str:
    if not p.get("samples"):
        return "none"
    return f"{_local(p['start'], tz)} → {_local(p['end'], tz)} ({p['samples']} samples)"


def _filter_text(f: dict[str, Any]) -> str:
    kind = f.get("kind", "none")
    if kind == "none":
        return "none"
    if kind == "ewma":
        return f"EWMA, half-life {f['halflife_s'] / 60:g} min"
    return f"{kind.replace('_', ' ')}, {f['window_s'] / 60:g} min"
