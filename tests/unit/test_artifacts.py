from __future__ import annotations

import json
import zipfile

import numpy as np
import pandas as pd
import pytest

from inferential_studio.core.artifacts import (
    PRED_INPUT,
    PRED_OK,
    ArtifactError,
    build_artifact,
    describe_feature,
    load_artifact,
    predict,
    save_artifact,
)
from inferential_studio.core.exports import (
    ExportError,
    feature_matrix_table,
    predictions_table,
    write_table,
)
from inferential_studio.core.features import TargetSamples, build_feature_matrix
from inferential_studio.core.models import train_models
from inferential_studio.core.report import build_report
from inferential_studio.core.schemas import FeatureSpec, ModelKind
from inferential_studio.core.splits import TEST
from inferential_studio.core.synthetic import build_sample_project


@pytest.fixture(scope="module")
def trained(small_dataset):
    ds = small_dataset
    project = build_sample_project(ds)
    project.notes = "<script>alert('x')</script> notes"
    lab = ds.lab[~ds.lab["Sample_ID"].isin(ds.truth["bad_lab_sample_ids"])]
    n = len(lab)
    target = TargetSamples(
        "Product_Composition", pd.DatetimeIndex(lab["Sample_Timestamp"]),
        lab["Product_Composition"].to_numpy(), np.zeros(n, bool), np.zeros(n, bool),
        sample_ids=lab["Sample_ID"].to_numpy(),
    )  # fmt: skip
    fm = build_feature_matrix(target, ds.process, list(project.features))
    run = train_models(fm, project.modelling)
    return ds, project, fm, run


def model_name(run, kind):
    return next(m.name for m in run.models if m.spec.kind is kind)


def test_artifact_roundtrip_and_exact_predictions(trained, tmp_path):
    ds, project, fm, run = trained
    for kind in ModelKind:
        name = model_name(run, kind)
        art = build_artifact(project, run, name, owner="Tester", purpose="Demo")
        path = save_artifact(art, tmp_path / f"{kind.value}.imsmodel")
        loaded = load_artifact(path)
        assert loaded.data == json.loads(json.dumps(art.data))
        assert TEST not in loaded.data["metrics"]  # hidden unless revealed
        m = next(x for x in run.models if x.name == name)
        times = m.predictions["sample_time"]
        out = predict(loaded, ds.process, pd.DatetimeIndex(times))
        assert (out.table["status"] == PRED_OK).all()
        np.testing.assert_allclose(
            out.table["prediction"].to_numpy(), m.predictions["predicted"].to_numpy(), rtol=1e-9
        )
    assert "f[Tray_15_Temp]" in loaded.equation()


def test_test_metrics_only_when_revealed(trained):
    _ds, project, _fm, run = trained
    art = build_artifact(project, run, model_name(run, ModelKind.PLS), include_test=True)
    assert TEST in art.data["metrics"] and art.data["split"]["test_revealed"]


def test_prediction_reports_missing_inputs_and_extrapolation(trained):
    ds, project, _fm, run = trained
    art = build_artifact(project, run, model_name(run, ModelKind.MLR))
    proc = ds.process.copy()
    t = proc.index[3000]
    proc.loc[t - pd.Timedelta(hours=2) : t, "Tray_15_Temp"] = np.nan  # input gap
    proc.loc[proc.index[4000] - pd.Timedelta(hours=1) : proc.index[4000], "Column_Pressure"] += 5.0
    out = predict(art, proc, pd.DatetimeIndex([t, proc.index[4000]]))
    row_gap, row_hot = out.table.iloc[0], out.table.iloc[1]
    assert row_gap["status"] == PRED_INPUT and np.isnan(row_gap["prediction"])
    assert "Tray_15_Temp" in row_gap["reason"]
    assert row_hot["status"] == PRED_OK and "Column_Pressure" in row_hot["outside_training_range"]
    assert any("outside the training range" in w for w in out.warnings)
    with pytest.raises(ArtifactError, match="missing"):
        predict(art, proc.drop(columns=["Tray_20_Temp"]), pd.DatetimeIndex([t]))
    with pytest.raises(ArtifactError, match="units"):
        predict(art, proc, pd.DatetimeIndex([t]), units={"Tray_15_Temp": "degF"})


def test_load_rejects_foreign_and_unsafe_files(trained, tmp_path):
    _ds, project, _fm, run = trained
    art = build_artifact(project, run, model_name(run, ModelKind.RIDGE))
    good = save_artifact(art, tmp_path / "m.imsmodel")
    bad = tmp_path / "bad.imsmodel"
    with zipfile.ZipFile(good) as src, zipfile.ZipFile(bad, "w") as dst:
        for info in src.infolist():
            dst.writestr(info, src.read(info))
        dst.writestr("model.pkl", b"\x80\x04")  # a pickle is never read
    with pytest.raises(ArtifactError, match="unexpected"):
        load_artifact(bad)
    newer = tmp_path / "newer.imsmodel"
    with zipfile.ZipFile(newer, "w") as zf:
        zf.writestr(
            "manifest.json",
            json.dumps({"format": "inferential-model-studio-model", "artifact_version": 99}),
        )
        zf.writestr("model.json", "{}")
    with pytest.raises(ArtifactError, match="newer"):
        load_artifact(newer)
    with pytest.raises(ArtifactError):
        save_artifact(art, tmp_path / "m.json")


def test_feature_description():
    text = describe_feature(FeatureSpec("X", window_s=1800, deadtime_s=1200))
    assert text.startswith("mean of X over (t − 50 min, t − 20 min]")


def test_report_is_self_contained_and_escaped(trained):
    _ds, project, _fm, run = trained
    name = model_name(run, ModelKind.PLS)
    m = next(x for x in run.models if x.name == name)
    art = build_artifact(project, run, name, owner="A & B", purpose="Soft sensor <demo>")
    html = build_report(art, project, predictions=m.predictions, limitations="Line one\nLine two")
    assert "<script" not in html.lower()  # the notes' markup is escaped
    assert "&lt;script&gt;" in html and "A &amp; B" in html
    assert "http://" not in html.replace("http://www.w3.org/2000/svg", "")  # no external refs
    for section in ("Summary", "Cleaning and exclusions", "Time alignment", "Equation",
                    "Validation", "Operating limits", "Known limitations", "Provenance"):  # fmt: skip
        assert section in html
    assert "<svg" in html and "VIP" in html
    assert "has not been evaluated" in html
    shown = build_report(build_artifact(project, run, name, include_test=True), project,
                         predictions=m.predictions)  # fmt: skip
    assert "Final test RMSE" in shown and "has not been evaluated" not in shown


def test_exports(trained, tmp_path):
    _ds, _project, fm, run = trained
    name = model_name(run, ModelKind.MLR)
    hidden = predictions_table(run, name, fm, timezone="Asia/Singapore")
    shown = predictions_table(run, name, fm, timezone="Asia/Singapore", include_test=True)
    assert TEST not in set(hidden["period"]) and TEST in set(shown["period"])
    assert {"sample_time_utc", "sample_time_local (Asia/Singapore)", "sample_id"} <= set(shown)
    assert shown["sample_time_utc"].iloc[0].endswith("Z")
    csv = write_table(shown, tmp_path / "p.csv")
    back = pd.read_csv(csv, encoding="utf-8-sig")
    assert len(back) == len(shown)
    xlsx = write_table(feature_matrix_table(fm, "Asia/Singapore"), tmp_path / "f.xlsx", notes=["n"])
    assert pd.read_excel(xlsx, sheet_name="Data").shape[0] == len(fm.y)
    with pytest.raises(ExportError):
        write_table(shown, tmp_path / "p.txt")


def test_results_summary_and_comparison(trained):
    import json as _json

    from inferential_studio.core.artifacts import compare_results, results_summary

    _ds, _project, fm, run = trained
    saved = _json.loads(_json.dumps(results_summary(run, test_revealed=False)))
    assert compare_results(saved, run) == []
    assert all(TEST not in m["metrics"] for m in saved["models"] if m["error"] is None)
    saved["models"][0]["coefficients"] = {
        k: v * 1.01 for k, v in saved["models"][0]["coefficients"].items()
    }
    assert any("coefficients differ" in d for d in compare_results(saved, run))
