#!/usr/bin/env python3
"""Check that no step ends without the output file Pegasus promised for it.

Why this exists
---------------
On 2026-09-03 this workflow ran to 71% and then stopped, with no failure to
look at. 20 of 28 jobs done, one job held for half an hour, seven that would
never start. The held job's reason:

    Transfer output files failure ... reading from file
    .../scratch/accuracy_results.json: (errno 2) No such file or directory

`evaluate_accuracy` had found no predictions to evaluate, logged an error, and
returned — exit 0, no output file. Pegasus declares that file as the job's
output and HTCondor transfers it when the job exits *whatever the exit code*,
so a missing declared output does not fail the workflow. It holds the job, and
DAGMan waits on a held job forever. Exiting 0 also meant nothing retried and
nothing reported a failure.

The rule, for every step in this workflow: **write the declared output file on
every path, then exit non-zero if there is nothing worth reporting.** A step
with nothing to say still has to say it in the file it promised.

This script drives the three paths where that used to be wrong. It needs only
pillow and numpy — `kagglehub` and `torch` are stubbed, since sample mode never
calls Kaggle and the model is not what is under test.

    python3 validate_outputs.py
"""
import json
import os
import subprocess
import sys
import tempfile
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PY = sys.executable
failures = []


def check(name, ok, detail=""):
    print(f"{'PASS' if ok else 'FAIL'}  {name}{(' — ' + detail) if detail else ''}")
    if not ok:
        failures.append(name)


def stub_kagglehub(tmp):
    """fetch_crop_images imports kagglehub at module scope; sample mode never uses it."""
    stub = Path(tmp) / "stubs"
    stub.mkdir(exist_ok=True)
    (stub / "kagglehub.py").write_text("def dataset_download(*a, **k):\n    raise RuntimeError\n")
    return dict(os.environ, PYTHONPATH=str(stub))


# --- fetch --source sample must produce images, not just catalog rows ------

def test_sample_mode_is_runnable():
    with tempfile.TemporaryDirectory() as tmp:
        proc = subprocess.run(
            [PY, str(ROOT / "fetch_crop_images.py"),
             "--source", "sample", "--output", "crop_catalog.csv",
             "--output-dir", "./images", "--archive-output", "images.tar.gz"],
            cwd=tmp, capture_output=True, text=True, env=stub_kagglehub(tmp),
        )
        images = sorted(Path(tmp, "images").glob("*.jpg"))
        catalog = Path(tmp, "crop_catalog.csv")
        check("sample mode exits 0", proc.returncode == 0, proc.stderr[-200:])
        check("sample mode writes one image per catalog row", len(images) == 21,
              f"{len(images)} images")
        check("sample mode writes the catalog and the archive",
              catalog.exists() and Path(tmp, "images.tar.gz").exists())
        # The bug was catalog rows naming files that were never created, which
        # left classification with nothing to read.
        named = {ln.split(",")[1] for ln in catalog.read_text().splitlines()[1:] if ln}
        check("every catalog row names a file that exists",
              named == {p.name for p in images},
              f"{len(named)} named vs {len(images)} on disk")


# --- evaluate_accuracy: the step that hung the DAG ------------------------

def test_evaluate_writes_output_then_fails():
    with tempfile.TemporaryDirectory() as tmp:
        Path(tmp, "predictions.json").write_text(json.dumps({"predictions": []}))
        Path(tmp, "crop_catalog.csv").write_text("filename,category\na.jpg,Tomato___healthy\n")
        proc = subprocess.run(
            [PY, str(ROOT / "bin" / "evaluate_accuracy.py"),
             "--predictions", "predictions.json", "--catalog", "crop_catalog.csv",
             "--output", "accuracy_results.json"],
            cwd=tmp, capture_output=True, text=True,
        )
        out = Path(tmp, "accuracy_results.json")
        check("evaluate fails on empty predictions", proc.returncode != 0,
              f"rc={proc.returncode}")
        check("evaluate writes its declared output anyway", out.exists())
        if out.exists():
            check("the output records the reason", bool(json.loads(out.read_text()).get("error")))


def test_evaluate_still_computes():
    """The failure path must not have broken the working one."""
    with tempfile.TemporaryDirectory() as tmp:
        Path(tmp, "predictions.json").write_text(json.dumps({"predictions": [
            {"filename": "a.jpg", "predicted_class": "Tomato___healthy"},
            {"filename": "b.jpg", "predicted_class": "Tomato___healthy"},
        ]}))
        Path(tmp, "crop_catalog.csv").write_text(
            "filename,category\na.jpg,Tomato___healthy\nb.jpg,Tomato___Early_blight\n")
        proc = subprocess.run(
            [PY, str(ROOT / "bin" / "evaluate_accuracy.py"),
             "--predictions", "predictions.json", "--catalog", "crop_catalog.csv",
             "--output", "accuracy_results.json"],
            cwd=tmp, capture_output=True, text=True,
        )
        check("evaluate succeeds on real predictions", proc.returncode == 0,
              proc.stderr[-200:])
        body = json.loads(Path(tmp, "accuracy_results.json").read_text())
        check("evaluate reports 50% for 1 of 2 correct",
              body.get("overall_accuracy") == 0.5 and body.get("total_evaluated") == 2,
              str(body.get("overall_accuracy")))


# --- classify_disease: where the emptiness used to start ------------------

def test_classify_writes_output_then_fails():
    with tempfile.TemporaryDirectory() as tmp:
        sys.path.insert(0, str(ROOT / "bin"))
        sys.modules.setdefault("kagglehub", types.ModuleType("kagglehub"))
        # classify_disease defines `class SimpleCNN(nn.Module)` at module scope,
        # so importing it needs torch present even to read the module. The model
        # is not under test here, so stub it.
        torch = types.ModuleType("torch")
        nn = types.ModuleType("torch.nn")
        nn.Module = type("Module", (), {})
        for name in ("Conv2d", "MaxPool2d", "Linear", "Dropout", "ReLU",
                     "Sequential", "BatchNorm2d", "AdaptiveAvgPool2d", "Flatten",
                     "CrossEntropyLoss"):
            setattr(nn, name, type(name, (), {"__init__": lambda self, *a, **k: None}))
        nn.functional = types.ModuleType("torch.nn.functional")
        torch.nn = nn
        sys.modules.update({"torch": torch, "torch.nn": nn,
                            "torch.nn.functional": nn.functional})
        import classify_disease

        classify_disease.load_model = lambda d: (object(), {"0": "Tomato___healthy"}, "pytorch")
        cwd = os.getcwd()
        os.chdir(tmp)
        try:
            code = 0
            try:
                classify_disease.run_inference(".", "./images", "predictions.json")
            except SystemExit as exit_:
                code = exit_.code
            out = Path(tmp, "predictions.json")
            check("classify fails when there are no images", code != 0, f"code={code}")
            check("classify writes its declared output anyway", out.exists())
            if out.exists():
                check("the output records the reason",
                      bool(json.loads(out.read_text()).get("error")))
        finally:
            os.chdir(cwd)


if __name__ == "__main__":
    test_sample_mode_is_runnable()
    test_evaluate_writes_output_then_fails()
    test_evaluate_still_computes()
    test_classify_writes_output_then_fails()
    print()
    if failures:
        print(f"{len(failures)} check(s) FAILED: {', '.join(failures)}")
        sys.exit(1)
    print("all checks passed")
