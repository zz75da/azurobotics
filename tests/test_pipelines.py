import json
import re
import threading
from http.server import ThreadingHTTPServer
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pytest

import app
import pipelines
from mlops import MLOpsService, _validate_hyperparameters
from pipelines import Pipeline, PipelineBusy, PipelineRunner, Task, topological_order, validate_run_request
from robot_data import make_sample_telemetry_csv

ROOT = Path(__file__).resolve().parents[1]


def make_pipeline(tasks, kind="logs", name="demo"):
    return {name: Pipeline(name=name, title=name, description="", tasks=tuple(tasks), kind=kind)}


def ok(ctx):
    return {"value": 1}


def boom(ctx):
    raise RuntimeError("secret C:\\internal\\path")


def wait(runner, run_id):
    assert runner.wait_idle(20)
    return runner.get_run(run_id)


def test_topological_order_respects_dependencies_and_rejects_cycles():
    tasks = [Task("c", "C", ok, ("b",)), Task("a", "A", ok), Task("b", "B", ok, ("a",))]
    assert [task.id for task in topological_order(tasks)] == ["a", "b", "c"]
    with pytest.raises(ValueError):
        topological_order([Task("a", "A", ok, ("b",)), Task("b", "B", ok, ("a",))])
    with pytest.raises(ValueError):
        topological_order([Task("a", "A", ok, ("missing",))])


def test_downstream_of_failed_task_is_skipped_and_errors_are_sanitised(tmp_path):
    order = []
    tasks = [
        Task("a", "A", lambda ctx: order.append("a") or {}),
        Task("b", "B", boom, ("a",)),
        Task("c", "C", lambda ctx: order.append("c") or {}, ("b",)),
        Task("d", "D", lambda ctx: order.append("d") or {}, ("a",)),
    ]
    runner = PipelineRunner(tmp_path / "db.sqlite3", None, make_pipeline(tasks))
    run = wait(runner, runner.start("demo", {})["id"])
    status = {task["id"]: task for task in run["tasks"]}
    assert run["status"] == "failed"
    assert [status[name]["status"] for name in "abcd"] == ["success", "failed", "skipped", "success"]
    assert status["c"]["error"] == "upstream_failed"
    assert order == ["a", "d"]
    assert "secret" not in json.dumps(run) and "Traceback" not in json.dumps(run)
    assert status["a"]["duration_seconds"] is not None


def test_run_succeeds_and_tracks_outputs(tmp_path):
    runner = PipelineRunner(tmp_path / "db.sqlite3", None, make_pipeline([Task("a", "A", ok)]))
    run = wait(runner, runner.start("demo", {})["id"])
    assert run["status"] == "success"
    assert run["tasks"][0]["outputs"] == {"value": 1}
    assert run["duration_seconds"] is not None


def test_second_run_is_rejected_while_busy_then_allowed(tmp_path):
    gate = threading.Event()
    runner = PipelineRunner(
        tmp_path / "db.sqlite3", None, make_pipeline([Task("a", "A", lambda ctx: gate.wait(10) and {})])
    )
    first = runner.start("demo", {})
    with pytest.raises(PipelineBusy):
        runner.start("demo", {})
    gate.set()
    assert wait(runner, first["id"])["status"] == "success"
    assert runner.start("demo", {})["id"] != first["id"]
    runner.wait_idle(20)


def test_run_timeout_skips_remaining_tasks(tmp_path):
    import time

    tasks = [Task("a", "A", lambda ctx: time.sleep(0.2) or {}), Task("b", "B", ok, ("a",))]
    runner = PipelineRunner(tmp_path / "db.sqlite3", None, make_pipeline(tasks), timeout_seconds=0.05)
    run = wait(runner, runner.start("demo", {})["id"])
    assert run["status"] == "failed"
    assert run["tasks"][1]["status"] == "skipped" and run["tasks"][1]["error"] == "timeout"


def test_history_is_bounded_and_persisted_across_restarts(tmp_path, monkeypatch):
    monkeypatch.setattr(pipelines, "MEMORY_HISTORY", 3)
    database = tmp_path / "db.sqlite3"
    runner = PipelineRunner(database, None, make_pipeline([Task("a", "A", ok)]))
    ids = []
    for _ in range(4):
        ids.append(runner.start("demo", {})["id"])
        runner.wait_idle(20)
    assert len(runner.list_runs()) == 3
    assert runner.get_run(ids[0])["status"] == "success"  # evicted from memory, still in SQLite
    reloaded = PipelineRunner(database, None, make_pipeline([Task("a", "A", ok)]))
    assert [item["id"] for item in reloaded.list_runs()] == ids[::-1][:3]
    assert reloaded.get_run(ids[-1])["tasks"][0]["outputs"] == {"value": 1}


def test_runs_interrupted_by_a_restart_are_closed(tmp_path):
    database = tmp_path / "db.sqlite3"
    gate = threading.Event()
    runner = PipelineRunner(database, None, make_pipeline([Task("a", "A", lambda ctx: gate.wait(10) and {})]))
    run_id = runner.start("demo", {})["id"]
    reloaded = PipelineRunner(database, None, make_pipeline([Task("a", "A", ok)]))
    assert reloaded.get_run(run_id)["status"] == "failed"
    assert reloaded.get_run(run_id)["tasks"][0]["status"] == "skipped"
    gate.set()
    runner.wait_idle(20)


def pipeline(kind):
    return pipelines.default_pipelines()[{"train": "train_anomaly_model", "score": "score_and_drift", "logs": "analyze_logs"}[kind]]


@pytest.mark.parametrize(
    "body",
    [
        [],
        {"unknown": 1},
        {"params": []},
        {"params": {"nope": 1}},
        {"params": {"dataset": "other"}},
        {"params": {"dataset": "uploaded"}},
        {"params": {"dataset": "synthetic"}, "filename": "a.csv", "content": "x"},
        {"params": {"algorithm": "svm"}},
        {"params": {"preset": "missing"}},
        {"params": {"hyperparameters": {"n_estimators": 5}}},
        {"params": {"hyperparameters": {"__class__": 1}}},
        {"params": {"hyperparameters": {"max_depth": 99}}},
        {"params": {"hyperparameters": "x"}},
        {"params": {"description": "x" * 501}},
        {"params": {"description": 3}},
        {"filename": "a.csv"},
        {"filename": "a.exe", "content": "x"},
        {"filename": "a.csv", "content": 5},
        {"filename": "a.csv", "content": "x" * (25 * 1024 * 1024 + 1)},
    ],
)
def test_train_parameter_validation_rejects_bad_input(body):
    with pytest.raises(ValueError):
        validate_run_request(pipeline("train"), body)


def test_parameter_validation_accepts_and_normalises():
    params, filename, content = validate_run_request(
        pipeline("train"),
        {"params": {"preset": pipelines.TUNED_PRESET, "description": " d\x00 "},
         "filename": "..\\..\\evil name.csv", "content": "a,b"},
    )
    assert filename == "evil_name.csv" and content == "a,b"
    assert params["dataset"] == "uploaded" and params["description"] == "d"
    assert params["hyperparameters"]["algorithm"] == "random_forest"
    with pytest.raises(ValueError):  # training parameters are not accepted by other pipelines
        validate_run_request(pipeline("score"), {"params": {"algorithm": "random_forest"}})
    with pytest.raises(ValueError):
        validate_run_request(pipeline("score"), {"filename": "a.jsonl", "content": "x"})
    assert validate_run_request(pipeline("logs"), {"filename": "a.jsonl", "content": "x"})[1] == "a.jsonl"


def test_tuned_preset_matches_sweep_results():
    best = json.loads((ROOT / "tools" / "results" / "ai4i_random_forest_sweep.json").read_text(encoding="utf-8"))[0]
    preset = pipelines.PRESETS[pipelines.TUNED_PRESET]
    hp = preset["hyperparameters"]
    assert preset["label"] == "Random Forest ajusté (AI4I)"
    assert hp["algorithm"] == "random_forest"
    for key in ("n_estimators", "max_depth", "min_samples_leaf", "class_weight", "max_features"):
        assert hp[key] == best[key]
    assert hp["decision_threshold"] == best["threshold"]
    _validate_hyperparameters(hp)


class FakeMlops:
    def __init__(self):
        self.train_calls = []

    def train(self, content, **options):
        self.train_calls.append((content, options))
        return {"version": "v1", "status": "promoted", "mlflow_run_id": "abc123",
                "metrics": {"f1": 0.9, "roc_auc": 0.95, "precision": 0.9, "recall": 0.9, "test_rows": 10},
                "thresholds": {"roc_auc": 0.75, "f1": 0.5}}


def test_tuning_results_are_embedded_only_for_preset_with_uploaded_csv(tmp_path):
    mlops = FakeMlops()
    runner = PipelineRunner(tmp_path / "db.sqlite3", mlops)
    csv = make_sample_telemetry_csv()
    body = {"params": {"preset": pipelines.TUNED_PRESET}, "filename": "labelled.csv", "content": csv}
    run = wait(runner, runner.start("train_anomaly_model", body)["id"])
    assert run["status"] == "success"
    content, options = mlops.train_calls[0]
    assert content == csv
    assert len(options["tuning_results"]) > 10
    assert options["hyperparameters"]["decision_threshold"] == 0.32
    assert options["run_tags"]["pipeline_run"] == run["id"]
    assert [task["status"] for task in run["tasks"]] == ["success"] * 4
    assert run["summary"]["promotion"] == "promoted" and run["summary"]["mlflow_run_id"] == "abc123"
    assert "content" not in run["params"]
    run = wait(runner, runner.start("train_anomaly_model", {"params": {"preset": pipelines.TUNED_PRESET}})["id"])
    assert run["status"] == "success"
    assert mlops.train_calls[1][0] is None and "tuning_results" not in mlops.train_calls[1][1]


def test_validate_data_failure_skips_training(tmp_path):
    mlops = FakeMlops()
    runner = PipelineRunner(tmp_path / "db.sqlite3", mlops)
    body = {"filename": "bad.csv", "content": "duration_ms,battery_pct,x_m,y_m\n1,2,3,4\n"}
    run = wait(runner, runner.start("train_anomaly_model", body)["id"])
    assert run["status"] == "failed"
    assert [task["status"] for task in run["tasks"]] == ["failed", "skipped", "skipped", "skipped"]
    assert mlops.train_calls == []


def test_route_labels_for_pipeline_paths_are_bounded():
    assert app._route_label("/api/pipelines") == "/api/pipelines"
    assert app._route_label("/api/pipelines/runs") == "/api/pipelines/runs"
    assert app._route_label("/api/pipelines/runs/0123456789ab") == "/api/pipelines/runs/id"
    assert app._route_label("/api/pipelines/analyze_logs/run") == "/api/pipelines/run"
    assert app._route_label("/api/pipelines/../x") == "/other"


@pytest.fixture(scope="module")
def server(tmp_path_factory):
    tmp_path = tmp_path_factory.mktemp("pipelines")
    service = MLOpsService(tmp_path / "mlops.sqlite3", tmp_path / "models", tmp_path / "evidently")
    runner = PipelineRunner(service.database_path, service)
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), app.Handler)
    previous = (app.PORT, app.MLOPS, app.PIPELINES, app.LOG_PATH)
    app.PORT, app.MLOPS, app.PIPELINES, app.LOG_PATH = httpd.server_address[1], service, runner, tmp_path / "app.jsonl"
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    yield runner
    httpd.shutdown()
    thread.join(timeout=5)
    httpd.server_close()
    app.PORT, app.MLOPS, app.PIPELINES, app.LOG_PATH = previous


def call(method, path, payload=None, origin=True):
    base = "http://127.0.0.1:{}".format(app.PORT)
    headers = {"Content-Type": "application/json"}
    if method == "POST" and origin:
        headers["Origin"] = base
    data = json.dumps(payload).encode("utf-8") if payload is not None else (b"" if method == "POST" else None)
    try:
        with urlopen(Request(base + path, data=data, headers=headers, method=method), timeout=60) as response:
            return response.status, json.loads(response.read().decode("utf-8"))
    except HTTPError as error:
        return error.code, json.loads(error.read().decode("utf-8"))


def test_http_pipelines_end_to_end(server):
    status, defs = call("GET", "/api/pipelines")
    assert status == 200
    assert [p["name"] for p in defs["pipelines"]] == ["train_anomaly_model", "score_and_drift", "analyze_logs"]
    assert [t["id"] for t in defs["pipelines"][0]["tasks"]] == ["validate_data", "train", "evaluate", "promote"]
    assert defs["presets"][1]["label"] == "Random Forest ajusté (AI4I)"

    assert call("POST", "/api/pipelines/nope/run", {})[0] == 404
    assert call("POST", "/api/pipelines/analyze_logs/run", {"params": {"x": 1}})[0] == 400
    assert call("POST", "/api/pipelines/analyze_logs/run", {}, origin=False)[0] in (200, 202, 400, 409)
    server.wait_idle(60)
    request = Request(
        "http://127.0.0.1:{}/api/pipelines/analyze_logs/run".format(app.PORT), data=b"{}", method="POST",
        headers={"Origin": "https://evil.example.com", "Content-Type": "application/json"},
    )
    with pytest.raises(HTTPError) as rejected:
        urlopen(request, timeout=10)
    assert rejected.value.code == 403

    for name, body in (
        ("analyze_logs", {}),
        ("score_and_drift", {}),  # fails first: no promoted model yet
        ("train_anomaly_model", {"params": {"description": "e2e"}}),
        ("score_and_drift", {}),
    ):
        status, started = call("POST", "/api/pipelines/{}/run".format(name), body)
        assert status == 202, started
        server.wait_idle(120)
        status, run = call("GET", "/api/pipelines/runs/" + started["id"])
        assert status == 200
        if name == "score_and_drift" and run["tasks"][1]["status"] == "failed":
            assert run["tasks"][2]["status"] == "skipped" and run["tasks"][2]["error"] == "upstream_failed"
            assert "promu" in run["tasks"][1]["error"]
        else:
            assert run["status"] == "success", run
    assert run["summary"]["records"] == 600

    status, history = call("GET", "/api/pipelines/runs")
    assert status == 200 and len(history["runs"]) == 5
    assert history["runs"][0]["id"] == run["id"]
    assert call("GET", "/api/pipelines/runs/zzzz")[0] == 404

    request = Request("http://127.0.0.1:{}/metrics".format(app.PORT))
    with urlopen(request, timeout=10) as response:
        metrics = response.read().decode("utf-8")
    assert 'robotics_pipeline_runs_total{pipeline="train_anomaly_model",status="success"} 1' in metrics
    assert "robotics_pipeline_run_duration_seconds_count" in metrics


def test_http_returns_409_when_a_run_is_active(server):
    gate = threading.Event()
    original = server.pipelines
    server.pipelines = make_pipeline([Task("a", "A", lambda ctx: gate.wait(10) and {})])
    try:
        status, first = call("POST", "/api/pipelines/demo/run", {})
        assert status == 202
        status, error = call("POST", "/api/pipelines/demo/run", {})
        assert status == 409 and "error" in error
        gate.set()
        server.wait_idle(20)
    finally:
        gate.set()
        server.wait_idle(20)
        server.pipelines = original


def test_pipeline_ui_section_renders_data_without_html_injection():
    html = (ROOT / "static" / "index.html").read_text(encoding="utf-8")
    script = html.split("// pipelines:begin")[1].split("// pipelines:end")[0]
    section = html.split('id="pipelinesSection"')[1].split("</section>")[0]
    for forbidden in ("innerHTML", "outerHTML", "insertAdjacentHTML", "document.write", "eval(", "new Function"):
        assert forbidden not in script
    assert not re.search(r"\son[a-z]+\s*=", section)
    assert "http://" not in script and "https://" not in script.replace("^https?:", "")
    assert "6 · Pipelines" in html and 'aria-live="polite"' in section
    assert "setTimeout(tick, 2000)" in script
