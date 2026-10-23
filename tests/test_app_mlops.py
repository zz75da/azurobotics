import json
import threading
from http.server import ThreadingHTTPServer
from urllib.request import Request, urlopen

import app
from mlops import MLOpsService
from robot_data import make_sample_telemetry_csv


def test_operations_polling_updates_metrics_without_appending_request_log(monkeypatch):
    handler = app.Handler.__new__(app.Handler)
    handler.command = "GET"
    handler._request_id = "poll-request"
    handler._request_started = 0
    handler._response_status = 200
    metrics = []
    logs = []
    monkeypatch.setattr(app.time, "monotonic", lambda: 1)
    monkeypatch.setattr(
        app.METRICS,
        "observe",
        lambda method, route, status, duration: metrics.append((method, route, status)),
    )
    monkeypatch.setattr(app.LOG, "info", lambda *args, **kwargs: logs.append(args))

    handler._finish_request("/api/operations")

    assert metrics == [("GET", "/api/operations", 200)]
    assert logs == []


def test_mlops_http_endpoints_train_predict_and_export_metrics(tmp_path):
    service = MLOpsService(
        tmp_path / "mlops.sqlite3",
        tmp_path / "models",
        tmp_path / "evidently",
    )
    server = ThreadingHTTPServer(("127.0.0.1", 0), app.Handler)
    previous_port, previous_mlops, previous_log = app.PORT, app.MLOPS, app.LOG_PATH
    app.PORT = server.server_address[1]
    app.MLOPS = service
    app.LOG_PATH = tmp_path / "app.jsonl"
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    def post(path, payload=b""):
        origin = "http://127.0.0.1:{}".format(app.PORT)
        request = Request(
            origin + path,
            data=payload,
            headers={"Origin": origin, "Content-Type": "application/json"},
            method="POST",
        )
        with urlopen(request, timeout=30) as response:
            return json.loads(response.read().decode("utf-8"))

    try:
        trained = post("/api/ml/train")
        csv_payload = json.dumps(
            {"filename": "sample.csv", "content": make_sample_telemetry_csv()}
        ).encode("utf-8")
        prediction = post("/api/ml/predict", csv_payload)
        request = Request(
            "http://127.0.0.1:{}/metrics".format(app.PORT),
            headers={"Host": "127.0.0.1:{}".format(app.PORT)},
        )
        with urlopen(request, timeout=10) as response:
            metrics = response.read().decode("utf-8")
    finally:
        server.shutdown()
        thread.join(timeout=5)
        server.server_close()
        app.PORT, app.MLOPS, app.LOG_PATH = previous_port, previous_mlops, previous_log

    assert trained["status"] == "promoted"
    assert prediction["records_processed"] == 600
    assert "robotics_ml_predictions_total" in metrics
