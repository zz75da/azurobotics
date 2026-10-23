import json
import logging

from observability import JsonFormatter, RequestMetrics, configure_logging, read_recent_logs


def test_json_formatter_emits_context_and_exception():
    record = logging.LogRecord(
        "robotics-demo",
        logging.INFO,
        __file__,
        10,
        "pipeline_run_succeeded",
        (),
        None,
    )
    record.request_id = "request-123"
    record.route = "/api/analyze"
    record.status = 200

    event = json.loads(JsonFormatter().format(record))

    assert event["event"] == "pipeline_run_succeeded"
    assert event["request_id"] == "request-123"
    assert event["route"] == "/api/analyze"
    assert event["status"] == 200


def test_metrics_are_labeled_and_log_rotation_is_bounded(tmp_path):
    metrics = RequestMetrics()
    metrics.observe("POST", "/api/analyze", 200, 0.25)
    exposition = metrics.prometheus_metrics()
    assert 'robotics_http_requests_total{method="POST",route="/api/analyze",status="200"} 1' in exposition
    assert 'robotics_http_request_duration_seconds_bucket{method="POST",route="/api/analyze",status="200",le="0.5"} 1' in exposition
    assert 'robotics_http_request_duration_seconds_sum{method="POST",route="/api/analyze",status="200"} 0.250000' in exposition

    path = configure_logging(tmp_path)
    logger = logging.getLogger("robotics-demo")
    try:
        for index in range(20):
            logger.info("bounded_log_event", extra={"request_id": str(index)})

        file_handler = next(handler for handler in logger.handlers if hasattr(handler, "maxBytes"))
        assert file_handler.maxBytes == 2 * 1024 * 1024
        assert file_handler.backupCount == 3
        assert path.exists()
        assert len(read_recent_logs(path, limit=10)) == 10
    finally:
        for handler in logger.handlers[:]:
            logger.removeHandler(handler)
            handler.close()
