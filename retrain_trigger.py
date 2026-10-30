"""Ask GitHub Actions to run retrain.yml when a prediction batch drifts.

Off unless GITHUB_RETRAIN_REPOSITORY (owner/name) and GITHUB_RETRAIN_TOKEN are set. The token only
needs "Actions: read and write" on that repository.
"""
from __future__ import annotations

import json
import logging
import os
import urllib.error
import urllib.request

LOG = logging.getLogger("robotics-demo.retrain")
WORKFLOW = "retrain.yml"
TIMEOUT_SECONDS = 10


def dispatch_retrain(drift_share: float, snapshot_id: str) -> str:
    """Return "dispatched", "disabled" or "failed"; a GitHub outage must not fail the caller."""
    repository = os.environ.get("GITHUB_RETRAIN_REPOSITORY", "").strip()
    token = os.environ.get("GITHUB_RETRAIN_TOKEN", "").strip()
    if not repository or not token:
        return "disabled"
    body = {
        "ref": os.environ.get("GITHUB_RETRAIN_REF", "main"),
        "inputs": {"description": "Drift-triggered retrain (share {}, snapshot {})".format(drift_share, snapshot_id)},
    }
    request = urllib.request.Request(
        "https://api.github.com/repos/{}/actions/workflows/{}/dispatches".format(repository, WORKFLOW),
        data=json.dumps(body).encode("utf-8"),
        method="POST",
        headers={
            "Accept": "application/vnd.github+json",
            "Authorization": "Bearer {}".format(token),
            "X-GitHub-Api-Version": "2022-11-28",
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=TIMEOUT_SECONDS):
            pass
    except OSError as error:  # URLError and timeouts are OSError subclasses.
        LOG.warning("retrain_dispatch_failed", extra={"error_type": type(error).__name__})
        return "failed"
    LOG.info("retrain_dispatched", extra={"evidently_snapshot_id": snapshot_id})
    return "dispatched"
