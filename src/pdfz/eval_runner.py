"""Server-side eval runner that wraps the eval suite and stores results as JSON."""
from __future__ import annotations

import json
import os
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent

_volume_mount = os.environ.get("RAILWAY_VOLUME_MOUNT_PATH")
if _volume_mount:
    RESULTS_PATH = Path(_volume_mount) / "eval_results.json"
else:
    RESULTS_PATH = PROJECT_ROOT / "data" / "eval_results.json"

# Global state for the current run
_current_run: dict | None = None


def get_current_run() -> dict | None:
    return _current_run


def get_latest_results() -> dict | None:
    if _current_run is not None:
        return _current_run
    if RESULTS_PATH.exists():
        return json.loads(RESULTS_PATH.read_text())
    return None


async def run_evals() -> dict:
    """Run the eval suite and store results. Returns the results dict."""
    global _current_run

    run_id = uuid.uuid4().hex[:8]
    _current_run = {
        "run_id": run_id,
        "status": "running",
        "started_at": datetime.now(timezone.utc).isoformat(),
        "completed_at": None,
        "cases": [],
    }

    try:
        # Add project root to path so evals module is importable
        root_str = str(PROJECT_ROOT)
        if root_str not in sys.path:
            sys.path.insert(0, root_str)
        from evals.run_evals import (
            dataset, retrieve_and_ask, check_prerequisites,
        )

        check_prerequisites()
        report = await dataset.evaluate(retrieve_and_ask)

        cases = []
        for rc in report.cases:
            assertions = {}
            for name, result in rc.assertions.items():
                assertions[name] = {
                    "passed": bool(result.value),
                    "reason": result.reason or "",
                }

            # Extract structured fields from RetrievalOutput
            output_str = ""
            pages_fetched = []
            gold_pages = []
            if hasattr(rc.output, 'answer'):
                output_str = rc.output.answer[:500]
                pages_fetched = rc.output.pages_fetched
            else:
                output_str = str(rc.output)[:500]
            if hasattr(rc.inputs, 'gold_pages'):
                gold_pages = rc.inputs.gold_pages

            cases.append({
                "name": rc.name,
                "passed": all(r.value for r in rc.assertions.values()),
                "assertions": assertions,
                "duration": round(rc.task_duration, 2),
                "input": rc.inputs.question if hasattr(rc.inputs, 'question') else str(rc.inputs),
                "output": output_str,
                "expected": str(rc.expected_output)[:500] if rc.expected_output else None,
                "pages_fetched": pages_fetched,
                "gold_pages": gold_pages,
            })

        _current_run["status"] = "completed"
        _current_run["completed_at"] = datetime.now(timezone.utc).isoformat()
        _current_run["cases"] = cases

    except Exception as e:
        _current_run["status"] = "failed"
        _current_run["completed_at"] = datetime.now(timezone.utc).isoformat()
        _current_run["error"] = str(e)

    # Persist to disk
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.write_text(json.dumps(_current_run, indent=2))

    result = _current_run
    _current_run = None
    return result
