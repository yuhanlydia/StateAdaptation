"""Isolated, fixed-protocol replay of one anomalous table state."""

import argparse
from copy import deepcopy
import json
from pathlib import Path

import numpy as np

from aperture_table.cli import child, gpu_batches
from aperture_table.protocol import make_jobs
from aperture_table.worker import seal_unit
from vigor_handoff.protocol import atomic_json, read_jsonl


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--domain", required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--arm", required=True)
    p.add_argument("--gpu", required=True)
    a = p.parse_args()
    config = json.loads(Path(a.config).read_text())
    root = Path(config["run_root"])
    original = {j["arm"]: j for j in make_jobs(config)
                if (j["model_key"], j["domain_key"], j["support_seed"])
                == (a.model, a.domain, a.seed)}
    if a.arm not in original:
        p.error("unknown model/domain/seed/arm combination")
    job = deepcopy(original[a.arm])
    job["job_id"] += "--diagnostic-repeat"
    output = root / "diagnostics" / f"{a.model}--{a.domain}--s{a.seed}--{a.arm}"
    job["output"] = str(output / "state")

    def work(_, gpu):
        child(a.config, root, job, "fit", gpu)
        paths = [Path(j["output"]) / "fit/calibration_predictions.jsonl"
                 for j in (original[a.arm], job)]
        before, after = (read_jsonl(path) for path in paths)
        if [(r["sample_id"], r["label_id"]) for r in before] != [
                (r["sample_id"], r["label_id"]) for r in after]:
            raise ValueError("calibration identities changed")
        old = np.asarray([r["mean_log_scores"] for r in before], dtype=float)
        new = np.asarray([r["mean_log_scores"] for r in after], dtype=float)
        delta = float(np.max(np.abs(old - new)))
        selection = output / "selection.json"
        seal_unit([job if arm == a.arm else j for arm, j in original.items()], selection)
        child(a.config, root, job, "evaluate", gpu, selection)
        pair = [json.loads((Path(j["output"]) / "evaluation/evaluation.json").read_text())
                for j in (original[a.arm], job)]
        fields = ("macro_f1", "nll", "confusion_matrix", "accuracy", "ece", "brier")
        result = {
            "job_id": original[a.arm]["job_id"],
            "calibration_score_max_abs_delta": delta,
            "original_raw": {key: pair[0]["raw"][key] for key in fields},
            "replay_raw": {key: pair[1]["raw"][key] for key in fields},
            "original_temperature": {key: pair[0]["temperature"][key] for key in fields},
            "replay_temperature": {key: pair[1]["temperature"][key] for key in fields},
            "query_metrics_used_for_selection": False,
        }
        atomic_json(output / "report.json", result)
        print(json.dumps(result, indent=2), flush=True)

    gpu_batches([job], [a.gpu], work)


if __name__ == "__main__":
    main()
