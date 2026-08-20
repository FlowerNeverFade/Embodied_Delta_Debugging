#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -ne 1 ]]; then
  echo "usage: $0 OUTPUT_DIR" >&2
  exit 2
fi

OUT="$1"
mkdir -p "$OUT"
echo "$$" > "$OUT/watch.pid"
echo "$(date -Is) watcher starting" | tee "$OUT/watch.log"

while true; do
  running=0
  for pid_file in "$OUT"/gpu*/run.pid; do
    [[ -f "$pid_file" ]] || continue
    pid="$(cat "$pid_file")"
    if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then
      running=$((running + 1))
    fi
  done
  echo "$(date -Is) running_shards=$running" | tee -a "$OUT/watch.log"
  [[ "$running" -eq 0 ]] && break
  sleep 60
done

/data2/yanghaoyun/envs/libero38/bin/python - <<'PY' "$OUT"
import json
import sys
from pathlib import Path

out = Path(sys.argv[1])
summary = {
    "schema_version": "source-aware-top-candidates-summary-v1",
    "output_dir": str(out),
    "shards": [],
    "num_cases": 0,
    "num_recorded_success": 0,
    "num_recorded_improvement": 0,
    "num_observed_success": 0,
    "num_observed_improvement": 0,
    "num_reported_vs_recorded_mismatch": 0,
    "num_missing_manifests": 0,
}

html = [
    "<!doctype html><meta charset='utf-8'><title>Source-aware top candidates</title>",
    "<h1>Source-aware top candidates</h1>",
    "<p>Each shard uses a 3x2 panel: original, minimal replay, raw-policy slot, language repair, visual repair, demo/success repair. In recorded_error mode the raw-policy slot follows the original failed actions and is not counted as repair evidence.</p>",
]

for shard_dir in sorted(item for item in out.glob("gpu*") if item.is_dir()):
    manifest_path = shard_dir / "review" / "review_manifest.json"
    shard = {
        "shard": shard_dir.name,
        "manifest_path": str(manifest_path),
        "review_index": str(shard_dir / "review" / "review_index.html"),
        "num_cases": 0,
        "cases": [],
        "status": "missing_manifest",
    }
    html.append(f"<h2>{shard_dir.name}</h2>")
    if not manifest_path.exists():
        summary["num_missing_manifests"] += 1
        html.append("<p>missing manifest</p>")
        summary["shards"].append(shard)
        continue
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    shard["status"] = "ok"
    shard["num_cases"] = data.get("num_cases", 0)
    html.append(f"<p><a href='{manifest_path.parent / 'review_index.html'}'>review_index.html</a></p>")
    html.append("<ul>")
    for case in data.get("cases", []):
        evidence = case.get("recorded_repair_evidence") or {}
        row = {
            "case_id": case.get("case_id"),
            "task_id": case.get("task_id"),
            "init_state_id": case.get("init_state_id"),
            "seed": case.get("seed"),
            "failure_type": case.get("failure_type"),
            "reported_full_success_repair_pass": case.get("full_success_repair_pass"),
            "recorded_any_success": evidence.get("any_success"),
            "recorded_any_improvement": evidence.get("any_improvement"),
            "observed_any_success": evidence.get("observed_any_success"),
            "observed_any_improvement": evidence.get("observed_any_improvement"),
            "coordinate_mismatch_count": evidence.get("coordinate_mismatch_count"),
            "reported_vs_recorded_mismatch": evidence.get("reported_vs_recorded_mismatch"),
            "case_review_path": case.get("case_review_path"),
        }
        shard["cases"].append(row)
        summary["num_cases"] += 1
        if row["recorded_any_success"] is True:
            summary["num_recorded_success"] += 1
        if row["recorded_any_improvement"] is True:
            summary["num_recorded_improvement"] += 1
        if row["observed_any_success"] is True:
            summary["num_observed_success"] += 1
        if row["observed_any_improvement"] is True:
            summary["num_observed_improvement"] += 1
        if row["reported_vs_recorded_mismatch"] is True:
            summary["num_reported_vs_recorded_mismatch"] += 1
        case_review = row["case_review_path"] or ""
        html.append(
            "<li>"
            f"{row['case_id']}: recorded_success={row['recorded_any_success']}, "
            f"improvement={row['recorded_any_improvement']}, "
            f"observed_success={row['observed_any_success']}, "
            f"coordinate_mismatch_count={row['coordinate_mismatch_count']}, "
            f"mismatch={row['reported_vs_recorded_mismatch']} "
            f"<a href='{case_review}'>case_review.json</a>"
            "</li>"
        )
    html.append("</ul>")
    summary["shards"].append(shard)

(out / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
(out / "review_index.html").write_text("\n".join(html), encoding="utf-8")
print(json.dumps(summary, indent=2, ensure_ascii=False))
PY

echo "$(date -Is) watcher finished" | tee -a "$OUT/watch.log"
