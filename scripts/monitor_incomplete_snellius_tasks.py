#!/usr/bin/env python3
"""Monitor incomplete Snellius full-dataset continuation tasks.

This script is intentionally conservative. It reports progress for the known
incomplete full-dataset methods and only auto-submits a follow-up job when the
action is non-duplicating and already decided by the experiment plan.
"""

from __future__ import annotations

import datetime as dt
import os
import re
import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DATASET_ROOT = Path("/gpfs/work3/0/prjs0839/data/UVG_CWI_DQPC/UVG-CWI-DQPC")
RESULTS_ROOT = Path("/gpfs/work3/0/prjs0839/results/PointCloudEnhancement/full_dataset/results")
REPORT_ROOT = REPO_ROOT / "logs" / "incomplete_task_monitor"

SEQUENCES = (
    "BlueSpeech",
    "BlueVolley",
    "BouncingBlue",
    "FitFluencer",
    "GoodVision",
    "Mannequin",
    "OrangeKettlebell",
    "PinkNoir",
    "TicTacToe",
    "TrumanShow",
    "VictoryHeart",
    "VirtualLife",
)

METHODS = {
    "PD-LTS": "pdlts_light_fbm_full",
    "SnowflakeNet-PU": "snowflakenet_pu_4x_full",
    "RepKPU": "repkpu_pu1k_4x_full",
    "Score-Denoise": "score_denoise_full",
    "MAG": "mag_full",
    "PU-Gaussian": "pu_gaussian_pu1k_4x_full",
    "PUDM": "pudm_pu1k_4x_chunk8192_step30_full",
}

WATCH_JOBS = ("24825178", "24825180", "24832268", "24832532", "24832534", "24832535", "24832618", "24833529")
SCORE_SMOKE_JOB = "24833529"
SCORE_TAIL_JOB_NAME = "score_tail_fix"
SCORE_TAIL_SCRIPT = REPO_ROOT / "jobs" / "score_denoise_bouncingblue_tail_emptyfix.slurm"


def run(cmd: list[str], check: bool = False) -> str:
    result = subprocess.run(cmd, cwd=REPO_ROOT, text=True, capture_output=True, check=check)
    return (result.stdout + result.stderr).strip()


def paired_frames(sequence: str) -> list[str]:
    cg = {path.stem.split("_")[-1] for path in (DATASET_ROOT / sequence / "cg" / "15fps").glob("*.ply")}
    he = {path.stem.split("_")[-1] for path in (DATASET_ROOT / sequence / "he" / "15fps").glob("*.ply")}
    return sorted(cg & he)


def output_frames(method_name: str, sequence: str) -> set[str]:
    out_dir = RESULTS_ROOT / "method_outputs" / method_name / sequence
    done: set[str] = set()
    if not out_dir.exists():
        return done
    for path in out_dir.rglob("*.ply"):
        match = re.search(r"(?:frame_)?(\d{4})", path.stem)
        if match:
            done.add(match.group(1))
    return done


def method_status(method_name: str) -> tuple[int, int, list[str]]:
    total_done = 0
    total_expected = 0
    lines = []
    for sequence in SEQUENCES:
        expected = paired_frames(sequence)
        done = output_frames(method_name, sequence)
        missing = [frame for frame in expected if frame not in done]
        total_done += len(done & set(expected))
        total_expected += len(expected)
        if missing:
            lines.append(
                f"  - {sequence}: remaining {len(missing)}/{len(expected)} "
                f"({missing[0]}..{missing[-1]})"
            )
    return total_done, total_expected, lines


def job_state(job_id: str) -> str:
    out = run(["sacct", "-j", job_id, "--format=JobID,JobName%24,State,Elapsed,ExitCode", "-P"])
    return out if out else f"{job_id}: no sacct record"


def has_active_job(job_name: str) -> bool:
    out = run(["squeue", "-u", os.environ.get("USER", ""), "-h", "-o", "%j"])
    return any(line.strip() == job_name for line in out.splitlines())


def maybe_submit_score_tail(report: list[str]) -> None:
    smoke_out = RESULTS_ROOT / "method_outputs" / "score_denoise_emptyfix_smoke" / "BouncingBlue" / "15fps" / "frame_0149.ply"
    if not smoke_out.exists():
        report.append("Score-Denoise action: smoke output is not present yet; no tail submission.")
        return
    if has_active_job(SCORE_TAIL_JOB_NAME):
        report.append("Score-Denoise action: BouncingBlue tail job is already active; no duplicate submission.")
        return
    expected = paired_frames("BouncingBlue")
    done = output_frames("score_denoise_full", "BouncingBlue")
    missing = [frame for frame in expected if frame not in done]
    target = [frame for frame in missing if "0149" <= frame <= "0156"]
    if not target:
        report.append("Score-Denoise action: BouncingBlue tail is already complete.")
        return
    if target != missing:
        report.append(
            "Score-Denoise action: BouncingBlue has unexpected missing frames "
            f"{missing}; needs manual decision."
        )
        return
    if not SCORE_TAIL_SCRIPT.exists():
        report.append(f"Score-Denoise action: missing {SCORE_TAIL_SCRIPT}; needs manual decision.")
        return
    submit = run(["sbatch", str(SCORE_TAIL_SCRIPT)])
    report.append(f"Score-Denoise action: submitted BouncingBlue tail for {target[0]}..{target[-1]}: {submit}")


def main() -> None:
    REPORT_ROOT.mkdir(parents=True, exist_ok=True)
    now = dt.datetime.now().astimezone()
    report = [f"# Snellius Incomplete Task Monitor", "", f"Time: {now.isoformat(timespec='seconds')}", ""]

    report.append("## Queue")
    report.append(run(["squeue", "-u", os.environ.get("USER", ""), "-o", "%.18i %.9P %.28j %.8T %.10M %.10l %.6D %R"]) or "No active jobs.")
    report.append("")

    report.append("## Known Job Accounting")
    for job_id in WATCH_JOBS:
        report.append(job_state(job_id))
        report.append("")

    report.append("## Incomplete Method Counts")
    for label, method_name in METHODS.items():
        done, expected, details = method_status(method_name)
        report.append(f"{label}: remaining {expected - done}/{expected} frames")
        report.extend(details[:12])
        report.append("")

    report.append("## Conservative Actions")
    maybe_submit_score_tail(report)
    report.append("")

    text = "\n".join(report).rstrip() + "\n"
    stamp = now.strftime("%Y%m%d_%H%M%S")
    (REPORT_ROOT / f"{stamp}.md").write_text(text, encoding="utf-8")
    (REPORT_ROOT / "latest.md").write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
