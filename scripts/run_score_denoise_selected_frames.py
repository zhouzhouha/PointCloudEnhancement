"""Run Score-Denoise on selected UVG-CWI-DQPC frames.

Score-Denoise has the same `test_large.py` interface as MAG. Reuse the MAG
adapter and only swap the official repository/checkpoint paths.
"""

from __future__ import annotations

from pathlib import Path

import run_mag_selected_frames as runner


SCORE_ROOT = runner.REPO_ROOT / "third_party" / "enhancement" / "score-denoise"

runner.MAG_ROOT = SCORE_ROOT
runner.MAG_CKPT = SCORE_ROOT / "pretrained" / "ckpt.pt"


if __name__ == "__main__":
    runner.main()
