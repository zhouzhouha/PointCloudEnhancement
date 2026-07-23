"""Run one included enhancement method on one full UVG-CWI-DQPC sequence."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
PROJECT_RESULTS = Path("/gpfs/work3/0/prjs0839/results/PointCloudEnhancement/full_dataset/results")
DEFAULT_DATASET_ROOT = Path("/gpfs/work3/0/prjs0839/data/UVG_CWI_DQPC/UVG-CWI-DQPC")


METHODS = {
    "pdlts": {
        "script": "scripts/run_pdlts_selected_frames.py",
        "method_name": "pdlts_light_fbm_full",
        "args": ["--variant", "light", "--patch-size", "1024", "--niters", "1", "--seed-k", "5"],
        "results_root": True,
    },
    "pathnet": {
        "script": "scripts/run_pathnet_selected_frames.py",
        "method_name": "pathnet_chunked_full",
        "args": ["--chunk-size", "50000", "--batch-size", "512", "--knn", "128", "--iterations", "2"],
        "results_root": True,
    },
    "apuldi": {
        "script": "scripts/run_apuldi_selected_frames.py",
        "method_name": "apuldi_local_pu1k_4x_2048_full",
        "args": ["--input-points", "2048", "--device", "cuda"],
        "results_root": True,
        "env": {"PCE_FORCE_APULDI_FALLBACKS": "1"},
    },
    "pc2pu": {
        "script": "scripts/run_pc2pu_selected_frames.py",
        "method_name": "pc2pu_4x_chunks256_full",
        "args": ["--chunk-size", "256", "--batch-chunks", "16", "--up-ratio", "4"],
        "results_root": True,
    },
    "neuralpoints": {
        "script": "scripts/run_neuralpoints_selected_frames.py",
        "method_name": "neuralpoints_16x_2048_full",
        "args": ["--input-points", "2048"],
    },
    "pdflow": {
        "script": "scripts/run_pdflow_selected_frames.py",
        "method_name": "pdflow_full_dataset",
        "args": [],
        "results_root": True,
    },
    "p2p_bridge": {
        "script": "scripts/run_p2p_bridge_selected_frames.py",
        "method_name": "p2p_bridge_pvds_punet_full",
        "args": ["--steps", "5", "--k", "3"],
        "results_root": True,
    },
    "pudm": {
        "script": "scripts/run_pudm_selected_frames.py",
        "method_name": "pudm_pu1k_4x_full",
        "args": [
            "--checkpoint",
            "third_party/enhancement/PUDM/pointnet2/pkls/pu1k.pkl",
            "--config",
            "third_party/enhancement/PUDM/pointnet2/exp_configs/PU1K.json",
            "--chunk-size",
            "2048",
            "--up-rate",
            "4",
            "--step",
            "30",
            "--gamma",
            "0.5",
        ],
        "env": {"PCE_FORCE_PUDM_FALLBACKS": "1"},
    },
    "pucrn": {
        "script": "scripts/run_pucrn_selected_frames.py",
        "method_name": "pucrn_pu1k_4x_full",
        "args": ["--chunk-size", "2048", "--up-ratio", "4"],
    },
    "repkpu": {
        "script": "scripts/run_repkpu_selected_frames.py",
        "method_name": "repkpu_pu1k_4x_full",
        "args": ["--chunk-size", "8192", "--up-rate", "4", "--patch-rate", "3"],
    },
    "spupmd": {
        "script": "scripts/run_spupmd_selected_frames.py",
        "method_name": "spupmd_pu1k_4x_full",
        "args": ["--chunk-size", "2048", "--num-point", "256", "--up-ratio", "4", "--patch-num-ratio", "3.0"],
        "env": {"PCE_FORCE_SPUPMD_FALLBACKS": "1"},
    },
    "pufm": {
        "script": "scripts/run_pufm_selected_frames.py",
        "method_name": "pufm_pugan_4x_full",
        "args": ["--model", "pufm", "--chunk-size", "2048", "--up-rate", "4", "--num-points", "256", "--steps", "5"],
        "env": {"PCE_FORCE_PUFM_FALLBACKS": "1"},
    },
    "puflow": {
        "script": "scripts/run_puflow_selected_frames.py",
        "method_name": "puflow_discrete_full",
        "args": ["--chunk-size", "8192", "--num-patch", "256", "--up-ratio", "4"],
    },
    "crcir": {
        "script": "scripts/run_crcir_selected_frames.py",
        "method_name": "crcir_aftercomp_4x_full",
        "args": ["--chunk-size", "2048", "--encoder-downsample", "2", "--decoder-multiplier", "4"],
    },
    "pointcleannet": {
        "script": "scripts/run_pointcleannet_selected_frames.py",
        "method_name": "pointcleannet_full",
        "args": ["--nrun", "3", "--workers", "1", "--cache-capacity", "1"],
    },
    "mag": {
        "script": "scripts/run_mag_selected_frames.py",
        "method_name": "mag_full",
        "args": ["--cluster-size", "30000"],
    },
    "score_denoise": {
        "script": "scripts/run_score_denoise_selected_frames.py",
        "method_name": "score_denoise_full",
        "args": ["--cluster-size", "30000"],
    },
    "gqenet": {
        "script": "scripts/run_gqenet_selected_frames.py",
        "method_name": "gqenet_full",
        "args": ["--test-batch-size", "8"],
    },
    "upsample_clean": {
        "script": "scripts/run_upsample_clean_selected_frames.py",
        "method_name": "upsample_clean_ounet_full",
        "args": [],
    },
    "spu": {
        "script": "scripts/run_spu_selected_frames.py",
        "method_name": "spu_pointnet_4x_full",
        "args": ["--chunk-size", "2048", "--scale", "4"],
    },
    "snowflakenet_pu": {
        "script": "scripts/run_snowflakenet_pu_selected_frames.py",
        "method_name": "snowflakenet_pu_4x_full",
        "args": ["--chunk-size", "8192", "--patch-num-ratio", "3", "--num-per-patch", "256", "--up-ratio", "4"],
    },
    "pu_gaussian": {
        "script": "scripts/run_pu_gaussian_selected_frames.py",
        "method_name": "pu_gaussian_pu1k_4x_full",
        "args": ["--checkpoint", "pu1k", "--patch-size", "10000", "--patch-rate", "3", "--up-ratio", "4", "--num-samples", "6"],
    },
    "gradpu": {
        "script": "scripts/run_gradpu_selected_frames.py",
        "method_name": "gradpu_chunked_4x_full",
        "args": ["--up-rate", "4", "--chunk-size", "2048"],
        "results_root": True,
    },
    "iterativepfn": {
        "script": "scripts/run_iterativepfn_selected_frames.py",
        "method_name": "iterativepfn_full",
        "args": [
            "--device", "cuda", "--patch-size", "1000", "--niters", "1",
            "--seed-k", "6", "--seed-k-alpha", "10",
        ],
        "results_root": True,
    },
}


def frame_id(path: Path) -> str:
    return path.stem.split("_")[-1]


def common_frames(dataset_root: Path, sequence: str) -> list[str]:
    cg_dir = dataset_root / sequence / "cg" / "15fps"
    he_dir = dataset_root / sequence / "he" / "15fps"
    cg = {frame_id(path) for path in cg_dir.glob("*.ply")}
    he = {frame_id(path) for path in he_dir.glob("*.ply")}
    frames = sorted(cg & he)
    if not frames:
        raise FileNotFoundError(f"No paired frames for {sequence}: {cg_dir} / {he_dir}")
    return frames


def ensure_project_symlinks(method_name: str, sequence: str, results_root: Path) -> None:
    links = [
        (
            REPO_ROOT / "results" / "method_outputs" / method_name,
            results_root / "method_outputs" / method_name,
        ),
        (
            REPO_ROOT / "results" / "work" / method_name,
            results_root / "work" / method_name,
        ),
        (
            REPO_ROOT / "results" / "uvg_cwi_dqpc" / sequence / method_name,
            results_root / "uvg_cwi_dqpc" / sequence / method_name,
        ),
    ]
    for link, target in links:
        target.mkdir(parents=True, exist_ok=True)
        link.parent.mkdir(parents=True, exist_ok=True)
        if link.exists() or link.is_symlink():
            if link.is_symlink() and link.resolve() == target:
                continue
            if link.is_dir() and not link.is_symlink():
                continue
            link.unlink()
        if not link.exists():
            link.symlink_to(target, target_is_directory=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", required=True, choices=sorted(METHODS))
    parser.add_argument(
        "--method-name",
        help="Override the configured result name, for immutable/resumable batch runs",
    )
    parser.add_argument("--sequence", required=True)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--results-root", type=Path, default=PROJECT_RESULTS)
    parser.add_argument("--frames", nargs="+", help="Optional paired frame IDs for compatibility smoke/resume runs")
    args = parser.parse_args()

    spec = METHODS[args.method]
    method_name = args.method_name or str(spec["method_name"])
    paired_frames = common_frames(args.dataset_root, args.sequence)
    if args.frames:
        missing = sorted(set(args.frames) - set(paired_frames))
        if missing:
            raise ValueError(f"Requested frames are not paired for {args.sequence}: {missing}")
        frames = args.frames
    else:
        frames = paired_frames
    ensure_project_symlinks(method_name, args.sequence, args.results_root)

    env = os.environ.copy()
    env.setdefault("TORCH_EXTENSIONS_DIR", str(args.results_root.parent / "torch_extensions"))
    env.setdefault("TMPDIR", str(args.results_root.parent / "tmp"))
    Path(env["TORCH_EXTENSIONS_DIR"]).mkdir(parents=True, exist_ok=True)
    Path(env["TMPDIR"]).mkdir(parents=True, exist_ok=True)
    env.update(spec.get("env", {}))

    method_args = list(spec["args"])
    if args.method == "iterativepfn" and os.environ.get("PCE_ITERATIVEPFN_PATCH_SIZE"):
        patch_size = os.environ["PCE_ITERATIVEPFN_PATCH_SIZE"]
        patch_index = method_args.index("--patch-size") + 1
        method_args[patch_index] = patch_size
    if args.method == "iterativepfn" and os.environ.get("PCE_ITERATIVEPFN_SEED_K_ALPHA"):
        seed_k_alpha = os.environ["PCE_ITERATIVEPFN_SEED_K_ALPHA"]
        alpha_index = method_args.index("--seed-k-alpha") + 1
        method_args[alpha_index] = seed_k_alpha

    cmd = [
        sys.executable,
        "-u",
        str(REPO_ROOT / str(spec["script"])),
        "--sequence",
        args.sequence,
        "--frames",
        *frames,
        "--dataset-root",
        str(args.dataset_root),
        "--method-name",
        method_name,
        *method_args,
    ]
    if spec.get("results_root"):
        cmd.extend(["--results-root", str(args.results_root)])

    print(f"[full-dataset] method={args.method} method_name={method_name} sequence={args.sequence} frames={len(frames)}")
    print("[cmd]", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=str(REPO_ROOT), env=env, check=True)


if __name__ == "__main__":
    main()
