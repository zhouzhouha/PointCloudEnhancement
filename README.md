# GrandChallenge — Usage

This repo contains a small CLI tool to compute point-cloud metrics between matched PLY files.

Launcher
--------

Use the PowerShell launcher `run_metrics.ps1` from the repository root. The script will ensure
the project's virtual environment exists, install `requirements.txt`, and run the metrics.

Examples:

Run a single pair (PLY files):

```powershell
.\run_metrics.ps1 -GtDir "UVG-CWI-DQPC/OrangeKettlebell/CG/15fps/OrangeKettlebell_..._0000.ply" -RecDir "UVG-CWI-DQPC/OrangeKettlebell/HE/15fps/OrangeKettlebell_..._0000.ply" -OutCsv onepair_results.csv -ColorWeight 0
```

Run directories (matching basenames):

```powershell
.\run_metrics.ps1 -GtDir "path\to\gt_dir" -RecDir "path\to\rec_dir" -OutCsv results.csv
```

Notes
-----
- The launcher has a `-ColorWeight` argument (default 0). Set to 0 to ignore color and compute
  distances on XYZ only.
- Use `-OutCsv` to control output path. Use `-Append` to append results to an existing CSV.
- The venv is created at `./venv` and used automatically by the launcher.
