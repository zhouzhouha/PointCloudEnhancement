<#
run_metrics.ps1

Usage: (from repo root)
  .\run_metrics.ps1 -GtDir <gt_path> -RecDir <rec_path> [-OutCsv results.csv]

This script ensures the project's virtual environment exists, installs `requirements.txt`,
and runs `pointcloud_metrics.py` with the provided arguments. It uses the venv python directly
so activation is not required.
#>

param(
  [Parameter(Mandatory=$true)][string]$GtDir,
  [Parameter(Mandatory=$true)][string]$RecDir,
  [string]$OutCsv = 'results.csv',
  [double]$ColorWeight = 0.0,
  [switch]$Append
)

Push-Location -Path (Split-Path -Parent $MyInvocation.MyCommand.Definition)

$venvDir = Join-Path $PWD 'venv'
$venvPython = Join-Path $venvDir 'Scripts\python.exe'

if (-not (Test-Path $venvPython)) {
    Write-Output "Creating virtual environment..."
    python -m venv $venvDir
}

Write-Output "Upgrading pip and installing requirements..."
& $venvPython -m pip install --upgrade pip
& $venvPython -m pip install -r requirements.txt

Write-Output "Running pointcloud_metrics.py"
  if ((Test-Path $GtDir -PathType Leaf) -and (Test-Path $RecDir -PathType Leaf)) {
  Write-Output "Detected file paths. Running onepair_runner.py for a single pair..."
  if ($Append.IsPresent) { $appendFlag = '--append' } else { $appendFlag = '' }
  & $venvPython .\onepair_runner.py $GtDir $RecDir $OutCsv --color_weight $ColorWeight $appendFlag
} else {
  if ($Append.IsPresent) { $appendFlag = '--append' } else { $appendFlag = '' }
  & $venvPython .\pointcloud_metrics.py --gt_dir $GtDir --rec_dir $RecDir --out_csv $OutCsv --color_weight $ColorWeight $appendFlag
}

Pop-Location
