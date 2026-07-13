#!/bin/bash
#SBATCH --job-name=uvg_dqpc_dl
#SBATCH --partition=staging
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=2-00:00:00
#SBATCH --output=logs/download_uvg_cwi_dqpc_he15_cgv2_15_%j.out

set -euo pipefail

cd /gpfs/home2/xzhou/code/PointCloudEnhancement

DATA_PARENT=/gpfs/work3/0/prjs0839/data/UVG_CWI_DQPC
DATA_ROOT="${DATA_PARENT}/UVG-CWI-DQPC"
ZIP_DIR="${DATA_ROOT}/__zip"
JSON_FILE="${DATA_ROOT}/UVG-CWI-DQPC.json"
JSON_URL="https://ultravideo.fi/UVG-CWI-DQPC/UVG-CWI-DQPC.json"

mkdir -p "${ZIP_DIR}" logs

echo "[info] downloading metadata ${JSON_URL}"
curl -fsSL "${JSON_URL}" -o "${JSON_FILE}"

echo "[info] selected types: HE_15, CGv2_15"
jq -r '
  .sequences[]
  | .sequence as $seq
  | .links
  | to_entries[]
  | select(.key == "HE_15" or .key == "CGv2_15")
  | [$seq, .key, .value] | @tsv
' "${JSON_FILE}" > "${DATA_ROOT}/download_manifest_he15_cgv2_15.tsv"

echo "[info] manifest:"
cat "${DATA_ROOT}/download_manifest_he15_cgv2_15.tsv"

while IFS=$'\t' read -r seq typ url; do
  archive="${ZIP_DIR}/$(basename "${url}")"
  echo "[download] ${seq} ${typ} ${url}"
  if [ ! -s "${archive}" ]; then
    curl -fL --retry 5 --retry-delay 20 -o "${archive}.part" "${url}"
    mv "${archive}.part" "${archive}"
  else
    echo "[skip] archive exists: ${archive}"
  fi

  echo "[unzip] ${archive}"
  unzip -q -o "${archive}" -d "${DATA_PARENT}"
done < "${DATA_ROOT}/download_manifest_he15_cgv2_15.tsv"

echo "[done] dataset root: ${DATA_ROOT}"
find "${DATA_ROOT}" -maxdepth 3 -type d | sort | sed -n '1,200p'
