#!/bin/bash


SEQ_FILTER="all"
TYPE_FILTER="all"
SKIP_CHECK=0
KEEP_ARCHIVES=0
FORCE_DL=0
INSTALL=0
INSTALL_PATH="$(pwd)"

info() {
  echo -e "\033[0;34m$*\033[0m"
}
error() {
  echo -e "\033[0;31m$*\033[0m"
}

usage() {
  cat <<EOF
Usage: $0 [options]

Options:
  -s <values>             Comma-separated sequence filter (default: all)
  -t <values>             Comma-separated type filter (default: all)
  --skip-check            Skip the user confirmation prompt
  --keep-archives         Download archives
  --force-download        Force to redownload archives
  --install[=<path>]      Install the script to a path (default: $INSTALL_PATH/UVG-CWI-DQPC)
  -h, --help              Show this help message and exit

Examples:
  $0 -s BlueSpeech,VirtualLife -t HE_15,CG_15
  $0 -s all -t HE_15,CG_15
  $0 -s all --skip-check
EOF
}

JSON_URL="https://ultravideo.fi/UVG-CWI-DQPC/UVG-CWI-DQPC.json"
JSON_FILE="UVG-CWI-DQPC.json"

for cmd in jq curl unzip; do
    if ! command -v $cmd >/dev/null 2>&1; then
        error "Error: '$cmd' is not installed. Please install it first."
        exit 1
    fi
done

curl -s -o "$JSON_FILE" "$JSON_URL"
if [ $? -ne 0 ]; then
  error "Error: Failed to download JSON."
  exit 1
fi


while [[ $# -gt 0 ]]; do
    key="$1"
    case $key in
        -s)
            SEQ_FILTER="$2"
            shift 2
            ;;
        -t)
            TYPE_FILTER="$2"
            shift 2
            ;;
        --skip-check)
            SKIP_CHECK=1
            shift
            ;;
        --keep_archives)
            KEEP_ARCHIVES=1
            shift
            ;;
        --force-download)
            FORCE_DL=1
            shift
            ;;
        --install=*)
            INSTALL=1
            INSTALL_PATH="${1#*=}"
            shift
            ;;
        --install)
            INSTALL=1
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            info "Unknown option: $1"
            usage
            exit 1
            ;;
    esac
done

build_filter() {
  local input="$1"
  if [ "$input" = "all" ]; then
    echo '[]'
  else
    IFS=',' read -r -a arr <<< "$input"
    printf '%s\n' "${arr[@]}" | sed 's/^[[:space:]]*//;s/[[:space:]]*$//' | jq -R . | jq -s .
  fi
}

SEQ_FILTER_JQ=$(build_filter "$SEQ_FILTER")
TYPE_FILTER_JQ=$(build_filter "$TYPE_FILTER")

JQ_FILTER='
  .sequences[]
  | .sequence as $seq
  | select(($seqfilter | length) == 0 or ($seqfilter | any(. == ($seq|tostring))))
  | select(.links != null)
  | .links | to_entries[]
  | .key as $key
  | select(($typefilter | length) == 0 or ($typefilter | any(. == ($key|tostring))))
  | "\($seq)|\(.key)|\(.value)"
'

if [ "$SKIP_CHECK" -eq 0 ]; then
    jq -r --argjson seqfilter "$SEQ_FILTER_JQ" --argjson typefilter "$TYPE_FILTER_JQ" "$JQ_FILTER" "$JSON_FILE" | while IFS='|' read -r SEQ TYPE LINK; do
    info "=> '$SEQ' (type: $TYPE): $LINK"
    FILENAME="${SEQ}_${TYPE}_$(basename "$LINK")"
    done
    
    read -p "will be downloaded, do you want to continue ? (Y/n)" answer
    answer=$(echo "$answer" | tr '[:upper:]' '[:lower:]')
    if [ "$answer" != "y" ]; then
        info "Aborting..."
        exit 1
    fi
fi

mkdir -p $INSTALL_PATH/UVG-CWI-DQPC/__zip
if [ $? -ne 0 ]; then
    error "Error: Failed to create download directory."
    exit 1
fi

jq -r --argjson seqfilter "$SEQ_FILTER_JQ" --argjson typefilter "$TYPE_FILTER_JQ" "$JQ_FILTER" "$JSON_FILE" | while IFS='|' read -r SEQ TYPE LINK; do
    info "Downloading $LINK for sequence '$SEQ' (type: $TYPE)..."
    FILENAME=$INSTALL_PATH/UVG-CWI-DQPC/__zip/$(basename $LINK)
    if [ -f "$FILENAME" ] && [ $FORCE_DL -eq 0 ]; then
        info "File '$FILENAME' already exists. Skipping download."
        continue
    else
        rm -f "$FILENAME"
    fi
    curl -L -o "$FILENAME" "$LINK"
    if [ $? -ne 0 ]; then
        error "Error: Failed to download $LINK"
        continue
    fi
    # if install then unzip to the install path
    if [ $INSTALL -eq 1 ]; then
        info "Unzipping $FILENAME to $INSTALL_PATH..."
        unzip -q "$FILENAME" -d $(dirname "$INSTALL_PATH")
        if [ $? -ne 0 ]; then
            error "Error: Failed to unzip $FILENAME"
            continue
        fi
        if [ $KEEP_ARCHIVES -eq 0 ]; then
            rm -f "$FILENAME"
        fi
    fi
done