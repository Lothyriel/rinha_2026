#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
RESOURCES_DIR="${1:-$ROOT_DIR/spec/resources}"
OUTPUT_PATH="index.mmap"
KMEANS_K="${RINHA_KMEANS_K:-2048}"
KMEANS_SEED="${RINHA_KMEANS_SEED:-67}"

mkdir -p "$(dirname "$OUTPUT_PATH")"

if [ ! -f "$OUTPUT_PATH" ]; then
	: >"$OUTPUT_PATH"
fi

printf 'Generating centroid mmap\n'
printf '  resources: %s\n' "$RESOURCES_DIR"
printf '  output:    %s\n' index.mmap
printf '  kmeans_k:  %s\n' "$KMEANS_K"
printf '  seed:      %s\n' "$KMEANS_SEED"

RINHA_KMEANS_K="$KMEANS_K" \
	RINHA_KMEANS_SEED="$KMEANS_SEED" \
	cargo run --release --bin build_index -- index.mmap "$OUTPUT_PATH" "$KMEANS_K"

printf 'Done: %s\n' "$OUTPUT_PATH"
