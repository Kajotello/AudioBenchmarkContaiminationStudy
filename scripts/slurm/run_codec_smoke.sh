#!/usr/bin/env bash
set -euo pipefail

# Minimal smoke test for the CoDeC contamination detection pipeline.
# Requires:
#   - the disk dataset produced by scripts/download_dataset.py
#     (./data/CLAPv2__Clotho/{train,validation}/{audio,metadata.jsonl})
#   - the project conda env activated (see setup_env.sh)

# --- AF3 ---
# python src/contamination.py \
#     model=audio_flamingo3 \
#     method=codec \
#     method.mode=full \
#     context_pool_size=4 \
#     max_member_samples=2 \
#     max_non_member_samples=2 \
#     batch_size=1

# python src/contamination.py \
#     model=audio_flamingo3 \
#     method=codec \
#     method.mode=no_audio \
#     context_pool_size=4 \
#     max_member_samples=2 \
#     max_non_member_samples=2 \
#     batch_size=1

# --- AF2 ---
python -m src.contamination \
    model=audio_flamingo2 \
    method=codec \
    method.mode=full \
    context_pool_size=4 \
    max_member_samples=2 \
    max_non_member_samples=2 \
    batch_size=1

python -m src.contamination \
    model=audio_flamingo2 \
    method=codec \
    method.mode=no_audio \
    context_pool_size=4 \
    max_member_samples=2 \
    max_non_member_samples=2 \
    batch_size=1
