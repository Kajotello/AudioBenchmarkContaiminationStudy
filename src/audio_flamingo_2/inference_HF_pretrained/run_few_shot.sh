#!/usr/bin/env bash
set -euo pipefail

MUSIC_DIR="../../../music"

AUDIO1="${MUSIC_DIR}/50213.wav"
AUDIO2="${MUSIC_DIR}/50367.wav"

PROMPT1="what instruments can you hear in this audio?"
ANSW1="the recording features a piano and light percussion with a steady rhythm"
PROMPT2="what instruments can you hear in this audio?"

python inference.py \
    --audio1 "${AUDIO1}" \
    --prompt1 "${PROMPT1}" \
    --answ1  "${ANSW1}" \
    --audio2 "${AUDIO2}" \
    --prompt2 "${PROMPT2}"
