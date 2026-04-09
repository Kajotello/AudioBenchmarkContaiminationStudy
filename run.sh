#!/bin/bash
module load Miniconda3
eval "$(conda shell.bash hook)"
conda activate /net/tscratch/people/plgrszczepaniak/conda/spzc
module load CUDA/12.8

export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
python src/eval.py
