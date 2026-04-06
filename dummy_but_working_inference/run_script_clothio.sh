module load Miniconda3
eval "$(conda shell.bash hook)"

conda activate /net/tscratch/people/plgwzarzecki/conda/wimu

module load CUDA/12.8 

python dummy_but_working_inference/script_clotho.py

