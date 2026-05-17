# Audio Contamination: Benchmark Contamination Detection for Audio Language Models
### Projekt WIMU 26L - zespół 10 - Kajetan Rożej, Rafał Szczepaniak, Wojciech Zarzecki

## Dependencies fun
### conclusion about 2 environments
#### cause
- HF removed from datasets > 4.0 the old soundfile/torchaudio backend and now requires torchcodec to decode the Audio feature. If torchcodec isn't installed you get ImportError: To support decoding audio data, please install 'torchcodec'. The problem is that torchcodec wheels are tightly pinned to a specific torch version (eg torchcodec 0.8 to PyTorch 2.9`) 
- Audio Flamingo 2 and 3 have pinned old torch versions 

#### solution two environments 
1) one for running model inference give path to physical file (eg .wav file) - soundfile loads file with no problem; Audio Flamingo can used pinned old torch
2) another to convert HF dataset to local files on disk - one which requires torchcodec with newer torch (or find drive with physical files)

### Dump AF3 on old GPU
After a longer while of struggling found on https://huggingface.co/nvidia/audio-flamingo-3-hf that: Supported Hardware: NVIDIA Ampere (A100), NVIDIA Hopper (H100) - so older GPUs like A5000 won't work :(


### How to run AF2
create python 3.10 env and install requirements from src/audio_flamingo_2/requirements.txt (AF2 native and project specific eg hydra)

## Outline

This project aims to adapt benchmark contamination detection methods — originally developed for LLMs and Vision-Language Models (VLMs) — to **Audio Language Models (ALMs)**. Building on the "slot guessing for perturbed captions" technique from *Both Text and Images Leaked!* [3] and multi-modal semantic perturbation approaches [5], we will design audio-domain equivalents of semantic perturbations (e.g., pitch shifts, tempo changes, noise injections) to probe whether a model was trained on supposedly held-out evaluation data. Using datasets with explicit train/test boundaries (Clotho-AQA, AudioMCQ), found contamination signals will be used to: (1) verify the feasibility of audio-domain contamination detection, and (2) evaluate detection robustness across multiple ALMs (Audio-Reasoner, Flamingo Audio).

## References

1. Y. Wang et al., "Beyond Boundaries: A Comprehensive Survey of Transferable Attacks on AI Systems," arXiv, 2024.
2. N. Carlini et al., "Membership Inference Attacks from First Principles," in *IEEE S&P*, 2022, doi: 10.1109/SP46214.2022.9833649.
3. Anonymous, "Both Text and Images Leaked! A Systematic Analysis of Multimodal LLM Data Contamination," EMNLP 2025.
4. Anonymous, "Detecting Data Contamination in LLMs via In-Context Learning," ICLR 2026.
5. Anonymous, "Contamination Detection for VLMs using Multi-Modal Semantic Perturbation," ICLR 2026.
6. Z. Xie et al., "Audio-Reasoner," HuggingFace: `zhifeixie/Audio-Reasoner`, 2025.
7. NVIDIA, "Flamingo Audio," arXiv, doi: 10.48550/arXiv.2511.10289, Nov. 2025.


## Planned Experiments and Schedule

Date	Milestone
25.03.2026	Literature overview — survey contamination detection in LLMs and VLMs; models and datasets analysis (Clotho-AQA, MMAU, AudioMCQ, Audio-Reasoner, Flamingo Audio)
01.04.2026	HPC cluster access setup and environment configuration (Conda, PyTorch, torchaudio, HF)
03.04–06.04.2026	Easter break
08.04.2026	Experimental setup — dataloader implementation, evaluation pipeline, logging with W&B
15.04.2026	Adaptation of Method 1 — slot guessing for perturbed audio captions (Both Text and Images Leaked!)
22.04.2026	Adaptation of Method 2 — in-context learning contamination detection for audio (Detecting Data Contamination via ICL)
29.04.2026	Adaptation of Method 3 — Membership Inference Attacks against Large Vision-Language Models 
01.05–04.05.2026	majówka (🚲, 🍻)
06.05.2026	Bug fixes, edge case handling, and pipeline stabilization
13.05.2026	Results analysis — ablation over perturbation types, cross-model comparison
20.05.2026	Write-up, figures, and final results consolidation
25.05.2026	final deadline


## Technological Stack

Python, PyTorch, HuggingFace Transformers, HuggingFace Datasets, torchaudio, librosa, Weights & Biases, Conda, ruff, make