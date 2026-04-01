# Audio Contamination: Benchmark Contamination Detection for Audio Language Models
### Projekt WIMU 26L - zespół 10 - Kajetan Rożej, Rafał Szczepaniak, Wojciech Zarzecki

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

## Literature Overview: Credibility of Benchmark Evaluation for Audio-Language Models

The main goal of this project is to review the credibility of benchmark evaluation for audio-language models (ALMs). We approach this from three angles: (1) methods for detecting benchmark data contamination in LLMs and VLMs, which focus on the outcomes of contamination (inflated benchmark scores); (2) membership inference attacks (MIAs), which focus on the cause of contamination by attempting to detect overlap between training and benchmark data; and (3) a survey of state-of-the-art audio-language models and their training/evaluation datasets, identifying where training and evaluation data are disclosed and where they overlap.

---

### 1. Benchmark Data Contamination Detection Methods for LLMs and VLMs

In assessing the generalization capabilities of multimodal models, a growing body of work investigates whether benchmark data has leaked into the training pipeline, leading to inflated performance metrics. We review three recent papers from CORE A* conferences that propose detection methods for this problem.

| **Paper** | **Venue** | **ArXiv / Paper Link** | **Method Name** | **Modality** | **Code Available** | **Key Metrics** | **Compute / Resources** | **Commentary** |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Song et al. (2025) | EMNLP 2025 | [arxiv.org/abs/2411.03823](http://arxiv.org/abs/2411.03823) | MM-Detect | VLM (text+image) | Yes: [github.com/MLLM-Data-Contamination/MM-Detect](http://github.com/MLLM-Data-Contamination/MM-Detect) | Correct Rate (CR), Perturbed Correct Rate (PCR), Phi contamination degree (minor/partial/severe) | Not reported in detail | First systematic analysis of multimodal contamination. Traces contamination to LLM pretraining phase. Evaluates 12 MLLMs on 5 benchmarks. Requires Java for POS tagger. |
| Zawalski et al. (2025) | ICLR 2026 (Poster) | [arxiv.org/abs/2510.27055](http://arxiv.org/abs/2510.27055) | CoDeC | LLM (text) | Yes: https://docs.nvidia.com/nemo/evaluator/latest/evaluation/benchmarks/catalog/all/harnesses/codec.html | Percentage-based contamination scores at dataset level | Lightweight: requires only forward passes on model being tested | Model-agnostic, no calibration needed. Works at dataset level without held-out data. Reveals memorization in models with undisclosed training corpora. Clean theoretical motivation from loss landscape perspective. |
| Park et al. (2025) | ICLR 2026 | [arxiv.org/abs/2511.03774](http://arxiv.org/abs/2511.03774) | Multi-Modal Semantic Perturbation | VLM (text+image) | Yes: [github.com/jadenpark0/mm-perturb](http://github.com/jadenpark0/mm-perturb) | Performance drop on perturbed vs. original benchmark (up to -45%) | GPT-4o for captions (~$1.50 per 1000 images); Flux+ControlNet on single 24GB GPU; can replace GPT-4o with Molmo-7B-D | Designed specifically for VLMs. Robust across LoRA and standard fine-tuning. Uses controlled image perturbation rather than text-only tricks. Existing text-based methods shown to fail on VLMs. |

---

### 2. Membership Inference Attacks (MIAs)

As a complementary approach to benchmark contamination detection, membership inference attacks aim to detect intersection between training datasets and benchmark data at the instance level. While contamination detection methods focus on the outcome (performance inflation), MIAs focus on the cause: determining whether a specific sample was part of the model's training set.

| **Paper** | **Venue** | **ArXiv / Paper Link** | **Method** | **Modality** | **Code Available** | **Key Metrics** | **Commentary** |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Yeom et al. (2018) | IEEE CSF 2018 | [arxiv.org/abs/1709.01604](http://arxiv.org/abs/1709.01604) | Perplexity threshold | General ML / LLM | available for later paper which compare to it | Membership inference accuracy, overfitting gap | Foundational work. Establishes the link between overfitting and privacy risk. Simple but still used as a baseline. |
| Shi et al. (2024) | ICLR 2024 | [arxiv.org/abs/2310.16789](http://arxiv.org/abs/2310.16789) | Min-K% Prob | LLM | available for later paper which compare to it | AUC, TPR@low FPR | Focuses on low-probability tokens. Effective for pretraining data detection. Widely adopted as a baseline in subsequent work. |
| Zhang et al. (2024) | Preprint | [arxiv.org/abs/2404.02936](http://arxiv.org/abs/2404.02936) | Min-K%++ | LLM | available for later paper which compare to it | AUC, TPR@low FPR | Adds calibration/normalization to Min-K%. More robust across domains and model sizes. |
| VL-MIA (2024) | NeurIPS 2024 | [arxiv.org/abs/2411.02902](http://arxiv.org/abs/2411.02902) | Entropy-based MIA for VLMs | VLM (text+image) | Yes: [github.com/LIONS-EPFL/VL-MIA](http://github.com/LIONS-EPFL/VL-MIA) | AUC, membership inference accuracy | First MIA study for VLMs. Key insight: highest-entropy positions amplify member/non-member signal. Directly relevant to auditing VLM training data. |

---

### 3. Audio-Language Models and Datasets

To investigate the credibility of benchmark evaluation in the audio domain, we survey state-of-the-art audio-language models that disclose their training and evaluation data, alongside frontier models from major labs whose training data is not publicly available but that report performance on open-source benchmarks.

#### Open-Source Models with Disclosed Data

The following models are fully open-source (weights, code, and training data composition disclosed), making them suitable candidates for contamination analysis. All three are from NVIDIA's Audio Flamingo line and share a common architecture (encoder + cross-attention/LLaVA + LLM backbone).

**Audio Reasoner (Xie et al., 2025)** is a large audio language model trained on CoTA, a dataset of 1.2M reasoning-rich samples created via chain-of-thought annotation with closed-source models. It achieves strong results on MMAU-mini, AIR-Bench, and MELD. Training data and code are open.

**Audio Flamingo 2 (Ghosh et al., ICML 2025)** is a 3B-parameter ALM with a custom CLAP encoder. It is trained on ~10M audio-text pairs using a 3-stage curriculum and achieves SOTA across 20+ benchmarks. It introduces AudioSkills (synthetic reasoning data) and LongAudio (long audio understanding up to 5 min). Trained exclusively on public datasets.

**Audio Flamingo 3 (Goel/Ghosh et al., NeurIPS 2025 Spotlight)** is a 7B-parameter model built on a custom Whisper encoder (AF-Whisper). It handles speech, sound, and music; supports multi-turn multi-audio chat and on-demand thinking. Trained on ~50M audio-text pairs with a 5-stage curriculum. Achieves new SOTA on 20+ benchmarks, surpassing Gemini and GPT-4o-audio.

#### Frontier / Closed-Data Models

The following models report benchmark results on the same open-source datasets, but their full training corpora are not publicly disclosed, making it impossible to directly verify the absence of benchmark contamination through data inspection.

- `Qwen/Qwen2-Audio-7B-Instruct` (Alibaba): Weights available on HuggingFace, but full training data composition undisclosed.
- `mistralai/Voxtral-Small-24B-2507` (Mistral AI): Weights available, training data undisclosed.

#### Training/Evaluation Dataset Overlap

The table below shows which publicly known datasets are used for both training and evaluation by the open-source models reviewed. This overlap is the primary vector for potential benchmark contamination.

| **Dataset** | **HF Link** | **Original Paper** | **Audio Reasoner (Train / Eval)** | **AF2 (Train / Eval)** | **AF3 (Train / Eval)** |
| --- | --- | --- | --- | --- | --- |
| AudioCaps | [OpenSound/AudioCaps](https://huggingface.co/datasets/OpenSound/AudioCaps) | Kim et al. NAACL 2019 | train (49,838) / test (975) | train (49,838) / test (975) | train (49,838) / test (975) |
| AudioSet | [agkphysics/AudioSet](https://huggingface.co/datasets/agkphysics/AudioSet) | Gemmeke et al. ICASSP 2017 | train (18.7k balanced) / test (17.1k eval) | train (18.7k balanced) / test (17.1k eval) | train (18.7k balanced) / test (17.1k eval) |
| Clotho-v2 | [CLAPv2/Clotho](https://huggingface.co/datasets/CLAPv2/Clotho) | Drossos et al. ICASSP 2020 | train (3,839) / evaluation (1,045) | train (3,839) / evaluation (1,045) | train (3,839) / evaluation (1,045) |
| MusicBench | [amaai-lab/MusicBench](https://huggingface.co/datasets/amaai-lab/MusicBench) | Melechovsky et al. 2023 | train (52,768) / test (400) | --- | train (52,768) / test (400) |
| CoVoST 2 (zh-en) | [facebook/covost2](https://huggingface.co/datasets/facebook/covost2) | Wang et al. 2020 | train / test | --- | --- |
| MELD | [ajyy/MELD_audio](https://huggingface.co/datasets/ajyy/MELD_audio) | Poria et al. ACL 2019 | train (9,989) / test (2,610) | --- | --- |
| MusicQA | [mu-llama/MusicQA](https://huggingface.co/datasets/mu-llama/MusicQA) | Liu et al. 2024 | --- | --- | Pretraining + Finetuning / Evaluation |
| LibriSpeech | [openslr/librispeech_asr](https://huggingface.co/datasets/openslr/librispeech_asr) | Panayotov et al. ICASSP 2015 | --- | --- | train (clean/other) / test (clean/other) |
| VoxPopuli | [facebook/voxpopuli](https://huggingface.co/datasets/facebook/voxpopuli) | Wang et al. ACL 2021 | --- | --- | train / test |

Additionally, Audio Reasoner uses a large combined training dataset `zhifeixie/Audio-Reasoner-CoTA` (1.2M samples with CoT annotations).

#### Model Summary Table

| **Model** | **Paper / ArXiv** | **Venue** | **Params** | **Weights Available** | **Code Available** | **Training Data Disclosed** | **Key Benchmarks Reported** | **Commentary** |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Audio Reasoner | [arxiv.org/abs/2503.02318](http://arxiv.org/abs/2503.02318) | Preprint (Mar 2025) | 7B | Yes (HF) | Yes: [github.com/xzf-thu/Audio-Reasoner](http://github.com/xzf-thu/Audio-Reasoner) | Yes (CoTA dataset open) | MMAU-mini, AIR-Bench, MELD | Focuses on CoT reasoning for audio. Uses closed-source models to generate training annotations. Good candidate for contamination analysis since CoTA dataset is open. |
| Audio Flamingo 2 | [arxiv.org/abs/2503.03983](http://arxiv.org/abs/2503.03983) | ICML 2025 | 3B (LLM) + 203M (encoder) | Yes (HF: nvidia/audio-flamingo-2-0.5B) | Yes: [github.com/NVIDIA/audio-flamingo](http://github.com/NVIDIA/audio-flamingo) | Yes (public datasets only) | 20+ benchmarks including AudioCaps, Clotho, AudioSet | Strong SOTA with smaller model. 3-stage curriculum. Introduces AudioSkills and LongAudio datasets. Trained exclusively on public data. |
| Audio Flamingo 3 | [arxiv.org/abs/2507.08128](http://arxiv.org/abs/2507.08128) | NeurIPS 2025 (Spotlight) | 7B (LLM) | Yes (HF) | Yes: [github.com/NVIDIA/audio-flamingo](http://github.com/NVIDIA/audio-flamingo) | Yes (~50M pairs, open-source audio data) | 20+ benchmarks; surpasses Gemini, GPT-4o-audio | Most capable open ALM. Handles speech+sound+music. 5-stage curriculum. Open-sources AudioSkills-XL, LongAudio-XL, AF-Think, AF-Chat datasets. |
| Qwen2-Audio-7B-Instruct | Alibaba technical report | N/A | 7B | Yes (HF: Qwen/Qwen2-Audio-7B-Instruct) | Partial | **No** (training data undisclosed) | Reports on AudioCaps, AudioSet, Clotho, etc. | Frontier model. Reports performance on public benchmarks but training data composition unknown. Key target for contamination auditing. |
| Voxtral-Small-24B | Mistral AI blog | N/A | 24B | Yes (HF: mistralai/Voxtral-Small-24B-2507) | No | **No** (training data undisclosed) | Reports on open-source benchmarks | Frontier model from Mistral. Closed training data. Key target for contamination auditing alongside Qwen2-Audio. |

---

### Summary

1. **Contamination is a real and documented problem in multimodal models.** MM-Detect (Song et al.) finds significant contamination in proprietary models on older benchmarks, with contamination sometimes originating in unimodal pretraining. CoDeC (Zawalski et al.) provides lightweight detection for LLMs via ICL. Park et al. show that VLM-specific perturbation methods are needed since text-only approaches fail.
2. **MIAs provide a complementary lens.** While contamination detection measures outcomes (inflated scores), MIAs (Min-K%, VL-MIA) directly test whether specific samples were in the training set. The VL-MIA work shows this is feasible for multimodal models via entropy analysis.
3. **Audio-language models present a clear opportunity.** Open models (Audio Reasoner, AF2, AF3) disclose training data that overlaps with evaluation benchmarks (AudioCaps, AudioSet, Clotho are used for both training and testing). Frontier models (Qwen2-Audio, Voxtral) report results on the same benchmarks but do not disclose training data. No existing work has applied contamination detection or MIA methods to audio-language models specifically.


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