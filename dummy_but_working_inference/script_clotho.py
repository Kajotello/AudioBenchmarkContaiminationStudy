import os
import tempfile

import numpy as np
import soundfile as sf
import torch
from datasets import Audio, load_dataset
from transformers import AudioFlamingo3ForConditionalGeneration, AutoProcessor

# ---------------------------------------------------------------------------
# Load one sample from CLAPv2/Clotho
# ---------------------------------------------------------------------------
print("Loading Clotho dataset sample...")
dataset = load_dataset("CLAPv2/Clotho", split="validation")
dataset = dataset.cast_column("audio", Audio())
sample = dataset[0]

audio_array = sample["audio"]["array"]
sampling_rate = sample["audio"]["sampling_rate"]
# Clotho stores all 5 captions as period-separated sentences in "text"
caption = sample["text"].split(".")[0].strip()

print(f"Audio shape : {np.array(audio_array).shape}, SR: {sampling_rate}")
print(f"Reference caption: {caption}")

# ---------------------------------------------------------------------------
# The processor expects a file path, so write to a temporary WAV file
# ---------------------------------------------------------------------------
with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
    tmp_path = tmp.name
sf.write(tmp_path, audio_array, sampling_rate)

try:
    # -----------------------------------------------------------------------
    # Load model
    # -----------------------------------------------------------------------
    model_id = "nvidia/audio-flamingo-3-hf"
    processor = AutoProcessor.from_pretrained(model_id)
    model = AudioFlamingo3ForConditionalGeneration.from_pretrained(
        model_id, device_map="auto", torch_dtype=torch.float16
    )

    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Describe the sounds in the audio."},
                {"type": "audio", "path": tmp_path},
            ],
        }
    ]

    inputs = processor.apply_chat_template(
        conversation,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
    ).to(model.device)
    inputs = {k: v.half() if v.dtype == torch.float32 else v for k, v in inputs.items()}

    outputs = model.generate(**inputs, max_new_tokens=500)
    decoded = processor.batch_decode(
        outputs[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True
    )

    print("\n--- Model output ---")
    print(decoded[0])
    print("\n--- Reference caption ---")
    print(caption)

finally:
    os.unlink(tmp_path)
