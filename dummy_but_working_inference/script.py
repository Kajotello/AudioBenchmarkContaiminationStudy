from transformers import AudioFlamingo3ForConditionalGeneration, AutoProcessor
import torch

model_id = "nvidia/audio-flamingo-3-hf"
processor = AutoProcessor.from_pretrained(model_id)
model = AudioFlamingo3ForConditionalGeneration.from_pretrained(model_id, device_map="auto", torch_dtype=torch.float16)

conversation = [
    {
        "role": "user",
        "content": [
            {"type": "text", "text": "Transcribe the input speech."},
            {"type": "audio", "path": "https://huggingface.co/datasets/nvidia/AudioSkills/resolve/main/assets/WhDJDIviAOg_120_10.mp3"},
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

decoded_outputs = processor.batch_decode(outputs[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)
print(decoded_outputs)
