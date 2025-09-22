import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
from datasets import load_dataset


device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

model_id = "openai/whisper-large-v3"

model = AutoModelForSpeechSeq2Seq.from_pretrained(
    model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
)
model.to(device)

processor = AutoProcessor.from_pretrained(model_id)

pipe = pipeline(
    "automatic-speech-recognition",
    model=model,
    tokenizer=processor.tokenizer,
    feature_extractor=processor.feature_extractor,
    torch_dtype=torch_dtype,
    device=device,
)


result = pipe("nestle.mp3", return_timestamps=True)
print(result["text"])

"""
Chocolate is scrunchies when it crunches. That's why I love Nestle Crunch. Chocolate is scrunchies when it crunches. That's why 
I love Nestle Crunch. The blend of Nestle's Creamy Milk Chocolate with Crunchies tastes just as good as it sounds. Chocolate is 
scrunchies when it crunches. That's why I love Nestle Crunch. Thank you.
"""