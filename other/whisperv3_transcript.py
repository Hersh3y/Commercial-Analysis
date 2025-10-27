import os
import torch
import ffmpeg
import soundfile as sf
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

# --- Setup device and precision ---
device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

# --- Model ID ---
model_id = "openai/whisper-large-v3"

# --- Load model & processor ---
model = AutoModelForSpeechSeq2Seq.from_pretrained(
    model_id,
    torch_dtype=torch_dtype,
    low_cpu_mem_usage=True,
    use_safetensors=True,
)
model.to(device)

processor = AutoProcessor.from_pretrained(model_id)

# --- Create pipeline ---
pipe = pipeline(
    "automatic-speech-recognition",
    model=model,
    tokenizer=processor.tokenizer,
    feature_extractor=processor.feature_extractor,
    torch_dtype=torch_dtype,
    device=device,
)

# --- Input video path ---
video_path = "24108666_TV_Only_Way_Is_Through_6.mp4"
audio_path = os.path.splitext(video_path)[0] + ".wav"

# --- Extract audio with ffmpeg ---
(
    ffmpeg
    .input(video_path)
    .output(audio_path, ac=1, ar=16000)  # mono, 16kHz (Whisper expects this)
    .overwrite_output()
    .run(quiet=True)
)

# --- Load extracted audio ---
audio, sample_rate = sf.read(audio_path)

# --- Run transcription ---
result = pipe({"array": audio, "sampling_rate": sample_rate}, return_timestamps=True)

segments = result["chunks"]
duration = audio.shape[0] / sample_rate

filtered = []
for i, seg in enumerate(segments):
    seg_text = seg["text"].strip()
    start, end = seg["timestamp"]

    # condition: tiny chunk near the end
    if i == len(segments) - 1 and len(seg_text.split()) <= 3 and end > duration - 2:
        continue

    filtered.append(seg_text)

cleaned_text = " ".join(filtered)
print(cleaned_text)
