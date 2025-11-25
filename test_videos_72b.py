import os
import torch
import ffmpeg
import soundfile as sf
import gc
import time
import json
import tempfile
from huggingface_hub import hf_hub_download
from pathlib import Path
from transformers import (
    AutoModelForSpeechSeq2Seq, 
    AutoProcessor, 
    pipeline,
    Qwen2_5_VLForConditionalGeneration,
    AutoTokenizer,
    AutoConfig
)
from qwen_vl_utils import process_vision_info
from awq import AutoAWQForCausalLM

# Clears 
def clear_memory():
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()

# Setup device and precision
device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.bfloat16

# Track total script start time
script_start_time = time.time()

# Video folder
videos_folder = "videos"
os.makedirs(videos_folder, exist_ok=True)

# Get all MP4 files
video_files = list(Path(videos_folder).glob("*.mp4"))

if not video_files:
    print(f"No MP4 files found in '{videos_folder}' folder")
    exit()

print(f"Found {len(video_files)} videos to process\n")

# Load Whisper model
model_id = "openai/whisper-large-v3"
whisper_model = AutoModelForSpeechSeq2Seq.from_pretrained(
    model_id,
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True,
    use_safetensors=True,
    attn_implementation="sdpa",
)
whisper_model.to(device)
whisper_processor = AutoProcessor.from_pretrained(model_id)

whisper_pipe = pipeline(
    "automatic-speech-recognition",
    model=whisper_model,
    tokenizer=whisper_processor.tokenizer,
    feature_extractor=whisper_processor.feature_extractor,
    torch_dtype=torch.float16,
    device=device,
    model_kwargs={"attn_implementation": "sdpa"},
)

qwen_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2.5-VL-72B-Instruct-AWQ",
    torch_dtype="auto",
    device_map="auto",
)

qwen_processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-72B-Instruct-AWQ")

# Process each video
for idx, video_file in enumerate(video_files, 1):
    start_time = time.time()  # start timing for this video

    video_path = str(video_file)
    video_filename = video_file.name
    # Save analysis file in script directory
    output_file = os.path.splitext(video_filename)[0] + "_analysis.txt"
    
    print(f"[{idx}/{len(video_files)}] Processing: {video_filename}")
    
    try:
        with open(output_file, "w", encoding="utf-8") as f:
            # Write video filename
            f.write(f"Video File: {video_filename}\n")
            f.write("=" * 80 + "\n\n")
            
            # Extract audio
            audio_path = str(video_file.with_suffix('.wav'))
            
            (
                ffmpeg
                .input(video_path, t=35)
                .output(audio_path, ac=1, ar=16000)
                .overwrite_output()
                .run(quiet=True)
            )
            
            # Load extracted audio
            audio, sample_rate = sf.read(audio_path)
            
            # Transcribe
            result = whisper_pipe(
                {"array": audio, "sampling_rate": sample_rate}, 
                return_timestamps=False,
                chunk_length_s=35,
                batch_size=1,
                generate_kwargs={
                    "task": "transcribe", 
                    "language": "en"
                }
            )
            
            transcript = result["text"]
            
            f.write("Transcript:\n")
            f.write(transcript + "\n\n")
            
            # Clear memory before vision analysis
            clear_memory()
            
            # Prepare questions
            questions = f"""Analyze this video and transcript to answer each question specifically and directly.

**ANALYSIS TASKS:**

1. CELEBRITIES
List any celebrities or public figures visible. Format: [Name] - [context/role in video]
If none, respond: "None identified"

2. PEOPLE COUNT
Exact number of distinct individuals visible in video: [NUMBER]
If crowd/unclear, provide: [approximate range] (e.g., "~15-20 people")

3. GENDER
List each person's apparent gender presentation:
Person 1: [Male/Female/Not clearly visible]
Person 2: [Male/Female/Not clearly visible]
Format: [Person number/identifier]: [Gender]

4. ETHNICITY
Describe observed ethnic/racial characteristics per person:
Person 1: [Specific observable characteristics - e.g., "Light skin tone" OR "Dark skin tone" OR "East Asian appearance"]
Person 2: [Observable characteristics]
Use descriptive physical traits, not demographic labels.

5. ACTIVITIES
List all visible activities/sports:
- [Activity 1]: [brief description]
- [Activity 2]: [brief description]
Include location/setting context.

6. TRANSCRIPT ANALYSIS
Main topics discussed (if transcript available):
- Topic 1: [specific detail]
- Topic 2: [specific detail]
- Topic 3: [specific detail]
Key speakers (if identifiable): [names or descriptions]
If no transcript: "No transcript available"

7. BRAND/PRODUCT
Identify the product or service being advertised along with the name of brand. 
Response should be <Name of the Brand><Name of sub-brand if present><Product/Service>

8. LOGO APPEARANCE COUNT
Does the advertised brand or logo appear throughout the entire video? If not how many times does it appear?	

9. THEME
What is the ad trying to communicate about the brand, service, or product

**TRANSCRIPT PROVIDED:**
{transcript}

**RESPONSE INSTRUCTIONS:**
- Answer only what you observe
- Be specific, not vague
- Use exact numbers where possible
- Describe visible characteristics objectively
- Do not speculate
- Label each answer with its number"""
            
            # Messages
            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "video",
                            "video": video_path,
                            "max_pixels": 600 * 360,
                            "fps": 5.0,
                            "duration": 35,
                        },
                        {"type": "text", "text": questions},
                    ],
                }
            ]
            
            # Preparation for inference
            text = qwen_processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            image_inputs, video_inputs = process_vision_info(messages)
            inputs = qwen_processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            )
            inputs = inputs.to("cuda")
            
            # Generation
            generated_ids = qwen_model.generate(
                **inputs, 
                max_new_tokens=512,
                do_sample=False,
                temperature=0.1,
                top_p=0.9,
                repetition_penalty=1.1,
                pad_token_id=qwen_processor.tokenizer.eos_token_id,
                num_beams=1,
                early_stopping=True,
            )
            
            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            output_text = qwen_processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )
            
            # Analysis Results
            analysis_result = output_text[0]
            
            f.write("\nAnalysis Results:\n")
            f.write(analysis_result + "\n\n")
            
            # Clean up intermediate tensors
            del generated_ids, generated_ids_trimmed, inputs, image_inputs, video_inputs
            clear_memory()
            
            # Clean up audio file
            if os.path.exists(audio_path):
                os.remove(audio_path)
        
        # Print timing for this video
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"Completed: {output_file} | Time: {elapsed_time:.2f} seconds\n")
        
    except Exception as e:
        print(f"Error processing {video_filename}: {str(e)}\n")
        continue

# Clean up models at the end
del whisper_model, whisper_processor, whisper_pipe
del qwen_model, qwen_processor
clear_memory()

# Print total runtime (seconds only)
total_time = time.time() - script_start_time
print(f"\nDone. Processed {len(video_files)} videos.")