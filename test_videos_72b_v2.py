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

# Clear memory
def clear_memory():
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()

# Setup device
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

    # starts time
    start_time = time.time()

    video_path = str(video_file)
    video_filename = video_file.name

    # Save results file
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
            )
            
            transcript = result["text"]
            
            f.write("Transcript:\n")
            f.write(transcript + "\n\n")
            
            # Clear memory before vision analysis
            clear_memory()
            
            # prompt
            prompt = f"""
Analyze this video and transcript to answer each question specifically and directly.

**ANALYSIS TASKS:**

1. CELEBRITIES
List any celebrities or public figures visible.
Format: [Name] - [Occupation] - [Context/role in video]
If none, respond: "None identified"

2. PEOPLE COUNT
Identify exact number of distinct individual humans visible in video. 
Response should be a whole number. 
If there is a crowd, provide the closest approximate number (e.g., "~15-20 people").

3. GENDER
Identify the gender of Human 1, Human 2, Human 3 and Human 4 within the video as identified in previous question. 
Categorize gender as Male, Female, Binary or Unclear. In case of no zero humans or crowd of humans, response should be "NA". 
In case of presence of humans, response should be: 
Human 1: <Human 1 Gender>; 
Human 2: <Human 2 gender if Human 2 was identified, else NA>;  
Human 3: <Human 3 gender if Human 3 was identified, else NA>;  
Human 4: <Human 4 gender if Human 4 was identified, else NA>.

4. BRAND/PRODUCT
Identify the product or service being advertised along with the name of brand. 
Response should be <Name of the Brand><Name of sub-brand if present><Product/Service>

5. LOGO APPEARANCE COUNT
Does the advertised brand or logo appear throughout the entire video? If not how many times does it appear?	

6. THEME
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
                            # Below 384 x 384 causes error
                            "max_pixels": 400 * 400,
                            "fps": 3.0,
                            "duration": 35,
                        },
                        {"type": "text", "text": prompt},
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
                # Don't make this too low, maybe switch to 512
                max_new_tokens=384,
                do_sample=False,
                temperature=0.1,
                top_p=0.9,
                repetition_penalty=1.0,
                pad_token_id=qwen_processor.tokenizer.eos_token_id,
                num_beams=1,
                early_stopping=True,
                use_cache=True,
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
            
            # Clean up
            del inputs, video_inputs, image_inputs, generated_ids  

            # Clear memory every 5 videos
            if idx % 5 == 0:
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

# clear memory
clear_memory()

# Print total runtime in seconds
total_time = time.time() - script_start_time
print(f"\nDone. Processed {len(video_files)} videos.")