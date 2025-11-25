import os
import torch
import ffmpeg
import soundfile as sf
import gc
from transformers import (
    AutoModelForSpeechSeq2Seq, 
    AutoProcessor, 
    pipeline,
    Qwen2_5_VLForConditionalGeneration,
    AutoTokenizer
)
from qwen_vl_utils import process_vision_info

def clear_memory():
    """Thoroughly clear GPU memory"""
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()

# Video path
video_path = "24204046_TV_Selection_Overview.mp4"
output_file = os.path.splitext(video_path)[0] + "_analysis.txt"

# Open output file
with open(output_file, "w", encoding="utf-8") as f:
    f.write("=" * 80 + "\n")
    f.write("VIDEO ANALYSIS REPORT\n")
    f.write("=" * 80 + "\n\n")

    # =========================================================================
    # STEP 1: TRANSCRIPTION
    # =========================================================================
    f.write("=" * 80 + "\n")
    f.write("STEP 1: TRANSCRIPTION\n")
    f.write("=" * 80 + "\n")

    # Setup device and precision
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    # Model ID
    model_id = "openai/whisper-large-v3"

    # Load model & processor
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_id,
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=True,
        use_safetensors=True,
    )
    model.to(device)

    processor = AutoProcessor.from_pretrained(model_id)

    # Create pipeline
    pipe = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        torch_dtype=torch_dtype,
        device=device,
    )

    # Extract audio
    audio_path = os.path.splitext(video_path)[0] + ".wav"
    f.write(f"Extracting audio to {audio_path}...\n")

    (
        ffmpeg
        .input(video_path)
        .output(audio_path, ac=1, ar=16000)
        .overwrite_output()
        .run(quiet=True)
    )

    # Load extracted audio
    f.write("Loading audio...\n")
    audio, sample_rate = sf.read(audio_path)

    # Run transcription
    f.write("Transcribing...\n")
    result = pipe({"array": audio, "sampling_rate": sample_rate}, return_timestamps=True)

    segments = result["chunks"]
    duration = audio.shape[0] / sample_rate

    # Filter out tiny chunks near the end
    filtered = []
    for i, seg in enumerate(segments):
        seg_text = seg["text"].strip()
        start, end = seg["timestamp"]

        if i == len(segments) - 1 and len(seg_text.split()) <= 3 and end > duration - 2:
            continue

        filtered.append(seg_text)

    transcript = " ".join(filtered)
    
    f.write("\nTranscript:\n")
    f.write("-" * 80 + "\n")
    f.write(transcript + "\n\n")
    
    # Clean up
    del model, processor, pipe
    clear_memory()

    # =========================================================================
    # STEP 2: VIDEO ANALYSIS
    # =========================================================================
    f.write("=" * 80 + "\n")
    f.write("STEP 2: VIDEO ANALYSIS\n")
    f.write("=" * 80 + "\n")
    
    # Clear memory before loading
    clear_memory()

    # Load the model with GPU-only device map
    f.write("Loading Qwen vision model...\n")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen2.5-VL-72B-Instruct-AWQ", 
        torch_dtype=torch.float16,
        device_map="cuda:0"
    )

    processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-72B-Instruct-AWQ")

    # Define the questions
    questions = f"""Analyze the video and give well-detailed and accurate answers to the following questions:
1. Is there a celebrity present in the video?
2. How many humans appear in the video (Give me the exact number of unique humans in the video)?
3. What is the gender of the humans shown in the video?
4. What is the ethnicity of any celebrities shown in the video?
5. What main activities are shown in the video?
6. Who is the protagonist of the video?
Transcript: [{transcript}]"""

    # Prepare messages with video input
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "video",
                    "video": video_path,
                    "max_pixels": 360 * 420,
                    "fps": 4.0,
                },
                {"type": "text", "text": questions},
            ],
        }
    ]

    # Preparation for inference
    f.write("Processing video...\n")
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to("cuda")

    # Inference
    f.write("Generating analysis...\n")
    generated_ids = model.generate(**inputs, max_new_tokens=512)
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )

    analysis_result = output_text[0]

    f.write("\nAnalysis Results:\n")
    f.write("-" * 80 + "\n")
    f.write(analysis_result + "\n\n")

    # Clean up
    del generated_ids, generated_ids_trimmed, inputs, image_inputs, video_inputs
    del model, processor
    clear_memory()

    f.write("=" * 80 + "\n")
    f.write("COMPLETE\n")
    f.write("=" * 80 + "\n")

print(f"Analysis complete! Results saved to: {output_file}")

"""
Transcript:
--------------------------------------------------------------------------------
There's a fog out there, and it's calling you by name. It wants to wrap its whole self around you, make you forget who you are. 
Let's be ready. But you won't forget. You've got to find your way through that fog. Through the unknown, there's only one way.

Analysis Results:
--------------------------------------------------------------------------------
1. **Is there a celebrity present in the video?**
   - No, there is no identifiable celebrity present in the video. The individuals shown are engaged in athletic activities, but 
   they are not recognizable as celebrities.

2. **How many humans appear in the video?**
   - There are two humans appearing in the video. One person is seen dribbling a basketball and performing various basketball-related 
   actions, while another person is shown running through a foggy environment.

3. **What is the gender of the humans shown in the video?**
   - Both humans shown in the video appear to be male. The individual dribbling the basketball and the one running are both depicted as men.

4. **What is the ethnicity of any celebrities shown in the video?**
   - Since there are no identifiable celebrities in the video, it is not possible to determine the ethnicity of any celebrities.

5. **What main activities are shown in the video?**
   - The main activities shown in the video include basketball-related actions such as dribbling and handling a basketball, and running 
   through a foggy environment. The video also features motivational text and branding from Under Armour, suggesting a focus on sports and athletic performance.
"""

"""
---------------------------------------------------------------------------------------------------------------------------------------------------------------------------
"""

import os
import torch
import ffmpeg
import soundfile as sf
import gc
from transformers import (
    AutoModelForSpeechSeq2Seq, 
    AutoProcessor, 
    pipeline,
    Qwen2_5_VLForConditionalGeneration,
    AutoTokenizer
)
from qwen_vl_utils import process_vision_info

# Function to clear memory
def clear_memory():
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()

# Video path
video_path = "23888455_TV_B_Burke_Podiatrist_Certified_Arch_Support.mp4"
output_file = os.path.splitext(video_path)[0] + "_analysis.txt"

# Open output file
with open(output_file, "w", encoding="utf-8") as f:

    # Setup device and precision
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16

    # Whisper large v3 Model
    model_id = "openai/whisper-large-v3"

    # Load model & processor with optimizations
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_id,
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=True,
        use_safetensors=True,
        attn_implementation="sdpa",
    )
    model.to(device)

    processor = AutoProcessor.from_pretrained(model_id)

    # Create pipeline
    pipe = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        torch_dtype=torch_dtype,
        device=device,
        model_kwargs={"attn_implementation": "sdpa"},
    )

    # Extract audio
    audio_path = os.path.splitext(video_path)[0] + ".wav"

    # Use first 35 seconds
    (
        ffmpeg
        .input(video_path, t=35)
        .output(audio_path, ac=1, ar=16000)
        .overwrite_output()
        .run(quiet=True)
    )

    # Load extracted audio
    audio, sample_rate = sf.read(audio_path)

    # Result
    result = pipe(
        {"array": audio, "sampling_rate": sample_rate}, 
        return_timestamps=False,
        chunk_length_s=35,  # Larger chunks for speed
        batch_size=1,  # Larger batch for speed
        generate_kwargs={
            "task": "transcribe", 
            "language": "en"
        }
    )

    transcript = result["text"]
    
    f.write("Transcript:\n")
    f.write(transcript + "\n\n")
    
    # Clean up
    del model, processor, pipe
    clear_memory()
    
    # Clear memory before loading
    clear_memory()

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen2.5-VL-7B-Instruct-AWQ",
        torch_dtype=torch.float16,
        device_map="cuda:0",
    )

    # Use the correct processor for the AWQ model
    processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct-AWQ")

    # EXACT SAME PROMPT
    questions = f"""Analyze the video and give well-detailed and accurate answers to the following questions:
1. Is there a celebrity present in the video?
2. How many humans appear in the video (Give me the exact number of unique humans in the video)?
3. What is the gender of the humans shown in the video?
4. What is the ethnicity of any celebrities shown in the video?
5. What main activities are shown in the video?
Transcript: [{transcript}]"""

    # Messages
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "video",
                    "video": video_path,
                    "max_pixels": 600 * 360,  # Resolution
                    "fps": 5.0,  # Frames per Second
                    "duration": 35,  # Duration
                },
                {"type": "text", "text": questions},
            ],
        }
    ]

    # Preparation for inference
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to("cuda")

    # Generation Setting (ADJUST SETTINGS HERE)
    generated_ids = model.generate(
        **inputs, 
        max_new_tokens=512,  # Tokens
        do_sample=False,  # Sample
        temperature=0.1,  # Temperature
        top_p=0.9,
        repetition_penalty=1.1,
        pad_token_id=processor.tokenizer.eos_token_id,
        num_beams=1,  # Number of Beams
        early_stopping=True,
    )
    
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )

    # Analysis Results
    analysis_result = output_text[0]

    f.write("\nAnalysis Results:\n")
    f.write(analysis_result + "\n\n")

    # Clean up
    del generated_ids, generated_ids_trimmed, inputs, image_inputs, video_inputs
    del model, processor
    clear_memory()

print(f"Done! File: {output_file}")