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

def transcribe_video(video_path):
    """Extract audio and transcribe video"""
    print("=" * 80)
    print("STEP 1: TRANSCRIPTION")
    print("=" * 80)
    
    # Setup device and precision
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    # Model ID
    model_id = "openai/whisper-large-v3"

    # Load model & processor
    print("Loading Whisper model...")
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
    print(f"Extracting audio to {audio_path}...")
    
    (
        ffmpeg
        .input(video_path)
        .output(audio_path, ac=1, ar=16000)
        .overwrite_output()
        .run(quiet=True)
    )

    # Load extracted audio
    print("Loading audio...")
    audio, sample_rate = sf.read(audio_path)

    # Run transcription
    print("Transcribing...")
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

    cleaned_text = " ".join(filtered)
    
    # Clean up
    del model, processor, pipe
    clear_memory()
    
    return cleaned_text

def analyze_video(video_path):
    """Analyze video with vision model"""
    print("\n" + "=" * 80)
    print("STEP 2: VIDEO ANALYSIS")
    print("=" * 80)
    
    # Clear memory before loading
    clear_memory()

    # Load the model with GPU-only device map
    print("Loading Qwen vision model...")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen2.5-VL-72B-Instruct-AWQ", 
        torch_dtype=torch.float16,
        device_map="cuda:0"
    )

    processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-72B-Instruct-AWQ")

    # Define the questions
    questions = """Please analyze this video and answer the following questions:
1. Is there a celebrity present in the video?
2. How many humans appear in the video?
3. What is the gender of the humans shown in the video?
4. What is the ethnicity of any celebrities shown in the video?
5. What main activities are shown in the video?
Please provide clear, detailed answers for each question."""

    # Prepare messages with video input
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "video",
                    "video": video_path,
                    "max_pixels": 360 * 420,
                    "fps": 1.0,
                },
                {"type": "text", "text": questions},
            ],
        }
    ]

    # Preparation for inference
    print("Processing video...")
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
    print("Generating analysis...")
    generated_ids = model.generate(**inputs, max_new_tokens=512)
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )

    analysis_result = output_text[0]

    # Clean up
    del generated_ids, generated_ids_trimmed, inputs, image_inputs, video_inputs
    del model, processor
    clear_memory()
    
    return analysis_result

def main():
    # Video path
    video_path = "24108666_TV_Only_Way_Is_Through_6.mp4"
    
    # Step 1: Transcribe
    transcript = transcribe_video(video_path)
    print("\nTranscript:")
    print("-" * 80)
    print(transcript)
    
    # Step 2: Analyze
    analysis = analyze_video(video_path)
    print("\nAnalysis Results:")
    print("-" * 80)
    print(analysis)
    
    print("\n" + "=" * 80)
    print("COMPLETE")
    print("=" * 80)

if __name__ == "__main__":
    main()

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