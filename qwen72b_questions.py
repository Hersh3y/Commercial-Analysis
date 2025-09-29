from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
import torch
import gc

def clear_memory():
    """Thoroughly clear GPU memory"""
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()

# Clear memory at the start
clear_memory()

# Load the model with GPU-only device map
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2.5-VL-72B-Instruct-AWQ", 
    torch_dtype=torch.float16,
    device_map="cuda:0"  # Force to use only GPU
)

# Default processor
processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-72B-Instruct-AWQ")

# Path to your video file
video_path = "24108666_TV_Only_Way_Is_Through_6.mp4"

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

# Inference: Generation of the output
generated_ids = model.generate(**inputs, max_new_tokens=512)
generated_ids_trimmed = [
    out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
]
output_text = processor.batch_decode(
    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
)

print("Analysis Results:")
print("=" * 80)
print(output_text[0])

# Clean up after inference
del generated_ids, generated_ids_trimmed, inputs, image_inputs, video_inputs
del model, processor
clear_memory()

print("\n" + "=" * 80)