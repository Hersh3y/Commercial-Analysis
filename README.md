# Commercial-Analysis

Automated video commercial analysis using Whisper for transcription and Qwen2.5-VL for visual analysis.

## Features

- **Audio Transcription**: Extracts and transcribes audio using Whisper Large V3
- **Visual Analysis**: Analyzes video content with Qwen2.5-VL-72B model
- **Celebrity Detection**: Identifies public figures and celebrities
- **Demographic Analysis**: Counts people and identifies gender
- **Brand Recognition**: Detects brands, logos, and products
- **Theme Extraction**: Determines advertising message and themes

## Usage

Place MP4 videos in the `videos/` folder and run:

Results are saved as `{video_name}_analysis.txt` for each video.

## Performance

- **Accuracy**: 81.25% (Celebrity), 100% (Brand Name & Ad Theme), 62.5% (Gender), 50% (Logo Count), 43.75% (Human Count)
- **Processing Time**: ~41s per video (Celebrity), ~31s (Demographics), ~67s (Brand analysis)
- **Total Runtime**: ~1h 20m for 32 videos