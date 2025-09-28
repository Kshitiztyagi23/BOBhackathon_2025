# Video Deepfake Detection Interface

This project provides both a command-line tool and an interactive web interface for detecting deepfakes in videos using AI.

## Features

### Command-Line Tool (`Untitled-1.py`)
- Process videos by sampling frames at configurable intervals
- Average predictions across all sampled frames
- Configurable frame stepping and processing limits
- Direct code editing for video path and parameters

### Web Interface (`video_interface.py`)
- **Drag & drop video upload** - Support for MP4, AVI, MOV, MKV, WEBM, etc.
- **Side-by-side display** - Original video alongside real-time analysis results
- **Interactive charts** - Frame-by-frame confidence scores over time
- **Configurable processing** - Adjust frame step and max frames via sidebar
- **Detailed results** - Overall scores, frame-level data, and processing stats

## Setup

1. Install Python 3.9 or newer
2. Install dependencies:

```powershell
pip install -r requirements.txt
```

## Usage

### Web Interface (Recommended)

Launch the Streamlit web interface:

```powershell
streamlit run video_interface.py
```

This opens a browser window where you can:
1. Upload your video file
2. Configure analysis settings (frame step, max frames)
3. View results side-by-side with the original video
4. Explore frame-by-frame analysis charts

### Command-Line Tool

Edit the configuration variables in `Untitled-1.py`:

```python
VIDEO_PATH = r"C:\path\to\your\video.mp4"
FRAME_STEP = 20  # Process every 20th frame
MAX_FRAMES = None  # Process all frames (or set to a number)
```

Then run:

```powershell
python Untitled-1.py
```

## Model Information

- **Model**: `prithivMLmods/deepfake-detector-model-v1` from Hugging Face
- **Labels**: `fake` (artificially generated) vs `real` (authentic content)
- **Processing**: Samples frames at configurable intervals and averages predictions
- **Device**: Automatically uses CUDA if available, otherwise CPU

## Interface Screenshots

The web interface provides:
- **Left Panel**: Original video player with file information
- **Right Panel**: Analysis results with confidence metrics
- **Bottom Panel**: Interactive timeline chart showing confidence over time
- **Expandable Details**: Frame-by-frame data table

## Troubleshooting

- **Slow processing**: Increase `frame_step` to sample fewer frames
- **Memory issues**: Set `max_frames` to limit processing
- **CUDA errors**: The interface will automatically fall back to CPU
- **Video format issues**: Try converting to MP4 with standard codecs

## Performance Tips

- **GPU acceleration**: Install CUDA-compatible PyTorch for faster processing
- **Frame stepping**: Use higher values (30-50) for faster analysis
- **Video resolution**: Lower resolution videos process faster
- **Max frames**: Limit to 50-100 frames for quick analysis of long videos