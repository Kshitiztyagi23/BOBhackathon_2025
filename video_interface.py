"""Interactive web interface for video deepfake detection with side-by-side display."""

import tempfile
import os
from typing import Dict, List, Tuple
import streamlit as st
import torch
from PIL import Image
from transformers import AutoImageProcessor, SiglipForImageClassification
import cv2
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd

# Configuration
MODEL_NAME = "mode.safetensor"
ID2LABEL = {"0": "fake", "1": "real"}
LABEL_IDS = tuple(sorted(ID2LABEL.keys(), key=int))

# Initialize session state
if 'model' not in st.session_state:
    st.session_state.model = None
if 'processor' not in st.session_state:
    st.session_state.processor = None

@st.cache_resource
def load_model():
    """Load and cache the deepfake detection model."""
    model = SiglipForImageClassification.from_pretrained(MODEL_NAME)
    processor = AutoImageProcessor.from_pretrained(MODEL_NAME)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    return model, processor, device

def classify_frame(image: Image.Image, model, processor, device) -> Dict[str, float]:
    """Classify a single frame."""
    inputs = processor(images=image.convert("RGB"), return_tensors="pt")
    inputs = {name: tensor.to(device) for name, tensor in inputs.items()}
    
    with torch.no_grad():
        logits = model(**inputs).logits
        probs = torch.nn.functional.softmax(logits, dim=1).squeeze(0).tolist()
    
    return {ID2LABEL[str(i)]: round(prob, 4) for i, prob in enumerate(probs)}

def process_video(video_path: str, frame_step: int = 30, max_frames: int = None) -> Tuple[Dict[str, float], List[Tuple[int, Dict[str, float]]], int]:
    """Process video frames and return predictions."""
    model, processor, device = load_model()
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        st.error("Could not open video file")
        return {}, [], 0
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    accumulated = [0.0, 0.0]  # fake, real
    frame_predictions = []
    processed_frames = 0
    frame_index = 0
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_index % frame_step == 0:
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image = Image.fromarray(rgb_frame)
                predictions = classify_frame(image, model, processor, device)
                
                # Accumulate predictions
                accumulated[0] += predictions["fake"]
                accumulated[1] += predictions["real"]
                
                # Store frame prediction with timestamp
                timestamp = frame_index / fps if fps > 0 else frame_index
                frame_predictions.append((frame_index, timestamp, predictions))
                processed_frames += 1
                
                # Update progress
                progress = min(frame_index / total_frames, 1.0)
                progress_bar.progress(progress)
                status_text.text(f"Processing frame {frame_index}/{total_frames} (sampled: {processed_frames})")
                
                if max_frames and processed_frames >= max_frames:
                    break
            
            frame_index += 1
    
    finally:
        cap.release()
        progress_bar.empty()
        status_text.empty()
    
    if processed_frames == 0:
        return {}, [], total_frames
    
    # Calculate averages
    avg_predictions = {
        "fake": accumulated[0] / processed_frames,
        "real": accumulated[1] / processed_frames
    }
    
    return avg_predictions, frame_predictions, total_frames

def create_results_chart(frame_predictions: List[Tuple[int, float, Dict[str, float]]]):
    """Create an interactive chart of frame predictions over time."""
    if not frame_predictions:
        return None
    
    # Prepare data for plotting
    timestamps = [pred[1] for pred in frame_predictions]  # Use timestamp instead of frame number
    fake_scores = [pred[2]["fake"] for pred in frame_predictions]
    real_scores = [pred[2]["real"] for pred in frame_predictions]
    
    # Create DataFrame
    df = pd.DataFrame({
        'Time (seconds)': timestamps,
        'Fake Score': fake_scores,
        'Real Score': real_scores
    })
    
    # Create line chart
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=df['Time (seconds)'],
        y=df['Fake Score'],
        mode='lines+markers',
        name='Fake Score',
        line=dict(color='red', width=2),
        marker=dict(size=4)
    ))
    
    fig.add_trace(go.Scatter(
        x=df['Time (seconds)'],
        y=df['Real Score'],
        mode='lines+markers',
        name='Real Score',
        line=dict(color='green', width=2),
        marker=dict(size=4)
    ))
    
    fig.update_layout(
        title="Deepfake Detection Scores Over Time",
        xaxis_title="Time (seconds)",
        yaxis_title="Confidence Score",
        yaxis=dict(range=[0, 1]),
        hovermode='x unified',
        height=400
    )
    
    return fig

def main():
    st.set_page_config(
        page_title="Video Deepfake Detector",
        page_icon="🎥",
        layout="wide"
    )
    
    st.title("🎥 Video Deepfake Detection Interface")
    st.markdown("Upload a video to analyze it for deepfake content using AI detection.")
    
    # Sidebar for configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        frame_step = st.slider("Frame Step (process every Nth frame)", 10, 100, 30, 5)
        max_frames = st.number_input("Max Frames to Process (0 = all)", 0, 1000, 50, 10)
        max_frames = max_frames if max_frames > 0 else None
        
        # Model info
        st.header("📊 Model Information")
        model, processor, device = load_model()
        st.info(f"**Model**: {MODEL_NAME}")
        st.info(f"**Device**: {device}")
    
    # File upload
    uploaded_file = st.file_uploader(
        "Choose a video file",
        type=['mp4', 'avi', 'mov', 'mkv', 'webm', 'mpg', 'mpeg', 'm4v'],
        help="Upload a video file to analyze for deepfake content"
    )
    
    if uploaded_file is not None:
        # Create temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp_file:
            tmp_file.write(uploaded_file.read())
            temp_video_path = tmp_file.name
        
        try:
            # Create two columns for side-by-side layout
            col1, col2 = st.columns([1, 1])
            
            with col1:
                st.subheader("📹 Original Video")
                st.video(uploaded_file)
                
                # Video info
                cap = cv2.VideoCapture(temp_video_path)
                if cap.isOpened():
                    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    fps = cap.get(cv2.CAP_PROP_FPS)
                    duration = total_frames / fps if fps > 0 else 0
                    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    cap.release()
                    
                    st.info(f"""
                    **Video Information:**
                    - Duration: {duration:.2f} seconds
                    - Total Frames: {total_frames}
                    - FPS: {fps:.2f}
                    - Resolution: {width}x{height}
                    """)
            
            with col2:
                st.subheader("🤖 Analysis Results")
                
                if st.button("🚀 Analyze Video", type="primary"):
                    with st.spinner("Processing video frames..."):
                        avg_predictions, frame_predictions, total_frames = process_video(
                            temp_video_path, frame_step, max_frames
                        )
                    
                    if avg_predictions:
                        # Overall results
                        st.success("Analysis completed!")
                        
                        # Big metric display
                        fake_score = avg_predictions["fake"]
                        real_score = avg_predictions["real"]
                        
                        col_fake, col_real = st.columns(2)
                        with col_fake:
                            st.metric(
                                "🚨 Fake Score",
                                f"{fake_score:.1%}",
                                delta=f"{fake_score - 0.5:.1%}" if fake_score != 0.5 else None
                            )
                        
                        with col_real:
                            st.metric(
                                "✅ Real Score", 
                                f"{real_score:.1%}",
                                delta=f"{real_score - 0.5:.1%}" if real_score != 0.5 else None
                            )
                        
                        # Verdict
                        if fake_score > real_score:
                            st.error(f"🚨 **LIKELY FAKE** (Confidence: {fake_score:.1%})")
                        else:
                            st.success(f"✅ **LIKELY REAL** (Confidence: {real_score:.1%})")
                        
                        # Processing stats
                        st.info(f"""
                        **Processing Statistics:**
                        - Frames analyzed: {len(frame_predictions)}
                        - Total frames: {total_frames}
                        - Frame step: {frame_step}
                        """)
            
            # Full-width chart below
            if 'frame_predictions' in locals() and frame_predictions:
                st.subheader("📈 Frame-by-Frame Analysis")
                chart = create_results_chart(frame_predictions)
                if chart:
                    st.plotly_chart(chart, use_container_width=True)
                
                # Detailed frame data (expandable)
                with st.expander("🔍 Detailed Frame Data"):
                    df_detailed = pd.DataFrame([
                        {
                            "Frame": pred[0],
                            "Time (s)": f"{pred[1]:.2f}",
                            "Fake Score": f"{pred[2]['fake']:.3f}",
                            "Real Score": f"{pred[2]['real']:.3f}",
                            "Prediction": "FAKE" if pred[2]['fake'] > pred[2]['real'] else "REAL"
                        }
                        for pred in frame_predictions
                    ])
                    st.dataframe(df_detailed, use_container_width=True)
        
        finally:
            # Clean up temporary file
            if os.path.exists(temp_video_path):
                os.unlink(temp_video_path)
    
    else:
        st.info("👆 Please upload a video file to get started!")
        
        # Example/demo section
        st.markdown("---")
        st.subheader("ℹ️ How it works")
        st.markdown("""
        1. **Upload** your video file using the file uploader above
        2. **Configure** the analysis settings in the sidebar (frame step, max frames)
        3. **Click** the "Analyze Video" button to start processing
        4. **View** results including:
           - Overall fake/real confidence scores
           - Frame-by-frame analysis chart
           - Detailed predictions for each sampled frame
        
        The AI model analyzes sampled frames from your video and provides confidence scores
        indicating whether the content appears to be real or artificially generated (deepfake).
        """)

if __name__ == "__main__":

    main()
