# Bank of Baroda Zero-Trust Hackathon: Hybrid Identity Monitoring & Deepfake-Resistant Verification

This repository contains the code for our submission to the Bank of Baroda Zero-Trust Hackathon. Our project provides a robust solution for hybrid identity monitoring and deepfake-resistant verification, featuring a transformer-based model and an interactive web interface for video analysis.

## 🚩 The Problem

The rise of AI-generated deepfakes poses a significant threat to digital identity verification processes like video KYC (vKYC). These sophisticated fakes can convincingly impersonate real individuals, bypassing traditional security measures. The core challenge is to implement a **Zero Trust** security model where every identity assertion is continuously re-validated to prevent fraud.

***

## 💡 Our Solution

We have developed a transformer-based deepfake detection model designed to be integrated into the vKYC pipeline. The model processes each image or video frame and outputs a probability score, flagging suspicious inputs for further review.

### Key Features:

Fine-tuned Architecture:** Utilizes the `SiglipForImageClassification` model, fine-tuned to distinguish between authentic and synthetic media. 
* [cite_start]**Vision-Language Model:** Leverages both visual patterns and contextual information for more accurate detection. [cite: 1]
* [cite_start]**Binary Classification:** Provides clear, actionable results with confidence scores for each prediction (Real vs. Fake). [cite: 1]
* [cite_start]**Specialized Training:** Trained on the `OpenDeepfake-Preview` dataset to recognize patterns specific to synthetic media. [cite: 1]

***

## ⚙️ How It Works

Our solution follows a structured pipeline to ensure high accuracy and reliability:

1.  [cite_start]**Model Architecture:** Built on Google's SigLIP vision-language transformer for state-of-the-art image classification. [cite: 3]
2.  [cite_start]**Training Data:** Fine-tuned on a dataset of approximately 20,000 real and AI-generated faces to ensure balanced performance. [cite: 3]
3.  [cite_start]**Inference Process:** Input images are resized, normalized, and scored by the model to determine if they are real or fake. [cite: 3]
4.  [cite_start]**Decision Logic:** Frames with scores above a set threshold trigger security flags and may require additional authentication. [cite: 4]
5.  [cite_start]**Zero Trust Alignment:** Every identity claim is explicitly verified with each access attempt, adhering to the Zero Trust security model. [cite: 4]

***

## 📊 Performance Metrics

The model was evaluated on a balanced dataset of **19,999 samples** and achieved the following performance:

| Metric | Fake Detection | Real Detection |
| :--- | :--- | :--- |
| **Precision** | [cite_start]97.18% [cite: 7, 8] | [cite_start]92.01% [cite: 7, 8] |
| **Recall** | [cite_start]91.55% [cite: 7, 8] | [cite_start]97.34% [cite: 7, 8] |
| **F1-Score** | [cite_start]94.28% [cite: 7, 8] | [cite_start]94.60% [cite: 7, 8] |

[cite_start]**Overall Accuracy: 94.44%** [cite: 7, 8]



***

## 🚀 Getting Started

To get started with our deepfake detection interface, follow these steps:

### Prerequisites

* Python 3.8+
* PyTorch
* Git

### Installation

1.  Clone the repository:
    ```bash
    git clone [https://github.com/Kshitiztyagi23/BOBhackathon_2025.git](https://github.com/Kshitiztyagi23/BOBhackathon_2025.git)
    cd BOBhackathon_2025
    ```
2.  Install the required packages from the `requirements.txt` file. This includes Streamlit, OpenCV, Plotly, and Transformers.
    ```bash
    pip install -r requirements.txt
    ```

***

## 🖥️ Usage: Interactive Web Interface

This project includes an interactive web application built with Streamlit for easy video analysis.

### Running the Application

1.  Navigate to the project directory in your terminal.
2.  Run the following command (assuming the script is named `app.py`):
    ```bash
    streamlit run app.py
    ```
3.  Your web browser will open with the application's interface.

### How to Use the Interface

1.  **Upload a Video:** Use the file uploader to select a video file (`.mp4`, `.mov`, `.avi`, etc.).
2.  **Configure Analysis:** In the sidebar, you can adjust:
    * **Frame Step:** Determines how frequently a frame is analyzed (e.g., a step of 30 processes every 30th frame).
    * **Max Frames:** Sets a limit on the total number of frames to process for a quick analysis.
3.  **Analyze:** Click the **"🚀 Analyze Video"** button to begin processing.
4.  **View Results:** The interface will display:
    * **Original Video:** A preview of the uploaded video.
    * **Overall Scores:** Metrics for the average "Fake" and "Real" scores across all analyzed frames.
    * **Final Verdict:** A clear "LIKELY FAKE" or "LIKELY REAL" assessment based on the average scores.
    * **Frame-by-Frame Chart:** An interactive Plotly chart showing the confidence scores over the video's timeline.
    * **Detailed Data:** An expandable table with the specific scores for each frame that was processed.



***

## 💼 Business Model

Our solution is designed for broad commercial application with a flexible business model:

* [cite_start]**API Monetization:** A pay-as-you-go model for deepfake detection calls, with tiered subscriptions for different usage levels. [cite: 6]
* [cite_start]**Enterprise Licensing:** On-premises deployment, co-development partnerships, and white-label licensing for financial institutions and government agencies. [cite: 6]
* [cite_start]**Target Markets:** [cite: 6]
    * [cite_start]**Financial Services:** Banks, payment providers, and fintech companies. [cite: 6]
    * [cite_start]**Telecom & Video Conferencing:** Platforms like Zoom and Microsoft Teams. [cite: 6]
    * [cite_start]**Government & Defense:** National security and critical infrastructure protection. [cite: 6]

***

## 🤝 Our Team

* **Anikeit Khanna** - `anikeitk23@iitk.ac.in`
* **Kshitiz Tyagi** - `ktyagi23@iitk.ac.in`
* **Ankit Kumar** - `ankitk23@iitk.ac.in`
