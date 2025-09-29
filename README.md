# Bank of Baroda Zero-Trust Hackathon: Hybrid Identity Monitoring & Deepfake-Resistant Verification

This repository contains the code for our submission to the Bank of Baroda Zero-Trust Hackathon. Our project provides a robust solution for hybrid identity monitoring and deepfake-resistant verification, featuring a transformer-based model and an interactive web interface for video analysis.

***

## 🚩 The Problem

The rise of AI-generated deepfakes poses a significant threat to digital identity verification processes like video KYC (vKYC). These sophisticated fakes can convincingly impersonate real individuals, bypassing traditional security measures. The core challenge is to implement a **Zero Trust** security model where every identity assertion is continuously re-validated to prevent fraud.

***

## 💡 Our Solution

We have developed a transformer-based deepfake detection model designed to be integrated into the vKYC pipeline. The model processes each image or video frame and outputs a probability score, flagging suspicious inputs for further review.

### Key Features:

* **Fine-tuned Architecture:** Utilizes the `SiglipForImageClassification` model, fine-tuned to distinguish between authentic and synthetic media.
* **Vision-Language Model:** Leverages both visual patterns and contextual information for more accurate detection.
* **Binary Classification:** Provides clear, actionable results with confidence scores for each prediction (Real vs. Fake).
* **Specialized Training:** Trained on the `OpenDeepfake-Preview` dataset to recognize patterns specific to synthetic media.

***

## ⚙️ How It Works

Our solution follows a structured pipeline to ensure high accuracy and reliability:

1.  **Model Architecture:** Built on Google's SigLIP vision-language transformer for state-of-the-art image classification.
2.  **Training Data:** Fine-tuned on a dataset of approximately 20,000 real and AI-generated faces to ensure balanced performance.
3.  **Inference Process:** Input images are resized, normalized, and scored by the model to determine if they are real or fake.
4.  **Decision Logic:** Frames with scores above a set threshold trigger security flags and may require additional authentication.
5.  **Zero Trust Alignment:** Every identity claim is explicitly verified with each access attempt, adhering to the Zero Trust security model.

***

## 📊 Performance Metrics

The model was evaluated on a balanced dataset of **19,999 samples** and achieved the following performance:

| Metric | Fake Detection | Real Detection |
| :--- | :--- | :--- |
| **Precision** | 97.18% | 92.01% |
| **Recall** | 91.55% | 97.34% |
| **F1-Score** | 94.28% | 94.60% |

**Overall Accuracy: 94.44%**



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


### 🎥 Demonstration Videos

Check out these videos to see the deepfake detector in action:

* **[Demonstration 1: Real Video Analysis with complex elements for harsh testing](https://drive.google.com/file/d/1M8LUAyxiKPSc79hLbV8pCcnKoNMp4bVU/view?usp=sharing)**
* **[Demonstration 2: Deepfake Video Analysis](https://drive.google.com/file/d/15GO0nxW0wlbHmZSWCYwsPGwQqFsIY9_8/view?usp=sharing)**




***

## 💼 Business Model

Our solution is designed for broad commercial application with a flexible business model:

* **API Monetization:** A pay-as-you-go model for deepfake detection calls, with tiered subscriptions for different usage levels.
* **Enterprise Licensing:** On-premises deployment, co-development partnerships, and white-label licensing for financial institutions and government agencies.
* **Target Markets:**
    * **Financial Services:** Banks, payment providers, and fintech companies.
    * **Telecom & Video Conferencing:** Platforms like Zoom and Microsoft Teams.
    * **Government & Defense:** National security and critical infrastructure protection.

***

## 🤝 Our Team

* **Anikeit Khanna** - `anikeitk23@iitk.ac.in`
* **Kshitiz Tyagi** - `ktyagi23@iitk.ac.in`
* **Ankit Kumar** - `ankitk23@iitk.ac.in`
