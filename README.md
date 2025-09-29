# Bank of Baroda Zero-Trust Hackathon: Hybrid Identity Monitoring & Deepfake-Resistant Verification

This repository contains the code and resources for our submission to the Bank of Baroda Zero-Trust Hackathon. Our project focuses on providing a robust solution for hybrid identity monitoring and deepfake-resistant verification, essential for securing digital banking systems like video KYC (vKYC).

## 🚩 The Problem

The rise of AI-generated deepfakes poses a significant threat to digital identity verification. These sophisticated fakes can convincingly impersonate real individuals, bypassing traditional security measures and compromising user trust. The core challenges include:

* **Bypassing vKYC:** AI-generated deepfakes can alter faces and voices to deceive verification systems.
* **Continuous Verification:** The Zero Trust paradigm requires that no user or device is inherently trusted, meaning every identity assertion must be continuously re-validated.

## 💡 Our Solution

We have developed a transformer-based deepfake detection model designed to be integrated into the vKYC pipeline. The model processes each image or video frame and outputs a probability score, flagging suspicious inputs for further review.

### Key Features:

* [cite_start]**Fine-tuned Architecture:** Utilizes the `SiglipForImageClassification` model, fine-tuned to distinguish between authentic and synthetic media. [cite: 1]
* [cite_start]**Vision-Language Model:** Leverages both visual patterns and contextual information for more accurate detection. [cite: 1]
* [cite_start]**Binary Classification:** Provides clear, actionable results with confidence scores for each prediction (Real vs. Fake). [cite: 1]
* [cite_start]**Specialized Training:** Trained on the `OpenDeepfake-Preview` dataset to recognize patterns specific to synthetic media. [cite: 1]

## ⚙️ How It Works

Our solution follows a structured pipeline to ensure high accuracy and reliability:

1.  [cite_start]**Model Architecture:** Built on Google's SigLIP vision-language transformer for state-of-the-art image classification. [cite: 2]
2.  [cite_start]**Training Data:** Fine-tuned on a dataset of approximately 20,000 real and AI-generated faces to ensure balanced performance. [cite: 2]
3.  [cite_start]**Inference Process:** Input images are resized, normalized, and scored by the model to determine if they are real or fake. [cite: 2]
4.  [cite_start]**Decision Logic:** Frames with scores above a set threshold trigger security flags and may require additional authentication. [cite: 3]
5.  [cite_start]**Zero Trust Alignment:** Every identity claim is explicitly verified with each access attempt, adhering to the Zero Trust security model. [cite: 3]

## 📊 Performance Metrics

The model was evaluated on a balanced dataset of 19,999 samples and achieved the following performance:

| Metric | Fake Detection | Real Detection |
| :--- | :--- | :--- |
| **Precision** | 97.18% | 92.01% |
| **Recall** | 91.55% | 97.34% |
| **F1-Score** | 94.28% | 94.60% |

**Overall Accuracy:** 94.44%

[cite_start]A detailed confusion matrix is also available in the presentation, showing 9,155 true positives for fakes and 9,733 for real images. [cite: 8]


## 🚀 Getting Started

To get started with our deepfake detection model, follow these steps:

### Prerequisites

* Python 3.8+
* PyTorch
* Transformers library
* Other dependencies listed in `requirements.txt`

### Installation

1.  Clone the repository:
    ```bash
    git clone [https://github.com/Kshitiztyagi23/BOBhackathon_2025.git](https://github.com/Kshitiztyagi23/BOBhackathon_2025.git)
    cd BOBhackathon_2025
    ```
2.  Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```

### Usage

To make a prediction on a new image, you can use the provided inference script. Make sure to place your image in the `data` folder.

```python
from main import predict_image

image_path = "data/sample_image.jpg"
prediction = predict_image(image_path)
print(f"Prediction: {prediction}")
