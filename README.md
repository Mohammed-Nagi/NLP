# SemEval-2026 Task 9: A Two-Stage Framework for Tackling Class Imbalance in Multilingual Polarization Detection

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](SemEval_2026_Task_9.ipynb)
![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=flat&logo=PyTorch&logoColor=white)

**Author:** Mohammed Nagi (AIMS South Africa)

## Project Overview

This repository contains the source code and implementation for the system description paper **"A Two-Stage Framework for Tackling Class Imbalance in Multilingual Polarization Detection"**, submitted to the SemEval-2026 Task 9 shared task.

The project addresses the detection of online polarization across three subtasks in English and Arabic:
1.  **Subtask 1:** Polarization Detection (Binary Classification).
2.  **Subtask 2:** Polarization Type Classification (Multi-label).
3.  **Subtask 3:** Polarization Manifestation Identification (Multi-label).

To address the challenge of extreme class imbalance (up to 16:1 ratio) and low-resource constraints, we implement a two-stage training framework utilizing domain-adapted transformer models and Focal Loss.

## Methodology

The system architecture prioritizes data efficiency and loss function engineering over parameter scaling.

### 1. Model Selection
We employ language-specific models pre-trained on domain-relevant corpora:
* **Arabic:** `UBC-NLP/MARBERTv2` (Pre-trained on 1 billion Arabic tweets) is used for all subtasks to capture dialectal morphology and informal lexicon.
* **English:**
    * `cardiffnlp/twitter-roberta-base-sentiment` is used for Subtasks 1 and 3 to leverage sentiment-aware pre-training for manifestation detection.
    * `microsoft/deberta-v3-base` is used for Subtask 2 to utilize disentangled attention for topic classification.

### 2. Imbalance Mitigation
* **Focal Loss:** Replaces standard Binary Cross-Entropy to dynamically down-weight easy negatives. The focusing parameter $\gamma$ is set to 1.5 for Arabic and 2.0 for English.
* **Threshold Optimization:** A post-processing step calculates the optimal decision threshold for each class independently on the validation set to maximize the Macro-F1 score.

### 3. Training Pipeline
The implementation follows a two-stage strategy:
* **Stage 1 (Validation):** Cross-validation is performed to derive optimal per-class probability thresholds.
* **Stage 2 (Inference):** The model is retrained on the full dataset (training + validation), and the optimized thresholds from Stage 1 are applied to the test set predictions.

## Repository Structure

* `SemEval_2026_Task_9.ipynb`: The primary notebook containing data loading, preprocessing, model definitions, training loops, and inference logic.
* `README.md`: System documentation.
* `requirements.txt`: Python dependencies.

## Usage Instructions

The code is designed to run in a Jupyter environment, specifically optimized for Google Colab with GPU acceleration.

### Prerequisites
* Python 3.8+
* PyTorch
* Transformers (Hugging Face)

### Execution
1.  **Environment Setup:** Install the required dependencies.
    ```bash
    pip install transformers datasets accelerate evaluate scikit-learn emoji pyarabic
    ```
2.  **Data Configuration:** The notebook assumes the dataset is located in Google Drive. Modify the `base_path` variable in the data loading section to point to your local or Drive directory.
3.  **Reproduction:** Run the cells sequentially to perform:
    * Text preprocessing (normalization and cleaning).
    * Model training (Subtasks 1, 2, and 3).
    * Generation of prediction files (`.csv`) for the leaderboard.

## Experimental Results

The following table reports the Macro-F1 scores achieved on the internal validation set using stratified 5-fold cross-validation.

| Subtask | Language | Model | Macro F1 |
| :--- | :--- | :--- | :--- |
| **S1: Detection** | English | Twitter-RoBERTa | **0.811** |
| | Arabic | MARBERTv2 | **0.795** |
| **S2: Type** | English | DeBERTa-v3 | **0.397** |
| | Arabic | MARBERTv2 | **0.602** |
| **S3: Manifestation** | English | Twitter-RoBERTa | **0.501** |
| | Arabic | MARBERTv2 | **0.567** |

## Citation

If you utilize this implementation in your research, please cite the associated report:

```bibtex
@techreport{nagi2026polarization,
  title={SemEval-2026 Task 9: A Two-Stage Framework for Tackling Class Imbalance in Multilingual Polarization Detection},
  author={Nagi, Mohammed},
  institution={African Institute for Mathematical Sciences (AIMS), South Africa},
  year={2026}
}
