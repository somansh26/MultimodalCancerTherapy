# Multimodal Cancer Therapy Framework

This repository implements a multimodal machine learning framework for personalized cancer therapy decision support. It integrates imaging, textual, and omics data using advanced neural network architectures with cross-modal attention and transformer-based fusion.

## Features
- **Imaging Module**: Processes and extracts features from radiological images using a convolutional neural network.
- **NLP Module**: Leverages pre-trained BERT models to extract semantic features from clinical narratives.
- **Omics Module**: Handles genomics and proteomics data with a fully connected neural network.
- **Cross-Modal Fusion**: Combines data from all modalities with cross-modal attention and transformer-based fusion.
- **Custom Metrics**: Computes ROC-AUC, F1-Score, and the Concordance Index (C-index) for survival prediction.
- **End-to-End Workflow**: Includes data preprocessing, model training, evaluation, and visualization.

---

## Repository Structure

```plaintext
MultimodalCancerTherapy/
│
├── data/               # Dataset management and loaders
│   ├── data_loader.py  # Downloads and preprocesses datasets
│   ├── sample_images/  # Example image files (for quick testing)
│   ├── sample_text.csv # Example text data (for quick testing)
│   └── sample_omics.csv# Example omics data (for quick testing)
│
├── experiments/        # Training and evaluation scripts
│   ├── train_model.py  # Script to train the model
│   └── evaluate_model.py # Script to evaluate the model
│
├── models/             # Model definitions
│   ├── fusion_model.py # Multimodal transformer-based fusion model
│   └── modules/        # Submodules for individual modalities
│       ├── imaging_module.py
│       ├── nlp_module.py
│       └── omics_module.py
│
├── preprocessing/      # Preprocessing scripts for each modality
│   └── preprocess_data.py
│
├── utils/              # Helper functions and utilities
│   ├── logger.py       # Logging and experiment tracking
│   ├── metrics.py      # Evaluation metrics
│   └── visualization.py # Visualization functions (e.g., ROC curve)
│
├── README.md           # Project overview and setup guide
└── requirements.txt    # Python dependencies



## Dataset Table

| Dataset   | Type          | Description                        | Link         |
|-----------|---------------|------------------------------------|--------------|
| TCIA      | Imaging       | Radiological images (CT, MRI)     | [Visit Site](https://www.cancerimagingarchive.net/) |
| MIMIC-CXR | Imaging, Text | Chest X-rays and radiology reports| [Visit Site](https://physionet.org/content/mimic-cxr/2.0.0/) |
| TCGA      | Omics         | Genomics and proteomics data      | [Visit Site](https://portal.gdc.cancer.gov/) |




