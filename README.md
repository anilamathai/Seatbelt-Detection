```
# Seatbelt Classifier: Real-Time Seatbelt Detection

## Overview

The Seatbelt Classifier is a lightweight, real-time image classifier designed to detect whether a person is wearing a seatbelt. It balances high accuracy with practical usability and handles ambiguous or very low-quality (VLQ) images by returning a confidence score of zero. The model uses **MobileNetV2** as its backbone for fast inference.

**Key Features:**

- User-facing predictions: `Seatbelt` or `No-Seatbelt`
- Internal class for training: `VLQ` (ambiguous/low-quality images)
- Lightweight backbone: MobileNetV2
- Confidence-based handling of ambiguous images

## Assumptions

- Training and validation images are organized in folders by class: `Seatbelt`, `No-Seatbelt`, `VLQ`.
- The VLQ class is not used in user-facing predictions; it is only for detecting ambiguous images.
- Images are RGB and of varying sizes; they will be resized to `224x224` pixels.
- Confidence threshold (default `0.6`) determines if a prediction is reliable; predictions below this threshold or classified as VLQ are returned with confidence `0`.

## Folder Structure

seatbelt_classifier/
├─ assets/               # Icons and example images for visualization
├─ checkpoint/           # Saved model weights
│  └─ best_model.pth
├─ data/                 # Training images
│  ├─ Seatbelt/
│  ├─ No-Seatbelt/
│  └─ VLQ/
├─ infer.py              # Inference script
├─ train.py              # Training script
├─ model.py              # Model definition
├─ utils.py              # Helper functions for config and checkpoint handling
├─ config.yaml           # Configuration file
├─ requirements.txt      # Python dependencies
└─ README.md             # Project description and instructions

## Configuration

All hyperparameters and paths are defined in `config.yaml`:

model:
  name: mobilenet_v2
  pretrained: true
  num_classes: 3

training:
  image_size: 224
  batch_size: 32
  epochs: 25
  learning_rate: 0.0003

inference:
  confidence_threshold: 0.6

paths:
  data_dir: data
  checkpoint_dir: checkpoint

## Approach

- **Data Preprocessing:** Images are resized to 224x224 and converted to tensors. Data augmentation can be added for robustness.
- **Model Architecture:** MobileNetV2 backbone with 3 output neurons (`Seatbelt`, `No-Seatbelt`, `VLQ`), optimized for fast inference.
- **Training:** Uses cross-entropy loss and Adam optimizer. Trains on the data folder organized by class. Best model saved in `checkpoint/best_model.pth`.
- **Inference:** Input image is preprocessed and fed through the model. Softmax probabilities are computed for all three classes.
  - VLQ probability > 0.4 → confidence = 0
  - Confidence < threshold → confidence = 0
  - Otherwise, returns class with max probability among `Seatbelt` / `No-Seatbelt`.

## Execution Steps

### 1. Install Dependencies
pip install -r requirements.txt

### 2. Training
python train.py
# Trains MobileNetV2 on your dataset and saves the best model in checkpoint/best_model.pth.

### 3. Inference
python infer.py --image path_to_image.jpg
# Example output:
{"prediction": "Seatbelt", "confidence": 0.987}

## Notes

- The VLQ class is used internally only and never returned to users.
- User-facing predictions are always `Seatbelt` or `No-Seatbelt`.
- Confidence handling ensures ambiguous or low-quality images return `0` confidence.
- Model is optimized for real-time inference on CPU/GPU.

## Cite

@misc{seatbelt2026,
title={Seatbelt Classifier: Real-Time Detection of Seatbelt Usage},
author={Anila Mathai},
year={2026},
note={GitHub repository: https://github.com/anilamathai/seatbelt_classifier}
}
```
