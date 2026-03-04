# NLP Coursework

This repository contains the implementation and results for the NLP coursework project.

## Project Structure

```
nlp_coursework/
└── BestModel/
    ├── train.py          # Training code
    ├── best_model.pt     # Trained model checkpoint
    ├── dev.txt           # Development set predictions
    ├── test.txt          # Test set predictions
    └── results.json      # Training results and metrics
```

## BestModel Directory

All the core files for the best performing model are located in the `BestModel/` directory.

### Code

- **`train.py`**: The main training script that implements:
  - RoBERTa-base model for binary classification
  - CrossEntropyLoss with label smoothing
  - Weighted random sampling for balanced batches
  - Differential learning rates for backbone and classification head
  - Cosine learning rate schedule with warmup

### Model Checkpoint

- **`best_model.pt`**: The trained model checkpoint saved after training. This file contains the complete model state and can be loaded for inference or further training.

### Prediction Results

- **`dev.txt`**: Predictions for the development set. Each line contains a binary prediction (0 or 1) for the corresponding development sample.

- **`test.txt`**: Predictions for the test set. Each line contains a binary prediction (0 or 1) for the corresponding test sample.

### Results

- **`results.json`**: Contains comprehensive training results including:
  - Model configuration and method used
  - Best validation F1 score
  - Final metrics (F1 macro, F1 positive, F1 negative)
  - Training history (loss curves, validation metrics across epochs)

## Model Details

- **Model**: RoBERTa-base
- **Loss Function**: CrossEntropyLoss with Label Smoothing (0.1)
- **Best Validation F1 (Macro)**: 0.7897
- **Training Epochs**: 10
- **Batch Size**: 32
- **Max Length**: 128 tokens

## Usage

To use the trained model for inference or to retrain:

```bash
cd BestModel
python train.py
```

Make sure to update the data paths in the `Config` class within `train.py` according to your environment.

## Results Summary

The model achieves the following performance on the validation set:
- **F1 Macro**: 0.7897
- **F1 Positive**: 0.6178
- **F1 Negative**: 0.9616

For detailed training history and metrics, please refer to `BestModel/results.json`.
