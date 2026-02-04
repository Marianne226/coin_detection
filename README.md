# Coin Detection and Classification

A computer vision project for detecting and classifying Swiss Franc (CHF) and Euro (EUR) coins from images with varying backgrounds.

## Project Overview

### Objectives

The main goal of this project is to automatically detect and classify coins in images taken under different conditions. The system handles three types of backgrounds:

- **Neutral Background**: Clean, white/uniform backgrounds
- **Hand-Held Background**: Coins held in a hand
- **Noisy Background**: Cluttered or textured backgrounds

The pipeline consists of four main stages:

1. **Background Recognition** - Classify the image background type using color thresholding (RGB/HSV analysis)
2. **Image Pre-processing** - Apply background-specific algorithms to isolate coins
3. **Coin Classification** - Use a fine-tuned ResNet-50 model to identify coin denominations
4. **Output Generation** - Export predictions in CSV format for evaluation

### Supported Coins

| Swiss Franc (CHF) | Euro (EUR) |
|-------------------|------------|
| 5 CHF | 2 EUR |
| 2 CHF | 1 EUR |
| 1 CHF | 0.50 EUR |
| 0.50 CHF | 0.20 EUR |
| 0.20 CHF | 0.10 EUR |
| 0.10 CHF | 0.05 EUR |
| 0.05 CHF | 0.02 EUR |
| | 0.01 EUR |

---

## Code Structure

```
coin_detection/
├── Project_group29.ipynb      # Main Jupyter notebook with complete pipeline
├── README.md                  # This file
├── requirements.txt           # Python dependencies
├── model28_05.pth             # Pre-trained classification model (generated/provided)
├── predicted_coins28_05.csv   # Output predictions (generated)
├── Report-IMG/                # Images used in the report/notebook
├── train/                     # Training dataset
│   ├── 1. neutral_bg/         # Neutral background images
│   ├── 2. noisy_bg/           # Noisy background images
│   ├── 3. hand/               # Hand-held background images
│   ├── 4. neutral_bg_outliers/
│   ├── 5. noisy_bg_outliers/
│   └── 6. hand_outliers/
├── test/                      # Test dataset for predictions
└── final_train_data_split/    # Processed training data (generated)
```

### Notebook Structure

The `Project_group29.ipynb` notebook is organized into the following sections:

| Section | Description |
|---------|-------------|
| **Package Importation** | Load required libraries and dependencies |
| **Background Detection** | Threshold-based classification using ROI color analysis |
| **Coin Isolation** | Three specialized pre-processing algorithms for each background type |
| **Coin Detection** | Contour detection and circular shape extraction |
| **Coin Extraction** | Extract individual coin images from detected regions |
| **Train Dataset Creation** | Organize training data for model training |
| **Training the Classification Model** | Fine-tune ResNet-50 for coin classification |
| **Predicting the Test Images** | Run inference on test set and generate output CSV |

---

## Installation

### Prerequisites

- Python 3.9
- CUDA-compatible GPU (optional, for faster training)

### Setup

1. **Clone the repository**

```bash
git clone <repository-url>
cd coin_detection
```

2. **Create a virtual environment** (recommended)

```bash
conda create -n coin_detection python=3.9
conda activate coin_detection
```

3. **Install dependencies**

```bash
pip install numpy matplotlib scikit-image pillow opencv-python
pip install torch torchvision
pip install datasets
```

Or install all at once:

```bash
pip install numpy matplotlib scikit-image pillow opencv-python torch torchvision datasets
```

### Required Data

Ensure the following folders are present in the project root:

- `train/` - Training images organized by background type
- `test/` - Test images for prediction
- `Report-IMG/` - Report images (optional, for visualization)

---

## Usage

### Running the Complete Pipeline

1. Open the Jupyter notebook:

```bash
jupyter notebook Project_group29.ipynb
```

2. Run all cells sequentially to:
   - Load and analyze the training data
   - Train the classification model (or load pre-trained weights)
   - Generate predictions for the test set

### Using a Pre-trained Model

If you have the pre-trained model (`model28_05.pth`), you can skip the training section and directly run the prediction cells:

```python
# Load the saved model
model.load_state_dict(torch.load('model28_05.pth'))
model.eval()
```

### Output Format

The predictions are saved as a CSV file (`predicted_coins28_05.csv`) with the following columns:

| Column | Description |
|--------|-------------|
| `id` | Image filename (without extension) |
| `5CHF`, `2CHF`, ... | Count of each CHF coin denomination |
| `2EUR`, `1EUR`, ... | Count of each EUR coin denomination |
| `OOD` | Out-of-distribution / unrecognized objects |

---

## Technical Details

### Background Classification

The algorithm uses a Region of Interest (ROI) at the bottom-middle of each image to classify backgrounds:

- **Neutral**: Low hue (< 17) and saturation (< 60)
- **Noisy**: Specific RGB/HSV thresholds indicating cluttered backgrounds
- **Hand**: All other cases (typically skin tones)

Achieves **100% accuracy** on the training set.

### Pre-processing Pipeline

Each background type has a specialized pre-processing algorithm:

| Background | Techniques |
|------------|------------|
| Neutral | Adaptive thresholding, morphological closing/opening |
| Hand | HSV + RGB masking, intensive morphological transformations |
| Noisy | HSV filtering, morphological operations |

All images are resized by a factor of 0.6 for computational efficiency.

### Classification Model

- **Architecture**: ResNet-50 (pre-trained on ImageNet)
- **Fine-tuning**: Final classification layer replaced for 23 coin classes
- **Input size**: 224 × 224 pixels
- **Normalization**: ImageNet mean/std ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

---

## License

This project was developed as part of the IAPR (Image Analysis and Pattern Recognition) course at EPFL.
