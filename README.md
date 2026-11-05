# ME3101 Mini-Project 2: CNN-Based Classification of Hyperspectral Polymer Images

**Student Name:** Kumar Yash Raj  
**Student Roll Number:** 2401ME38  
**Course:** ME3101 — Machine Learning for Mechanical Engineering (Autumn 2026)  
**Task:** Design, train, and evaluate a Convolutional Neural Network (CNN) in MATLAB to classify near-infrared (NIR) hyperspectral images into four polymer categories:
- **HDPE** (High-Density Polyethylene)
- **PET** (Polyethylene Terephthalate)
- **PP** (Polypropylene)
- **PS** (Polystyrene)

---

## 1. Directory Structure

```text
2401ME38_Kumar_Yash_Raj_project_2/
├── README.md                       # Comprehensive documentation & execution guide
├── Code/                           # MATLAB source code & helper scripts
│   ├── load_hyperspectral_cube.m   # Robust reader for ENVI BIL/HDR hyperspectral cubes
│   ├── select_pca_bands.m          # PCA band selection on training data only
│   ├── pad_hyperspectral_image.m   # Balanced zero-padding to HT x WT (74x74)
│   ├── create_datastores.m         # Constructs MATLAB datastores with custom reader
│   ├── build_cnn_model.m           # 2D CNN architecture definition (Shallow, Baseline, Deep)
│   ├── train_cnn.m                 # Main MATLAB CNN training script (Adam, BatchNorm, Dropout)
│   ├── evaluate_cnn.m              # Held-out evaluation: 4x4 confusion matrix, precision, recall, F1
│   ├── run_experiments.m           # Displays controlled comparisons across depth, dropout, LR
│   ├── classify_unseen_image.m     # Task 8 unseen BIL/HDR inference routine without retraining
│   ├── readfile.m                  # Supplied helper script for BIL/HDR inspection
│   └── reduceHypercubePCA.m        # Supplied helper script for PCA band reduction
├── Model/                          # Pre-trained models & preprocessing parameters
│   ├── trained_cnn.mat             # Trained Baseline CNN weights & parameters (ready for inference)
│   ├── preprocessing_params.mat    # 10 PCA band indices, target size (74x74), scale factor
│   └── split_info.json             # Stratified train (70%), val (15%), test (15%) file lists
├── Results/                        # Experimental tables & publication plots
│   ├── experimental_results.csv    # Performance metrics across all 7 controlled CNN experiments
│   ├── experimental_results.mat    # MATLAB struct containing experimental results
│   ├── confusion_matrix.mat        # 4x4 test confusion matrix and summary metrics
│   └── Plots/                      # 300-DPI publication figures
│       ├── representative_spectra.png
│       ├── pca_variance_explained.png
│       ├── pca_band_loadings.png
│       ├── sample_raw_vs_padded.png
│       ├── training_curves_final_cnn.png
│       ├── experiment_comparison_accuracy.png
│       ├── experiment_comparison_loss.png
│       ├── confusion_matrix_heatmap.png
│       ├── class_wise_metrics.png
│       ├── unseen_image_prediction.png
│       └── unseen_image_prediction_matlab.png
└── Report/                         # Technical Report
    └── Technical_Report.pdf        # Complete technical report detailing theory, experiments, and answers
```

---

## 2. MATLAB & Toolbox Requirements

- **MATLAB Version:** MATLAB R2022b or later (tested on MATLAB R2026b).
- **Supported Toolboxes:**
  - *Deep Learning Toolbox* (for `trainnet` / `trainNetwork`, `layerGraph`, `imageDatastore`).
  - *Image Processing Toolbox* (for `imhypercube`, `enviinfo`).
  - *Statistics and Machine Learning Toolbox* (for `pca`).
- **Universal Portability (Zero-Toolbox Native Fallback):**
  - All scripts include a built-in native MATLAB fallback.
  - If the Deep Learning Toolbox or Statistics Toolbox is absent on the evaluation machine, `load_hyperspectral_cube.m`, `select_pca_bands.m`, `evaluate_cnn.m`, and `classify_unseen_image.m` automatically execute using vectorized native array math. Any MATLAB installation can run inference and evaluation seamlessly.

---

## 3. Dataset Source and Layout

The dataset consists of 200 hyperspectral fragments (.bil images with .hdr ENVI headers) collected with a Resonon Pika IR benchtop line-scan system (900–1700 nm, 168 bands, 14-bit depth):
```text
Dataset/
├── HDPE/   (50 .bil and .hdr files, lines: 26–68, samples: 27–73)
├── PET/    (50 .bil and .hdr files)
├── PP/     (50 .bil and .hdr files)
└── PS/     (50 .bil and .hdr files)
```
- Each scan (`scan0001` through `scan0010`) represents an acquisition group of 20 fragments (5 per class).

---

## 4. Execution Order (Step-by-Step)

Open MATLAB, navigate to the unzipped project folder, and run:

### Step 1: Add Code Folder to Path
```matlab
addpath('Code');
```

### Step 2: Evaluate Trained Model on Held-Out Test Set (Task 7)
```matlab
evalResults = evaluate_cnn;
```
*Loads the 30 held-out test images, computes and displays the $4 \times 4$ confusion matrix, class-wise precision, recall, F1-scores, and overall accuracy.*

### Step 3: Classify an Unseen Hyperspectral Image (Task 8)
```matlab
% Default demonstration on unseen PET fragment:
classify_unseen_image('Dataset/PET/scan0001_frag01.bil');

% Or test on an HDPE fragment:
classify_unseen_image('Dataset/HDPE/scan0002_frag11.bil');
```
*Loads preprocessing parameters and trained weights, extracts the 10 PCA bands, applies balanced padding, and predicts the polymer class with confidence probabilities without retraining.*

### Step 4: Display Controlled Experiments Comparison (Task 6)
```matlab
run_experiments;
```
*Displays the quantitative results table comparing all 7 CNN configurations across depth, dropout, and learning rate.*

### Step 5: (Optional) Re-Train the Baseline CNN (Task 5)
```matlab
train_cnn;
```
*Re-trains the 3-block CNN for 35 epochs using Adam optimizer and updates `Model/trained_cnn.mat`.*

---

## 5. Configurable Preprocessing and Hyperparameters

All pipeline parameters are stored in `Model/preprocessing_params.mat`:

| Parameter | Value | Engineering Justification |
| :--- | :---: | :--- |
| `targetHeight` ($H_T$) | **74** | Standardized height; covers max fragment line count (68) with balanced padding. |
| `targetWidth` ($W_T$) | **74** | Standardized width; covers max sample count (73) with balanced padding. |
| `selectedBands` | **[159..168]** | 10 common original spectral bands (1672.87–1719.25 nm) selected via PCA on training data only ($2\nu_{\text{CH}}$ C-H overtone region). |
| `normScale` | **65535.0** | Normalization divisor scaling raw 16-bit counts to $[0, 1]$. |
| `padValue` | **0.0** | Constant zero-padding simulating dark zero-radiance sensor background. |
| `batchSize` | **16** | Mini-batch size for training. |
| `initialLearnRate` | **1e-3** | Adam initial learning rate for stable convergence. |
| `weightDecay` ($L_2$) | **1e-4** | Regularization penalty preventing weight explosion. |
| `dropoutRate` | **0.3** | Prevents feature co-adaptation before final FC layer. |

---

## 6. Industrial Sorting Economics: Metric Prioritization

For an automated polymer recycling line, **Precision must be prioritized over Recall**:
- **False Positives (High Severity):** Ejecting a foreign polymer (e.g., PVC or PP) into a high-purity recycled PET stream causes severe degradation during melt re-extrusion, resulting in the rejection of entire multi-ton resin batches ($>\$10,000$ financial loss).
- **False Negatives (Tolerable Severity):** Missing a recyclable bottle simply directs it to the reject conveyor, representing a minor recoverable yield loss.
- Therefore, optical sorters are tuned for **maximum precision ($\ge 99.5\%$ purity)** to protect product quality.
