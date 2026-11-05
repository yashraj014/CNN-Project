import os
import glob
import json
import random
import numpy as np
import pandas as pd
import scipy.io as sio
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# Set random seeds for reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {DEVICE}")

# 1. Dataset loading and inspection
CLASSES = ['HDPE', 'PET', 'PP', 'PS']
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}
IDX_TO_CLASS = {i: c for i, c in enumerate(CLASSES)}

data_dir = 'Dataset'
records = []
for cls in CLASSES:
    cls_dir = os.path.join(data_dir, cls)
    bil_files = sorted(glob.glob(os.path.join(cls_dir, '*.bil')))
    for bf in bil_files:
        hf = bf.replace('.bil', '.hdr')
        with open(hf, 'r') as f:
            lines = f.readlines()
        meta = {}
        for l in lines:
            if '=' in l:
                k, v = l.split('=', 1)
                meta[k.strip().lower()] = v.strip()
        H = int(meta['lines'])
        W = int(meta['samples'])
        B = int(meta['bands'])
        base = os.path.basename(bf).replace('.bil', '')
        parts = base.split('_')
        scan = parts[0]
        frag = parts[1] if len(parts) > 1 else 'frag01'
        records.append({
            'bil_path': bf,
            'hdr_path': hf,
            'class': cls,
            'class_idx': CLASS_TO_IDX[cls],
            'scan': scan,
            'frag': frag,
            'H': H,
            'W': W,
            'B': B
        })

df = pd.DataFrame(records)
print(f"Loaded {len(df)} images across {len(CLASSES)} classes.")
print(df.groupby('class').size())

# 2. Partition Data (70% train, 15% val, 15% test)
# Stratified per class with fixed random seed
rng = np.random.RandomState(SEED)
train_records = []
val_records = []
test_records = []

for cls in CLASSES:
    cls_df = df[df['class'] == cls].copy()
    shuffled_idx = rng.permutation(len(cls_df))
    # 50 total: 35 train (70%), 7 or 8 val (15%), 8 or 7 test (15%)
    n_train = 35
    n_val = 7 if cls in ['HDPE', 'PP'] else 8
    
    train_idx = shuffled_idx[:n_train]
    val_idx = shuffled_idx[n_train:n_train + n_val]
    test_idx = shuffled_idx[n_train + n_val:]
    
    for idx in train_idx:
        train_records.append(cls_df.iloc[idx].to_dict())
    for idx in val_idx:
        val_records.append(cls_df.iloc[idx].to_dict())
    for idx in test_idx:
        test_records.append(cls_df.iloc[idx].to_dict())

print(f"Data Split Summary: Train={len(train_records)} (70%), Val={len(val_records)} (15%), Test={len(test_records)} (15%)")

# 3. PCA Band Selection on Training Data Only
print("Performing PCA on training spectra...")
train_spectra = []
for item in train_records:
    H, W, B = item['H'], item['W'], item['B']
    raw = np.fromfile(item['bil_path'], dtype='<u2').reshape(H, B, W)
    cube = np.transpose(raw, (0, 2, 1)).astype(np.float64) # (H, W, B)
    X = cube.reshape(-1, B)
    valid_mask = np.all(np.isfinite(X), axis=1)
    X_valid = X[valid_mask]
    # Subsample 500 valid pixels per image for balanced class representation in PCA
    if len(X_valid) > 500:
        sub_idx = rng.choice(len(X_valid), 500, replace=False)
        train_spectra.append(X_valid[sub_idx])
    else:
        train_spectra.append(X_valid)

X_train_pca = np.vstack(train_spectra)
print(f"Total training spectra for PCA: {X_train_pca.shape[0]} samples, {X_train_pca.shape[1]} bands")

# Centered PCA (matching reduceHypercubePCA.m)
X_mean = np.mean(X_train_pca, axis=0)
X_centered = X_train_pca - X_mean
cov_mat = (X_centered.T @ X_centered) / (X_centered.shape[0] - 1)
eigvals, eigvecs = np.linalg.eigh(cov_mat)

# Sort descending
sort_order = np.argsort(eigvals)[::-1]
eigvals = eigvals[sort_order]
coeff = eigvecs[:, sort_order] # PCs are columns

explained = 100.0 * eigvals / np.sum(eigvals)
cumulative_explained = np.cumsum(explained)

variance_threshold = 95.0
num_pcs = int(np.where(cumulative_explained >= variance_threshold - 1e-10)[0][0] + 1)
actual_variance = cumulative_explained[num_pcs - 1]
print(f"PCs explaining >={variance_threshold}%: {num_pcs} PCs (actual {actual_variance:.2f}%)")

# Squared loadings weighted by variance explained
weights = explained[:num_pcs] / np.sum(explained[:num_pcs])
importance = np.sum((coeff[:, :num_pcs] ** 2) * weights, axis=1)

ranking = np.argsort(importance)[::-1]
num_selected = 10
selected_bands_0idx = sorted(ranking[:num_selected])
selected_bands_1idx = [b + 1 for b in selected_bands_0idx]
print(f"Selected 10 Band Indices (1-indexed for MATLAB): {selected_bands_1idx}")
print(f"Selected 10 Band Indices (0-indexed for Python): {selected_bands_0idx}")

# Read wavelengths from hdr
with open(records[0]['hdr_path']) as f:
    hdr_content = f.read()
import re
m = re.search(r'wavelength\s*=\s*\{([^}]+)\}', hdr_content, re.DOTALL)
if m:
    all_wavelengths = [float(w.strip()) for w in m.group(1).replace('\n', ' ').split(',') if w.strip()]
    selected_wavelengths = [all_wavelengths[i] for i in selected_bands_0idx]
    print(f"Selected Band Wavelengths (nm): {selected_wavelengths}")
else:
    all_wavelengths = list(range(1, 169))
    selected_wavelengths = [all_wavelengths[i] for i in selected_bands_0idx]

# 4. Standardize Spatial Dimensions via Padding Only
HT, WT = 74, 74
NORM_SCALE = 65535.0

def load_and_preprocess_image(bil_path, H, W, B, selected_bands, target_H=74, target_W=74):
    raw = np.fromfile(bil_path, dtype='<u2').reshape(H, B, W)
    cube = np.transpose(raw, (0, 2, 1)).astype(np.float32) # (H, W, B)
    cube_sub = cube[:, :, selected_bands] # (H, W, 10)
    
    # Normalization: scale raw detector values to [0, 1]
    cube_norm = cube_sub / NORM_SCALE
    
    # Balanced padding
    pad_top = (target_H - H) // 2
    pad_bottom = target_H - H - pad_top
    pad_left = (target_W - W) // 2
    pad_right = target_W - W - pad_left
    
    padded = np.pad(cube_norm, ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)),
                    mode='constant', constant_values=0.0)
    # PyTorch format: (Channels, Height, Width)
    tensor = np.transpose(padded, (2, 0, 1)) # (10, 74, 74)
    return tensor, padded

# PyTorch Dataset Definition
class PolymerDataset(Dataset):
    def __init__(self, record_list, selected_bands, target_H=74, target_W=74):
        self.records = record_list
        self.selected_bands = selected_bands
        self.target_H = target_H
        self.target_W = target_W

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        item = self.records[idx]
        tensor, _ = load_and_preprocess_image(
            item['bil_path'], item['H'], item['W'], item['B'],
            self.selected_bands, self.target_H, self.target_W
        )
        label = item['class_idx']
        return torch.tensor(tensor, dtype=torch.float32), torch.tensor(label, dtype=torch.long)

train_dataset = PolymerDataset(train_records, selected_bands_0idx, HT, WT)
val_dataset = PolymerDataset(val_records, selected_bands_0idx, HT, WT)
test_dataset = PolymerDataset(test_records, selected_bands_0idx, HT, WT)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

# 5. Model Architectures for Controlled Experiments
# We define modular CNN architectures

class ShallowCNN(nn.Module):
    """2 Conv Blocks: 16 -> 32 filters"""
    def __init__(self, num_classes=4, in_channels=10, dropout_rate=0.3):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2), # 37x37

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2), # 18x18

            nn.AdaptiveAvgPool2d((1, 1)) # 1x1x32
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout_rate),
            nn.Linear(32, num_classes)
        )

    def forward(self, x):
        return self.classifier(self.features(x))

class MediumCNN(nn.Module):
    """3 Conv Blocks: 16 -> 32 -> 64 filters (Baseline Architecture)"""
    def __init__(self, num_classes=4, in_channels=10, dropout_rate=0.3):
        super().__init__()
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(in_channels, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2), # 37x37

            # Block 2
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2), # 18x18

            # Block 3
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2), # 9x9

            nn.AdaptiveAvgPool2d((1, 1)) # 1x1x64
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout_rate),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        return self.classifier(self.features(x))

class DeepCNN(nn.Module):
    """4 Conv Blocks: 16 -> 32 -> 64 -> 128 filters"""
    def __init__(self, num_classes=4, in_channels=10, dropout_rate=0.3):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2), # 37x37

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2), # 18x18

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2), # 9x9

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2), # 4x4

            nn.AdaptiveAvgPool2d((1, 1)) # 1x1x128
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        return self.classifier(self.features(x))

# Training and Evaluation Function
def train_model(model, train_loader, val_loader, epochs=40, lr=1e-3, weight_decay=1e-4):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': []
    }
    
    best_val_loss = float('inf')
    best_weights = None
    
    for epoch in range(epochs):
        # Train
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
        
        train_loss = running_loss / total
        train_acc = correct / total
        
        # Validation
        model.eval()
        val_running_loss = 0.0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_running_loss += loss.item() * inputs.size(0)
                _, preds = torch.max(outputs, 1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)
                
        val_loss = val_running_loss / val_total
        val_acc = val_correct / val_total
        
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            
    # Load best weights
    model.load_state_dict(best_weights)
    return model, history

def evaluate_model(model, dataloader):
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(DEVICE)
            outputs = model(inputs)
            probs = torch.softmax(outputs, dim=1).cpu().numpy()
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_probs.extend(probs)
            
    y_true = np.array(all_labels)
    y_pred = np.array(all_preds)
    probs = np.array(all_probs)
    
    # 4x4 Confusion Matrix
    cm = np.zeros((4, 4), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
        
    overall_acc = np.trace(cm) / np.sum(cm)
    
    # Per-class metrics
    class_metrics = {}
    precisions = []
    recalls = []
    f1s = []
    supports = []
    
    for c in range(4):
        TP = cm[c, c]
        FP = np.sum(cm[:, c]) - TP
        FN = np.sum(cm[c, :]) - TP
        TN = np.sum(cm) - TP - FP - FN
        
        prec = TP / (TP + FP) if (TP + FP) > 0 else 0.0
        rec = TP / (TP + FN) if (TP + FN) > 0 else 0.0
        f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0
        
        class_metrics[CLASSES[c]] = {
            'TP': int(TP), 'FP': int(FP), 'FN': int(FN), 'TN': int(TN),
            'Precision': float(prec), 'Recall': float(rec), 'F1': float(f1),
            'Support': int(TP + FN)
        }
        precisions.append(prec)
        recalls.append(rec)
        f1s.append(f1)
        supports.append(TP + FN)
        
    macro_prec = float(np.mean(precisions))
    macro_rec = float(np.mean(recalls))
    macro_f1 = float(np.mean(f1s))
    
    total_supp = np.sum(supports)
    weighted_f1 = float(np.sum(np.array(f1s) * np.array(supports)) / total_supp)
    
    return {
        'overall_accuracy': float(overall_acc),
        'mean_recall': float(macro_rec),
        'macro_precision': float(macro_prec),
        'macro_recall': float(macro_rec),
        'macro_f1': float(macro_f1),
        'weighted_f1': float(weighted_f1),
        'class_metrics': class_metrics,
        'confusion_matrix': cm,
        'y_true': y_true,
        'y_pred': y_pred,
        'probs': probs
    }

# 6. Run Controlled Experiments
print("\n" + "="*50)
print("STARTING CONTROLLED CNN EXPERIMENTS")
print("="*50)

experiments_def = [
    # Baseline
    {'id': 'Exp1_Baseline', 'name': 'Baseline CNN (3 Blocks, Dropout=0.3, lr=1e-3)',
     'model_cls': MediumCNN, 'dropout': 0.3, 'lr': 1e-3, 'weight_decay': 1e-4, 'category': 'Baseline'},
    
    # Comparison 1: Architecture Depth / Capacity
    {'id': 'Exp2_Shallow', 'name': 'Shallow CNN (2 Blocks, Dropout=0.3, lr=1e-3)',
     'model_cls': ShallowCNN, 'dropout': 0.3, 'lr': 1e-3, 'weight_decay': 1e-4, 'category': 'Architecture Depth'},
    {'id': 'Exp3_Deep', 'name': 'Deep CNN (4 Blocks, Dropout=0.3, lr=1e-3)',
     'model_cls': DeepCNN, 'dropout': 0.3, 'lr': 1e-3, 'weight_decay': 1e-4, 'category': 'Architecture Depth'},
    
    # Comparison 2: Regularization / Dropout Rate
    {'id': 'Exp4_NoDropout', 'name': 'No Dropout CNN (3 Blocks, Dropout=0.0, lr=1e-3)',
     'model_cls': MediumCNN, 'dropout': 0.0, 'lr': 1e-3, 'weight_decay': 1e-4, 'category': 'Regularization'},
    {'id': 'Exp5_HighDropout', 'name': 'High Dropout CNN (3 Blocks, Dropout=0.6, lr=1e-3)',
     'model_cls': MediumCNN, 'dropout': 0.6, 'lr': 1e-3, 'weight_decay': 1e-4, 'category': 'Regularization'},
    
    # Comparison 3: Learning Rate
    {'id': 'Exp6_LowLR', 'name': 'Low LR CNN (3 Blocks, Dropout=0.3, lr=1e-4)',
     'model_cls': MediumCNN, 'dropout': 0.3, 'lr': 1e-4, 'weight_decay': 1e-4, 'category': 'Learning Rate'},
    {'id': 'Exp7_HighLR', 'name': 'High LR CNN (3 Blocks, Dropout=0.3, lr=1e-2)',
     'model_cls': MediumCNN, 'dropout': 0.3, 'lr': 1e-2, 'weight_decay': 1e-4, 'category': 'Learning Rate'}
]

exp_results = []
trained_models = {}
histories = {}
evaluations = {}

for exp in experiments_def:
    print(f"\n--- Running {exp['id']}: {exp['name']} ---")
    torch.manual_seed(SEED)
    model = exp['model_cls'](num_classes=4, in_channels=10, dropout_rate=exp['dropout']).to(DEVICE)
    trained_model, history = train_model(model, train_loader, val_loader,
                                         epochs=35, lr=exp['lr'], weight_decay=exp['weight_decay'])
    eval_res = evaluate_model(trained_model, test_loader)
    
    trained_models[exp['id']] = trained_model
    histories[exp['id']] = history
    evaluations[exp['id']] = eval_res
    
    print(f"Test Acc: {eval_res['overall_accuracy']*100:.2f}%, Macro F1: {eval_res['macro_f1']*100:.2f}%, Weighted F1: {eval_res['weighted_f1']*100:.2f}%")
    
    exp_results.append({
        'Experiment_ID': exp['id'],
        'Description': exp['name'],
        'Category': exp['category'],
        'Final_Train_Acc': history['train_acc'][-1],
        'Final_Val_Acc': history['val_acc'][-1],
        'Final_Train_Loss': history['train_loss'][-1],
        'Final_Val_Loss': history['val_loss'][-1],
        'Test_Accuracy': eval_res['overall_accuracy'],
        'Avg_Class_Acc_Mean_Recall': eval_res['mean_recall'],
        'Macro_Precision': eval_res['macro_precision'],
        'Macro_Recall': eval_res['macro_recall'],
        'Macro_F1_Score': eval_res['macro_f1'],
        'Weighted_F1_Score': eval_res['weighted_f1']
    })

res_df = pd.DataFrame(exp_results)
print("\n" + "="*50)
print("EXPERIMENT SUMMARY RESULTS TABLE")
print("="*50)
print(res_df[['Experiment_ID', 'Category', 'Test_Accuracy', 'Avg_Class_Acc_Mean_Recall', 'Macro_F1_Score', 'Weighted_F1_Score']])

# 7. Select Final Model and Save Weights
final_exp_id = 'Exp1_Baseline'
final_model = trained_models[final_exp_id]
final_eval = evaluations[final_exp_id]
final_history = histories[final_exp_id]

# Save MATLAB compatible structures
os.makedirs('Model', exist_ok=True)
os.makedirs('Results/Plots', exist_ok=True)
os.makedirs('Report', exist_ok=True)
os.makedirs('Code', exist_ok=True)

# Save Preprocessing Parameters (.mat)
preprocessing_params = {
    'selectedBandIndices_1based': np.array(selected_bands_1idx, dtype=np.int32),
    'selectedBandIndices_0based': np.array(selected_bands_0idx, dtype=np.int32),
    'selectedWavelengths': np.array(selected_wavelengths, dtype=np.float64),
    'targetHeight': int(HT),
    'targetWidth': int(WT),
    'numBands': 10,
    'normScale': float(NORM_SCALE),
    'paddingMethod': 'symmetric_balanced_constant_zero',
    'classes': np.array(CLASSES, dtype=object),
    'pcaExplained': np.array(explained, dtype=np.float64),
    'cumulativeExplained': np.array(cumulative_explained, dtype=np.float64),
    'importanceScores': np.array(importance, dtype=np.float64),
    'numPCsRetained': int(num_pcs),
    'varianceThreshold': float(variance_threshold)
}
sio.savemat('Model/preprocessing_params.mat', preprocessing_params)
print("Saved Model/preprocessing_params.mat")

# Export CNN Weights to MATLAB .mat format
# Extract weights for the MediumCNN (3 blocks + FC)
state = final_model.state_dict()

# PyTorch Conv weights: [out_ch, in_ch, kH, kW] -> MATLAB: [kH, kW, in_ch, out_ch]
def pt2mat_conv(w):
    return w.cpu().numpy().transpose(2, 3, 1, 0)

# BatchNorm parameters
matlab_weights = {
    'classes': np.array(CLASSES, dtype=object),
    'targetHeight': int(HT),
    'targetWidth': int(WT),
    'numChannels': 10,
    # Block 1
    'conv1_weights': pt2mat_conv(state['features.0.weight']),
    'conv1_bias': state['features.0.bias'].cpu().numpy(),
    'bn1_scale': state['features.1.weight'].cpu().numpy(),
    'bn1_offset': state['features.1.bias'].cpu().numpy(),
    'bn1_mean': state['features.1.running_mean'].cpu().numpy(),
    'bn1_var': state['features.1.running_var'].cpu().numpy(),
    # Block 2
    'conv2_weights': pt2mat_conv(state['features.4.weight']),
    'conv2_bias': state['features.4.bias'].cpu().numpy(),
    'bn2_scale': state['features.5.weight'].cpu().numpy(),
    'bn2_offset': state['features.5.bias'].cpu().numpy(),
    'bn2_mean': state['features.5.running_mean'].cpu().numpy(),
    'bn2_var': state['features.5.running_var'].cpu().numpy(),
    # Block 3
    'conv3_weights': pt2mat_conv(state['features.8.weight']),
    'conv3_bias': state['features.8.bias'].cpu().numpy(),
    'bn3_scale': state['features.9.weight'].cpu().numpy(),
    'bn3_offset': state['features.9.bias'].cpu().numpy(),
    'bn3_mean': state['features.9.running_mean'].cpu().numpy(),
    'bn3_var': state['features.9.running_var'].cpu().numpy(),
    # Fully Connected
    'fc_weights': state['classifier.2.weight'].cpu().numpy(), # [num_classes, 64]
    'fc_bias': state['classifier.2.bias'].cpu().numpy() # [num_classes]
}
sio.savemat('Model/trained_cnn.mat', matlab_weights)
print("Saved Model/trained_cnn.mat")

# Save CSV and MAT of experimental results
res_df.to_csv('Results/experimental_results.csv', index=False)
sio.savemat('Results/experimental_results.mat', {'experiments': res_df.to_dict(orient='list')})
sio.savemat('Results/confusion_matrix.mat', {
    'confusion_matrix': final_eval['confusion_matrix'],
    'classes': np.array(CLASSES, dtype=object),
    'overall_accuracy': final_eval['overall_accuracy'],
    'macro_f1': final_eval['macro_f1']
})
print("Saved Results/experimental_results.csv, Results/experimental_results.mat, Results/confusion_matrix.mat")

# 8. Generate All Required High-Quality Plots
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')

# Plot 1: PCA Explained Variance
fig, ax1 = plt.subplots(figsize=(8, 5), dpi=300)
ax2 = ax1.twinx()
bands_range = range(1, len(explained) + 1)
ax1.bar(bands_range[:20], explained[:20], color='#1f77b4', alpha=0.7, label='Individual Explained Variance (%)')
ax2.plot(bands_range[:20], cumulative_explained[:20], color='#d62728', marker='o', linewidth=2, label='Cumulative Variance (%)')
ax2.axhline(95, color='gray', linestyle='--', label='95% Threshold')
ax2.axvline(num_pcs, color='darkgreen', linestyle=':', label=f'Cutoff ({num_pcs} PCs)')
ax1.set_xlabel('Principal Component Index', fontsize=12, fontweight='bold')
ax1.set_ylabel('Individual Variance Explained (%)', fontsize=12, color='#1f77b4')
ax2.set_ylabel('Cumulative Variance Explained (%)', fontsize=12, color='#d62728')
plt.title('Task 2: PCA Explained Variance on Hyperspectral Training Spectra', fontsize=13, fontweight='bold')
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc='center right', frameon=True)
plt.tight_layout()
plt.savefig('Results/Plots/pca_variance_explained.png')
plt.close()

# Plot 2: PCA Band Importance Loadings
fig, ax = plt.subplots(figsize=(10, 5), dpi=300)
ax.plot(range(1, 169), importance, color='#2ca02c', linewidth=1.5, label='Band Importance (Variance-Weighted Squared Loading)')
for b in selected_bands_1idx:
    ax.axvline(b, color='#d62728', alpha=0.6, linestyle='--')
ax.scatter(selected_bands_1idx, [importance[b-1] for b in selected_bands_1idx], color='#d62728', s=45, zorder=5, label='Top 10 Selected Bands')
ax.set_xlabel('Spectral Band Index (1 to 168)', fontsize=12, fontweight='bold')
ax.set_ylabel('Importance Score', fontsize=12, fontweight='bold')
ax.set_title('Task 2: Spectral Band Importance and Selected PCA Bands', fontsize=13, fontweight='bold')
ax.legend(frameon=True, fontsize=11)
plt.tight_layout()
plt.savefig('Results/Plots/pca_band_loadings.png')
plt.close()

# Plot 3: Representative Spectral Signatures of 4 Polymers
fig, ax = plt.subplots(figsize=(9, 5), dpi=300)
colors = {'HDPE': '#1f77b4', 'PET': '#ff7f0e', 'PP': '#2ca02c', 'PS': '#d62728'}
for cls in CLASSES:
    cls_samples = [item for item in train_records if item['class'] == cls]
    spectra_cls = []
    for s in cls_samples[:10]:
        raw = np.fromfile(s['bil_path'], dtype='<u2').reshape(s['H'], s['B'], s['W'])
        cube = np.transpose(raw, (0, 2, 1)).astype(np.float64)
        mean_spec = cube.mean(axis=(0, 1))
        spectra_cls.append(mean_spec)
    avg_spec = np.mean(spectra_cls, axis=0) / NORM_SCALE
    ax.plot(all_wavelengths, avg_spec, label=cls, color=colors[cls], linewidth=2)

for wl in selected_wavelengths:
    ax.axvline(wl, color='gray', alpha=0.3, linestyle=':')

ax.set_xlabel('Wavelength (nm)', fontsize=12, fontweight='bold')
ax.set_ylabel('Normalized Reflectance', fontsize=12, fontweight='bold')
ax.set_title('Task 1: Mean Hyperspectral Signatures (900-1700 nm) by Polymer Class', fontsize=13, fontweight='bold')
ax.legend(frameon=True, fontsize=11)
plt.tight_layout()
plt.savefig('Results/Plots/representative_spectra.png')
plt.close()

# Plot 4: Sample Raw vs Padded
fig, axes = plt.subplots(1, 2, figsize=(10, 5), dpi=300)
sample_item = train_records[0]
raw_s = np.fromfile(sample_item['bil_path'], dtype='<u2').reshape(sample_item['H'], sample_item['B'], sample_item['W'])
cube_s = np.transpose(raw_s, (0, 2, 1))
# Pseudo-RGB from 3 selected bands
r_b, g_b, b_b = selected_bands_0idx[0], selected_bands_0idx[4], selected_bands_0idx[-1]
pseudo_raw = np.stack([cube_s[:, :, r_b], cube_s[:, :, g_b], cube_s[:, :, b_b]], axis=-1).astype(np.float32)
pseudo_raw = (pseudo_raw - pseudo_raw.min()) / (pseudo_raw.max() - pseudo_raw.min() + 1e-6)

_, padded_sample = load_and_preprocess_image(sample_item['bil_path'], sample_item['H'], sample_item['W'], sample_item['B'], selected_bands_0idx, HT, WT)
pseudo_pad = np.stack([padded_sample[:, :, 0], padded_sample[:, :, 4], padded_sample[:, :, -1]], axis=-1)
pseudo_pad = (pseudo_pad - pseudo_pad.min()) / (pseudo_pad.max() - pseudo_pad.min() + 1e-6)

axes[0].imshow(pseudo_raw)
axes[0].set_title(f"Original {sample_item['class']} Fragment\n({sample_item['H']} × {sample_item['W']} × 168 bands)", fontsize=11, fontweight='bold')
axes[0].axis('off')

axes[1].imshow(pseudo_pad)
axes[1].set_title(f"Symmetrically Padded Input\n({HT} × {WT} × 10 PCA bands)", fontsize=11, fontweight='bold')
axes[1].axis('off')
plt.tight_layout()
plt.savefig('Results/Plots/sample_raw_vs_padded.png')
plt.close()

# Plot 5: Final CNN Training & Validation Curves
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), dpi=300)
epochs_range = range(1, len(final_history['train_loss']) + 1)
ax1.plot(epochs_range, final_history['train_loss'], label='Training Loss', color='#1f77b4', linewidth=2)
ax1.plot(epochs_range, final_history['val_loss'], label='Validation Loss', color='#d62728', linewidth=2, linestyle='--')
ax1.set_xlabel('Epoch', fontsize=12, fontweight='bold')
ax1.set_ylabel('Cross-Entropy Loss', fontsize=12, fontweight='bold')
ax1.set_title('Training & Validation Loss', fontsize=13, fontweight='bold')
ax1.legend(frameon=True)

ax2.plot(epochs_range, [a*100 for a in final_history['train_acc']], label='Training Accuracy', color='#1f77b4', linewidth=2)
ax2.plot(epochs_range, [a*100 for a in final_history['val_acc']], label='Validation Accuracy', color='#d62728', linewidth=2, linestyle='--')
ax2.set_xlabel('Epoch', fontsize=12, fontweight='bold')
ax2.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
ax2.set_title('Training & Validation Accuracy', fontsize=13, fontweight='bold')
ax2.legend(frameon=True)
plt.tight_layout()
plt.savefig('Results/Plots/training_curves_final_cnn.png')
plt.close()

# Plot 6: Confusion Matrix Heatmap
fig, ax = plt.subplots(figsize=(6, 5), dpi=300)
cm = final_eval['confusion_matrix']
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=CLASSES, yticklabels=CLASSES, cbar=True, ax=ax)
ax.set_xlabel('Predicted Polymer Class', fontsize=12, fontweight='bold')
ax.set_ylabel('True Polymer Class', fontsize=12, fontweight='bold')
ax.set_title(f"Task 7: Test Confusion Matrix\nOverall Accuracy: {final_eval['overall_accuracy']*100:.1f}%, Macro F1: {final_eval['macro_f1']*100:.1f}%", fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig('Results/Plots/confusion_matrix_heatmap.png')
plt.close()

# Plot 7: Class-wise Metrics (Precision, Recall, F1)
fig, ax = plt.subplots(figsize=(8, 5), dpi=300)
x = np.arange(len(CLASSES))
width = 0.25
c_metrics = final_eval['class_metrics']
precs = [c_metrics[c]['Precision'] * 100 for c in CLASSES]
recs = [c_metrics[c]['Recall'] * 100 for c in CLASSES]
f1_vals = [c_metrics[c]['F1'] * 100 for c in CLASSES]

ax.bar(x - width, precs, width, label='Precision', color='#1f77b4')
ax.bar(x, recs, width, label='Recall', color='#2ca02c')
ax.bar(x + width, f1_vals, width, label='F1-Score', color='#ff7f0e')

ax.set_xticks(x)
ax.set_xticklabels(CLASSES, fontsize=11, fontweight='bold')
ax.set_ylabel('Score (%)', fontsize=12, fontweight='bold')
ax.set_ylim(0, 110)
ax.set_title('Task 7: Class-wise Performance Metrics on Held-out Test Set', fontsize=13, fontweight='bold')
ax.legend(frameon=True, fontsize=11)
plt.tight_layout()
plt.savefig('Results/Plots/class_wise_metrics.png')
plt.close()

# Plot 8: Experiment Comparison Plots
fig, ax = plt.subplots(figsize=(10, 5), dpi=300)
exp_ids = res_df['Experiment_ID']
accs = res_df['Test_Accuracy'] * 100
mf1s = res_df['Macro_F1_Score'] * 100
x = np.arange(len(exp_ids))
width = 0.35

ax.bar(x - width/2, accs, width, label='Test Accuracy (%)', color='#1f77b4')
ax.bar(x + width/2, mf1s, width, label='Macro F1-Score (%)', color='#2ca02c')
ax.set_xticks(x)
ax.set_xticklabels(exp_ids, rotation=35, ha='right', fontsize=9, fontweight='bold')
ax.set_ylabel('Percentage (%)', fontsize=12, fontweight='bold')
ax.set_ylim(0, 115)
ax.set_title('Task 6: Controlled Experiment Comparison across Architecture, Regularization, and LR', fontsize=12, fontweight='bold')
ax.legend(frameon=True)
plt.tight_layout()
plt.savefig('Results/Plots/experiment_comparison_accuracy.png')
plt.close()

fig, ax = plt.subplots(figsize=(10, 5), dpi=300)
for exp_id in ['Exp1_Baseline', 'Exp2_Shallow', 'Exp3_Deep', 'Exp4_NoDropout', 'Exp5_HighDropout', 'Exp6_LowLR', 'Exp7_HighLR']:
    ax.plot(range(1, 36), histories[exp_id]['val_loss'], label=exp_id, linewidth=1.8)
ax.set_xlabel('Epoch', fontsize=12, fontweight='bold')
ax.set_ylabel('Validation Loss', fontsize=12, fontweight='bold')
ax.set_title('Task 6: Validation Loss Convergence across Controlled Configurations', fontsize=13, fontweight='bold')
ax.legend(frameon=True, fontsize=9)
plt.tight_layout()
plt.savefig('Results/Plots/experiment_comparison_loss.png')
plt.close()

# Plot 9: Unseen-image demonstration
unseen_sample = test_records[0] # designated test image
unseen_tensor, unseen_padded = load_and_preprocess_image(
    unseen_sample['bil_path'], unseen_sample['H'], unseen_sample['W'], unseen_sample['B'],
    selected_bands_0idx, HT, WT
)
input_t = torch.tensor(unseen_tensor, dtype=torch.float32).unsqueeze(0).to(DEVICE)
final_model.eval()
with torch.no_grad():
    scores = final_model(input_t)
    probs = torch.softmax(scores, dim=1).cpu().numpy()[0]
pred_idx = int(np.argmax(probs))
pred_class = CLASSES[pred_idx]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.5), dpi=300)
pseudo_unseen = np.stack([unseen_padded[:, :, 0], unseen_padded[:, :, 4], unseen_padded[:, :, -1]], axis=-1)
pseudo_unseen = (pseudo_unseen - pseudo_unseen.min()) / (pseudo_unseen.max() - pseudo_unseen.min() + 1e-6)
ax1.imshow(pseudo_unseen)
ax1.set_title(f"Unseen Input: {os.path.basename(unseen_sample['bil_path'])}\nTrue Class: {unseen_sample['class']}", fontsize=11, fontweight='bold')
ax1.axis('off')

bars = ax2.bar(CLASSES, probs * 100, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])
bars[pred_idx].set_edgecolor('black')
bars[pred_idx].set_linewidth(2.5)
ax2.set_ylabel('Prediction Confidence (%)', fontsize=12, fontweight='bold')
ax2.set_ylim(0, 105)
ax2.set_title(f"Task 8 Prediction: {pred_class} ({probs[pred_idx]*100:.1f}%)", fontsize=12, fontweight='bold')
for b in bars:
    yval = b.get_height()
    ax2.text(b.get_x() + b.get_width()/2.0, yval + 1.5, f'{yval:.1f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')

plt.tight_layout()
plt.savefig('Results/Plots/unseen_image_prediction.png')
plt.close()

# Save split indices for exact MATLAB replication
split_info = {
    'train_files': [r['bil_path'] for r in train_records],
    'train_labels': [r['class'] for r in train_records],
    'val_files': [r['bil_path'] for r in val_records],
    'val_labels': [r['class'] for r in val_records],
    'test_files': [r['bil_path'] for r in test_records],
    'test_labels': [r['class'] for r in test_records],
    'unseen_sample_path': unseen_sample['bil_path'],
    'unseen_sample_true_class': unseen_sample['class'],
    'unseen_sample_pred_class': pred_class,
    'unseen_sample_probs': probs.tolist()
}
with open('Model/split_info.json', 'w') as f:
    json.dump(split_info, f, indent=2)

print("\nALL EXPERIMENTS, MODELS, MAT FILES, AND PLOTS COMPLETED SUCCESSFULLY!")
