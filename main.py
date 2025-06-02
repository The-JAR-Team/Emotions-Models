# Single-file script for training a Transformer Encoder for emotion recognition
# using MediaPipe landmarks, with Hugging Face Trainer, and ONNX export.
#
# Dataset: Local FER+ (priority) or microsoft/ferplus from Hub (fallback)
# Input: 3D MediaPipe Landmarks (478 landmarks, x,y,z)

import os
import shutil
import time
import json
import glob # For finding checkpoints
import pandas as pd # For loading local CSV

import torch
import torch.nn as nn
from torch.utils.data import Dataset 

from datasets import load_dataset # For fallback or if not using local
from transformers import TrainingArguments, Trainer, PreTrainedModel, PretrainedConfig, EarlyStoppingCallback
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

import cv2
import mediapipe as mp
import numpy as np
import math
from PIL import Image 

# --- Configuration ---
# LOCAL DATA CONFIGURATION
USE_LOCAL_FERPLUS_DATA = True # SET TO True TO USE YOUR LOCAL COPY
# IMPORTANT: Update this path to where your "FERPLUS" directory is located.
# This directory should contain fer2013Plus.csv and subfolders like FER2013Train, FER2013Valid, FER2013Test
LOCAL_FERPLUS_BASE_PATH = r"./FERPLUS" # Example: "C:/Users/User/Emo_Model/Emotions-Models/FERPLUS"
                                        # Or relative: "./FERPLUS" if FERPLUS dir is in the same dir as this script

MODEL_DIR = "./emotion_transformer_ferplus_local_model_small_v1_1" # Updated for smaller model
DATASET_CACHE_DIR = "./dataset_cache_ferplus_hub_v4" 
ONNX_MODEL_PATH = os.path.join(MODEL_DIR, "emotion_transformer_small.onnx")
REPORT_PATH = os.path.join(MODEL_DIR, "training_report_small.json")

# FER+ has 8 emotion classes:
# 0:neutral, 1:happiness, 2:surprise, 3:sadness, 4:anger, 5:disgust, 6:fear, 7:contempt
NUM_CLASSES = 8
EMOTION_COLUMNS = ['neutral', 'happiness', 'surprise', 'sadness', 'anger', 'disgust', 'fear', 'contempt'] 

NUM_LANDMARKS = 478 
LANDMARK_DIM = 3 

# Model Hyperparameters (Slightly reduced for a smaller model)
D_MODEL = 192               # Was 256, now 192 (must be divisible by NHEAD)
NHEAD = 4                   # Kept at 4
NUM_ENCODER_LAYERS = 3      # Was 4, now 3
DIM_FEEDFORWARD = 192       # Was 256, now 192
DROPOUT = 0.05 # Can keep dropout, or slightly reduce if model is too small and underfits

# --- Data Augmentation Configuration for MediaPipe Landmarks ---
# Set to True to enable data augmentation on the extracted MediaPipe landmarks.
USE_LANDMARK_AUGMENTATION = True 

# Parameters for data augmentation, grouped for easier control.
# These are the default "light" augmentation values.
DATA_AUG_CONFIG = {
    "noise_std_dev": 0.005,      # Standard deviation for Gaussian noise (x, y, z)
    "scale_factor_range": 0.02,  # Range for random scaling, e.g., 0.02 for [0.98, 1.02]
    "rotation_max_degrees": 5,   # Maximum rotation angle in degrees for each axis (x, y, z)
    "translation_max_offset": 0.01 # Maximum offset for random translation (x, y, z)
}

# --- Staged Training Configuration ---
# Define multiple training stages. Each dictionary represents a stage's hyperparameters.
# The 'num_train_epochs' and 'learning_rate' are typical parameters to vary.
# Other TrainingArguments can be added here as needed for each stage.
# You can also add 'augment_config_override' to any stage to use different
# augmentation parameters for that specific stage.
TRAINING_STAGES = [
    {
        "stage_name": "Stage 1: Initial Training (Warm-up)",
        "num_train_epochs": 15,
        "per_device_train_batch_size": 64,
        "per_device_eval_batch_size": 124,
        "learning_rate": 5e-5,
        "early_stopping_patience": 5, # Allow early stop if performance plateaus quickly
        "weight_decay": 0.0001,
        "warmup_ratio": 0.1,
        "fp16": True,
        # Uses global DATA_AUG_CONFIG for light augmentation
    },
    {
        "stage_name": "Stage 2: Main Training (Standard Augmentation)",
        "num_train_epochs": 30, # More epochs
        "per_device_train_batch_size": 64,
        "per_device_eval_batch_size": 124,
        "learning_rate": 2e-5, # Lower learning rate for fine-tuning
        "early_stopping_patience": 8, # More patience than Stage 1
        "weight_decay": 0.0001,
        "warmup_ratio": 0.05, # Less warmup
        "fp16": True,
        "augment_config_override": { # Slightly increased augmentation
            "noise_std_dev": 0.008,
            "scale_factor_range": 0.03,
            "rotation_max_degrees": 7,
            "translation_max_offset": 0.015
        }
    },
    {
        "stage_name": "Stage 3: Deep Fine-tuning (Aggressive Augmentation)",
        "num_train_epochs": 10, # Fewer epochs, but with aggressive augmentation
        "per_device_train_batch_size": 64,
        "per_device_eval_batch_size": 124,
        "learning_rate": 5e-6, # Very low learning rate
        "early_stopping_patience": None, # Disable early stopping to force completion of this stage
        "weight_decay": 0.0001,
        "warmup_ratio": 0.02, # Minimal warmup
        "fp16": True,
        "augment_config_override": { # More aggressive augmentation
            "noise_std_dev": 0.01,
            "scale_factor_range": 0.03,
            "rotation_max_degrees": 10,
            "translation_max_offset": 0.02
        }
    }
]

# Common Training Hyperparameters (can be overridden by stages)
LOGGING_STRATEGY = "steps" 
LOGGING_STEPS = 100
SAVE_TOTAL_LIMIT = 3
LOAD_BEST_MODEL_AT_END = True
METRIC_FOR_BEST_MODEL = "accuracy" 
GREATER_IS_BETTER = True
DATALOADER_NUM_WORKERS = 4
DATALOADER_PIN_MEMORY = True
REPORT_TO = "tensorboard"
EARLY_STOPPING_PATIENCE = 10 # Global default for early stopping patience
START_FROM_CHECKPOINT = False # Controls initial resume, subsequent stages resume from previous

# --- MediaPipe Landmark Extraction ---
try:
    mp_face_mesh_solution = mp.solutions.face_mesh
    face_mesh_processor = mp_face_mesh_solution.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5
    )
except Exception as e:
    print(f"Error initializing MediaPipe FaceMesh: {e}")
    face_mesh_processor = None

def extract_landmarks_from_image(image_np_rgb, processor=face_mesh_processor):
    # Corrected: Use 'is None' for None comparisons in Python
    if processor is None:
        return np.zeros((NUM_LANDMARKS, LANDMARK_DIM), dtype=np.float32), False
    results = processor.process(image_np_rgb)
    if results.multi_face_landmarks:
        face_landmarks_mp = results.multi_face_landmarks[0]
        image_height, image_width, _ = image_np_rgb.shape
        landmarks_abs_3d = np.array(
            [(lm.x * image_width, lm.y * image_height, lm.z * image_width) 
             for lm in face_landmarks_mp.landmark],
            dtype=np.float32)
        if landmarks_abs_3d.shape != (NUM_LANDMARKS, 3): return np.zeros((NUM_LANDMARKS, LANDMARK_DIM), dtype=np.float32), False
        nose_tip_3d = landmarks_abs_3d[1].copy() 
        landmarks_centered_3d = landmarks_abs_3d - nose_tip_3d
        p_left_eye_inner_xy = landmarks_centered_3d[133, :2]
        p_right_eye_inner_xy = landmarks_centered_3d[362, :2]
        inter_ocular_distance = np.linalg.norm(p_left_eye_inner_xy - p_right_eye_inner_xy)
        if inter_ocular_distance < 1e-6:
            inter_ocular_distance = image_width / 4.0 
            if inter_ocular_distance < 1e-6: inter_ocular_distance = 1.0
        landmarks_normalized_3d = landmarks_centered_3d / inter_ocular_distance
        if landmarks_normalized_3d.shape != (NUM_LANDMARKS, LANDMARK_DIM): return np.zeros((NUM_LANDMARKS, LANDMARK_DIM), dtype=np.float32), False
        return landmarks_normalized_3d, True
    return np.zeros((NUM_LANDMARKS, LANDMARK_DIM), dtype=np.float32), False

# --- Data Augmentation Functions ---
# These functions now accept a config dictionary to get their parameters
def apply_random_noise(landmarks, config):
    """
    Adds Gaussian noise to landmark coordinates.
    Args:
        landmarks (np.ndarray): Normalized 3D landmarks (NUM_LANDMARKS, LANDMARK_DIM).
        config (dict): Configuration dictionary containing 'noise_std_dev'.
    Returns:
        np.ndarray: Augmented landmarks.
    """
    std_dev = config.get("noise_std_dev", 0.0) # Default to 0 if not found
    noise = np.random.normal(loc=0.0, scale=std_dev, size=landmarks.shape).astype(np.float32)
    return landmarks + noise

def apply_random_scaling(landmarks, config):
    """
    Applies a random scaling factor to landmarks.
    Args:
        landmarks (np.ndarray): Normalized 3D landmarks (NUM_LANDMARKS, LANDMARK_DIM).
        config (dict): Configuration dictionary containing 'scale_factor_range'.
    Returns:
        np.ndarray: Augmented landmarks.
    """
    scale_range = config.get("scale_factor_range", 0.0)
    scale_factor = np.random.uniform(1.0 - scale_range, 1.0 + scale_range)
    return landmarks * scale_factor

def apply_random_rotation(landmarks, config):
    """
    Applies a random rotation (Euler angles) around the origin to landmarks.
    Args:
        landmarks (np.ndarray): Normalized 3D landmarks (NUM_LANDMARKS, LANDMARK_DIM).
        config (dict): Configuration dictionary containing 'rotation_max_degrees'.
    Returns:
        np.ndarray: Augmented landmarks.
    """
    max_degrees = config.get("rotation_max_degrees", 0.0)
    # Convert degrees to radians
    max_radians = np.deg2rad(max_degrees)
    
    # Generate random rotation angles for x, y, z axes
    alpha = np.random.uniform(-max_radians, max_radians) # Rotation around X-axis
    beta = np.random.uniform(-max_radians, max_radians)  # Rotation around Y-axis
    gamma = np.random.uniform(-max_radians, max_radians) # Rotation around Z-axis

    # Rotation matrices for each axis
    Rx = np.array([[1, 0, 0],
                   [0, np.cos(alpha), -np.sin(alpha)],
                   [0, np.sin(alpha), np.cos(alpha)]], dtype=np.float32)
    
    Ry = np.array([[np.cos(beta), 0, np.sin(beta)],
                   [0, 1, 0],
                   [-np.sin(beta), 0, np.cos(beta)]], dtype=np.float32)
    
    Rz = np.array([[np.cos(gamma), -np.sin(gamma), 0],
                   [np.sin(gamma), np.cos(gamma), 0],
                   [0, 0, 1]], dtype=np.float32)
    
    # Combined rotation matrix (order matters: ZYX usually)
    R = Rz @ Ry @ Rx
    
    # Apply rotation to landmarks
    return landmarks @ R.T # Transpose R because landmarks are (N, 3) and we want (3, 3) * (3, N)

def apply_random_translation(landmarks, config):
    """
    Applies a random translation to landmarks.
    Args:
        landmarks (np.ndarray): Normalized 3D landmarks (NUM_LANDMARKS, LANDMARK_DIM).
        config (dict): Configuration dictionary containing 'translation_max_offset'.
    Returns:
        np.ndarray: Augmented landmarks.
    """
    max_offset = config.get("translation_max_offset", 0.0)
    translation_vector = np.random.uniform(-max_offset, max_offset, size=(1, LANDMARK_DIM)).astype(np.float32)
    return landmarks + translation_vector


# --- PyTorch Dataset for Local FER+ Data ---
class EmotionLandmarkLocalDataset(Dataset):
    def __init__(self, dataframe, image_base_path, split_name="", augment=False, aug_config=None):
        self.df = dataframe
        self.image_base_path = image_base_path 
        self.split_name = split_name
        self.successful_extractions = 0
        self.failed_extractions = 0
        self.to_numpy = np.array
        self.usage_to_folder = {'Training': 'FER2013Train', 'PublicTest': 'FER2013Valid', 'PrivateTest': 'FER2013Test'}
        self.augment = augment # Store augmentation flag
        self.aug_config = aug_config if aug_config is not None else {} # Store augmentation config
        if self.split_name == "train" or len(self.df) > 1000:
            print(f"INFO: Landmark extraction for local '{self.split_name}' set ({len(self.df)} samples) is on-the-fly.")
            print("      This will be slow. For large-scale training, PRE-PROCESSING LANDMARKS IS CRITICAL.")
            if self.augment:
                print("      Data augmentation is ENABLED for this dataset split.")

    def __len__(self): return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        image_name_val = row['Image name']
        if pd.isna(image_name_val): 
            self.failed_extractions += 1
            landmarks = np.zeros((NUM_LANDMARKS, LANDMARK_DIM), dtype=np.float32)
            return {"pixel_values": torch.tensor(landmarks, dtype=torch.float32), "labels": torch.tensor(0, dtype=torch.long)} 
        image_name = str(image_name_val) 

        usage_val = row['Usage']
        usage = str(usage_val) 

        label = row['derived_label']

        folder_name = self.usage_to_folder.get(usage)
        if not folder_name:
            self.failed_extractions +=1
            landmarks = np.zeros((NUM_LANDMARKS, LANDMARK_DIM), dtype=np.float32)
            return {"pixel_values": torch.tensor(landmarks, dtype=torch.float32), "labels": torch.tensor(0, dtype=torch.long)}

        image_path = os.path.join(self.image_base_path, folder_name, image_name)
        try: 
            pil_image = Image.open(image_path)
        except FileNotFoundError:
            self.failed_extractions += 1
            landmarks = np.zeros((NUM_LANDMARKS, LANDMARK_DIM), dtype=np.float32)
            return {"pixel_values": torch.tensor(landmarks, dtype=torch.float32), "labels": torch.tensor(0, dtype=torch.long)}
        
        if pil_image.mode != 'RGB': pil_image = pil_image.convert('RGB')
        image_np_rgb = self.to_numpy(pil_image)
        h, w, _ = image_np_rgb.shape
        target_size = 224 
        if h < target_size or w < target_size:
            if h < w: new_h, new_w = target_size, int(w * (target_size / h))
            else: new_w, new_h = target_size, int(h * (target_size / w))
            image_np_resized_rgb = cv2.resize(image_np_rgb, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        else: image_np_resized_rgb = image_np_rgb
        
        landmarks, success = extract_landmarks_from_image(image_np_resized_rgb)
        if success: 
            self.successful_extractions += 1
            # Apply data augmentation if enabled and it's a training split
            if self.augment and self.split_name == "train_local_ferplus":
                landmarks = apply_random_noise(landmarks, self.aug_config)
                landmarks = apply_random_scaling(landmarks, self.aug_config)
                landmarks = apply_random_rotation(landmarks, self.aug_config)
                landmarks = apply_random_translation(landmarks, self.aug_config)
        else: 
            self.failed_extractions += 1
            # If landmark extraction fails, return zeros, no augmentation
            landmarks = np.zeros((NUM_LANDMARKS, LANDMARK_DIM), dtype=np.float32)
        
        return {"pixel_values": torch.tensor(landmarks, dtype=torch.float32), "labels": torch.tensor(label, dtype=torch.long)}

    def print_extraction_stats(self):
        total_processed = self.successful_extractions + self.failed_extractions
        if total_processed == 0: print(f"Landmark extraction stats for '{self.split_name}': No items processed."); return
        success_rate = (self.successful_extractions / total_processed * 100) if total_processed > 0 else 0
        print(f"Landmark extraction stats for '{self.split_name}': Successfully extracted: {self.successful_extractions}/{total_processed} ({success_rate:.2f}%)")

# --- PyTorch Dataset for Hugging Face datasets (Fallback) ---
class EmotionLandmarkHFDataset(Dataset):
    def __init__(self, hf_dataset_split, split_name="", augment=False, aug_config=None):
        self.hf_dataset_split = hf_dataset_split
        self.split_name = split_name
        self.successful_extractions = 0
        self.failed_extractions = 0
        self.to_numpy = np.array
        self.augment = augment # Store augmentation flag
        self.aug_config = aug_config if aug_config is not None else {} # Store augmentation config
        if self.split_name == "train" or len(self.hf_dataset_split) > 1000:
            print(f"INFO (HF): Landmark extraction for '{self.split_name}' set ({len(self.hf_dataset_split)} samples) is on-the-fly.")
            print("      This will be slow. PRE-PROCESSING LANDMARKS IS CRITICAL for large datasets.")
            if self.augment:
                print("      Data augmentation is ENABLED for this dataset split.")

    def __len__(self): return len(self.hf_dataset_split)

    def __getitem__(self, idx):
        item = self.hf_dataset_split[idx]
        pil_image, label = item['image'], item['label'] 
        if pil_image.mode != 'RGB': pil_image = pil_image.convert('RGB')
        image_np_rgb = self.to_numpy(pil_image)
        h, w, _ = image_np_rgb.shape
        target_size = 224 
        if h < target_size or w < target_size:
            if h < w: new_h, new_w = target_size, int(w * (target_size / h))
            else: new_w, new_h = target_size, int(h * (target_size / w))
            image_np_resized_rgb = cv2.resize(image_np_rgb, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        else: image_np_resized_rgb = image_np_rgb
        
        landmarks, success = extract_landmarks_from_image(image_np_resized_rgb)
        if success: 
            self.successful_extractions += 1
            # Apply data augmentation if enabled and it's a training split
            if self.augment and self.split_name == "train": # 'train' is the split name for HF dataset
                landmarks = apply_random_noise(landmarks, self.aug_config)
                landmarks = apply_random_scaling(landmarks, self.aug_config)
                landmarks = apply_random_rotation(landmarks, self.aug_config)
                landmarks = apply_random_translation(landmarks, self.aug_config)
        else: 
            self.failed_extractions += 1
            # If landmark extraction fails, return zeros, no augmentation
            landmarks = np.zeros((NUM_LANDMARKS, LANDMARK_DIM), dtype=np.float32)

        return {"pixel_values": torch.tensor(landmarks, dtype=torch.float32), "labels": torch.tensor(label, dtype=torch.long)}

    def print_extraction_stats(self):
        total_processed = self.successful_extractions + self.failed_extractions
        if total_processed == 0: print(f"Landmark extraction stats for '{self.split_name}': No items processed."); return
        success_rate = (self.successful_extractions / total_processed * 100) if total_processed > 0 else 0
        print(f"Landmark extraction stats for '{self.split_name}': Successfully extracted: {self.successful_extractions}/{total_processed} ({success_rate:.2f}%)")

# --- Transformer Model ---
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=DROPOUT, max_len=NUM_LANDMARKS + 50): # Use global DROPOUT
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout) # Use the global DROPOUT from config
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
    def forward(self, x): x = x + self.pe[:, :x.size(1), :]; return self.dropout(x)

class EmotionTransformerConfig(PretrainedConfig):
    model_type = "emotion_transformer"
    def __init__(self, num_landmarks=NUM_LANDMARKS, landmark_dim=LANDMARK_DIM, d_model=D_MODEL,
                 nhead=NHEAD, num_encoder_layers=NUM_ENCODER_LAYERS, dim_feedforward=DIM_FEEDFORWARD,
                 dropout=DROPOUT, num_classes=NUM_CLASSES, **kwargs):
        self.num_landmarks, self.landmark_dim, self.d_model, self.nhead = num_landmarks, landmark_dim, d_model, nhead
        self.num_encoder_layers, self.dim_feedforward, self.dropout, self.num_classes = num_classes, dim_feedforward, dropout, num_classes
        super().__init__(**kwargs)

class EmotionTransformerModel(PreTrainedModel):
    config_class = EmotionTransformerConfig
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.input_embedding = nn.Linear(config.landmark_dim, config.d_model)
        self.pos_encoder = PositionalEncoding(config.d_model, config.dropout, max_len=config.num_landmarks)
        encoder_layer = nn.TransformerEncoderLayer(d_model=config.d_model, nhead=config.nhead,
                                                   dim_feedforward=config.dim_feedforward, dropout=config.dropout, batch_first=True,
                                                   activation=nn.GELU())
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=config.num_encoder_layers)
        self.fc_out = nn.Linear(config.d_model, config.num_classes)
    def forward(self, pixel_values, labels=None): 
        x = self.input_embedding(pixel_values) * math.sqrt(self.config.d_model)
        x = self.pos_encoder(x)
        transformer_output = self.transformer_encoder(x)
        embedding = transformer_output.mean(dim=1) 
        logits = self.fc_out(embedding)
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.config.num_classes), labels.view(-1))
        return {"loss": loss, "logits": logits, "embedding": embedding}

# --- Metrics Computation ---
def compute_metrics_fn(pred):
    labels = pred.label_ids
    if isinstance(pred.predictions, tuple): logits = pred.predictions[0]
    else: logits = pred.predictions 
    preds = np.argmax(logits, axis=-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='weighted', zero_division=0)
    acc = accuracy_score(labels, preds)
    return {'accuracy': acc, 'f1': f1, 'precision': precision, 'recall': recall}

# --- Usage Instructions & Checkpoint Functions ---
def print_usage_instructions(model_dir, onnx_path, num_lm, lm_dim):
    print("\n--- How to use the saved models ---")
    print(f"PyTorch model (and config) can be loaded using:")
    print(f"  # from your_script_name import EmotionTransformerModel, EmotionTransformerConfig")
    print(f"  # config = EmotionTransformerConfig.from_pretrained('{model_dir}')")
    print(f"  # model = EmotionTransformerModel.from_pretrained('{model_dir}', config=config)")
    print(f"\nONNX model can be loaded using onnxruntime:")
    print(f"  import onnxruntime\n  import numpy as np")
    print(f"  ort_session = onnxruntime.InferenceSession('{onnx_path}')")
    print(f"  # landmarks_input = np.random.randn(1, {num_lm}, {lm_dim}).astype(np.float32)")
    print(f"  # ort_inputs = {{ort_session.get_inputs()[0].name: landmarks_input}}")
    print(f"  # ort_outs = ort_session.run(None, ort_inputs)\n  # logits = ort_outs[0]\n  # embedding = ort_outs[1]")
    print("\n--- To get embeddings from the PyTorch model ---")
    print("  # model.eval()\n  # with torch.no_grad():")
    print("  #     # Assuming 'landmarks_tensor' is your input: (batch_size, NUM_LANDMARKS, LANDMARK_DIM)")
    print("  #     outputs = model(pixel_values=landmarks_tensor)")
    print("  #     embedding = outputs['embedding']\n  #     logits = outputs['logits']")

def get_latest_checkpoint(checkpoint_dir_base):
    checkpoint_dir = os.path.join(checkpoint_dir_base, "training_checkpoints")
    if not os.path.isdir(checkpoint_dir): print(f"Checkpoint directory not found: {checkpoint_dir}"); return None
    checkpoints = glob.glob(os.path.join(checkpoint_dir, "checkpoint-*"))
    if not checkpoints: print(f"No checkpoints found in {checkpoint_dir}"); return None
    latest_checkpoint = max(checkpoints, key=lambda p: int(p.split('-')[-1]))
    print(f"Found latest checkpoint: {latest_checkpoint}")
    return latest_checkpoint

# --- Main Script ---
if __name__ == "__main__":
    print("Starting Emotion Recognition Training...")
    print(f"Device: {'cuda' if torch.cuda.is_available() else 'cpu'}")
    if face_mesh_processor is None: print("CRITICAL ERROR: MediaPipe FaceMesh not initialized. Exiting."); exit()
    
    train_emotion_dataset, eval_emotion_dataset, test_emotion_dataset = None, None, None
    dataset_source_name = ""
    df_train, df_val, df_test = None, None, None # Initialize these for broader scope
    train_hf_data, eval_hf_data, test_hf_data = None, None, None # Initialize these for broader scope

    if USE_LOCAL_FERPLUS_DATA:
        print(f"\n--- Attempting to load LOCAL FER+ dataset from: {LOCAL_FERPLUS_BASE_PATH} ---")
        local_csv_path = os.path.join(LOCAL_FERPLUS_BASE_PATH, "fer2013Plus.csv")
        if os.path.exists(local_csv_path) and os.path.exists(LOCAL_FERPLUS_BASE_PATH):
            try:
                df = pd.read_csv(local_csv_path)
                for col in EMOTION_COLUMNS:
                    if col not in df.columns: raise ValueError(f"Required emotion column '{col}' not found in {local_csv_path}")
                    df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
                df['derived_label'] = df[EMOTION_COLUMNS].idxmax(axis=1).map({emotion: i for i, emotion in enumerate(EMOTION_COLUMNS)})
                df_train = df[df['Usage'] == 'Training'].reset_index(drop=True)
                df_val = df[df['Usage'] == 'PublicTest'].reset_index(drop=True)
                df_test = df[df['Usage'] == 'PrivateTest'].reset_index(drop=True)
                if not (df_train.empty or df_val.empty or df_test.empty):
                    # Initial dataset creation (will be re-created per stage for augmentation)
                    train_emotion_dataset = EmotionLandmarkLocalDataset(df_train, LOCAL_FERPLUS_BASE_PATH, "train_local_ferplus", augment=USE_LANDMARK_AUGMENTATION, aug_config=DATA_AUG_CONFIG)
                    eval_emotion_dataset = EmotionLandmarkLocalDataset(df_val, LOCAL_FERPLUS_BASE_PATH, "eval_local_ferplus", augment=False) # No augmentation for evaluation
                    test_emotion_dataset = EmotionLandmarkLocalDataset(df_test, LOCAL_FERPLUS_BASE_PATH, "test_local_ferplus", augment=False) # No augmentation for testing
                    dataset_source_name = f"Local FER+ ({LOCAL_FERPLUS_BASE_PATH})"
                    print(f"Successfully prepared local FER+ splits: Train={len(df_train)}, Val={len(df_val)}, Test={len(df_test)}")
                else:
                    print("WARNING: Could not create all splits from local FER+ CSV. 'Usage' column might be missing or have unexpected values.")
                    USE_LOCAL_FERPLUS_DATA = False 
            except Exception as e:
                print(f"ERROR loading local FER+ data: {e}. Will attempt Hugging Face Hub fallback.")
                USE_LOCAL_FERPLUS_DATA = False 
        else:
            print(f"WARNING: Local FER+ path or CSV not found: {local_csv_path}. Will attempt Hugging Face Hub fallback.")
            USE_LOCAL_FERPLUS_DATA = False 

    if not USE_LOCAL_FERPLUS_DATA or train_emotion_dataset is None:
        print("\n--- Loading Dataset from Hugging Face Hub (microsoft/ferplus) ---")
        try:
            ferplus_raw = load_dataset("microsoft/ferplus", cache_dir=DATASET_CACHE_DIR)
            train_hf_data, eval_hf_data, test_hf_data = ferplus_raw['train'], ferplus_raw['validation'], ferplus_raw['test']
            # Initial dataset creation (will be re-created per stage for augmentation)
            train_emotion_dataset = EmotionLandmarkHFDataset(train_hf_data, "train", augment=USE_LANDMARK_AUGMENTATION, aug_config=DATA_AUG_CONFIG)
            eval_emotion_dataset = EmotionLandmarkHFDataset(eval_hf_data, "validation", augment=False) # No augmentation for evaluation
            test_emotion_dataset = EmotionLandmarkHFDataset(test_hf_data, "test", augment=False) # No augmentation for testing
            dataset_source_name = "microsoft/ferplus (Hugging Face Hub)"
            print("Successfully loaded FER+ from Hugging Face Hub.")
        except Exception as e:
            print(f"CRITICAL ERROR: Failed to load FER+ dataset from local path AND Hugging Face Hub: {e}"); exit()
    
    print(f"Using Train dataset size: {len(train_emotion_dataset)}")
    print(f"Using Validation dataset size: {len(eval_emotion_dataset)}")
    print(f"Using Test dataset size: {len(test_emotion_dataset)}")

    print("\n--- Initializing Model ---")
    config = EmotionTransformerConfig(num_classes=NUM_CLASSES, 
                                      d_model=D_MODEL, nhead=NHEAD, 
                                      num_encoder_layers=NUM_ENCODER_LAYERS, 
                                      dim_feedforward=DIM_FEEDFORWARD, dropout=DROPOUT) # Pass all model hyperparams
    model = EmotionTransformerModel(config)
    print(f"Model initialized with {sum(p.numel() for p in model.parameters() if p.requires_grad):,} trainable parameters.")

    # --- Staged Training Loop ---
    current_checkpoint = None
    for i, stage_config in enumerate(TRAINING_STAGES):
        print(f"\n--- Starting {stage_config['stage_name']} (Stage {i+1}/{len(TRAINING_STAGES)}) ---")

        # Determine the data augmentation config for the current stage
        # If 'augment_config_override' is present, merge it with the global config.
        # This allows for partial overrides.
        if "augment_config_override" in stage_config:
            print(f"Applying stage-specific data augmentation parameters for {stage_config['stage_name']}.")
            # Create a new dictionary by copying global and then updating with stage-specific overrides
            current_aug_config = {**DATA_AUG_CONFIG, **stage_config["augment_config_override"]}
        else:
            current_aug_config = DATA_AUG_CONFIG # Use the global default

        # Re-initialize dataset with the current stage's augmentation config
        # This is crucial because augmentation parameters are part of the dataset's __init__
        if USE_LOCAL_FERPLUS_DATA:
            train_emotion_dataset = EmotionLandmarkLocalDataset(df_train, LOCAL_FERPLUS_BASE_PATH, "train_local_ferplus", augment=USE_LANDMARK_AUGMENTATION, aug_config=current_aug_config)
        else:
            train_emotion_dataset = EmotionLandmarkHFDataset(train_hf_data, "train", augment=USE_LANDMARK_AUGMENTATION, aug_config=current_aug_config)
        
        # Update training arguments for the current stage
        # Extract early_stopping_patience separately as it's for the callback, not TrainingArguments
        # Use .get() with a default to handle cases where it might not be explicitly set in stage_config
        current_early_stopping_patience = stage_config.get("early_stopping_patience", EARLY_STOPPING_PATIENCE)

        # Explicitly define parameters for TrainingArguments, excluding 'early_stopping_patience'
        current_training_args_params = {
            "num_train_epochs": stage_config.get("num_train_epochs"),
            "per_device_train_batch_size": stage_config.get("per_device_train_batch_size"),
            "per_device_eval_batch_size": stage_config.get("per_device_eval_batch_size"),
            "learning_rate": stage_config.get("learning_rate"),
            "warmup_ratio": stage_config.get("warmup_ratio"),
            "weight_decay": stage_config.get("weight_decay"),
            "fp16": stage_config.get("fp16") and torch.cuda.is_available(),
        }

        # Re-calculate steps per epoch in case batch size changed
        steps_per_epoch = math.ceil(len(train_emotion_dataset) / current_training_args_params["per_device_train_batch_size"])
        effective_logging_steps = LOGGING_STEPS
        if LOGGING_STRATEGY == "epoch":
            effective_logging_steps = steps_per_epoch
        elif LOGGING_STRATEGY == "steps":
            effective_logging_steps = LOGGING_STEPS 

        training_args = TrainingArguments(
            output_dir=os.path.join(MODEL_DIR, "training_checkpoints"),
            logging_strategy=LOGGING_STRATEGY,
            logging_steps=effective_logging_steps,
            eval_strategy="epoch", 
            save_strategy="epoch",       
            save_total_limit=SAVE_TOTAL_LIMIT,
            load_best_model_at_end=LOAD_BEST_MODEL_AT_END, 
            metric_for_best_model=METRIC_FOR_BEST_MODEL, 
            greater_is_better=GREATER_IS_BETTER,
            report_to=REPORT_TO,
            dataloader_num_workers=DATALOADER_NUM_WORKERS, 
            dataloader_pin_memory=DATALOADER_PIN_MEMORY,
            remove_unused_columns=False,
            **current_training_args_params # Unpack stage-specific parameters
        )
        
        # Re-initialize Trainer for each stage to pick up new training_args and potentially new dataset
        # Only add EarlyStoppingCallback if patience is not None
        callbacks = []
        if current_early_stopping_patience is not None:
            callbacks.append(EarlyStoppingCallback(early_stopping_patience=current_early_stopping_patience))

        trainer = Trainer(model=model, args=training_args, train_dataset=train_emotion_dataset,
                          eval_dataset=eval_emotion_dataset, compute_metrics=compute_metrics_fn,
                          callbacks=callbacks) # Pass the list of callbacks

        start_time = time.time()
        
        # Determine checkpoint for current stage
        resume_checkpoint_path = None
        if i == 0 and START_FROM_CHECKPOINT:
            # Only try to load an external checkpoint for the very first stage
            # This is for resuming a multi-stage run that was previously stopped
            # or starting from a pre-trained model not from this script's previous stages.
            resume_checkpoint_path = get_latest_checkpoint(MODEL_DIR)
            if resume_checkpoint_path:
                print(f"Attempting to resume Stage 1 from external checkpoint: {resume_checkpoint_path}")
            else:
                print("No valid external checkpoint found for Stage 1. Starting from scratch.")
        # For subsequent stages (i > 0), the 'model' object is already updated by the previous trainer.
        # No need to load from disk again for intermediate stages.
        # The 'model' variable itself carries the state from the previous stage.


        try:
            trainer.train(resume_from_checkpoint=resume_checkpoint_path)
        except Exception as e:
            print(f"An error occurred during training in {stage_config['stage_name']}: {e}")
            print("This often happens if you changed model hyperparameters (D_MODEL, NHEAD, etc.)")
            print(f"and are trying to resume from a checkpoint saved by a model with a different architecture.")
            print(f"Please delete the '{MODEL_DIR}' directory and try running the script again if this is the case.")
            exit()
        
        training_time = time.time() - start_time
        print(f"{stage_config['stage_name']} finished in {training_time / 60:.2f} minutes.")

        if hasattr(train_emotion_dataset, 'print_extraction_stats'): train_emotion_dataset.print_extraction_stats()
        if hasattr(eval_emotion_dataset, 'print_extraction_stats'): eval_emotion_dataset.print_extraction_stats()

        # Removed the 'break' condition here. Training will now proceed to the next stage
        # even if early stopping was triggered in the current stage.
        if trainer.control.should_training_stop:
            print(f"Note: Early stopping was triggered in {stage_config['stage_name']}, but continuing to next stage as per configuration.")


        # After each stage, save the model (Trainer does this automatically if save_strategy="epoch")
        # And ensure the model instance is updated to the best one if LOAD_BEST_MODEL_AT_END is True
        # The trainer.model will already be the best model if load_best_model_at_end is True.
        # For next stage, we just use the current model instance.
        model = trainer.model # Ensure the model object is the one from the trainer (potentially best checkpoint)
        print(f"Model state updated after {stage_config['stage_name']}.")

    print("\n--- All training stages completed. ---")

    print("\n--- Evaluating on Test Set ---")
    test_results = trainer.predict(test_emotion_dataset)
    if hasattr(test_emotion_dataset, 'print_extraction_stats'): test_emotion_dataset.print_extraction_stats()
    
    print("\nTest Set Metrics:")
    final_test_metrics = test_results.metrics
    for key, value in (final_test_metrics or {}).items(): print(f"  {key}: {value:.4f}")

    print("\n--- Saving Models and Report ---")
    trainer.save_model(MODEL_DIR) 
    print(f"PyTorch model and config saved to {MODEL_DIR}")

    model_for_onnx = trainer.model.to("cpu"); model_for_onnx.eval()
    dummy_input_landmarks = torch.randn(1, NUM_LANDMARKS, LANDMARK_DIM, device="cpu")
    try:
        print(f"Exporting model to ONNX at {ONNX_MODEL_PATH}...")
        dynamic_axes = {'pixel_values': {0: 'batch_size'}, 'logits': {0: 'batch_size'}, 'embedding': {0: 'batch_size'}}
        class OnnxWrapper(nn.Module):
            def __init__(self, model_to_wrap): super().__init__(); self.model_to_wrap = model_to_wrap
            def forward(self, pixel_values): outputs = self.model_to_wrap(pixel_values=pixel_values); return outputs['logits'], outputs['embedding']
        onnx_exportable_model = OnnxWrapper(model_for_onnx); onnx_exportable_model.eval()
        torch.onnx.export(onnx_exportable_model, dummy_input_landmarks, ONNX_MODEL_PATH,
                          input_names=['pixel_values'], output_names=['logits', 'embedding'],
                          dynamic_axes=dynamic_axes, opset_version=14, export_params=True)
        print("ONNX model exported successfully.")
        try:
            import onnxruntime
            ort_session = onnxruntime.InferenceSession(ONNX_MODEL_PATH, providers=['CPUExecutionProvider'])
            ort_inputs = {ort_session.get_inputs()[0].name: dummy_input_landmarks.cpu().numpy()}
            ort_outs = ort_session.run(None, ort_inputs)
            print(f"ONNX model loaded and verified. Logits shape: {ort_outs[0].shape}, Embedding shape: {ort_outs[1].shape}")
        except ImportError: print("onnxruntime not installed. Skipping ONNX verification.")
        except Exception as e: print(f"Error during ONNX verification: {e}")
    except Exception as e: print(f"Error exporting to ONNX: {e}")

    report = {
        "dataset_used": dataset_source_name,
        "model_configuration": model_for_onnx.config.to_dict(),
        "training_arguments": {k: str(v) if not isinstance(v, (int, float, str, bool, list, dict, type(None))) else v 
                               for k, v in training_args.to_dict().items()},
        "training_time_minutes": training_time / 60,
        "test_set_metrics": final_test_metrics or {},
        "landmark_extraction_stats": {}
    }
    for ds_obj, name in [(train_emotion_dataset, "train"), (eval_emotion_dataset, "validation"), (test_emotion_dataset, "test")]:
        if hasattr(ds_obj, 'successful_extractions') and hasattr(ds_obj, 'failed_extractions'):
            total = ds_obj.successful_extractions + ds_obj.failed_extractions
            report["landmark_extraction_stats"][name] = {
                "successful": ds_obj.successful_extractions, "failed": ds_obj.failed_extractions,
                "total": total, "success_rate_percent": (ds_obj.successful_extractions / total * 100) if total > 0 else 0
            }
    with open(REPORT_PATH, 'w') as f: json.dump(report, f, indent=4)
    print(f"Training report saved to {REPORT_PATH}")
    print_usage_instructions(MODEL_DIR, ONNX_MODEL_PATH, NUM_LANDMARKS, LANDMARK_DIM)
    print("\n--- Script Finished ---")    
    if face_mesh_processor: face_mesh_processor.close()
