# Single-file script for training a Transformer Encoder for emotion recognition
# using MediaPipe landmarks, with Hugging Face Trainer, and ONNX export.
# This script is modified to resume training from the latest checkpoint.

import os
import shutil
import time
import json
import glob # For finding files
import pandas as pd

# Kaggle dataset loading
import kagglehub
from kagglehub import KaggleDatasetAdapter

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms # Not used directly for landmarks, but often for image datasets

from datasets import load_dataset, concatenate_datasets, Image as HFImage # HFImage for casting
from transformers import TrainingArguments, Trainer, PreTrainedModel, PretrainedConfig
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import train_test_split # For custom dataset splits

import cv2
import mediapipe as mp
import numpy as np
import math
from PIL import Image # For handling images from datasets

# --- Configuration ---
MODEL_DIR = "./emotion_transformer_affectnet_model"
DATASET_CACHE_DIR = "./dataset_cache_affectnet"
ONNX_MODEL_PATH = os.path.join(MODEL_DIR, "emotion_transformer.onnx")
# PYTORCH_MODEL_PATH = os.path.join(MODEL_DIR, "emotion_transformer.pth") # Trainer saves full model
REPORT_PATH = os.path.join(MODEL_DIR, "training_report.json")

# AffectNet has 8 emotion classes:
# 0: neutral, 1: happiness, 2: surprise, 3: sadness, 4: anger, 5: disgust, 6: fear, 7: contempt
NUM_CLASSES = 8

NUM_LANDMARKS = 478 # Using refine_landmarks=True for FaceMesh
LANDMARK_DIM = 2 # Using (x, y) coordinates, normalized

# Model Hyperparameters (can be tuned)
D_MODEL = 128       # Embedding dimension for Transformer
NHEAD = 4           # Number of attention heads
NUM_ENCODER_LAYERS = 3 # Number of Transformer encoder layers
DIM_FEEDFORWARD = 256 # Dimension of feedforward network in Transformer
DROPOUT = 0.1

# Training Hyperparameters
BATCH_SIZE = 64 # Increased slightly, adjust based on GPU memory
LEARNING_RATE = 1e-4
NUM_EPOCHS = 10 # Increased for a bit more training, still relatively short for full convergence
WEIGHT_DECAY = 0.01
DATALOADER_NUM_WORKERS = 0 # As requested

# --- MediaPipe Landmark Extraction ---
# Initialize MediaPipe FaceMesh solution
# It's better to initialize it once globally if possible,
# or pass it around if state needs to be managed carefully.
# For simplicity in a single file, a global instance is often used.
try:
    mp_face_mesh_solution = mp.solutions.face_mesh
    face_mesh_processor = mp_face_mesh_solution.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=True, # Enables iris landmarks, total 478
        min_detection_confidence=0.5
    )
except Exception as e:
    print(f"Error initializing MediaPipe FaceMesh: {e}")
    print("Please ensure MediaPipe is installed correctly.")
    face_mesh_processor = None


def extract_landmarks_from_image(image_np_rgb, processor=face_mesh_processor):
    """
    Extracts 478 MediaPipe face landmarks from a single RGB image.
    Normalizes landmarks by centering on nose and scaling by inter-ocular distance (more robust).
    """
    if processor is None:
        return np.zeros((NUM_LANDMARKS, LANDMARK_DIM), dtype=np.float32), False

    results = processor.process(image_np_rgb)
    
    if results.multi_face_landmarks:
        face_landmarks_mp = results.multi_face_landmarks[0]
        
        landmarks_px = np.array(
            [(lm.x * image_np_rgb.shape[1], lm.y * image_np_rgb.shape[0]) for lm in face_landmarks_mp.landmark],
            dtype=np.float32
        )

        if landmarks_px.shape[0] != NUM_LANDMARKS: # Should not happen with refine_landmarks=True
            # print(f"Warning: Expected {NUM_LANDMARKS} landmarks, got {landmarks_px.shape[0]}. Skipping.")
            return np.zeros((NUM_LANDMARKS, LANDMARK_DIM), dtype=np.float32), False

        # Normalization:
        # 1. Center around a stable point (e.g., nose tip - landmark 1)
        nose_tip = landmarks_px[1].copy()
        landmarks_normalized = landmarks_px - nose_tip
        
        # 2. Scale by a robust distance, e.g., inter-ocular distance
        # Left eye: landmark 473 (outer corner), Right eye: landmark 468 (outer corner)
        # These are iris landmarks. For eye corners, better use e.g., 33 (left) and 263 (right) for outer eye corners
        # Or for pupils: 468 (right pupil), 473 (left pupil) when refine_landmarks=True
        # Let's use outer eye corners for scaling (more stable than mouth for all expressions)
        # Check MediaPipe landmark map: left eye outer: 33, right eye outer: 263
        # Or if using refined landmarks: left eye outer contour: 130, right eye outer contour: 359
        # Let's use specific points that should be consistently available.
        # Example: Left eye (landmark indices around 33, 133), Right eye (landmark indices around 263, 362)
        # For simplicity and consistency:
        # Left eye corner (approx): landmarks_normalized[33]
        # Right eye corner (approx): landmarks_normalized[263]
        # This assumes these indices are valid within the 478 set.
        # A quick check on mediapipe viz:
        # Left eye outer corner: 33
        # Right eye outer corner: 263
        # Left eye inner corner: 133
        # Right eye inner corner: 362
        # Let's use distance between 133 and 362 (inner eye corners)
        p_left_eye_inner = landmarks_normalized[133]
        p_right_eye_inner = landmarks_normalized[362]
        
        inter_ocular_distance = np.linalg.norm(p_left_eye_inner - p_right_eye_inner)
        
        if inter_ocular_distance < 1e-6: # Avoid division by zero
            # print("Warning: Inter-ocular distance is near zero. Using default scale.")
            inter_ocular_distance = image_np_rgb.shape[1] / 4.0 # Fallback scale
            if inter_ocular_distance < 1e-6: inter_ocular_distance = 1.0

        landmarks_normalized /= inter_ocular_distance
        
        return landmarks_normalized[:, :LANDMARK_DIM], True # Return (x,y) and success
    return np.zeros((NUM_LANDMARKS, LANDMARK_DIM), dtype=np.float32), False

# --- PyTorch Dataset ---
class EmotionLandmarkHFDataset(Dataset):
    def __init__(self, hf_dataset_split, split_name=""):
        self.hf_dataset_split = hf_dataset_split
        self.split_name = split_name
        self.successful_extractions = 0
        self.failed_extractions = 0
        
        # PIL to NumPy transform
        self.to_numpy = np.array

        if self.split_name == "train" or len(self.hf_dataset_split) > 1000: # Show warning for large sets
            print(f"INFO: Landmark extraction for the '{self.split_name}' set ({len(self.hf_dataset_split)} samples) is on-the-fly.")
            print("      This will be slow. For AffectNet or large-scale training, PRE-PROCESSING LANDMARKS IS CRITICAL.")

    def __len__(self):
        return len(self.hf_dataset_split)

    def __getitem__(self, idx):
        item = self.hf_dataset_split[idx]
        pil_image = item['image']
        label = item['label']

        # Convert PIL Image to NumPy array (RGB)
        if pil_image.mode != 'RGB':
            pil_image = pil_image.convert('RGB')
        image_np_rgb = self.to_numpy(pil_image)
        
        # Resize small images (like FER+ 48x48) for potentially better landmark detection
        # This is a heuristic. Optimal size might vary.
        h, w, _ = image_np_rgb.shape
        target_size = 224 # A common size for face processing
        if h < target_size or w < target_size:
            if h < w:
                new_h = target_size
                new_w = int(w * (target_size / h))
            else:
                new_w = target_size
                new_h = int(h * (target_size / w))
            image_np_resized_rgb = cv2.resize(image_np_rgb, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        else:
            image_np_resized_rgb = image_np_rgb

        landmarks, success = extract_landmarks_from_image(image_np_resized_rgb)

        if success:
            self.successful_extractions += 1
        else:
            self.failed_extractions += 1
            # print(f"Warning: Failed to extract landmarks for item {idx} in {self.split_name} set.")

        return {"pixel_values": torch.tensor(landmarks, dtype=torch.float32), "labels": torch.tensor(label, dtype=torch.long)}

    def print_extraction_stats(self):
        total_processed = self.successful_extractions + self.failed_extractions
        if total_processed == 0:
            print(f"Landmark extraction stats for '{self.split_name}': No items processed by this instance yet.")
            return
            
        success_rate = (self.successful_extractions / total_processed * 100) if total_processed > 0 else 0
        print(f"Landmark extraction stats for '{self.split_name}':")
        print(f"  Successfully extracted: {self.successful_extractions}/{total_processed} ({success_rate:.2f}%)")


# --- AffectNet Kaggle Dataset ---
class AffectNetKaggleDataset(Dataset):
    def __init__(self, dataframe, split_name="affectnet_kaggle"):
        self.df = dataframe
        self.split_name = split_name
        self.successful_extractions = 0
        self.failed_extractions = 0
        self.to_numpy = np.array
        
        print(f"INFO: Initializing AffectNetKaggleDataset for '{split_name}' with {len(self.df)} samples.")
        if self.split_name == "train" or len(self.df) > 1000:
            print(f"      Landmark extraction for the '{self.split_name}' set ({len(self.df)} samples) is on-the-fly.")
            print("      This will be slow. For large-scale training, PRE-PROCESSING LANDMARKS IS CRITICAL.")

    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        # Get label first
        if 'mapped_emotion' in row:
            label = int(row['mapped_emotion'])
        elif 'label' in row:
            label = int(row['label'])
        elif 'emotion' in row:
            label = int(row['emotion'])
        elif 'expression' in row:
            label = int(row['expression'])
        else:
            label = 0  # Default neutral if no label found
        
        # Load image
        if 'image' in row:
            pil_image = row['image']
        elif 'Image' in row:
            pil_image = row['Image']
        elif 'image_path' in row:
            # Load image from file path
            img_path = row['image_path']
            try:
                if os.path.exists(img_path):
                    pil_image = Image.open(img_path)
                else:
                    # Try alternative path construction
                    print(f"Warning: Image not found at {img_path}")
                    pil_image = Image.new('RGB', (224, 224), color=(128, 128, 128))
            except Exception as e:
                print(f"Error loading image {img_path}: {e}")
                pil_image = Image.new('RGB', (224, 224), color=(128, 128, 128))
        else:
            # If image path is provided instead of image data
            img_path = row.get('pth', '')
            try:
                if img_path and os.path.exists(img_path):
                    pil_image = Image.open(img_path)
                else:
                    pil_image = Image.new('RGB', (224, 224), color=(128, 128, 128))
            except:
                # Create dummy image if path doesn't work
                pil_image = Image.new('RGB', (224, 224), color=(128, 128, 128))

        # Convert PIL Image to NumPy array (RGB)
        if pil_image.mode != 'RGB':
            pil_image = pil_image.convert('RGB')
        image_np_rgb = self.to_numpy(pil_image)
        
        # Resize for better landmark detection
        h, w, _ = image_np_rgb.shape
        target_size = 224
        if h < target_size or w < target_size:
            if h < w:
                new_h = target_size
                new_w = int(w * (target_size / h))
            else:
                new_w = target_size
                new_h = int(h * (target_size / w))
            image_np_resized_rgb = cv2.resize(image_np_rgb, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        else:
            image_np_resized_rgb = image_np_rgb

        landmarks, success = extract_landmarks_from_image(image_np_resized_rgb)

        if success:
            self.successful_extractions += 1
        else:
            self.failed_extractions += 1

        return {"pixel_values": torch.tensor(landmarks, dtype=torch.float32), "labels": torch.tensor(label, dtype=torch.long)}

    def print_extraction_stats(self):
        total_processed = self.successful_extractions + self.failed_extractions
        if total_processed == 0:
            print(f"Landmark extraction stats for '{self.split_name}': No items processed by this instance yet.")
            return
            
        success_rate = (self.successful_extractions / total_processed * 100) if total_processed > 0 else 0
        print(f"Landmark extraction stats for '{self.split_name}':")
        print(f"  Successfully extracted: {self.successful_extractions}/{total_processed} ({success_rate:.2f}%)")

# --- Transformer Model (Identical to previous version) ---
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=NUM_LANDMARKS + 50): # Ensure max_len is sufficient
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)

class EmotionTransformerConfig(PretrainedConfig):
    model_type = "emotion_transformer"
    def __init__(self, num_landmarks=NUM_LANDMARKS, landmark_dim=LANDMARK_DIM, d_model=D_MODEL,
                 nhead=NHEAD, num_encoder_layers=NUM_ENCODER_LAYERS, dim_feedforward=DIM_FEEDFORWARD,
                 dropout=DROPOUT, num_classes=NUM_CLASSES, **kwargs):
        self.num_landmarks, self.landmark_dim, self.d_model, self.nhead = num_landmarks, landmark_dim, d_model, nhead
        self.num_encoder_layers, self.dim_feedforward, self.dropout, self.num_classes = num_encoder_layers, dim_feedforward, dropout, num_classes
        super().__init__(**kwargs)

class EmotionTransformerModel(PreTrainedModel):
    config_class = EmotionTransformerConfig
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.input_embedding = nn.Linear(config.landmark_dim, config.d_model)
        self.pos_encoder = PositionalEncoding(config.d_model, config.dropout, max_len=config.num_landmarks)
        encoder_layer = nn.TransformerEncoderLayer(d_model=config.d_model, nhead=config.nhead,
                                                   dim_feedforward=config.dim_feedforward, dropout=config.dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=config.num_encoder_layers)
        self.fc_out = nn.Linear(config.d_model, config.num_classes)

    def forward(self, pixel_values, labels=None): # pixel_values are landmarks here
        x = self.input_embedding(pixel_values) * math.sqrt(self.config.d_model) # Scaling input embedding
        x = self.pos_encoder(x)
        transformer_output = self.transformer_encoder(x)
        embedding = transformer_output.mean(dim=1) # Mean pooling
        logits = self.fc_out(embedding)
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.config.num_classes), labels.view(-1))
        return {"loss": loss, "logits": logits, "embedding": embedding}

# --- Metrics Computation ---
def compute_metrics_fn(pred):
    labels = pred.label_ids
    # pred.predictions can be a tuple (logits, embedding). We need logits.
    if isinstance(pred.predictions, tuple):
        logits = pred.predictions[0]
    else: # Should be just logits if model output dict is handled by Trainer
        logits = pred.predictions 
    
    preds = np.argmax(logits, axis=-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='weighted', zero_division=0)
    acc = accuracy_score(labels, preds)
    return {'accuracy': acc, 'f1': f1, 'precision': precision, 'recall': recall}

# --- Usage Instructions Function ---
def print_usage_instructions(model_dir, onnx_path, num_lm, lm_dim):
    print("\n--- How to use the saved models ---")
    print(f"PyTorch model (and config) can be loaded using:")
    print(f"  from transformers import PreTrainedModel # (Actually your EmotionTransformerModel class)")
    print(f"  # from your_script_name import EmotionTransformerModel, EmotionTransformerConfig")
    print(f"  # config = EmotionTransformerConfig.from_pretrained('{model_dir}')")
    print(f"  # model = EmotionTransformerModel.from_pretrained('{model_dir}', config=config)")
    
    print(f"\nONNX model can be loaded using onnxruntime:")
    print(f"  import onnxruntime")
    print(f"  import numpy as np")
    print(f"  ort_session = onnxruntime.InferenceSession('{onnx_path}')")
    print(f"  # landmarks_input = np.random.randn(1, {num_lm}, {lm_dim}).astype(np.float32)")
    print(f"  # ort_inputs = {{ort_session.get_inputs()[0].name: landmarks_input}}")
    print(f"  # ort_outs = ort_session.run(None, ort_inputs)")
    print(f"  # logits = ort_outs[0]")
    print(f"  # embedding = ort_outs[1]")

    print("\n--- To get embeddings from the PyTorch model ---")
    print("  # model.eval()")
    print("  # with torch.no_grad():")
    print("  #     # Assuming 'landmarks_tensor' is your input: (batch_size, NUM_LANDMARKS, LANDMARK_DIM)")
    print("  #     outputs = model(pixel_values=landmarks_tensor)")
    print("  #     embedding = outputs['embedding'] # Shape: (batch_size, d_model)")
    print("  #     logits = outputs['logits']")

# --- Function to find the latest checkpoint ---
def get_latest_checkpoint(checkpoint_dir):
    if not os.path.isdir(checkpoint_dir):
        return None
    
    checkpoints = glob.glob(os.path.join(checkpoint_dir, "checkpoint-*"))
    if not checkpoints:
        return None
        
    latest_checkpoint = max(checkpoints, key=os.path.getmtime)
    print(f"Found latest checkpoint: {latest_checkpoint}")
    return latest_checkpoint

# --- Main Script ---
if __name__ == "__main__":
    print("Starting Emotion Recognition Training (Resuming from Checkpoint)...")
    print(f"Device: {'cuda' if torch.cuda.is_available() else 'cpu'}")
    
    if face_mesh_processor is None:
        print("CRITICAL ERROR: MediaPipe FaceMesh could not be initialized. Exiting.")
        exit()
    
    # --- 1. Load and Prepare Dataset ---
    print("\n--- Loading Dataset (Kaggle AffectNet via kagglehub) ---")
    # AffectNet labels: 0:Neutral, 1:Happy, 2:Sad, 3:Surprise, 4:Fear, 5:Disgust, 6:Anger, 7:Contempt
    # This matches NUM_CLASSES = 8
    try:
        # Define the path to the parent of 'archive (3)' directory
        local_dataset_main_dir = r"D:\Documnts2\forJohn\archive"
        # Define the specific data directory name (where labels.csv, Test/, Train/ reside)
        data_sub_dir_name = "archive (3)"
        # Construct the full path to the data directory to be walked
        dataset_root_to_walk = os.path.join(local_dataset_main_dir, data_sub_dir_name)
        
        print(f"Attempting to use local dataset at: {dataset_root_to_walk}")
        if not os.path.isdir(dataset_root_to_walk):
            raise Exception(f"Local dataset directory not found: {dataset_root_to_walk}. Please check the path.")

        all_files_in_local_dir = []
        for root, dirs, files in os.walk(dataset_root_to_walk): # Walk the local directory
            for file in files:
                all_files_in_local_dir.append(os.path.join(root, file))
        
        # Look for CSV files first in the local directory
        csv_files = [f for f in all_files_in_local_dir if f.lower().endswith('.csv')]
        
        if csv_files:
            csv_file_to_load = csv_files[0] 
            preferred_csv_file = os.path.join(dataset_root_to_walk, "labels.csv")
            if preferred_csv_file in csv_files:
                csv_file_to_load = preferred_csv_file
            elif any("labels.csv" in os.path.basename(f) for f in csv_files):
                 try:
                    csv_file_to_load = next(f for f in csv_files if "labels.csv" in os.path.basename(f))
                 except StopIteration: 
                    pass 

            print(f"Loading: {csv_file_to_load}")
            df = pd.read_csv(csv_file_to_load)
        else:
            parquet_files = [f for f in all_files_in_local_dir if f.lower().endswith('.parquet')]
            json_files = [f for f in all_files_in_local_dir if f.lower().endswith(('.json', '.jsonl'))]
            
            if parquet_files:
                df = pd.read_parquet(parquet_files[0])
            elif json_files:
                df = pd.read_json(json_files[0])
            else:
                raise Exception("No supported data format found in the local dataset directory")
        
        # Helper function to determine the correct image path
        def determine_actual_image_path(pth_value, root_dir_for_images):
            normalized_pth = pth_value.replace('/', os.sep)
            path_components = normalized_pth.split(os.sep)
            test_image_path = os.path.join(root_dir_for_images, 'Test', *path_components)
            if os.path.exists(test_image_path):
                return test_image_path
            train_image_path = os.path.join(root_dir_for_images, 'Train', *path_components)
            if os.path.exists(train_image_path):
                return train_image_path
            return None

        df['image_path'] = df['pth'].apply(lambda x: determine_actual_image_path(x, dataset_root_to_walk))
        original_len = len(df)
        df.dropna(subset=['image_path'], inplace=True)
        new_len = len(df)
        if new_len < original_len:
            print(f"INFO: Dropped {original_len - new_len} rows because images were not found.")
        
        if len(df) == 0:
            print("ERROR: No usable image paths found. Exiting.")
            exit()
        
        emotion_col = None
        if 'emotion' in df.columns: emotion_col = 'emotion'        
        elif 'label' in df.columns: emotion_col = 'label'
        elif 'expression' in df.columns: emotion_col = 'expression'
        else:
            for col in df.columns:
                if 'emotion' in col.lower() or 'expression' in col.lower() or 'label' in col.lower():
                    emotion_col = col
                    break
        if emotion_col is None:
            print("ERROR: Could not find emotion/label column. Exiting.")
            exit()
        
        emotion_mapping = {'neutral': 0, 'happy': 1, 'sad': 2, 'surprise': 3, 'fear': 4, 'disgust': 5, 'anger': 6, 'contempt': 7}
        valid_emotions = list(emotion_mapping.keys())
        df = df[df[emotion_col].isin(valid_emotions)]
        if len(df) == 0:
            print("ERROR: No valid emotion samples after filtering. Exiting.")
            exit()
        df['mapped_emotion'] = df[emotion_col].map(emotion_mapping)
        
        train_df, temp_df = train_test_split(df, test_size=0.3, stratify=df['mapped_emotion'], random_state=42)
        val_df, test_df = train_test_split(temp_df, test_size=0.5, stratify=temp_df['mapped_emotion'], random_state=42)
        
    except Exception as e:
        print(f"Failed to load AffectNet dataset: {e}")
        exit()
    
    print(f"Using Train dataset size: {len(train_df)}")
    print(f"Using Validation dataset size: {len(val_df)}")
    print(f"Using Test dataset size: {len(test_df)}")

    print("\n--- Initializing PyTorch Datasets with Landmark Extraction ---")
    train_emotion_dataset = AffectNetKaggleDataset(train_df, "train")
    eval_emotion_dataset = AffectNetKaggleDataset(val_df, "validation")
    test_emotion_dataset = AffectNetKaggleDataset(test_df, "test")

    # --- 2. Initialize Model ---
    print("\n--- Initializing Model ---")
    config = EmotionTransformerConfig(num_classes=NUM_CLASSES)
    model = EmotionTransformerModel(config)
    print(f"Model initialized with {sum(p.numel() for p in model.parameters() if p.requires_grad):,} trainable parameters.")

    # --- 3. Training ---
    print("\n--- Setting up Trainer ---")
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)

    checkpoint_dir = os.path.join(MODEL_DIR, "training_checkpoints")
    latest_checkpoint_path = get_latest_checkpoint(checkpoint_dir)

    training_args = TrainingArguments(
        output_dir=checkpoint_dir,
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        warmup_ratio=0.1,
        weight_decay=WEIGHT_DECAY,
        learning_rate=LEARNING_RATE,
        logging_dir=os.path.join(MODEL_DIR, "training_logs"),
        logging_steps=max(10, int(len(train_emotion_dataset) / (BATCH_SIZE * 10))),
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        greater_is_better=True,
        fp16=torch.cuda.is_available(),
        report_to="tensorboard",
        dataloader_num_workers=DATALOADER_NUM_WORKERS,
        dataloader_pin_memory=True,
        remove_unused_columns=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_emotion_dataset,
        eval_dataset=eval_emotion_dataset,
        compute_metrics=compute_metrics_fn,
    )

    print("\n--- Starting Training ---")
    start_time = time.time()
    try:
        # Pass latest_checkpoint_path to trainer.train()
        # If latest_checkpoint_path is None, it starts fresh (or from model_init if specified)
        trainer.train(resume_from_checkpoint=latest_checkpoint_path) 
    except Exception as e:
        print(f"An error occurred during training: {e}")
        if "CUDA out of memory" in str(e):
            print("CUDA OOM: Try reducing BATCH_SIZE or model size.")
        exit()
        
    training_time = time.time() - start_time
    print(f"Training finished in {training_time / 60:.2f} minutes.")

    if hasattr(train_emotion_dataset, 'print_extraction_stats'): train_emotion_dataset.print_extraction_stats()
    if hasattr(eval_emotion_dataset, 'print_extraction_stats'): eval_emotion_dataset.print_extraction_stats()

    # --- 4. Evaluation ---
    print("\n--- Evaluating on Test Set ---")
    test_results = trainer.predict(test_emotion_dataset)
    if hasattr(test_emotion_dataset, 'print_extraction_stats'): test_emotion_dataset.print_extraction_stats()
    
    print("\nTest Set Metrics:")
    final_test_metrics = test_results.metrics
    for key, value in final_test_metrics.items():
        print(f"  {key}: {value:.4f}")

    # --- 5. Save Model & Report ---
    print("\n--- Saving Models and Report ---")
    model.save_pretrained(MODEL_DIR) # Saves the best model due to load_best_model_at_end=True
    print(f"PyTorch model and config saved to {MODEL_DIR}")

    model.eval()
    model.to("cpu")
    dummy_input_landmarks = torch.randn(1, NUM_LANDMARKS, LANDMARK_DIM)
    
    try:
        print(f"Exporting model to ONNX at {ONNX_MODEL_PATH}...")
        dynamic_axes = {'pixel_values': {0: 'batch_size'}, 'logits': {0: 'batch_size'}, 'embedding': {0: 'batch_size'}}
        
        class OnnxWrapper(nn.Module):
            def __init__(self, model_to_wrap):
                super().__init__()
                self.model_to_wrap = model_to_wrap
            def forward(self, pixel_values):
                outputs = self.model_to_wrap(pixel_values=pixel_values)
                return outputs['logits'], outputs['embedding']

        onnx_exportable_model = OnnxWrapper(model)
        onnx_exportable_model.eval()

        torch.onnx.export(
            onnx_exportable_model,
            dummy_input_landmarks,
            ONNX_MODEL_PATH,
            input_names=['pixel_values'],
            output_names=['logits', 'embedding'],
            dynamic_axes=dynamic_axes,
            opset_version=14, # Changed to 14 to support aten::scaled_dot_product_attention
            export_params=True
        )
        print("ONNX model exported successfully.")

        try:
            import onnxruntime
            ort_session = onnxruntime.InferenceSession(ONNX_MODEL_PATH, providers=['CPUExecutionProvider'])
            ort_inputs = {ort_session.get_inputs()[0].name: dummy_input_landmarks.numpy()}
            ort_outs = ort_session.run(None, ort_inputs)
            print(f"ONNX model loaded and verified. Logits shape: {ort_outs[0].shape}, Embedding shape: {ort_outs[1].shape}")
        except ImportError: print("onnxruntime not installed. Skipping ONNX verification.")
        except Exception as e: print(f"Error during ONNX verification: {e}")
    except Exception as e: print(f"Error exporting to ONNX: {e}")

    report = {
        "dataset_used": "Local AffectNet (D:\\Documnts2\\forJohn\\archive\\archive (3))",
        "model_configuration": model.config.to_dict(),
        "training_arguments": {k: str(v) if not isinstance(v, (int, float, str, bool, list, dict, type(None))) else v 
                               for k, v in training_args.to_dict().items()},
        "training_time_minutes": training_time / 60,
        "test_set_metrics": final_test_metrics,
        "resumed_from_checkpoint": latest_checkpoint_path,
        "landmark_extraction_stats": {}
    }
    if hasattr(train_emotion_dataset, 'successful_extractions'):
        report["landmark_extraction_stats"]["train"] = {
            "successful": train_emotion_dataset.successful_extractions, "failed": train_emotion_dataset.failed_extractions,
            "total": train_emotion_dataset.successful_extractions + train_emotion_dataset.failed_extractions
        }
    if hasattr(eval_emotion_dataset, 'successful_extractions'):
        report["landmark_extraction_stats"]["validation"] = {
            "successful": eval_emotion_dataset.successful_extractions, "failed": eval_emotion_dataset.failed_extractions,
             "total": eval_emotion_dataset.successful_extractions + eval_emotion_dataset.failed_extractions
        }
    if hasattr(test_emotion_dataset, 'successful_extractions'):
        report["landmark_extraction_stats"]["test"] = {
            "successful": test_emotion_dataset.successful_extractions, "failed": test_emotion_dataset.failed_extractions,
            "total": test_emotion_dataset.successful_extractions + test_emotion_dataset.failed_extractions
        }

    with open(REPORT_PATH, 'w') as f: json.dump(report, f, indent=4)
    print(f"Training report saved to {REPORT_PATH}")

    print_usage_instructions(MODEL_DIR, ONNX_MODEL_PATH, NUM_LANDMARKS, LANDMARK_DIM)
    print("\n--- Script Finished ---")    
    if face_mesh_processor:
        face_mesh_processor.close()
