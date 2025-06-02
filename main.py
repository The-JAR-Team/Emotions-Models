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

MODEL_DIR = "./emotion_transformer_ferplus_local_model_v4" 
DATASET_CACHE_DIR = "./dataset_cache_ferplus_hub_v4" 
ONNX_MODEL_PATH = os.path.join(MODEL_DIR, "emotion_transformer.onnx")
REPORT_PATH = os.path.join(MODEL_DIR, "training_report.json")

# FER+ has 8 emotion classes:
# 0:neutral, 1:happiness, 2:surprise, 3:sadness, 4:anger, 5:disgust, 6:fear, 7:contempt
NUM_CLASSES = 8
EMOTION_COLUMNS = ['neutral', 'happiness', 'surprise', 'sadness', 'anger', 'disgust', 'fear', 'contempt'] 

NUM_LANDMARKS = 478 
LANDMARK_DIM = 3 

# Model Hyperparameters
D_MODEL = 512
NHEAD = 8
NUM_ENCODER_LAYERS = 8
DIM_FEEDFORWARD = 1024
DROPOUT = 0.2

# Training Hyperparameters
PER_DEVICE_TRAIN_BATCH_SIZE = 64
PER_DEVICE_EVAL_BATCH_SIZE = 124
LEARNING_RATE = 5e-5
NUM_TRAIN_EPOCHS = 50
WARMUP_RATIO = 0.1
WEIGHT_DECAY = 0.01
LOGGING_STRATEGY = "steps" 
LOGGING_STEPS = 100
SAVE_TOTAL_LIMIT = 3
LOAD_BEST_MODEL_AT_END = True
METRIC_FOR_BEST_MODEL = "accuracy" 
GREATER_IS_BETTER = True
FP16_TRAINING = True 
DATALOADER_NUM_WORKERS = 4
DATALOADER_PIN_MEMORY = True
REPORT_TO = "tensorboard"
EARLY_STOPPING_PATIENCE = 10 
START_FROM_CHECKPOINT = False

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

# --- PyTorch Dataset for Local FER+ Data ---
class EmotionLandmarkLocalDataset(Dataset):
    def __init__(self, dataframe, image_base_path, split_name=""):
        self.df = dataframe
        self.image_base_path = image_base_path 
        self.split_name = split_name
        self.successful_extractions = 0
        self.failed_extractions = 0
        self.to_numpy = np.array
        self.usage_to_folder = {'Training': 'FER2013Train', 'PublicTest': 'FER2013Valid', 'PrivateTest': 'FER2013Test'}
        if self.split_name == "train" or len(self.df) > 1000:
            print(f"INFO: Landmark extraction for local '{self.split_name}' set ({len(self.df)} samples) is on-the-fly.")
            print("      This will be slow. For large-scale training, PRE-PROCESSING LANDMARKS IS CRITICAL.")
    def __len__(self): return len(self.df)
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        image_name_val = row['Image name']
        if pd.isna(image_name_val): # Handle potential NaN values
            # print(f"Warning: NaN found in 'Image name' for row index {idx} in {self.split_name}. Skipping item.")
            self.failed_extractions += 1
            landmarks = np.zeros((NUM_LANDMARKS, LANDMARK_DIM), dtype=np.float32)
            return {"pixel_values": torch.tensor(landmarks, dtype=torch.float32), "labels": torch.tensor(0, dtype=torch.long)} # Dummy item
        image_name = str(image_name_val) # Ensure image_name is a string

        usage_val = row['Usage']
        usage = str(usage_val) # Ensure usage is a string for dict key

        label = row['derived_label']

        folder_name = self.usage_to_folder.get(usage)
        if not folder_name:
            # print(f"Warning: Unknown usage type '{usage}' for image {image_name} (row index {idx}) in {self.split_name}. Skipping item.")
            self.failed_extractions +=1
            landmarks = np.zeros((NUM_LANDMARKS, LANDMARK_DIM), dtype=np.float32)
            return {"pixel_values": torch.tensor(landmarks, dtype=torch.float32), "labels": torch.tensor(0, dtype=torch.long)}

        image_path = os.path.join(self.image_base_path, folder_name, image_name)
        try: 
            pil_image = Image.open(image_path)
        except FileNotFoundError:
            # print(f"ERROR: Image not found at {image_path} (row index {idx}) in {self.split_name}. Skipping item.")
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
        if success: self.successful_extractions += 1
        else: self.failed_extractions += 1
        
        return {"pixel_values": torch.tensor(landmarks, dtype=torch.float32), "labels": torch.tensor(label, dtype=torch.long)}

    def print_extraction_stats(self):
        total_processed = self.successful_extractions + self.failed_extractions
        if total_processed == 0: print(f"Landmark extraction stats for '{self.split_name}': No items processed."); return
        success_rate = (self.successful_extractions / total_processed * 100) if total_processed > 0 else 0
        print(f"Landmark extraction stats for '{self.split_name}': Successfully extracted: {self.successful_extractions}/{total_processed} ({success_rate:.2f}%)")

# --- PyTorch Dataset for Hugging Face datasets (Fallback) ---
class EmotionLandmarkHFDataset(Dataset):
    def __init__(self, hf_dataset_split, split_name=""):
        self.hf_dataset_split = hf_dataset_split
        self.split_name = split_name
        self.successful_extractions = 0
        self.failed_extractions = 0
        self.to_numpy = np.array
        if self.split_name == "train" or len(self.hf_dataset_split) > 1000:
            print(f"INFO (HF): Landmark extraction for '{self.split_name}' set ({len(self.hf_dataset_split)} samples) is on-the-fly.")
            print("      This will be slow. PRE-PROCESSING LANDMARKS IS CRITICAL for large datasets.")
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
        if success: self.successful_extractions += 1
        else: self.failed_extractions += 1
        return {"pixel_values": torch.tensor(landmarks, dtype=torch.float32), "labels": torch.tensor(label, dtype=torch.long)}
    def print_extraction_stats(self):
        total_processed = self.successful_extractions + self.failed_extractions
        if total_processed == 0: print(f"Landmark extraction stats for '{self.split_name}': No items processed."); return
        success_rate = (self.successful_extractions / total_processed * 100) if total_processed > 0 else 0
        print(f"Landmark extraction stats for '{self.split_name}': Successfully extracted: {self.successful_extractions}/{total_processed} ({success_rate:.2f}%)")

# --- Transformer Model ---
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=DROPOUT, max_len=NUM_LANDMARKS + 50):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
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
                    train_emotion_dataset = EmotionLandmarkLocalDataset(df_train, LOCAL_FERPLUS_BASE_PATH, "train_local_ferplus")
                    eval_emotion_dataset = EmotionLandmarkLocalDataset(df_val, LOCAL_FERPLUS_BASE_PATH, "eval_local_ferplus")
                    test_emotion_dataset = EmotionLandmarkLocalDataset(df_test, LOCAL_FERPLUS_BASE_PATH, "test_local_ferplus")
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
            train_emotion_dataset = EmotionLandmarkHFDataset(train_hf_data, "train_hub_ferplus")
            eval_emotion_dataset = EmotionLandmarkHFDataset(eval_hf_data, "eval_hub_ferplus")
            test_emotion_dataset = EmotionLandmarkHFDataset(test_hf_data, "test_hub_ferplus")
            dataset_source_name = "microsoft/ferplus (Hugging Face Hub)"
            print("Successfully loaded FER+ from Hugging Face Hub.")
        except Exception as e:
            print(f"CRITICAL ERROR: Failed to load FER+ dataset from local path AND Hugging Face Hub: {e}"); exit()
    
    print(f"Using Train dataset size: {len(train_emotion_dataset)}")
    print(f"Using Validation dataset size: {len(eval_emotion_dataset)}")
    print(f"Using Test dataset size: {len(test_emotion_dataset)}")

    print("\n--- Initializing Model ---")
    config = EmotionTransformerConfig(num_classes=NUM_CLASSES)
    model = EmotionTransformerModel(config)
    print(f"Model initialized with {sum(p.numel() for p in model.parameters() if p.requires_grad):,} trainable parameters.")

    print("\n--- Setting up Trainer ---")
    if not os.path.exists(MODEL_DIR): os.makedirs(MODEL_DIR)

    # Calculate steps_per_epoch for epoch-wise evaluation and saving
    steps_per_epoch = math.ceil(len(train_emotion_dataset) / PER_DEVICE_TRAIN_BATCH_SIZE)
    print(f"Approximate steps per epoch: {steps_per_epoch}")
    
    # Ensure LOGGING_STEPS from config is used, or default to a fraction of steps_per_epoch
    effective_logging_steps = LOGGING_STEPS
    if LOGGING_STRATEGY == "epoch": # If user explicitly wants epoch logging (not in current config)
        effective_logging_steps = steps_per_epoch
    elif LOGGING_STRATEGY == "steps":
        effective_logging_steps = LOGGING_STEPS # Use the one from config

    training_args = TrainingArguments(
        output_dir=os.path.join(MODEL_DIR, "training_checkpoints"),
        num_train_epochs=NUM_TRAIN_EPOCHS,
        per_device_train_batch_size=PER_DEVICE_TRAIN_BATCH_SIZE,
        per_device_eval_batch_size=PER_DEVICE_EVAL_BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        warmup_ratio=WARMUP_RATIO, 
        weight_decay=WEIGHT_DECAY, 
        # Arguments from user config:
        logging_strategy=LOGGING_STRATEGY,
        logging_steps=effective_logging_steps,
        # Use eval_steps and save_steps for epoch-like behavior
        # This replaces evaluation_strategy="epoch" and save_strategy="epoch" for compatibility
        evaluation_strategy="steps", # Must be "steps" if eval_steps is used
        eval_steps=steps_per_epoch,  
        save_strategy="steps",       # Must be "steps" if save_steps is used
        save_steps=steps_per_epoch,  
        save_total_limit=SAVE_TOTAL_LIMIT,
        load_best_model_at_end=LOAD_BEST_MODEL_AT_END, 
        metric_for_best_model=METRIC_FOR_BEST_MODEL, 
        greater_is_better=GREATER_IS_BETTER,
        fp16=FP16_TRAINING and torch.cuda.is_available(), 
        report_to=REPORT_TO,
        dataloader_num_workers=DATALOADER_NUM_WORKERS, 
        dataloader_pin_memory=DATALOADER_PIN_MEMORY,
        remove_unused_columns=False,
    )
    trainer = Trainer(model=model, args=training_args, train_dataset=train_emotion_dataset,
                      eval_dataset=eval_emotion_dataset, compute_metrics=compute_metrics_fn,
                      callbacks=[EarlyStoppingCallback(early_stopping_patience=EARLY_STOPPING_PATIENCE)])

    print("\n--- Starting Training ---")
    start_time = time.time()
    checkpoint_to_resume = None
    if START_FROM_CHECKPOINT:
        checkpoint_to_resume = get_latest_checkpoint(MODEL_DIR)
        if checkpoint_to_resume: print(f"Attempting to resume training from: {checkpoint_to_resume}")
        else: print("No valid checkpoint found or START_FROM_CHECKPOINT is False. Starting from scratch.")
    try:
        trainer.train(resume_from_checkpoint=checkpoint_to_resume)
    except Exception as e:
        print(f"An error occurred during training: {e}")
        if "CUDA out of memory" in str(e): print("CUDA OOM: Try reducing BATCH_SIZE or model size.")
        exit()
    training_time = time.time() - start_time
    print(f"Training finished in {training_time / 60:.2f} minutes.")

    if hasattr(train_emotion_dataset, 'print_extraction_stats'): train_emotion_dataset.print_extraction_stats()
    if hasattr(eval_emotion_dataset, 'print_extraction_stats'): eval_emotion_dataset.print_extraction_stats()

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
