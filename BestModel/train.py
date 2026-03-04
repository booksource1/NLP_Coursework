import os
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from transformers import AutoTokenizer, AutoModel, get_cosine_schedule_with_warmup
from sklearn.metrics import f1_score, classification_report
from tqdm import tqdm
from datetime import datetime
import argparse


# ==================== CONFIG ====================
class Config:
    # Paths (update these for your environment)
    DATA_DIR = " ./NLPLabs-2024/Dont_Patronize_Me_Trainingset"
    TRAIN_SPLIT_DIR = "./dontpatronizeme/semeval-2022/practice splits"
    RESULTS_DIR = "./results/CrossEntropy_RoBERTa_v2"

    # Model
    MODEL_NAME = "roberta-base"

    # Training
    EPOCHS = 10
    BATCH_SIZE = 32
    MAX_LENGTH = 128

    # Optimizer: differential learning rates
    LR_BACKBONE = 2e-5
    LR_HEAD = 1e-4
    WEIGHT_DECAY = 0.01

    # Warmup: 6% of total steps
    WARMUP_RATIO = 0.06

    # Label smoothing
    LABEL_SMOOTHING = 0.1

    # Class weights for CrossEntropy (computed from data: neg/pos ratio)
    # Will be computed dynamically from training data
    USE_CLASS_WEIGHTS = True

    # Use WeightedRandomSampler for balanced mini-batches
    USE_WEIGHTED_SAMPLER = True

    # Use differential learning rates (different LR for backbone and head)
    USE_DIFFERENTIAL_LR = True

    # Use Cosine LR Decay with Warmup
    USE_COSINE_WARMUP = True

    # Fixed threshold for binary classification
    THRESHOLD = 0.5

    # Device
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Seed
    SEED = 42



# ==================== DATASET ====================
class PCLDataset(Dataset):
    """PCL binary classification dataset (Task 1 only)."""

    def __init__(self, full_df: pd.DataFrame, split_df: pd.DataFrame,
                 tokenizer, max_length: int = 128):
        """
        Args:
            full_df: Full dataset with columns [par_id, paragraph, label].
            split_df: Official split file with columns [par_id, label].
                      'label' here is the 7-dim category vector (Task 2),
                      but we only use par_id to join with full_df.
            tokenizer: HuggingFace tokenizer.
            max_length: Maximum token sequence length.
        """
        # Rename to avoid column collision
        split_df = split_df.rename(columns={'label': 'label_st2'})
        full_df  = full_df.rename(columns={'label': 'label_pcl'})

        merged = split_df.merge(full_df, on='par_id', how='left')
        merged = merged.dropna(subset=['paragraph'])

        self.texts  = merged['paragraph'].tolist()
        self.labels = (merged['label_pcl'] >= 2).astype(int).values

        self.tokenizer  = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx],
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        return {
            'input_ids':      encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label':          torch.tensor(self.labels[idx], dtype=torch.long),
        }

    def get_class_weights(self) -> torch.Tensor:
        """Compute inverse-frequency class weights."""
        n_neg = (self.labels == 0).sum()
        n_pos = (self.labels == 1).sum()
        total = len(self.labels)
        w_neg = total / (2.0 * n_neg)
        w_pos = total / (2.0 * n_pos)
        return torch.tensor([w_neg, w_pos], dtype=torch.float)

    def get_sample_weights(self) -> np.ndarray:
        """Per-sample weights for WeightedRandomSampler."""
        class_weights = self.get_class_weights().numpy()
        return class_weights[self.labels]


# ==================== MODEL ====================
class PCLClassifier(nn.Module):
    """
    RoBERTa-based binary classifier for PCL detection.
    Uses the standard RoBERTa classification head architecture.
    """

    def __init__(self, model_name: str, dropout: float = 0.1):
        super().__init__()
        self.roberta = AutoModel.from_pretrained(model_name)
        hidden_size  = self.roberta.config.hidden_size

        # Standard RoBERTa classification head
        self.dropout  = nn.Dropout(dropout)
        self.dense    = nn.Linear(hidden_size, hidden_size)
        self.out_proj = nn.Linear(hidden_size, 2)

        # Xavier initialization for the head
        nn.init.xavier_uniform_(self.dense.weight)
        nn.init.zeros_(self.dense.bias)
        nn.init.xavier_uniform_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    def forward(self, input_ids: torch.Tensor,
                attention_mask: torch.Tensor) -> torch.Tensor:
        outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = outputs.last_hidden_state[:, 0, :]   # [CLS] token
        x = self.dropout(cls_output)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        logits = self.out_proj(x)
        return logits


# ==================== EVALUATION ====================
def evaluate(model: nn.Module, data_loader: DataLoader,
             device: torch.device, threshold: float = 0.5,
             criterion=None) -> dict:
    """
    Evaluate the model on a data loader.

    Returns:
        dict with keys: loss, f1_macro, f1_positive, f1_negative,
                        precision_positive, recall_positive, report
    """
    model.eval()
    all_probs  = []
    all_labels = []
    total_loss = 0.0

    with torch.no_grad():
        for batch in data_loader:
            input_ids      = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels         = batch['label'].to(device)

            logits = model(input_ids, attention_mask)

            if criterion is not None:
                loss = criterion(logits, labels)
                total_loss += loss.item()

            probs = torch.softmax(logits, dim=-1)[:, 1]
            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    all_probs  = np.array(all_probs)
    all_labels = np.array(all_labels)
    all_preds  = (all_probs >= threshold).astype(int)

    f1_macro    = f1_score(all_labels, all_preds, average='macro',    zero_division=0)
    f1_positive = f1_score(all_labels, all_preds, average='binary',   zero_division=0)
    f1_negative = f1_score(all_labels, all_preds, pos_label=0,        zero_division=0)
    report      = classification_report(all_labels, all_preds,
                                        target_names=['Negative', 'Positive'],
                                        zero_division=0)

    avg_loss = total_loss / len(data_loader) if len(data_loader) > 0 else 0.0

    return {
        'loss':               avg_loss,
        'f1_macro':           f1_macro,
        'f1_positive':        f1_positive,
        'f1_negative':        f1_negative,
        'report':             report,
        'threshold':          threshold,
    }


# ==================== TRAINER ====================
class Trainer:
    """Training loop for a fixed number of epochs without validation during training."""

    def __init__(self, model: nn.Module, train_loader: DataLoader,
                 val_loader: DataLoader, config: Config,
                 class_weights: torch.Tensor = None):
        self.model       = model.to(config.DEVICE)
        self.train_loader = train_loader
        self.val_loader   = val_loader
        self.config       = config

        # Loss function: CrossEntropyLoss with label smoothing and class weights
        self.criterion = nn.CrossEntropyLoss(
            weight=class_weights.to(config.DEVICE) if (config.USE_CLASS_WEIGHTS and class_weights is not None) else None,
            label_smoothing=config.LABEL_SMOOTHING if config.LABEL_SMOOTHING > 0 else 0.0,
        )

        # Optimizer: differential learning rates or uniform LR
        if config.USE_DIFFERENTIAL_LR:
            backbone_params = list(model.roberta.parameters())
            head_params     = (list(model.dense.parameters()) +
                               list(model.out_proj.parameters()))
            self.optimizer = torch.optim.AdamW([
                {'params': backbone_params, 'lr': config.LR_BACKBONE,
                 'weight_decay': config.WEIGHT_DECAY},
                {'params': head_params,     'lr': config.LR_HEAD,
                 'weight_decay': 0.0},
            ])
        else:
            # Use uniform learning rate for all parameters
            self.optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=config.LR_BACKBONE,
                weight_decay=config.WEIGHT_DECAY
            )

        # Cosine LR scheduler with linear warmup or fixed LR
        if config.USE_COSINE_WARMUP:
            total_steps  = len(train_loader) * config.EPOCHS
            warmup_steps = int(total_steps * config.WARMUP_RATIO)
            self.scheduler = get_cosine_schedule_with_warmup(
                self.optimizer,
                num_warmup_steps=warmup_steps,
                num_training_steps=total_steps,
            )
        else:
            # Use fixed learning rate (no scheduler)
            self.scheduler = None

        # History (only training loss during training)
        self.history = {
            'train_loss': [],
        }

    def train_epoch(self, epoch: int) -> float:
        self.model.train()
        total_loss = 0.0

        pbar = tqdm(self.train_loader,
                    desc=f"Epoch {epoch+1}/{self.config.EPOCHS} [Train]",
                    leave=False)
        for batch in pbar:
            input_ids      = batch['input_ids'].to(self.config.DEVICE)
            attention_mask = batch['attention_mask'].to(self.config.DEVICE)
            labels         = batch['label'].to(self.config.DEVICE)

            self.optimizer.zero_grad()
            logits = self.model(input_ids, attention_mask)
            loss   = self.criterion(logits, labels)
            loss.backward()

            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            self.optimizer.step()
            if self.scheduler is not None:
                self.scheduler.step()

            total_loss += loss.item()
            if self.scheduler is not None:
                current_lr = self.scheduler.get_last_lr()[0]
            else:
                # Get current LR from optimizer
                current_lr = self.optimizer.param_groups[0]['lr']
            pbar.set_postfix({'loss': f"{loss.item():.4f}",
                              'lr': f"{current_lr:.2e}"})

        return total_loss / len(self.train_loader)

    def fit(self, output_dir: str) -> dict:
        os.makedirs(output_dir, exist_ok=True)
        print(f"\nOutput directory: {output_dir}")
        print(f"Training for all {self.config.EPOCHS} epochs on {self.config.DEVICE}")
        print(f"Loss: CrossEntropyLoss")
        print(f"Label smoothing: {self.config.LABEL_SMOOTHING}")
        print(f"Class weights: {self.config.USE_CLASS_WEIGHTS}")
        print(f"Weighted sampler: {self.config.USE_WEIGHTED_SAMPLER}")
        print(f"Differential LR: {self.config.USE_DIFFERENTIAL_LR}")
        print(f"Cosine Warmup: {self.config.USE_COSINE_WARMUP}")
        print(f"Threshold: {self.config.THRESHOLD} (fixed)")
        print("Note: No validation during training. Evaluation will be performed after training completes.")
        print("=" * 60)

        # --- Training loop (no validation) ---
        for epoch in range(self.config.EPOCHS):
            # --- Train only ---
            train_loss = self.train_epoch(epoch)

            # --- Log training loss only ---
            self.history['train_loss'].append(train_loss)

            print(f"Epoch {epoch+1:>2}/{self.config.EPOCHS} | "
                  f"Train Loss: {train_loss:.4f}")

        # --- Final evaluation after training completes ---
        print("\n" + "=" * 60)
        print("Training completed. Evaluating on validation set...")
        print("=" * 60)

        final_metrics = evaluate(
            self.model, self.val_loader, self.config.DEVICE,
            threshold=self.config.THRESHOLD, criterion=self.criterion)

        print(f"Threshold:      {self.config.THRESHOLD:.4f} (fixed)")
        print(f"Macro F1:       {final_metrics['f1_macro']:.4f}")
        print(f"F1+ (positive): {final_metrics['f1_positive']:.4f}")
        print(f"F1- (negative): {final_metrics['f1_negative']:.4f}")
        print("\nClassification Report:")
        print(final_metrics['report'])

        # Save results
        method_parts = ['CrossEntropyLoss']
        if self.config.LABEL_SMOOTHING > 0:
            method_parts.append('LabelSmoothing')
        if self.config.USE_CLASS_WEIGHTS:
            method_parts.append('ClassWeights')
        if self.config.USE_WEIGHTED_SAMPLER:
            method_parts.append('WeightedSampler')
        if self.config.USE_DIFFERENTIAL_LR:
            method_parts.append('DifferentialLR')
        if self.config.USE_COSINE_WARMUP:
            method_parts.append('CosineWarmup')
        
        results = {
            'method':         ' + '.join(method_parts),
            'model':          self.config.MODEL_NAME,
            'timestamp':      datetime.now().isoformat(),
            'threshold':      self.config.THRESHOLD,
            'config':         {
                'label_smoothing': self.config.LABEL_SMOOTHING,
                'use_class_weights': self.config.USE_CLASS_WEIGHTS,
                'use_weighted_sampler': self.config.USE_WEIGHTED_SAMPLER,
                'use_differential_lr': self.config.USE_DIFFERENTIAL_LR,
                'use_cosine_warmup': self.config.USE_COSINE_WARMUP,
            },
            'final_metrics':  {k: v for k, v in final_metrics.items() if k != 'report'},
            'history':        self.history,
        }
        with open(os.path.join(output_dir, 'results.json'), 'w') as f:
            json.dump(results, f, indent=2)

        return results


# ==================== DATA LOADING ====================
def load_data(config: Config):
    """Load train and validation data from official splits."""
    full_df = pd.read_csv(
        os.path.join(config.DATA_DIR, "dontpatronizeme_pcl.tsv"),
        sep='\t', skiprows=4, header=None,
        names=['par_id', 'article_id', 'keyword', 'country_code', 'paragraph', 'label']
    )

    train_split = pd.read_csv(
        os.path.join(config.TRAIN_SPLIT_DIR, "train_semeval_parids-labels.csv"))
    val_split = pd.read_csv(
        os.path.join(config.TRAIN_SPLIT_DIR, "dev_semeval_parids-labels.csv"))

    print(f"Full dataset:  {len(full_df):,} paragraphs")
    print(f"Train split:   {len(train_split):,} samples")
    print(f"Val split:     {len(val_split):,} samples")

    return full_df, train_split, val_split


# ==================== MAIN ====================
def main():
    parser = argparse.ArgumentParser(description='PCL Detection v2: CrossEntropyLoss + Label Smoothing')
    parser.add_argument('--data_dir',       type=str, default=None,
                        help='Override DATA_DIR in Config')
    parser.add_argument('--split_dir',      type=str, default=None,
                        help='Override TRAIN_SPLIT_DIR in Config')
    parser.add_argument('--output_dir',     type=str, default=None,
                        help='Output directory')
    parser.add_argument('--epochs',         type=int, default=None)
    parser.add_argument('--batch_size',     type=int, default=None)
    parser.add_argument('--lr_backbone',    type=float, default=None)
    parser.add_argument('--label_smoothing',type=float, default=None)
    parser.add_argument('--no_label_smoothing', action='store_true',
                        help='Disable label smoothing (set to 0)')
    parser.add_argument('--no_class_weights', action='store_true',
                        help='Disable class weighting')
    parser.add_argument('--no_weighted_sampler', action='store_true',
                        help='Disable WeightedRandomSampler (use standard shuffle)')
    parser.add_argument('--no_differential_lr', action='store_true',
                        help='Disable differential learning rates (use uniform LR)')
    parser.add_argument('--no_cosine_warmup', action='store_true',
                        help='Disable Cosine LR Decay with Warmup (use fixed LR)')
    parser.add_argument('--threshold',      type=float, default=None,
                        help='Decision threshold (default: 0.5)')
    args = parser.parse_args()

    config = Config()

    # Apply CLI overrides
    if args.data_dir:       config.DATA_DIR         = args.data_dir
    if args.split_dir:      config.TRAIN_SPLIT_DIR  = args.split_dir
    if args.epochs:         config.EPOCHS           = args.epochs
    if args.batch_size:     config.BATCH_SIZE       = args.batch_size
    if args.lr_backbone:    config.LR_BACKBONE      = args.lr_backbone
    if args.label_smoothing is not None: config.LABEL_SMOOTHING = args.label_smoothing
    if args.no_label_smoothing:          config.LABEL_SMOOTHING = 0.0
    if args.no_class_weights:           config.USE_CLASS_WEIGHTS = False
    if args.no_weighted_sampler:        config.USE_WEIGHTED_SAMPLER = False
    if args.no_differential_lr:         config.USE_DIFFERENTIAL_LR = False
    if args.no_cosine_warmup:           config.USE_COSINE_WARMUP = False
    if args.threshold is not None:      config.THRESHOLD = args.threshold

    # Output directory
    if args.output_dir:
        output_dir = args.output_dir
    else:
        output_dir = os.path.join(
            config.RESULTS_DIR,
            f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )

    # Reproducibility
    torch.manual_seed(config.SEED)
    np.random.seed(config.SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.SEED)

    print("=" * 60)
    print("PCL Detection v2: CrossEntropyLoss + Label Smoothing")
    print("=" * 60)
    print(f"Device:          {config.DEVICE}")
    print(f"Model:           {config.MODEL_NAME}")
    print(f"Epochs:          {config.EPOCHS}")
    print(f"Batch size:      {config.BATCH_SIZE}")
    print(f"LR backbone:     {config.LR_BACKBONE}")
    print(f"LR head:         {config.LR_HEAD}")
    print(f"Label smoothing: {config.LABEL_SMOOTHING}")
    print(f"Class weights:   {config.USE_CLASS_WEIGHTS}")
    print(f"Weighted sampler: {config.USE_WEIGHTED_SAMPLER}")
    print(f"Differential LR: {config.USE_DIFFERENTIAL_LR}")
    print(f"Cosine Warmup: {config.USE_COSINE_WARMUP}")
    print(f"Threshold:       {config.THRESHOLD} (fixed)")
    print("=" * 60)

    # Load data
    full_df, train_split, val_split = load_data(config)

    # Tokenizer
    print("\nLoading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(config.MODEL_NAME)

    # Datasets
    train_dataset = PCLDataset(full_df, train_split, tokenizer, config.MAX_LENGTH)
    val_dataset   = PCLDataset(full_df, val_split,   tokenizer, config.MAX_LENGTH)

    n_pos = train_dataset.labels.sum()
    n_neg = len(train_dataset.labels) - n_pos
    print(f"\nTrain: {len(train_dataset):,} samples | Pos: {n_pos} | Neg: {n_neg}")
    print(f"Val:   {len(val_dataset):,} samples   | Pos: {val_dataset.labels.sum()} | "
          f"Neg: {len(val_dataset.labels) - val_dataset.labels.sum()}")

    # Class weights
    class_weights = train_dataset.get_class_weights()
    print(f"Class weights: neg={class_weights[0]:.3f}, pos={class_weights[1]:.3f}")

    # WeightedRandomSampler for balanced mini-batches or standard shuffle
    if config.USE_WEIGHTED_SAMPLER:
        sample_weights = train_dataset.get_sample_weights()
        sampler = WeightedRandomSampler(
            weights=torch.from_numpy(sample_weights).float(),
            num_samples=len(train_dataset),
            replacement=True,
        )
        train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE,
                                  sampler=sampler, num_workers=0)
    else:
        train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE,
                                  shuffle=True, num_workers=0)
    val_loader   = DataLoader(val_dataset,   batch_size=config.BATCH_SIZE,
                              shuffle=False,  num_workers=0)

    # Model
    print("\nLoading model...")
    model = PCLClassifier(config.MODEL_NAME)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {n_params:,}")

    # Train
    trainer = Trainer(model, train_loader, val_loader, config,
                      class_weights=class_weights)
    results = trainer.fit(output_dir)

    print(f"\n✓ Training complete. Results saved to: {output_dir}")
    return results


if __name__ == '__main__':
    main()
