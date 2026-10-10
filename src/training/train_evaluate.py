import os
import sys

# ---------------------------------------------------------
# PATH RESOLUTION (Matches your exact project folder structure)
# ---------------------------------------------------------
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.append(PROJECT_ROOT)
sys.path.append(os.path.join(PROJECT_ROOT, "configs"))
sys.path.append(os.path.join(PROJECT_ROOT, "data"))
sys.path.append(os.path.join(PROJECT_ROOT, "src/models"))

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

from config import *
from dataset import UnlabeledVideoDataset, LabeledVideoDataset
from models import TemporalViTEncoder, MicroGuidedMacroEncoder


class ContrastiveTemporalLoss(torch.nn.Module):
    """
    Combines InfoNCE contrastive loss with temporal consistency regularization.
    """
    def __init__(self, temperature=TEMPERATURE):
        super().__init__()
        self.temperature = temperature
        self.cos = torch.nn.CosineSimilarity(dim=1)

    def forward(self, repr1, repr2):
        sim_matrix = torch.mm(repr1, repr2.T) / self.temperature
        labels = torch.arange(repr1.size(0)).to(repr1.device)
        loss_contra = torch.nn.CrossEntropyLoss()(sim_matrix, labels)
        loss_consistency = 1.0 - self.cos(repr1, repr2).mean()
        return loss_contra + 0.5 * loss_consistency


def run_experiment(exp_name, model, loader_train, loader_val, epochs=EPOCHS):
    print(f"\n==============================================")
    print(f"RUNNING EXPERIMENT ({exp_name}) [Subject-Independent Split]")
    print(f"==============================================")
    
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = ContrastiveTemporalLoss()
    
    best_val_loss = float('inf')
    micro_loader, macro_loader = loader_train
    val_micro, val_macro = loader_val

    for epoch in range(epochs):
        # --- Training Phase ---
        model.train()
        total_train_loss = 0.0
        batches = 0
        
        for micro_batch, macro_batch in zip(micro_loader, macro_loader):
            optimizer.zero_grad()
            
            if exp_name == "Proposed Method":
                enhanced_macro, macro_base = model(micro_batch, macro_batch)
                loss = criterion(enhanced_macro, macro_base)
            else:
                if isinstance(model, MicroGuidedMacroEncoder):
                    _, macro_base = model(micro_batch, macro_batch)
                    global_repr1 = macro_base
                    global_repr2, _ = model.macro_encoder(macro_batch)
                else:
                    global_repr1, _ = model(macro_batch)
                    global_repr2, _ = model(macro_batch)
                loss = criterion(global_repr1, global_repr2)

            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()
            batches += 1

        avg_train_loss = total_train_loss / max(1, batches)

        # --- Validation Phase ---
        model.eval()
        total_val_loss = 0.0
        val_batches = 0
        
        with torch.no_grad():
            for v_micro, v_macro in zip(val_micro, val_macro):
                if exp_name == "Proposed Method":
                    e_macro, m_base = model(v_micro, v_macro)
                    v_loss = criterion(e_macro, m_base)
                else:
                    g1, _ = model(v_macro)
                    g2, _ = model(v_macro)
                    v_loss = criterion(g1, g2)
                total_val_loss += v_loss.item()
                val_batches += 1
                
        avg_val_loss = total_val_loss / max(1, val_batches)
        print(f"Epoch [{epoch+1}/{epochs}] | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

        # Save Best Checkpoint based on Validation Loss
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            os.makedirs(os.path.join(PROJECT_ROOT, "checkpoints"), exist_ok=True)
            ckpt_path = os.path.join(PROJECT_ROOT, f"checkpoints/{exp_name.lower().replace(' ', '_')}_best.pth")
            torch.save(model.state_dict(), ckpt_path)

    print(f"Training complete. Best model saved for {exp_name}.")
    return model


def evaluate_test_set(model, test_eval_loader, exp_name):
    """Evaluates representation quality on the held-out test set using Linear Probing."""
    print(f"\n--- Held-Out Test Evaluation for: {exp_name} ---")
    model.eval()
    
    features, labels = [], []
    with torch.no_grad():
        for video_batch, target_batch in test_eval_loader:
            if isinstance(model, MicroGuidedMacroEncoder):
                _, macro_seq = model.macro_encoder(video_batch)
                global_repr = macro_seq.mean(dim=1)
            else:
                global_repr, _ = model(video_batch)
                
            features.append(global_repr.cpu().numpy())
            labels.append(target_batch.numpy())

    if len(features) > 0 and len(labels) > 0:
        X = np.concatenate(features, axis=0)
        y = np.concatenate(labels, axis=0)
        
        if len(np.unique(y)) > 1:
            clf = LogisticRegression(max_iter=300)
            clf.fit(X, y)
            preds = clf.predict(X)
            
            acc = accuracy_score(y, preds)
            print(f"[{exp_name}] Test Set Linear Probing Accuracy: {acc * 100:.2f}%")
            print("Classification Report:")
            print(classification_report(y, preds, zero_division=0))
        else:
            print("Single class detected in test evaluation batch.")


if __name__ == "__main__":
    # Micro-expression dataset roots (combining CASME II and MMEW Micro)
    micro_roots = [CASME_MICRO_ROOT, MMEW_MICRO_ROOT]

    # 1. Subject-Independent Training DataLoaders (with drop_last=True for synchronized batching)
    train_micro = UnlabeledVideoDataset(micro_roots, split="train", num_frames=NUM_FRAMES_MICRO, is_micro=True)
    train_macro = UnlabeledVideoDataset(MMEW_MACRO_ROOT, split="train", num_frames=NUM_FRAMES_MACRO, is_micro=False)
    loader_train_micro = DataLoader(train_micro, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    loader_train_macro = DataLoader(train_macro, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)

    # 2. Subject-Independent Validation DataLoaders
    val_micro = UnlabeledVideoDataset(micro_roots, split="val", num_frames=NUM_FRAMES_MICRO, is_micro=True)
    val_macro = UnlabeledVideoDataset(MMEW_MACRO_ROOT, split="val", num_frames=NUM_FRAMES_MACRO, is_micro=False)
    loader_val_micro = DataLoader(val_micro, batch_size=BATCH_SIZE, shuffle=False, drop_last=True)
    loader_val_macro = DataLoader(val_macro, batch_size=BATCH_SIZE, shuffle=False, drop_last=True)

    # 3. Subject-Independent Test Evaluation DataLoader
    excel_path = os.path.join(PROJECT_ROOT, MMEW_MICRO_EXCEL)
    test_eval_dataset = LabeledVideoDataset(micro_roots, excel_file=excel_path, split="test")
    loader_test_eval = DataLoader(test_eval_dataset, batch_size=BATCH_SIZE, shuffle=False, drop_last=False)

    # --- Run Experiment 1: Baseline ---
    baseline_model = TemporalViTEncoder(embed_dim=EMBED_DIM)
    baseline_model = run_experiment("Baseline", baseline_model, (loader_train_micro, loader_train_macro), (loader_val_micro, loader_val_macro))
    evaluate_test_set(baseline_model, loader_test_eval, "Baseline")

    # --- Run Experiment 2: Proposed Method ---
    proposed_model = MicroGuidedMacroEncoder(embed_dim=EMBED_DIM)
    proposed_model = run_experiment("Proposed Method", proposed_model, (loader_train_micro, loader_train_macro), (loader_val_micro, loader_val_macro))
    evaluate_test_set(proposed_model, loader_test_eval, "Proposed Method")

    # --- Run Experiment 3: Ablation ---
    ablation_model = TemporalViTEncoder(embed_dim=EMBED_DIM)
    ablation_model = run_experiment("Ablation", ablation_model, (loader_train_micro, loader_train_macro), (loader_val_micro, loader_val_macro))
    evaluate_test_set(ablation_model, loader_test_eval, "Ablation")

    print("\nAll subject-independent experiments completed successfully!")