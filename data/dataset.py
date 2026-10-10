import os
import glob
import pandas as pd
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T

def get_subject_splits(subjects, train_ratio=0.70, val_ratio=0.15, seed=42):
    np.random.seed(seed)
    subjects = np.unique(subjects)
    np.random.shuffle(subjects)
    
    n_total = len(subjects)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)
    
    train_subs = set(subjects[:n_train])
    val_subs = set(subjects[n_train:n_train + n_val])
    test_subs = set(subjects[n_train + n_val:])
    return train_subs, val_subs, test_subs


class UnlabeledVideoDataset(Dataset):
    """
    Loads unlabeled video folders from one or multiple root directories 
    (e.g., combining CASME II and MMEW Micro).
    """
    def __init__(self, root_dirs, split="train", num_frames=16, is_micro=True):
        if isinstance(root_dirs, str):
            root_dirs = [root_dirs]
            
        self.num_frames = num_frames
        self.split = split
        self.is_micro = is_micro  # Fixed attribute error
        self.video_paths = []
        
        all_paths = []
        for root_dir in root_dirs:
            if os.path.exists(root_dir):
                for path, subdirs, files in os.walk(root_dir):
                    images = [f for f in files if f.endswith(('.jpg', '.png', '.jpeg'))]
                    if len(images) >= 4:
                        all_paths.append(path)
                        
        if len(all_paths) > 0:
            def extract_sub(p):
                parts = p.split(os.sep)
                for part in parts:
                    if part.lower().startswith(('sub', 's')) and part[3:].isdigit():
                        return part
                return parts[-2] if len(parts) >= 2 else "unknown"

            subjects = [extract_sub(p) for p in all_paths]
            train_s, val_s, test_s = get_subject_splits(subjects)
            
            for p, sub in zip(all_paths, subjects):
                if split == "train" and sub in train_s:
                    self.video_paths.append(p)
                elif split == "val" and sub in val_s:
                    self.video_paths.append(p)
                elif split == "test" and sub in test_s:
                    self.video_paths.append(p)
                    
        print(f"[{split.upper()}] Loaded {len(self.video_paths)} video folders (is_micro={self.is_micro})")
        
        self.transform = T.Compose([
            T.Resize((112, 112)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return max(1, len(self.video_paths))

    def __getitem__(self, idx):
        if len(self.video_paths) == 0:
            return torch.randn(self.num_frames, 3, 112, 112)
            
        v_path = self.video_paths[idx % len(self.video_paths)]
        image_files = sorted(glob.glob(os.path.join(v_path, "*.jpg")) + glob.glob(os.path.join(v_path, "*.png")))
        
        total_frames = len(image_files)
        if total_frames >= self.num_frames:
            if self.is_micro:
                start_idx = np.random.randint(0, total_frames - self.num_frames + 1)
                selected_indices = range(start_idx, start_idx + self.num_frames)
            else:
                selected_indices = np.linspace(0, total_frames - 1, self.num_frames, dtype=int)
        else:
            selected_indices = list(range(total_frames)) + [total_frames - 1] * (self.num_frames - total_frames)

        frames = [self.transform(Image.open(image_files[i]).convert("RGB")) for i in selected_indices]
        return torch.stack(frames)


class LabeledVideoDataset(Dataset):
    """Evaluation dataset using metadata Excel sheets for held-out testing."""
    def __init__(self, root_dirs, excel_file, split="test", num_frames=32):
        if isinstance(root_dirs, str):
            root_dirs = [root_dirs]
            
        self.root_dirs = root_dirs
        self.num_frames = num_frames
        self.samples = []
        self.label_map = {}
        
        if os.path.exists(excel_file):
            df = pd.read_excel(excel_file)
            if 'Estimated Emotion' in df.columns and 'Subject' in df.columns:
                unique_emotions = sorted(df['Estimated Emotion'].dropna().unique())
                self.label_map = {emo: i for i, emo in enumerate(unique_emotions)}
                
                subjects = df['Subject'].unique()
                train_s, val_s, test_s = get_subject_splits(subjects)
                target_subjects = train_s if split == "train" else (val_s if split == "val" else test_s)
                
                for _, row in df.iterrows():
                    sub = row['Subject']
                    if sub in target_subjects:
                        filename = str(row['Filename'])
                        emotion = str(row['Estimated Emotion'])
                        if emotion in self.label_map:
                            self.samples.append((filename, self.label_map[emotion]))

        self.transform = T.Compose([
            T.Resize((112, 112)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return max(1, len(self.samples))

    def __getitem__(self, idx):
        if len(self.samples) == 0:
            return torch.randn(self.num_frames, 3, 112, 112), 0
            
        filename, label = self.samples[idx % len(self.samples)]
        
        matching_paths = []
        for rd in self.root_dirs:
            matching_paths = glob.glob(os.path.join(rd, f"**/{filename}"), recursive=True)
            if matching_paths:
                break
                
        if matching_paths:
            v_path = matching_paths[0]
            image_files = sorted(glob.glob(os.path.join(v_path, "*.jpg")) + glob.glob(os.path.join(v_path, "*.png")))
            total_frames = len(image_files)
            if total_frames > 0:
                selected_indices = np.linspace(0, total_frames - 1, self.num_frames, dtype=int)
                frames = [self.transform(Image.open(image_files[i]).convert("RGB")) for i in selected_indices]
                return torch.stack(frames), label

        return torch.randn(self.num_frames, 3, 112, 112), label