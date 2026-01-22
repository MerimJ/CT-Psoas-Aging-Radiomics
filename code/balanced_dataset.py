#!/usr/bin/env python3
"""
Balanced Dataset with Positive and Negative Patches
Includes 50% positive (with psoas) and 50% negative (background only) patches.
This helps the model learn what NOT to segment, reducing false positives.
"""

import os
import numpy as np
import nibabel as nib
import pandas as pd
import torch
from torch.utils.data import Dataset
from pathlib import Path
import gc
import random


class BalancedPsoasDataset(Dataset):
    """
    Balanced dataset with 50% positive and 50% negative patches.
    
    Why this reduces FPs:
    - Negative patches teach the model what background looks like
    - Model learns to distinguish psoas from other soft tissues
    - Prevents over-segmentation by showing what NOT to segment
    """
    
    def __init__(self, pairs_df, train=True, max_slices_per_volume=30, 
                 negative_ratio=0.5, patch_size=None, base_path=None):
        """
        Args:
            pairs_df: DataFrame with ct_path and mask_path
            train: Whether this is training set
            max_slices_per_volume: Max slices per volume
            negative_ratio: Ratio of negative patches (0.5 = 50% negative)
            patch_size: If None, use full slice. Otherwise crop patches.
            base_path: Base path for relative paths
        """
        self.pairs = pairs_df.reset_index(drop=True)
        self.train = train
        self.max_slices_per_volume = max_slices_per_volume
        self.negative_ratio = negative_ratio
        self.patch_size = patch_size
        self.base_path = base_path or Path(__file__).parent
        
        # Lists: (vol_idx, slice_idx, is_positive)
        self.positive_list = []
        self.negative_list = []
        
        self._build_slice_lists()
        self._balance_dataset()
    
    def _build_slice_lists(self):
        """Build lists of positive and negative slices."""
        print(f"\nBuilding {'training' if self.train else 'validation'} dataset...")
        print(f"Target: {self.negative_ratio*100:.0f}% negative patches")
        
        for vol_idx, row in self.pairs.iterrows():
            ct_path = row['ct_path']
            mask_path = row['mask_path']
            
            if not os.path.isabs(ct_path):
                ct_path = self.base_path / ct_path
            if not os.path.isabs(mask_path):
                mask_path = self.base_path / mask_path
            
            if not os.path.exists(str(ct_path)) or not os.path.exists(str(mask_path)):
                continue
            
            try:
                mask_nii = nib.load(str(mask_path))
                shape = mask_nii.shape
                
                positive_slices = []
                negative_slices = []
                
                # Sample slices
                all_slices = list(range(shape[2]))
                if len(all_slices) > self.max_slices_per_volume:
                    # Sample evenly
                    step = len(all_slices) // self.max_slices_per_volume
                    sampled_indices = all_slices[::step][:self.max_slices_per_volume]
                else:
                    sampled_indices = all_slices
                
                # Classify slices as positive or negative
                for z in sampled_indices:
                    mask_slice = np.array(mask_nii.dataobj[:, :, z])
                    has_psoas = np.sum(mask_slice > 0) > 0
                    
                    if has_psoas:
                        positive_slices.append(z)
                    else:
                        negative_slices.append(z)
                
                # Add to lists
                for z in positive_slices:
                    self.positive_list.append((vol_idx, z, True))
                for z in negative_slices:
                    self.negative_list.append((vol_idx, z, False))
                
                print(f"Volume {row.get('patient_id', vol_idx)}: "
                      f"{len(positive_slices)} positive, {len(negative_slices)} negative slices")
                
                del mask_nii
                gc.collect()
                
            except Exception as e:
                print(f"Error processing volume {vol_idx}: {e}")
                continue
        
        print(f"\nTotal positive slices: {len(self.positive_list)}")
        print(f"Total negative slices: {len(self.negative_list)}")
    
    def _balance_dataset(self):
        """Balance positive and negative samples."""
        n_positive = len(self.positive_list)
        n_negative = len(self.negative_list)
        
        if n_negative == 0:
            print("Warning: No negative slices found. Using only positive slices.")
            self.slice_list = self.positive_list
            return
        
        # Calculate target numbers
        if self.negative_ratio > 0:
            # Target: negative_ratio of total should be negative
            # n_negative / (n_positive + n_negative) = negative_ratio
            # Solve for n_negative: n_negative = negative_ratio * n_positive / (1 - negative_ratio)
            target_negative = int(n_positive * self.negative_ratio / (1 - self.negative_ratio))
            target_negative = min(target_negative, n_negative)
        else:
            target_negative = 0
        
        # Sample negative slices if needed
        if len(self.negative_list) > target_negative:
            self.negative_list = random.sample(self.negative_list, target_negative)
        
        # Combine lists
        self.slice_list = self.positive_list + self.negative_list
        
        # Shuffle
        random.shuffle(self.slice_list)
        
        n_pos = sum(1 for _, _, is_pos in self.slice_list if is_pos)
        n_neg = sum(1 for _, _, is_pos in self.slice_list if not is_pos)
        
        print(f"\nBalanced dataset:")
        print(f"  Positive: {n_pos} ({n_pos/len(self.slice_list)*100:.1f}%)")
        print(f"  Negative: {n_neg} ({n_neg/len(self.slice_list)*100:.1f}%)")
        print(f"  Total: {len(self.slice_list)}")
    
    def __len__(self):
        return len(self.slice_list)
    
    def __getitem__(self, idx):
        vol_idx, slice_idx, is_positive = self.slice_list[idx]
        row = self.pairs.iloc[vol_idx]
        
        ct_path = row['ct_path']
        mask_path = row['mask_path']
        
        if not os.path.isabs(ct_path):
            ct_path = self.base_path / ct_path
        if not os.path.isabs(mask_path):
            mask_path = self.base_path / mask_path
        
        ct_nii = nib.load(str(ct_path))
        mask_nii = nib.load(str(mask_path))
        
        ct_slice = np.array(ct_nii.dataobj[:, :, slice_idx])
        mask_slice = np.array(mask_nii.dataobj[:, :, slice_idx])
        
        # Normalize CT
        ct_slice = np.clip(ct_slice, -1000, 1000)
        ct_slice = (ct_slice + 1000) / 2000
        
        # Binarize mask
        if is_positive:
            mask_slice = (mask_slice > 0).astype(np.float32)
        else:
            # Negative patch: all zeros
            mask_slice = np.zeros_like(mask_slice, dtype=np.float32)
        
        # Optional: Crop patch if patch_size is specified
        if self.patch_size is not None:
            h, w = ct_slice.shape
            if h > self.patch_size and w > self.patch_size:
                # Random crop for training, center crop for validation
                if self.train:
                    top = random.randint(0, h - self.patch_size)
                    left = random.randint(0, w - self.patch_size)
                else:
                    top = (h - self.patch_size) // 2
                    left = (w - self.patch_size) // 2
                
                ct_slice = ct_slice[top:top+self.patch_size, left:left+self.patch_size]
                mask_slice = mask_slice[top:top+self.patch_size, left:left+self.patch_size]
        
        item = (
            torch.tensor(ct_slice[None, ...], dtype=torch.float32),
            torch.tensor(mask_slice[None, ...], dtype=torch.float32)
        )
        
        del ct_nii, mask_nii
        gc.collect()
        
        return item
