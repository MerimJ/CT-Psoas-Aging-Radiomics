#!/usr/bin/env python3
"""
Weighted Loss Functions for Severe Class Imbalance
Handles the case where psoas (white) is 0.72% and background (black) is 99.28%
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class WeightedBCELoss(nn.Module):
    """
    Pixel-level weighted Binary Cross Entropy Loss.
    
    Weights each pixel based on its class:
    - White pixels (psoas): weight = pos_weight (high, e.g., 0.98 or 139.0)
    - Black pixels (background): weight = 1.0 (or low, e.g., 0.02)
    
    Args:
        pos_weight: Weight for positive (white/psoas) pixels
        neg_weight: Weight for negative (black/background) pixels
    """
    def __init__(self, pos_weight=139.0, neg_weight=1.0):
        super().__init__()
        self.pos_weight = pos_weight  # Weight for white pixels (psoas)
        self.neg_weight = neg_weight  # Weight for black pixels (background)
    
    def forward(self, pred, target):
        # Create weight map: pos_weight for white pixels, neg_weight for black
        weights = torch.where(target > 0.5, 
                              torch.tensor(self.pos_weight, device=pred.device),
                              torch.tensor(self.neg_weight, device=pred.device))
        
        # Binary cross entropy with pixel-level weights
        bce = F.binary_cross_entropy(pred, target, reduction='none')
        weighted_bce = (weights * bce).mean()
        
        return weighted_bce


class WeightedDiceLoss(nn.Module):
    """
    Pixel-level weighted Dice Loss.
    
    Weights the foreground (white/psoas) pixels more heavily.
    
    Args:
        pos_weight: Weight for positive (white/psoas) pixels in Dice calculation
    """
    def __init__(self, pos_weight=139.0, smooth=1e-6):
        super().__init__()
        self.pos_weight = pos_weight
        self.smooth = smooth
    
    def forward(self, pred, target):
        # Calculate Dice for foreground (white pixels)
        pred_flat = pred.view(-1)
        target_flat = target.view(-1)
        
        # Intersection and union for foreground
        intersection = (pred_flat * target_flat).sum()
        pred_sum = pred_flat.sum()
        target_sum = target_flat.sum()
        
        # Weighted Dice: emphasize foreground
        dice = (2. * intersection * self.pos_weight + self.smooth) / \
               (pred_sum * self.pos_weight + target_sum + self.smooth)
        
        return 1 - dice


class WeightedCombinedLoss(nn.Module):
    """
    Combined Weighted BCE + Weighted Dice Loss.
    
    Best of both worlds:
    - Weighted BCE: Pixel-level weighting (white=high, black=low)
    - Weighted Dice: Emphasizes foreground overlap
    
    Args:
        pos_weight: Weight for white/psoas pixels
        neg_weight: Weight for black/background pixels
        dice_weight: Weight for Dice loss component
        bce_weight: Weight for BCE loss component
    """
    def __init__(self, pos_weight=139.0, neg_weight=1.0, 
                 dice_weight=0.5, bce_weight=0.5):
        super().__init__()
        self.pos_weight = pos_weight
        self.neg_weight = neg_weight
        self.dice_weight = dice_weight
        self.bce_weight = bce_weight
        
        self.weighted_bce = WeightedBCELoss(pos_weight, neg_weight)
        self.weighted_dice = WeightedDiceLoss(pos_weight)
    
    def forward(self, pred, target):
        bce = self.weighted_bce(pred, target)
        dice = self.weighted_dice(pred, target)
        
        return self.bce_weight * bce + self.dice_weight * dice


def calculate_class_weights(mask_paths, normalize=True):
    """
    Calculate optimal class weights from mask statistics.
    
    Args:
        mask_paths: List of paths to mask files
        normalize: If True, normalize weights to sum to 1.0
    
    Returns:
        pos_weight, neg_weight: Optimal weights for positive and negative classes
    """
    import nibabel as nib
    import numpy as np
    
    total_psoas = 0
    total_background = 0
    
    for mask_path in mask_paths:
        mask = nib.load(mask_path)
        data = mask.get_fdata()
        
        psoas_count = np.sum(data > 0)
        bg_count = np.sum(data == 0)
        
        total_psoas += psoas_count
        total_background += bg_count
    
    total = total_psoas + total_background
    
    # Inverse frequency weighting
    pos_weight = total / (2 * total_psoas) if total_psoas > 0 else 1.0
    neg_weight = total / (2 * total_background) if total_background > 0 else 1.0
    
    if normalize:
        # Normalize so they sum to 1.0 (like user suggested: 0.98 + 0.02)
        total_weight = pos_weight + neg_weight
        pos_weight = pos_weight / total_weight
        neg_weight = neg_weight / total_weight
    
    return pos_weight, neg_weight


# Example usage
if __name__ == '__main__':
    # Test weighted loss
    pred = torch.rand(2, 1, 256, 256)
    target = (torch.rand(2, 1, 256, 256) > 0.01).float()  # 1% positive
    
    # Option 1: User's suggestion (normalized)
    loss1 = WeightedBCELoss(pos_weight=0.98, neg_weight=0.02)
    print(f"Normalized weights (0.98/0.02): {loss1(pred, target).item():.4f}")
    
    # Option 2: Inverse frequency (recommended)
    loss2 = WeightedBCELoss(pos_weight=139.0, neg_weight=1.0)
    print(f"Inverse frequency (139.0/1.0): {loss2(pred, target).item():.4f}")
    
    # Option 3: Combined
    loss3 = WeightedCombinedLoss(pos_weight=139.0, neg_weight=1.0)
    print(f"Combined (Dice+BCE): {loss3(pred, target).item():.4f}")
