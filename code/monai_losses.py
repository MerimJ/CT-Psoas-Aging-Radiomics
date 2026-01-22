#!/usr/bin/env python3
"""
MONAI-based Loss Functions for Medical Image Segmentation
Focuses on reducing false positives and over-segmentation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from monai.losses import DiceFocalLoss, TverskyLoss, FocalLoss, DiceLoss
    MONAI_AVAILABLE = True
except ImportError:
    MONAI_AVAILABLE = False
    print("Warning: MONAI not available. Using PyTorch implementations.")


class TverskyLossFP(nn.Module):
    """
    Tversky Loss with alpha > beta to penalize false positives more.
    
    Tversky Index = TP / (TP + alpha*FP + beta*FN)
    - alpha=0.7, beta=0.3: Penalizes FPs 2.3x more than FNs
    - Higher alpha = more FP penalty = better precision
    
    Why this reduces FPs:
    - Directly penalizes false positives in the loss function
    - alpha > beta means FP errors cost more than FN errors
    """
    def __init__(self, alpha=0.7, beta=0.3, smooth=1e-6):
        super().__init__()
        self.alpha = alpha  # FP weight (higher = more FP penalty)
        self.beta = beta   # FN weight
        self.smooth = smooth
    
    def forward(self, pred, target):
        pred_flat = pred.view(-1)
        target_flat = target.view(-1)
        
        # True Positives, False Positives, False Negatives
        tp = (pred_flat * target_flat).sum()
        fp = (pred_flat * (1 - target_flat)).sum()
        fn = ((1 - pred_flat) * target_flat).sum()
        
        # Tversky Index (higher is better)
        tversky = (tp + self.smooth) / (tp + self.alpha * fp + self.beta * fn + self.smooth)
        
        # Return loss (1 - Tversky)
        return 1 - tversky


class DiceFocalLossFP(nn.Module):
    """
    Combined Dice + Focal Loss to handle class imbalance and penalize FPs.
    
    Loss = lambda_dice * DiceLoss + lambda_focal * FocalLoss
    
    Why this reduces FPs:
    - Dice loss: Handles overlap, good for segmentation
    - Focal loss: Focuses on hard examples (often FPs), handles class imbalance
    - Combined: Gets benefits of both
    """
    def __init__(self, lambda_dice=0.5, lambda_focal=0.5, gamma=2.0, alpha=0.25):
        super().__init__()
        self.lambda_dice = lambda_dice
        self.lambda_focal = lambda_focal
        self.gamma = gamma
        self.alpha = alpha
    
    def dice_loss(self, pred, target, smooth=1e-6):
        pred_flat = pred.view(-1)
        target_flat = target.view(-1)
        intersection = (pred_flat * target_flat).sum()
        return 1 - (2. * intersection + smooth) / (pred_flat.sum() + target_flat.sum() + smooth)
    
    def focal_loss(self, pred, target):
        bce = F.binary_cross_entropy(pred, target, reduction='none')
        pt = torch.where(target == 1, pred, 1 - pred)
        focal_term = (1 - pt) ** self.gamma
        alpha_t = torch.where(target == 1, self.alpha, 1 - self.alpha)
        return (alpha_t * focal_term * bce).mean()
    
    def forward(self, pred, target):
        dice = self.dice_loss(pred, target)
        focal = self.focal_loss(pred, target)
        return self.lambda_dice * dice + self.lambda_focal * focal


class CombinedLossFP(nn.Module):
    """
    Comprehensive loss combining Tversky, Dice, and Focal losses.
    Maximum FP penalty.
    
    Loss = w_tversky * TverskyLoss + w_dice * DiceLoss + w_focal * FocalLoss
    
    Why this reduces FPs:
    - Tversky: Direct FP penalty (alpha > beta)
    - Dice: Good overlap metric
    - Focal: Handles hard examples and imbalance
    - All three together provide maximum FP reduction
    """
    def __init__(self, 
                 w_tversky=0.4, w_dice=0.3, w_focal=0.3,
                 tversky_alpha=0.7, tversky_beta=0.3,
                 focal_gamma=2.0, focal_alpha=0.25):
        super().__init__()
        self.w_tversky = w_tversky
        self.w_dice = w_dice
        self.w_focal = w_focal
        
        self.tversky = TverskyLossFP(alpha=tversky_alpha, beta=tversky_beta)
        self.focal = FocalLossFP(gamma=focal_gamma, alpha=focal_alpha)
    
    def dice_loss(self, pred, target, smooth=1e-6):
        pred_flat = pred.view(-1)
        target_flat = target.view(-1)
        intersection = (pred_flat * target_flat).sum()
        return 1 - (2. * intersection + smooth) / (pred_flat.sum() + target_flat.sum() + smooth)
    
    def forward(self, pred, target):
        tversky = self.tversky(pred, target)
        dice = self.dice_loss(pred, target)
        focal = self.focal(pred, target)
        
        return (self.w_tversky * tversky + 
                self.w_dice * dice + 
                self.w_focal * focal)


class FocalLossFP(nn.Module):
    """Focal Loss for handling class imbalance and hard examples."""
    def __init__(self, gamma=2.0, alpha=0.25):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
    
    def forward(self, pred, target):
        bce = F.binary_cross_entropy(pred, target, reduction='none')
        pt = torch.where(target == 1, pred, 1 - pred)
        focal_term = (1 - pt) ** self.gamma
        alpha_t = torch.where(target == 1, self.alpha, 1 - self.alpha)
        return (alpha_t * focal_term * bce).mean()


def get_loss_function(loss_type='tversky', **kwargs):
    """
    Get loss function by name.
    
    Args:
        loss_type: 'tversky', 'dice_focal', 'combined', 'focal', or 'monai_dice_focal'
        **kwargs: Loss-specific parameters
    
    Returns:
        Loss function
    """
    if loss_type == 'tversky':
        alpha = kwargs.get('alpha', 0.7)
        beta = kwargs.get('beta', 0.3)
        return TverskyLossFP(alpha=alpha, beta=beta)
    
    elif loss_type == 'dice_focal':
        lambda_dice = kwargs.get('lambda_dice', 0.5)
        lambda_focal = kwargs.get('lambda_focal', 0.5)
        gamma = kwargs.get('gamma', 2.0)
        return DiceFocalLossFP(lambda_dice=lambda_dice, lambda_focal=lambda_focal, gamma=gamma)
    
    elif loss_type == 'combined':
        return CombinedLossFP(**kwargs)
    
    elif loss_type == 'focal':
        gamma = kwargs.get('gamma', 2.0)
        alpha = kwargs.get('alpha', 0.25)
        return FocalLossFP(gamma=gamma, alpha=alpha)
    
    elif loss_type == 'monai_dice_focal' and MONAI_AVAILABLE:
        lambda_dice = kwargs.get('lambda_dice', 0.5)
        lambda_focal = kwargs.get('lambda_focal', 0.5)
        gamma = kwargs.get('gamma', 2.0)
        return DiceFocalLoss(
            include_background=True,
            to_onehot_y=False,
            sigmoid=True,
            lambda_dice=lambda_dice,
            lambda_focal=lambda_focal,
            gamma=gamma
        )
    
    elif loss_type == 'monai_tversky' and MONAI_AVAILABLE:
        alpha = kwargs.get('alpha', 0.7)
        beta = kwargs.get('beta', 0.3)
        return TverskyLoss(
            include_background=True,
            to_onehot_y=False,
            sigmoid=True,
            alpha=alpha,
            beta=beta
        )
    
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")


# Example usage
if __name__ == '__main__':
    # Test loss functions
    pred = torch.rand(2, 1, 256, 256)
    target = (torch.rand(2, 1, 256, 256) > 0.1).float()
    
    # Test Tversky Loss
    tversky_loss = get_loss_function('tversky', alpha=0.7, beta=0.3)
    loss = tversky_loss(pred, target)
    print(f"Tversky Loss: {loss.item():.4f}")
    
    # Test Dice+Focal Loss
    dice_focal_loss = get_loss_function('dice_focal', lambda_dice=0.5, lambda_focal=0.5)
    loss = dice_focal_loss(pred, target)
    print(f"Dice+Focal Loss: {loss.item():.4f}")
