import torch
import torch.nn.functional as F
"""
Custom loss function to ignore values that equal -1

"""
def masked_loss_function(y_true, y_pred):
    mask_value = -1

    # Create mask (1 for valid values, 0 for masked values)
    mask = (y_true != mask_value).float()

    y_true_clamped = torch.clamp(y_true, min=0, max=1)

    # Compute BCE loss for all elements
    loss = F.binary_cross_entropy(y_pred, y_true_clamped, reduction='none')  # Keep per-element loss

    # Apply mask (zero out masked values)
    loss = loss * mask

    #loss_final = loss.sum() / mask.sum()

    loss_final = loss.sum() / mask.sum() if mask.sum() > 0 else torch.tensor(0.0) # If NaN, set to 0

    return loss_final