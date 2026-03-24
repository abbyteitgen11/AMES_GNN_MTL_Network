import torch
import torch.nn.functional as F
"""
Custom loss function to ignore values that equal -1

"""
def masked_loss_function(y_true, y_pred, class_weights):
    mask_value = -1

    # Create mask (1 for valid values, 0 for masked values)
    mask = (y_true != mask_value).float()

    y_true_clamped = torch.clamp(y_true, min=0, max=1)

    # Compute BCE loss for all elements
    loss = F.binary_cross_entropy(y_pred, y_true_clamped, reduction='none')  # Keep per-element loss

    weights_tensor = torch.where(
        y_true == 1, class_weights[1],
        torch.where(y_true == 0, class_weights[0], 0.0)
    )

    # Apply weights
    weighted_loss = loss * weights_tensor * mask

    # Apply mask (zero out masked values)
    #loss = loss * mask

    #loss_final = loss.sum() / mask.sum()

    #loss_final = loss.sum() / mask.sum() if mask.sum() > 0 else torch.tensor(0.0) # If NaN, set to 0
    total_mask = (mask * weights_tensor).sum()
    loss_final = weighted_loss.sum() / total_mask if total_mask > 0 else torch.tensor(0.0)


    return loss_final