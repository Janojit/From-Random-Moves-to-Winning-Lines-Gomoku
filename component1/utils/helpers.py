import torch
import numpy as np

def mask_logits(logits, board):
    mask = torch.tensor(board.flatten() != 0, device=logits.device)
    logits[mask] = -1e9
    return logits

def masked_softmax(logits, board):
    logits = mask_logits(logits, board)
    return torch.softmax(logits, dim=-1)
