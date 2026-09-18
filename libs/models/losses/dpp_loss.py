import torch
import torch.nn.functional as F
from mmdet.registry import MODELS

@MODELS.register_module()
class DPPLoss(torch.nn.Module):
    """
    Determinantal Point Process Loss for Distribution Diversity
    
    The loss encourages diversity among the predicted distribution features.
    as the log-scale, minimizing it will maximize the determinant of the similarity matrix (diversity).
    
    loss_weight: weight for the dpp loss
    eps: small value for numerical stability
    temp: temperature for scaling the similarity matrix
    """
    def __init__(self, loss_weight=0.1, eps=1e-5, temp=1.0):
        super(DPPLoss, self).__init__()
        self.loss_weight = loss_weight
        self.eps = eps
        self.temp = temp
    
    def forward(self, preds):
        """
        preds: (B, Nr, dim) distribution features
        """
        
        # Normalize for cosine-like kernel
        Phi = F.normalize(preds, dim=-1) # (B, Nr, dim)
        L = (Phi @ Phi.transpose(1, 2)) / self.temp # (B, Nr, Nr)
        I = torch.eye(L.size(-1), device=L.device, dtype=L.dtype)[None]
        
        # use slogdet for stability
        sign, logdet = torch.slogdet(I + L + self.eps * I)
        
        # sign should be +1; if numerical issues, clamp
        logdet = torch.where(sign > 0, logdet, torch.zeros_like(logdet))
        return -(logdet.mean() / preds.size(1)) * self.loss_weight