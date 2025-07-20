import jittor.nn as nn
import numpy as np
import jittor as jt

class SigmoidFocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, weight=None, ignore_index=255):
        super(SigmoidFocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.weight = weight
        self.ignore_index = ignore_index

    def execute(self, inputs, targets):
        # Manual implementation of BCE with logits for numerical stability
        # BCE with logits: max(x, 0) - x * z + log(1 + exp(-abs(x)))
        # where x is inputs and z is targets
        max_val = jt.clamp(inputs, min_v=0)
        loss = max_val - inputs * targets + jt.log(1 + jt.exp(-jt.abs(inputs)))
        
        # Apply sigmoid to get probabilities
        p = jt.sigmoid(inputs)
        
        # Calculate focal loss weight
        # pt = p when target = 1, (1-p) when target = 0
        pt = jt.where(targets == 1, p, 1 - p)
        
        # Focal loss = -alpha * (1-pt)^gamma * log(pt)
        # But we use BCE loss instead of -log(pt)
        focal_weight = self.alpha * jt.pow(1 - pt, self.gamma)
        
        # Apply focal weighting
        loss = focal_weight * loss
        
        return loss.mean()

def SoftIoULoss( pred, target):
        # Old One
        pred = jt.sigmoid(pred)
        smooth = 1

        # print("pred.shape: ", pred.shape)
        # print("target.shape: ", target.shape)

        intersection = pred * target
        loss = (intersection.sum() + smooth) / (pred.sum() + target.sum() -intersection.sum() + smooth)

        # loss = (intersection.sum(axis=(1, 2, 3)) + smooth) / \
        #        (pred.sum(axis=(1, 2, 3)) + target.sum(axis=(1, 2, 3))
        #         - intersection.sum(axis=(1, 2, 3)) + smooth)

        loss = 1 - loss.mean()
        # loss = (1 - loss).mean()

        return loss

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count 