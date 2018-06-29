import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, output, target):
        pred = F.sigmoid(output)
        intersection = (pred * target).sum()
        eps = 1
        return 1. - (2. * intersection + eps) / (pred.sum() + target.sum() + eps)


class JaccardLoss(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, output, target):
        pred = F.sigmoid(output)
        inter = (pred * target).sum()
        A = pred * (1. - target)
        B = target * (1. - pred)
        C = target + pred
        union = .5 * (A.sum() + B.sum() + C.sum())
        eps = 1e-9
        iou = 1. - inter / (union + eps)
        return iou
        
        #intersection = (pred * target).sum()
        #union = pred.sum() + target.sum() - intersection
        #return intersection / (union + 1e-9)
    

class BCEDiceLoss(nn.Module):
    def __init__(self, w_bce=.5, w_dice=.5, bce_weight=None):
        super().__init__()
        self.dice = DiceLoss()
        self.bce = nn.BCEWithLogitsLoss(weight=bce_weight)
        self.w_bce = w_bce
        self.w_dice = w_dice
        self.hist = {'bce': [], 'dice': []}

    def forward(self, input, target):
        bce = self.w_bce * self.bce(input, target)
        dice = self.w_dice * self.dice(input, target)
        self.hist['bce'].append(bce.data[0])
        self.hist['dice'].append(dice.data[0])
        return bce + dice
    
    
class BCEJaccardLoss(nn.Module):
    def __init__(self, w_bce=.5, w_jaccard=.5):
        super().__init__()
        self.jaccard = JaccardLoss()
        self.bce = nn.BCEWithLogitsLoss()
        self.w_bce = w_bce
        self.w_jaccard = w_jaccard
        self.hist = {'bce': [], 'jaccard': []}
        
    def forward(self, input, target):
        bce = self.w_bce * self.bce(input, target)
        jaccard = self.w_jaccard * self.jaccard(input, target)
        self.hist['bce'].append(bce.data[0])
        self.hist['jaccard'].append(jaccard.data[0])
        return bce + jaccard