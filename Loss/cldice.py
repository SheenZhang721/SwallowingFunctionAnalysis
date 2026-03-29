import torch
import torch.nn as nn
import torch.nn.functional as F
from .soft_skeleton import SoftSkeletonize

class soft_cldice(nn.Module):
    def __init__(self, iter_=3, smooth = 1., exclude_background=False):
        super(soft_cldice, self).__init__()
        self.iter = iter_
        self.smooth = smooth
        self.soft_skeletonize = SoftSkeletonize(num_iter=10)
        self.exclude_background = exclude_background

    def forward(self, y_true, y_pred):
        if self.exclude_background:
            y_true = y_true[:, 1:, :, :]
            y_pred = y_pred[:, 1:, :, :]
        skel_pred = self.soft_skeletonize(y_pred)
        skel_true = self.soft_skeletonize(y_true)
        tprec = (torch.sum(torch.multiply(skel_pred, y_true))+self.smooth)/(torch.sum(skel_pred)+self.smooth)
        tsens = (torch.sum(torch.multiply(skel_true, y_pred))+self.smooth)/(torch.sum(skel_true)+self.smooth)
        cl_dice = 1.- 2.0*(tprec*tsens)/(tprec+tsens)
        return cl_dice


def soft_dice(y_true, y_pred):
    """[function to compute dice loss]

    Args:
        y_true ([float32]): [ground truth image]
        y_pred ([float32]): [predicted image]

    Returns:
        [float32]: [loss value]
    """
    smooth = 1
    intersection = torch.sum((y_true * y_pred))
    coeff = (2. *  intersection + smooth) / (torch.sum(y_true) + torch.sum(y_pred) + smooth)
    return (1. - coeff)

class WeightedDiceLoss(nn.Module):
    def __init__(self, weights=[0.5, 0.5],n_labels=1): # W_pos=0.8, W_neg=0.2
        super(WeightedDiceLoss, self).__init__()
        self.weights = weights
        self.n_labels = n_labels

    def _show_dice(self, inputs, targets):
        inputs[inputs>=0.5] = 1
        inputs[inputs<0.5] = 0
        # print("2",np.sum(tmp))
        targets[targets>0] = 1
        targets[targets<=0] = 0
        hard_dice_coeff = 1.0 - self.dice_loss(inputs, targets)
        return hard_dice_coeff

    def forward(self, logit, truth, smooth=1e-5):

        if(self.n_labels==1):
            batch_size = len(logit)
            logit = logit.view(batch_size,-1)
            truth = truth.view(batch_size,-1)
            assert(logit.shape==truth.shape)
            p = logit.view(batch_size,-1)
            t = truth.view(batch_size,-1)
            w = truth.detach()
            w = w*(self.weights[1]-self.weights[0])+self.weights[0]
            # p = w*(p*2-1)  #convert to [0,1] --> [-1, 1]
            # t = w*(t*2-1)
            p = w*(p)
            t = w*(t)
            intersection = (p * t).sum(-1)
            union =  (p * p).sum(-1) + (t * t).sum(-1)
            dice  = 1 - (2*intersection + smooth) / (union +smooth)
            # print "------",dice.data

            loss = dice.mean()
            return loss


class soft_dice_cldice(nn.Module):
    def __init__(self, iter_=3, alpha=0.5, smooth = 1., exclude_background=False):
        super(soft_dice_cldice, self).__init__()
        self.iter = iter_
        self.smooth = smooth
        self.alpha = alpha
        self.soft_skeletonize = SoftSkeletonize(num_iter=10)
        self.exclude_background = exclude_background

    def forward(self, y_true, y_pred):
        # disable gradients
        y_true = y_true.detach()
        y_pred = y_pred.detach()
        if self.exclude_background:
            y_true = y_true[:, 1:, :, :]
            y_pred = y_pred[:, 1:, :, :]
        dice = soft_dice(y_true, y_pred)
        skel_true = self.soft_skeletonize(y_true)
        skel_pred = self.soft_skeletonize(y_pred)
        tprec = (torch.sum(torch.multiply(skel_pred, y_true))+self.smooth)/(torch.sum(skel_pred)+self.smooth)
        tsens = (torch.sum(torch.multiply(skel_true, y_pred))+self.smooth)/(torch.sum(skel_true)+self.smooth)
        cl_dice = 1.- 2.0*(tprec*tsens)/(tprec+tsens)
        return (1.0-self.alpha)*dice+self.alpha*cl_dice, (1.0 - dice), (1.0 - cl_dice)