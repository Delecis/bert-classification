# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
from torch.nn import functional as F

class FGM():
    def __init__(self, model):
        self.model = model
        self.backup = {}

    def attack(self, epsilon=1., emb_name='word_embeddings'):
        # emb_name这个参数要换成你模型中embedding的参数名
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name:
                self.backup[name] = param.data.clone()
                norm = torch.norm(param.grad)
                if norm != 0:
                    r_at = epsilon * param.grad / norm
                    param.data.add_(r_at)

    def restore(self, emb_name='word_embeddings'):
        # emb_name这个参数要换成你模型中embedding的参数名
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name:
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}

class PGD():
    def __init__(self, model):
        self.model = model
        self.emb_backup = {}
        self.grad_backup = {}

    def attack(self, epsilon=1., alpha=0.3, emb_name='word_embeddings', is_first_attack=False):
        # emb_name这个参数要换成你模型中embedding的参数名
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name:
                if is_first_attack:
                    self.emb_backup[name] = param.data.clone()
                norm = torch.norm(param.grad)
                if norm != 0 and not torch.isnan(norm):
                    r_at = alpha * param.grad / norm
                    param.data.add_(r_at)
                    param.data = self.project(name, param.data, epsilon)

    def restore(self, emb_name='word_embeddings'):
        # emb_name这个参数要换成你模型中embedding的参数名
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name:
                assert name in self.emb_backup
                param.data = self.emb_backup[name]
        self.emb_backup = {}

    def project(self, param_name, param_data, epsilon):
        r = param_data - self.emb_backup[param_name]
        if torch.norm(r) > epsilon:
            r = epsilon * r / torch.norm(r)
        return self.emb_backup[param_name] + r

    def backup_grad(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.grad_backup[name] = param.grad.clone()

    def restore_grad(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.grad = self.grad_backup[name]


# # 用于二分类，防止正负样本不平衡，focal_loss,表达式为：FL(pt) = - at * (1 - pt)^(gammma) * log(pt)
# class FocalLoss(nn.Module):
#     def __init__(self, alpha=1, gamma=2, logits=False, reduce=True):
#         super(FocalLoss, self).__init__()
#         self.alpha = alpha
#         self.gamma = gamma
#         self.logits = logits
#         self.reduce = reduce
#
#     def forward(self, inputs, targets):
#         if self.logits:
#             BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduce=False)
#         else:
#             BCE_loss = F.binary_cross_entropy(inputs, targets, reduce=False)
#         pt = torch.exp(-BCE_loss)
#         F_loss = self.alpha * (1-pt)**self.gamma * BCE_loss
#
#         if self.reduce:
#             return torch.mean(F_loss)
#         else:
#             return F_loss


# # 用于标签平滑
# class LabelSmoothingLoss(nn.Module):
#     '''LabelSmoothingLoss
#     '''
#     def __init__(self, classes, smoothing=0.05, dim=-1):
#         super(LabelSmoothingLoss, self).__init__()
#         self.confidence = 1.0 - smoothing
#         self.smoothing = smoothing
#         self.cls = classes
#         self.dim = dim
#
#     def forward(self, pred, target):
#         pred = pred.log_softmax(dim=self.dim)
#         with torch.no_grad():
#             # true_dist = pred.data.clone()
#             true_dist = torch.zeros_like(pred)
#             true_dist.fill_(self.smoothing / (self.cls - 1))
#             true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
#         return torch.mean(torch.sum(-true_dist * pred, dim=self.dim))
#
#
# ## 可以新建一个class
# def dice_loss(target,predictive,ep=1e-8):
#     intersection = 2 * torch.sum(predictive * target) + ep
#     union = torch.sum(predictive) + torch.sum(target) + ep
#     loss = 1 - intersection / union
#     return loss

class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, eps=0.1, reduction='mean', ignore_index=-100):
        super(LabelSmoothingCrossEntropy, self).__init__()
        self.eps = eps
        self.reduction = reduction
        self.ignore_index = ignore_index

    def forward(self, output, target):
        c = output.size()[-1]
        log_pred = torch.log_softmax(output, dim=-1)
        if self.reduction == 'sum':
            loss = -log_pred.sum()
        else:
            loss = -log_pred.sum(dim=-1)
            if self.reduction == 'mean':
                loss = loss.mean()


        return loss * self.eps / c + (1 - self.eps) * torch.nn.functional.nll_loss(log_pred, target,
                                                                                   reduction=self.reduction,
                                                                                   ignore_index=self.ignore_index)

class FocalLoss(nn.Module):
    """Multi-class Focal loss implementation"""
    def __init__(self, gamma=2, weight=None, reduction='mean', ignore_index=-100):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.weight = weight
        self.ignore_index = ignore_index
        self.reduction = reduction

    def forward(self, input, target):
        """
        input: [N, C]
        target: [N, ]
        """
        log_pt = torch.log_softmax(input, dim=1)
        pt = torch.exp(log_pt)
        log_pt = (1 - pt) ** self.gamma * log_pt
        loss = torch.nn.functional.nll_loss(log_pt, target, self.weight, reduction=self.reduction, ignore_index=self.ignore_index)
        return loss



# class LabelSmoothingCrossEntropy(nn.Module):
#     def __init__(self, eps=0.1, reduction='mean', ignore_index=-100):
#         super(LabelSmoothingCrossEntropy, self).__init__()
#         self.eps = eps
#         self.reduction = reduction
#         self.ignore_index = ignore_index
#
#     def forward(self, output, target):
#         c = output.size()[-1]
#         log_pred = torch.log_softmax(output, dim=-1)
#         if self.reduction == 'sum':
#             loss = -log_pred.sum()
#         else:
#             loss = -log_pred.sum(dim=-1)
#             if self.reduction == 'mean':
#                 loss = loss.mean()
#
#
#         return loss * self.eps / c + (1 - self.eps) * torch.nn.functional.nll_loss(log_pred, target,
#                                                                                    reduction=self.reduction,
#                                                                                    ignore_index=self.ignore_index)
#
# class FocalLoss(nn.Module):
#     """Multi-class Focal loss implementation"""
#     def __init__(self, gamma=2, weight=None, reduction='mean', ignore_index=-100):
#         super(FocalLoss, self).__init__()
#         self.gamma = gamma
#         self.weight = weight
#         self.ignore_index = ignore_index
#         self.reduction = reduction
#
#     def forward(self, input, target):
#         """
#         input: [N, C]
#         target: [N, ]
#         """
#         log_pt = torch.log_softmax(input, dim=1)
#         pt = torch.exp(log_pt)
#         log_pt = (1 - pt) ** self.gamma * log_pt
#         loss = torch.nn.functional.nll_loss(log_pt, target, self.weight, reduction=self.reduction, ignore_index=self.ignore_index)
#         return loss
#
# class SpatialDropout(nn.Module):
#     """
#     对字级别的向量进行丢弃
#     """
#     def __init__(self, drop_prob):
#         super(SpatialDropout, self).__init__()
#         self.drop_prob = drop_prob
#
#     @staticmethod
#     def _make_noise(input):
#         return input.new().resize_(input.size(0), *repeat(1, input.dim() - 2), input.size(2))
#
#     def forward(self, inputs):
#         output = inputs.clone()
#         if not self.training or self.drop_prob == 0:
#             return inputs
#         else:
#             noise = self._make_noise(inputs)
#             if self.drop_prob == 1:
#                 noise.fill_(0)
#             else:
#                 noise.bernoulli_(1 - self.drop_prob).div_(1 - self.drop_prob)
#             noise = noise.expand_as(inputs)
#             output.mul_(noise)
#         return output
#
# class ConditionalLayerNorm(nn.Module):
#     def __init__(self,
#                  normalized_shape,
#                  cond_shape,
#                  eps=1e-12):
#         super().__init__()
#
#         self.eps = eps
#
#         self.weight = nn.Parameter(torch.Tensor(normalized_shape))
#         self.bias = nn.Parameter(torch.Tensor(normalized_shape))
#
#         self.weight_dense = nn.Linear(cond_shape, normalized_shape, bias=False)
#         self.bias_dense = nn.Linear(cond_shape, normalized_shape, bias=False)
#
#         self.reset_weight_and_bias()
#
#     def reset_weight_and_bias(self):
#         """
#         此处初始化的作用是在训练开始阶段不让 conditional layer norm 起作用
#         """
#         nn.init.ones_(self.weight)
#         nn.init.zeros_(self.bias)
#
#         nn.init.zeros_(self.weight_dense.weight)
#         nn.init.zeros_(self.bias_dense.weight)
#
#     def forward(self, inputs, cond=None):
#         assert cond is not None, 'Conditional tensor need to input when use conditional layer norm'
#         cond = torch.unsqueeze(cond, 1)  # (b, 1, h*2)
#
#         weight = self.weight_dense(cond) + self.weight  # (b, 1, h)
#         bias = self.bias_dense(cond) + self.bias  # (b, 1, h)
#
#         mean = torch.mean(inputs, dim=-1, keepdim=True)  # （b, s, 1）
#         outputs = inputs - mean  # (b, s, h)
#
#         variance = torch.mean(outputs ** 2, dim=-1, keepdim=True)
#         std = torch.sqrt(variance + self.eps)  # (b, s, 1)
#
#         outputs = outputs / std  # (b, s, h)
#
#         outputs = outputs * weight + bias
#
#         return outputs
