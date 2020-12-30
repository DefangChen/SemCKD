from __future__ import print_function

import torch
import torch.nn as nn


class IRGLoss(nn.Module):
    """Knowledge Distillation via Instance Relationship Graph, CVPR2019
    Code from author: https://github.com/yufanLIU/IRG"""

    def __init__(self, w_graph=1, w_transform=1):
        super(IRGLoss, self).__init__()
        self.mseloss = nn.MSELoss()
        self.w_graph = w_graph
        self.w_transform = w_transform

    def forward(self, f_s, f_t, transform_s, transform_t, no_edge_transform=False):
        edge_transform = not no_edge_transform
        student = f_s.view(f_s.shape[0], -1)
        teacher = f_t.view(f_t.shape[0], -1)

        # vertex and edge loss
        with torch.no_grad():
            t_d = self.pdist(teacher, squared=True, normalization='max')
        d = self.pdist(student, squared=True, normalization='max')
        loss = self.mseloss(d, t_d)
        if f_s.shape == f_t.shape:
            loss += self.mseloss(f_s, f_t)
        loss *= self.w_graph

        # transform loss
        transform_zip = list(zip(transform_s, transform_t))
        for (l1_s, l1_t), (l2_s, l2_t) in list(zip(transform_zip, transform_zip[1:]))[::2]:
            loss += self.transform_loss(l1_s, l2_s, l1_t,
                                        l2_t, edge_transform) * self.w_transform

        return loss

    def transform_loss(self, l1_s, l2_s, l1_t, l2_t, edge_transform=True):
        loss = []
        if edge_transform:
            dl1_s = self.pdist(
                l1_s.view(l1_s.shape[0], -1), squared=True, normalization='max')
            dl2_s = self.pdist(
                l2_s.view(l2_s.shape[0], -1), squared=True, normalization='max')
            with torch.no_grad():
                dl1_t = self.pdist(
                    l1_t.view(l1_t.shape[0], -1), squared=True, normalization='max')
                dl2_t = self.pdist(
                    l2_t.view(l2_t.shape[0], -1), squared=True, normalization='max')
            loss.append(self.mseloss(self.mseloss(
                dl1_s, dl2_s), self.mseloss(dl1_t, dl2_t)))
        if l1_s.shape == l2_s.shape and l1_t.shape == l2_t.shape:
            with torch.no_grad():
                lossv_t = self.mseloss(l1_t, l2_t)
            loss.append(self.mseloss(self.mseloss(l1_s, l2_s), lossv_t))
        return sum(loss)

    @staticmethod
    def pdist(e, squared=False, eps=1e-12, normalization='max'):
        e_square = e.pow(2).sum(dim=1)
        prod = e @ e.t()
        res = (e_square.unsqueeze(1) +
               e_square.unsqueeze(0) - 2 * prod).clamp(min=eps)

        if not squared:
            res = res.sqrt()

        res = res.clone()
        res[range(len(e)), range(len(e))] = 0

        if normalization == 'max':
            res_max = res.max() + eps
            res = res / res_max

        return res
