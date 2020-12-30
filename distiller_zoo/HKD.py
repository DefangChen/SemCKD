import numpy as np
import torch
import torch.nn as nn


def prob_loss(teacher_features, student_features, eps=1e-6, kernel_parameters={}):
    """Heterogeneous Knowledge Distillation using Information Flow Modeling, CVPR2020
    Code from author: https://github.com/passalis/pkth/blob/master/nn/pkt_transfer.py"""
    # Teacher kernel
    if kernel_parameters['teacher'] == 'rbf':
        teacher_d = pairwise_distances(teacher_features)
        if 'teacher_sigma' in kernel_parameters:
            sigma = kernel_parameters['teacher_sigma']
        else:
            sigma = 1
        teacher_s = torch.exp(-teacher_d / sigma)
    elif kernel_parameters['teacher'] == 'adaptive_rbf':
        teacher_d = pairwise_distances(teacher_features)
        sigma = torch.mean(teacher_d).detach()
        teacher_s = torch.exp(-teacher_d / sigma)
    elif kernel_parameters['teacher'] == 'cosine':
        teacher_s = cosine_pairwise_similarities(teacher_features)
    elif kernel_parameters['teacher'] == 'student_t':
        teacher_d = pairwise_distances(teacher_features)
        if 'teacher_d' in kernel_parameters:
            d = kernel_parameters['teacher_d']
        else:
            d = 1
        teacher_s = 1.0 / (1 + teacher_d ** d)
    elif kernel_parameters['teacher'] == 'cauchy':
        teacher_d = pairwise_distances(teacher_features)
        if 'teacher_sigma' in kernel_parameters:
            sigma = kernel_parameters['teacher_sigma']
        else:
            sigma = 1
        teacher_s = 1.0 / (1 + (teacher_d ** 2 / sigma ** 2))
    elif kernel_parameters['teacher'] == 'combined':
        teacher_d = pairwise_distances(teacher_features)
        if 'teacher_d' in kernel_parameters:
            d = kernel_parameters['teacher_d']
        else:
            d = 1
        teacher_s_2 = 1.0 / (1 + teacher_d ** d)
        teacher_s_1 = cosine_pairwise_similarities(teacher_features)
    else:
        assert False

    # Student kernel
    if kernel_parameters['student'] == 'rbf':
        student_d = pairwise_distances(student_features)
        if 'student_sigma' in kernel_parameters:
            sigma = kernel_parameters['student_sigma']
        else:
            sigma = 1
        student_s = torch.exp(-student_d / sigma)

    elif kernel_parameters['student'] == 'adaptive_rbf':
        student_d = pairwise_distances(student_features)
        sigma = torch.mean(student_d).detach()
        student_s = torch.exp(-student_d / sigma)

    elif kernel_parameters['student'] == 'cosine':
        student_s = cosine_pairwise_similarities(student_features)

    elif kernel_parameters['student'] == 'student_t':
        student_d = pairwise_distances(student_features)
        if 'student_d' in kernel_parameters:
            d = kernel_parameters['student_d']
        else:
            d = 1
        student_s = 1.0 / (1 + student_d ** d)

    elif kernel_parameters['student'] == 'cauchy':
        student_d = pairwise_distances(student_features)
        if 'student_sigma' in kernel_parameters:
            sigma = kernel_parameters['student_sigma']
        else:
            sigma = 1
        student_s = 1.0 / (1 + (student_d ** 2 / sigma ** 2))

    elif kernel_parameters['student'] == 'combined':
        student_d = pairwise_distances(student_features)
        if 'student_d' in kernel_parameters:
            d = kernel_parameters['student_d']
        else:
            d = 1
        student_s_2 = 1.0 / (1 + student_d ** d)
        student_s_1 = cosine_pairwise_similarities(student_features)
    else:
        assert False

    if kernel_parameters['teacher'] == 'combined':
        # Transform them into probabilities
        teacher_s_1 = teacher_s_1 / torch.sum(teacher_s_1, dim=1, keepdim=True)
        student_s_1 = student_s_1 / torch.sum(student_s_1, dim=1, keepdim=True)

        teacher_s_2 = teacher_s_2 / torch.sum(teacher_s_2, dim=1, keepdim=True)
        student_s_2 = student_s_2 / torch.sum(student_s_2, dim=1, keepdim=True)

    else:
        # Transform them into probabilities
        teacher_s = teacher_s / torch.sum(teacher_s, dim=1, keepdim=True)
        student_s = student_s / torch.sum(student_s, dim=1, keepdim=True)

    if 'loss' in kernel_parameters:
        if kernel_parameters['loss'] == 'kl':
            loss = teacher_s * torch.log(eps + (teacher_s) / (eps + student_s))
        elif kernel_parameters['loss'] == 'abs':
            loss = torch.abs(teacher_s - student_s)
        elif kernel_parameters['loss'] == 'squared':
            loss = (teacher_s - student_s) ** 2
        elif kernel_parameters['loss'] == 'jeffreys':
            loss = (teacher_s - student_s) * (torch.log(teacher_s) - torch.log(student_s))
        elif kernel_parameters['loss'] == 'exponential':
            loss = teacher_s * (torch.log(teacher_s) - torch.log(student_s)) ** 2
        elif kernel_parameters['loss'] == 'kagan':
            loss = ((teacher_s - student_s) ** 2) / teacher_s
        elif kernel_parameters['loss'] == 'combined':
            # Jeffrey's  combined
            loss1 = (teacher_s_1 - student_s_1) * (torch.log(teacher_s_1) - torch.log(student_s_1))
            loss2 = (teacher_s_2 - student_s_2) * (torch.log(teacher_s_2) - torch.log(student_s_2))
        else:
            assert False
    else:
        loss = teacher_s * torch.log(eps + (teacher_s) / (eps + student_s))

    if 'loss' in kernel_parameters and kernel_parameters['loss'] == 'combined':
        loss = torch.mean(loss1) + torch.mean(loss2)
    else:
        loss = torch.mean(loss)

    return loss


def pairwise_distances(a, b=None, eps=1e-6):
    """
    Calculates the pairwise distances between matrices a and b (or a and a, if b is not set)
    :param a:
    :param b:
    :return:
    """
    if b is None:
        b = a

    aa = torch.sum(a ** 2, dim=1)
    bb = torch.sum(b ** 2, dim=1)

    aa = aa.expand(bb.size(0), aa.size(0)).t()
    bb = bb.expand(aa.size(0), bb.size(0))

    AB = torch.mm(a, b.transpose(0, 1))

    dists = aa + bb - 2 * AB
    dists = torch.clamp(dists, min=0, max=np.inf)
    dists = torch.sqrt(dists + eps)
    return dists


def cosine_pairwise_similarities(features, eps=1e-6, normalized=True):
    features_norm = torch.sqrt(torch.sum(features ** 2, dim=1, keepdim=True))
    features = features / (features_norm + eps)
    features[features != features] = 0
    similarities = torch.mm(features, features.transpose(0, 1))

    if normalized:
        similarities = (similarities + 1.0) / 2.0
    return similarities


class HKDLoss(nn.Module):
    """Heterogeneous Knowledge Distillation using Information Flow Modeling, CVPR2020"""
    def __init__(self, init_weight=100, decay=0.7):
        super(HKDLoss, self).__init__()
        self.init_weight = init_weight
        self.decay = decay

    def forward(self, f_s, f_t):
        kernel_parameters = {'teacher': 'combined', 'student': 'combined', 'loss': 'combined'}
        for i, (teacher, student) in enumerate(zip(f_t, f_s)):
            teacher = teacher.view(teacher.shape[0], -1)
            student = student.view(student.shape[0], -1)
            if i == 0:
                weight = self.init_weight
                loss = weight * prob_loss(teacher, student, kernel_parameters=kernel_parameters)
            else:
                weight *= self.decay
                loss += prob_loss(teacher, student, kernel_parameters=kernel_parameters)
        return loss