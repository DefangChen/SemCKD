from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F


class SemCKDLoss(nn.Module):
    # """Similarity-Preserving Knowledge Distillation, ICCV2019, verified by original author"""
    def __init__(self):
        super(SemCKDLoss, self).__init__()
        self.crit = nn.MSELoss(reduction='none')
        
    def forward(self, s_value, f_target, weight):
        loss = 0
        bsz, num_stu, num_tea = weight.shape
        ind_loss = torch.zeros(bsz, num_stu, num_tea).cuda()
        #print("bsz: " + str(bsz))
        #print("num_stu: " + str(num_stu))
        #print("num_tea: " +str(num_tea))
        #print("s_value: " + str(len(s_value)))
        for i in range(num_stu):
        #    print("s_value: " + str(i) + " : " + str(len(s_value[i])))
            for j in range(num_tea):
                ind_loss[:, i, j] = self.crit(s_value[i][j], f_target[i][j]).reshape(bsz,-1).mean(-1)

        loss = (weight * ind_loss).sum()/(1.0*bsz*num_stu)
        #f_s = torch.nn.functional.normalize(s_value[:,:,i].squeeze(), dim=1)
        #f_t = torch.nn.functional.normalize(f_target[:,:,i].squeeze(), dim=1)
        # G_diff = f_t - f_s
        # loss = loss + (G_diff * G_diff).view(-1, 1).mean()
        
        #G_diff = f_target[:,:,i].squeeze() - s_value[:,:,i].squeeze()
        #loss = loss + (G_diff * G_diff).view(-1, 1).mean()
        return loss