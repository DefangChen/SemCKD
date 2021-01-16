# https://github.com/KaiyuYue/mgd/blob/master/mgd/builder.py

import math
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from scipy.stats import norm
from ortools.graph import pywrapgraph


__all__ = [
    'MGDistiller',
    'SMDistiller',
    'get_margin_from_BN',
    'distillation_loss',
    'mgd_update',
]


class MGDistiller(nn.Module):
    """
    Matching Guided Distiller.

    Feature Reducer:
        rd  = Random Drop
        amp = Absolute Max Pooling
        sp  = Simple Pooling with Kernel MaxPool or AvgPool
    """

    def __init__(self,
                 t_net,
                 s_net,
                 t_channels=[],
                 s_channels=[],
                 t_ids=[],
                 s_ids=[],
                 ignore_inds=[],
                 reducer='amp',
                 sync_bn=False,
                 preReLU=True,
                 distributed=False,
                 ):
        super(MGDistiller, self).__init__()
        self.t_net = t_net
        self.s_net = s_net
        self.t_ids = t_ids
        self.s_ids = s_ids
        self.sync_bn = sync_bn
        self.preReLU = preReLU
        self.distributed = distributed
        self.ignore_inds = ignore_inds

        # select reducer
        self.reducer = getattr(self, reducer)

        # init vars
        self.t_channels = t_channels
        self.s_channels = s_channels

        # init margins
        self.init_margins()

        # build nets
        norm_layer = nn.SyncBatchNorm if self.sync_bn else nn.BatchNorm2d

        self.BNs = nn.ModuleList([norm_layer(
            32, s) if norm_layer == nn.GroupNorm else norm_layer(s) for s in self.s_channels])

        # init flow matrix
        self.init_flow()

    def init_margins(self):
        print('mgd info: init margins')
        q = self.t_net.get_bn_before_relu()
        q = [q[x-1] for x in self.t_ids]
        margins = [get_margin_from_BN(bn) for bn in q]
        for i, margin in enumerate(margins):
            self.register_buffer(
                'margin%d' % (i+1),
                margin.unsqueeze(1).unsqueeze(2).unsqueeze(0).detach().cuda()
            )

    def init_flow(self):
        print('mgd info: init flow')
        reminders = 0
        self.adj_matrix = []
        for s, t in zip(self.s_channels, self.t_channels):
            self.adj_matrix.append(np.zeros((s, t)))
            reminders += t % s

        # When the number of student channels can't be divisible by
        # the number of teacher channels, we shave the reminders.
        self.shave = False if reminders == 0 else True
        print('mgd info: shave matrix ? : {}'.format(self.shave))

        self.num_tracked_imgs = 0

    def extract_feature(self, x):
        t_feats, _ = self.t_net(x, is_feat=True, preact=self.preReLU)
        s_feats, _ = self.s_net(x, is_feat=True, preact=self.preReLU)
        self.track_running_stats([t_feats[x] for x in self.t_ids], [s_feats[x] for x in self.s_ids])

    def track_running_stats(self, t_feats, s_feats):
        feat_num = min(len(t_feats), len(s_feats))

        for i in range(feat_num):
            if i in self.ignore_inds:
                continue

            t_feat, s_feat = t_feats[i], s_feats[i]

            b, tc = t_feat.shape[0:2]
            _, sc = s_feat.shape[0:2]

            t_feat = F.normalize(t_feat.reshape(b, tc, -1), p=2, dim=2)
            s_feat = F.normalize(s_feat.reshape(b, sc, -1), p=2, dim=2)

            # print(s_feat.shape, t_feat.shape)
            cost = 2 - 2 * torch.bmm(s_feat, t_feat.transpose(1, 2))
            # print(self.adj_matrix[i].shape)
            # print(cost.sum(dim=0).cpu().data.numpy().shape)
            self.adj_matrix[i] += cost.sum(dim=0).cpu().data.numpy()

        self.num_tracked_imgs += b

    def update_flow(self):
        print('mgd info: update flow')
        feat_num = len(self.adj_matrix)

        self.guided_inds = []

        for i in range(feat_num):
            _col_ind = []

            sc, tc = self.adj_matrix[i].shape

            _adj_mat = []
            if sc != tc:
                _adj_mat = np.concatenate(
                    [self.adj_matrix[i] for _ in range(tc // sc)],
                    axis=0
                )
            else:
                _adj_mat = self.adj_matrix[i]

            cost = _adj_mat / self.num_tracked_imgs
            start = time.time()
            assignment = pywrapgraph.LinearSumAssignment()

            rows, cols = cost.shape

            # shave
            cols = rows if self.shave else cols

            for r in range(rows):
                for c in range(cols):
                    assignment.AddArcWithCost(r, c, int(1e5 * cost[r][c]))

            solve_status = assignment.Solve()
            if solve_status == assignment.OPTIMAL:
                _col_ind = [
                    assignment.RightMate(n)
                    for n in range(0, assignment.NumNodes())
                ]
                cost_sum = sum(assignment.AssignmentCost(n)
                               for n in range(0, assignment.NumNodes())
                               )
            print('mgd info: solve assignment for stage {}\tflow matrix shape: {}\ttime: {:.5f}\tcost: {:.5f}'.format(
                i, cost.shape, time.time()-start, 1e-5 * cost_sum)
            )

            if self.distributed:
                flow_inds = torch.from_numpy(
                    np.asarray(_col_ind)).long().cuda()
                # broadcast to all gpus
                torch.distributed.broadcast(flow_inds, src=0)
            else:
                flow_inds = torch.from_numpy(np.asarray(_col_ind)).long()
            self.guided_inds.append(flow_inds)

    def rd(self, i, t_feats, s_feats, margins):
        """
        Random Drop for channels reduction.
        """
        b, sc, h, w = s_feats[i].shape
        _, tc, _, _ = t_feats[i].shape

        groups = tc // sc

        t = []
        m = []

        for c in range(0, tc, sc):
            if c == (tc // sc) * sc and self.shave:
                continue

            t.append(t_feats[i][:, self.guided_inds[i][c:c+sc].detach(), :, :])
            m.append(margins[:, self.guided_inds[i][c:c+sc].detach(), :, :])

        t = torch.stack(t, dim=2)
        m = torch.stack(m, dim=2)

        t = t.reshape(b, sc, groups, -1).permute(0, 1, 3, 2)
        m = m.reshape(1, sc, groups, -1).permute(0, 1, 3, 2)

        # random drop mask
        _min = 0
        _max = groups

        t_mask = torch.randint(
            _min,
            _max,
            (b, sc, h * w, 1),
            dtype=torch.long,
            device=t.device
        )
        m_mask = torch.randint(
            _min,
            _max,
            (1, sc, 1, 1),
            dtype=torch.long,
            device=m.device
        )

        t = t.gather(3, t_mask)
        m = m.gather(3, m_mask)

        t = t.reshape(b, sc, h, w)
        m = m.reshape(1, sc, 1, 1)

        return t, m

    def amp(self, i, t_feats, s_feats, margins):
        """
        Absolute Max Pooling for channels reduction.
        """
        b, sc, h, w = s_feats[i].shape
        _, tc, _, _ = t_feats[i].shape

        groups = tc // sc

        t = []
        m = []
        for c in range(0, tc, sc):
            if c == (tc // sc) * sc and self.shave:
                continue

            t.append(t_feats[i][:, self.guided_inds[i][c:c+sc].detach(), :, :])
            m.append(margins[:, self.guided_inds[i][c:c+sc].detach(), :, :])

        t = torch.stack(t, dim=2)
        m = torch.stack(m, dim=2)

        t = t.reshape(b, sc, groups, -1)
        m = m.reshape(1, sc, groups, -1)

        t_inds = torch.argmax(t, dim=2)

        t = t.gather(2, t_inds.unsqueeze(2))
        m = m.mean(dim=2)

        t = t.reshape(b, sc, h, w)
        m = m.reshape(1, sc, 1, 1)

        return t, m

    def sp(self, i, t_feats, s_feats, margins, pooling_kernel='max'):
        """
        Simple Pooling for channels reduction, including max pooling and avg pooling.
        """
        b, sc, h, w = s_feats[i].shape
        _, tc, _, _ = t_feats[i].shape

        t = []
        m = []

        for c in range(0, tc, sc):
            if c == (tc // sc) * sc and len(self.ignore_inds) > 0:
                continue
            if c == (tc // sc) * sc and self.shave:
                continue

            t.append(t_feats[i][:, self.guided_inds[i][c:c+sc].detach(), :, :])
            m.append(margins[:, self.guided_inds[i][c:c+sc].detach(), :, :])

        t = torch.stack(t, dim=2)
        m = torch.stack(m, dim=2)

        # pooling_kernel: max F.adaptive_max_pool3d | avg F.adaptive_avg_pool3d
        t = F.adaptive_max_pool3d(t, (1, h, w)).squeeze(2)
        m = F.adaptive_max_pool3d(m, (1, 1, 1)).squeeze(2)

        return t, m

    def kd(self, t_out, s_out):
        """
        Knowledge Distillation: https://www.cs.toronto.edu/~hinton/absps/distillation.pdf

            - KD  distills student using final logits
            - MGD distills student using feature maps
        """
        loss_kd = F.kl_div(
            F.log_softmax(s_out/self.T, dim=1),
            F.softmax(t_out/self.T, dim=1).detach(),
            reduction='batchmean'
        ) * (self.T ** 2)

        return loss_kd

    def forward(self, s_feats, t_feats):
        t_feats = [t_feats[x] for x in self.t_ids]
        s_feats = [s_feats[x] for x in self.s_ids]
        
        feat_num = min(len(t_feats), len(s_feats))

        loss_factors = [2 ** (feat_num - i - 1) for i in range(feat_num)]
        loss_distill = 0
        for i in range(feat_num):
            if i in self.ignore_inds:
                continue

            # margins
            margins = getattr(self, 'margin%d' % (i+1))

            # bn for student features
            s_feats[i] = self.BNs[i](s_feats[i])

            # reduce channels
            t, m = self.reducer(i, t_feats, s_feats, margins)

            # accumulate loss
            loss_distill += distillation_loss(
                s_feats[i], t.detach(), m) / loss_factors[i]

        return loss_distill


class SMDistiller(nn.Module):
    """
    Matching Guided Distiller with Sparse Matching Reduction.
    """

    def __init__(self,
                 t_net,
                 s_net,
                 ignore_inds=[],
                 reducer='sm',
                 sync_bn=False,
                 with_kd=False,
                 preReLU=True,
                 distributed=False,
                 det=False
                 ):
        super(SMDistiller, self).__init__()
        self.t_net = t_net
        self.s_net = s_net
        self.sync_bn = sync_bn
        self.preReLU = preReLU
        self.distributed = distributed
        self.ignore_inds = ignore_inds
        self.det = det

        # kd setting
        self.with_kd = with_kd
        self.T = 1.5
        if self.with_kd:
            print(
                'mgd info: KD will be used together, KD Temperature = {:.3f}'.format(self.T))

        # select reducer
        assert reducer == 'sm'
        self.reducer = getattr(self, reducer)

        # init vars
        self.t_channels = self.t_net.get_channel_num()
        self.s_channels = self.s_net.get_channel_num()

        # init margins
        self.init_margins()

        # build nets
        norm_layer = nn.SyncBatchNorm if self.sync_bn else nn.BatchNorm2d

        if self.det:
            from detectron2.layers.batch_norm import NaiveSyncBatchNorm
            norm_layer = NaiveSyncBatchNorm if self.sync_bn else nn.GroupNorm

        self.BNs = nn.ModuleList([norm_layer(
            32, s) if norm_layer == nn.GroupNorm else norm_layer(s) for s in self.s_channels])

        # init flow matrix
        self.init_flow()

        # loss factors in retinanet
        if self.det:
            # [c2, c3, c4, c5, p3, p4, p5, p6, p7]
            self.loss_factors = list(np.asarray(
                [1, 1, 1e-1, 1e-1, 1, 1, 1e-2, 1e-1 * 0, 1e-1 * 0]) * 5e7)

    def init_margins(self):
        print('mgd info: init margins')
        margins = [get_margin_from_BN(bn)
                   for bn in self.t_net.get_bn_before_relu()]
        for i, margin in enumerate(margins):
            self.register_buffer(
                'margin%d' % (i+1),
                margin.unsqueeze(1).unsqueeze(2).unsqueeze(0).detach().cuda()
            )

    def init_flow(self):
        print('mgd info: init flow')
        t_channels = self.t_net.get_channel_num()
        s_channels = self.s_net.get_channel_num()
        self.adj_matrix = [
            np.zeros((s, t))
            for s, t in zip(s_channels, t_channels)
        ]
        self.num_tracked_imgs = 0

    def extract_feature(self, x):
        t_feats, _ = self.t_net.extract_feature(x, preReLU=self.preReLU)
        s_feats, _ = self.s_net.extract_feature(x, preReLU=self.preReLU)
        self.track_running_stats(t_feats, s_feats)

    def track_running_stats(self, t_feats, s_feats):
        feat_num = len(t_feats)

        for i in range(feat_num):
            if i in self.ignore_inds:
                continue

            t_feat, s_feat = t_feats[i], s_feats[i]

            b, tc = t_feat.shape[0:2]
            _, sc = s_feat.shape[0:2]

            t_feat = F.normalize(t_feat.reshape(b, tc, -1), p=2, dim=2)
            s_feat = F.normalize(s_feat.reshape(b, sc, -1), p=2, dim=2)

            cost = 2 - 2 * torch.bmm(s_feat, t_feat.transpose(1, 2))
            self.adj_matrix[i] += cost.sum(dim=0).cpu().data.numpy()

        self.num_tracked_imgs += b

    def update_flow(self):
        print('mgd info: update flow')
        feat_num = len(self.adj_matrix)

        self.guided_inds = []

        for i in range(feat_num):
            _col_ind = []

            sc, tc = self.adj_matrix[i].shape
            if sc != tc:
                _adj_mat = np.full((tc, tc), 1e10)
                _adj_mat[:sc, :tc] = self.adj_matrix[i]
            else:
                _adj_mat = self.adj_matrix[i]

            cost = _adj_mat / self.num_tracked_imgs
            start = time.time()
            assignment = pywrapgraph.LinearSumAssignment()

            rows, cols = cost.shape
            for r in range(rows):
                for c in range(cols):
                    assignment.AddArcWithCost(r, c, int(1e5 * cost[r][c]))

            solve_status = assignment.Solve()
            if solve_status == assignment.OPTIMAL:
                _col_ind = [assignment.RightMate(
                    n) for n in range(0, assignment.NumNodes())]
                cost_sum = 0
                for n in range(0, assignment.NumNodes()):
                    if n <= sc:
                        cost_sum += assignment.AssignmentCost(n)
            print('mgd info: solve assignment for {}\tshape: {}\ttime: {:.5f}\tcost: {:.5f}'.format(
                  i, cost.shape, time.time()-start, 1e-5 * cost_sum)
                  )

            if self.distributed:
                flow_inds = torch.from_numpy(
                    np.asarray(_col_ind)).long().cuda()
                # broadcast to all gpus
                torch.distributed.broadcast(flow_inds, src=0)
            else:
                flow_inds = torch.from_numpy(np.asarray(_col_ind)).long()
            self.guided_inds.append(flow_inds)

    def sm(self, i, t_feats, s_feats, margins):
        """
        Sparse Matching for channels reduction.
        """
        b, sc, h, w = s_feats[i].shape
        _, tc, _, _ = t_feats[i].shape

        # indices
        ind = self.guided_inds[i][:sc]

        # matching
        t = t_feats[i][:, ind.detach(), :, :]
        m = margins[:, ind.detach(), :, :]

        return t, m

    def kd(self, t_out, s_out):
        """
        Knowledge Distillation: https://www.cs.toronto.edu/~hinton/absps/distillation.pdf

            - KD  distills student using final logits
            - MGD distills student using feature maps
        """
        loss_kd = F.kl_div(
            F.log_softmax(s_out/self.T, dim=1),
            F.softmax(t_out/self.T, dim=1).detach(),
            reduction='batchmean'
        ) * (self.T ** 2)

        return loss_kd

    def forward(self, x):
        t_feats, t_out = self.t_net.extract_feature(x, preReLU=self.preReLU)
        s_feats, s_out = self.s_net.extract_feature(x, preReLU=self.preReLU)
        feat_num = len(t_feats)

        loss_factors = self.loss_factors if self.det else [
            2 ** (feat_num - i - 1) for i in range(feat_num)]
        loss_distill = 0
        for i in range(feat_num):
            if i in self.ignore_inds:
                continue

            # margins
            margins = getattr(self, 'margin%d' % (i+1))

            # bn for student features
            s_feats[i] = self.BNs[i](s_feats[i])

            # reduce channels
            t, m = self.reducer(i, t_feats, s_feats, margins)

            # accumulate loss
            loss_distill += distillation_loss(
                s_feats[i], t.detach(), m) / loss_factors[i]

        if self.with_kd:
            return s_out, [loss_distill, self.kd(t_out, s_out)]

        if self.det:
            s_out['loss_distill'] = loss_distill

        return s_out, loss_distill


def distillation_loss(source, target, margin):
    target = torch.max(target, margin)
    loss = torch.nn.functional.mse_loss(source, target, reduction="none")
    loss = loss * ((source > target) | (target > 0)).float()
    return loss.sum()


def get_margin_from_BN(bn):
    margin = []
    std = bn.weight.data
    mean = bn.bias.data
    for (s, m) in zip(std, mean):
        s = abs(s.item())
        m = m.item()
        if norm.cdf(-m / s) > 0.001:
            margin.append(- s * math.exp(- (m / s) ** 2 / 2) /
                          math.sqrt(2 * math.pi) / norm.cdf(-m / s) + m)
        else:
            margin.append(-3 * s)

    return torch.FloatTensor(margin).to(std.device)


def mgd_update(extra_loader, model, args):
    # switch to evaluate mode
    model.eval()

    # init flow
    model.init_flow()

    with torch.no_grad():
        for i, (images, _) in enumerate(extra_loader):
            images = images.cuda(args.gpu if args.multiprocessing_distributed else 0, non_blocking=True)

            # running for tracking status
            model.extract_feature(images)

            # break for ImageNet-1K
            if args.batch_size * i > 20000:
                break

        # update transpose/flow matrix
        model.update_flow()
