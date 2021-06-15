import os
import torch
import numpy as np
from abc import ABC, abstractmethod
from os.path import dirname
import scipy.io

class BaseModel(ABC):
    def __init__(self, opt, log_file):
        self.opt = opt
        self.gpu_ids = opt.gpu_ids
        self.model_names = []
        # Seed and CUDA
        self.device = torch.device('cuda:{}'.format(self.gpu_ids[0])) if self.gpu_ids else torch.device('cpu')
        self.opt.device = self.device
        self.save_dir = dirname(log_file)
        posDistThr = 25
        self.posDistSqThr = posDistThr * posDistThr

    def set_input(self, batch):
        sate_ims = batch['satellite']
        pano_ims = batch['street']
        self.satellite = sate_ims.to(self.device)
        self.street = pano_ims.to(self.device)

    def set_input_cvact(self, data, utm):
        sate_ims = data['satellite']
        pano_ims = data['street']
        self.satellite = sate_ims.to(self.device)
        self.street = pano_ims.to(self.device)
        self.in_batch_dis = torch.zeros(utm.shape[0], utm.shape[0]).to(self.device)
        for k in range(utm.shape[0]):
            for j in range(utm.shape[0]):
                self.in_batch_dis[k, j] = (utm[k,0] - utm[j,0])*(utm[k,0] - utm[j,0]) + (utm[k, 1] - utm[j, 1])*(utm[k, 1] - utm[j, 1])


    def mutual_topk_acc(self, dists, topk=1):
        pos_dists = torch.diag(dists)
        N = len(pos_dists)
        # Distances smaller than positive pair
        dist_s2p = pos_dists.unsqueeze(1) - dists
        dist_p2s = pos_dists - dists

        acc_s2p = 100.0 * ((dist_s2p > 0).sum(1) < topk).sum().float() / N
        acc_p2s = 100.0 * ((dist_p2s > 0).sum(0) < topk).sum().float() / N
        return acc_p2s.item(), acc_s2p.item()

    def soft_margin_triplet_loss(self, sate_vecs, pano_vecs, loss_weight=10, hard_topk_ratio=1.0):
        dists = 2 - 2 * torch.matmul(sate_vecs, pano_vecs.permute(1, 0))  # Pairwise matches within batch
        pos_dists = torch.diag(dists)
        N = len(pos_dists)
        diag_ids = np.arange(N)
        num_hard_triplets = int(hard_topk_ratio * (N * (N - 1))) if hard_topk_ratio < 1.0 else N * (N - 1)

        # Match from satellite to street pano
        triplet_dist_s2p = pos_dists.unsqueeze(1) - dists
        loss_s2p = torch.log(1 + torch.exp(loss_weight * triplet_dist_s2p))
        loss_s2p[diag_ids, diag_ids] = 0  # Ignore diagnal losses

        if hard_topk_ratio < 1.0:  # Hard negative mining
            loss_s2p = loss_s2p.view(-1)
            loss_s2p, s2p_ids = torch.topk(loss_s2p, num_hard_triplets)
        loss_s2p = loss_s2p.sum() / num_hard_triplets

        # Match from street pano to satellite
        triplet_dist_p2s = pos_dists - dists
        loss_p2s = torch.log(1 + torch.exp(loss_weight * triplet_dist_p2s))
        loss_p2s[diag_ids, diag_ids] = 0  # Ignore diagnal losses

        if hard_topk_ratio < 1.0:  # Hard negative mining
            loss_p2s = loss_p2s.view(-1)
            loss_p2s, p2s_ids = torch.topk(loss_p2s, num_hard_triplets)
        loss_p2s = loss_p2s.sum() / num_hard_triplets
        # Total loss
        loss = (loss_s2p + loss_p2s) / 2.0
        return loss

    def compute_cvact_loss(self, sate_vecs, pano_vecs, utms_x, UTMthres, loss_weight=10, hard_topk_ratio=1.0):

        dists = 2 - 2 * torch.matmul(sate_vecs, pano_vecs.permute(1, 0))  # Pairwise matches within batch
        pos_dists = torch.diag(dists)
        N = len(pos_dists)
        diag_ids = np.arange(N)
        useful_pairs = torch.ge(utms_x[:,:], UTMthres)
        useful_pairs = useful_pairs.float()
        pair_n = useful_pairs.sum()
        num_hard_triplets = int(hard_topk_ratio * (N * (N - 1))) if int(hard_topk_ratio * (N * (N - 1))) < pair_n else pair_n

        # Match from satellite to street pano
        triplet_dist_s2p = (pos_dists.unsqueeze(1) - dists) * useful_pairs
        loss_s2p = torch.log(1 + torch.exp(loss_weight * triplet_dist_s2p))
        loss_s2p[diag_ids, diag_ids] = 0
        if num_hard_triplets != pair_n:
            loss_s2p = loss_s2p.view(-1)
            loss_s2p, s2p_ids = torch.topk(loss_s2p, num_hard_triplets)
        loss_s2p = loss_s2p.sum() / num_hard_triplets

        # Match from street pano to satellite
        triplet_dist_p2s = (pos_dists - dists) * useful_pairs
        loss_p2s = torch.log(1 + torch.exp(loss_weight * triplet_dist_p2s))
        loss_p2s[diag_ids, diag_ids] = 0
        if num_hard_triplets != pair_n:
            loss_p2s = loss_p2s.view(-1)
            loss_p2s, p2s_ids = torch.topk(loss_p2s, num_hard_triplets)
        loss_p2s = loss_p2s.sum() / num_hard_triplets

        # Total loss
        loss = (loss_s2p + loss_p2s) / 2.0
        return loss

    def load_weights(self, weights_dir, device, key='state_dict'):
        map_location = lambda storage, loc: storage.cuda(device.index) if torch.cuda.is_available() else storage
        weights_dict = None
        if weights_dir is not None:
            weights_dict = torch.load(weights_dir, map_location=map_location)
        return weights_dict

    def set_requires_grad(self, nets, requires_grad=False):

        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad

