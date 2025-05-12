import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class AdaptiveGraphLayer(nn.Module):
    def __init__(self, in_features, out_features, num_keypoints=17, 
                 learn_graph=True, use_pose=True, dist_method='l2', 
                 gamma=0.1, k=4, temperature=0.1):

        super(AdaptiveGraphLayer, self).__init__()
        self.num_keypoints = num_keypoints
        self.learn_graph = learn_graph
        self.use_pose = use_pose
        self.dist_method = dist_method
        self.gamma = gamma
        self.k = k
        self.temperature = temperature
        
        self.fc = nn.Linear(in_features, out_features)
        
        if learn_graph:
            self.emb_q = nn.Linear(out_features, out_features)
            self.emb_k = nn.Linear(out_features, out_features)
            
        if use_pose:
            self.pose_weight = nn.Parameter(torch.ones(1))
            
    def get_sim_matrix(self, x):
        if self.dist_method == 'dot':
            emb_q = self.emb_q(x)
            emb_k = self.emb_k(x)
            sim_matrix = torch.bmm(emb_q, emb_k.transpose(1, 2))
        else:  # l2
            distmat = torch.pow(x, 2).sum(dim=2).unsqueeze(1) + \
                     torch.pow(x, 2).sum(dim=2).unsqueeze(2)
            distmat -= 2 * torch.bmm(x, x.transpose(1, 2))
            distmat = distmat.clamp(1e-12).sqrt()
            sim_matrix = torch.exp(-distmat / self.temperature)
            
        return sim_matrix
        
    def get_knn_graph(self, sim_matrix):
        N = sim_matrix.size(0)

        _, topk_indices = torch.topk(sim_matrix, self.k, dim=2)
        
        adj = torch.zeros_like(sim_matrix)
        for i in range(N):
            for j in range(self.num_keypoints):
                adj[i, j, topk_indices[i, j]] = 1
                
        adj = adj + torch.eye(self.num_keypoints, device=adj.device)
        
        return adj
        
    def forward(self, x, pose_adj=None):
        x = self.fc(x)
        
        if self.learn_graph:
            sim_matrix = self.get_sim_matrix(x)
            
            learned_adj = self.get_knn_graph(sim_matrix)
            
            if self.use_pose and pose_adj is not None:
                adj = (pose_adj + self.gamma * learned_adj) / (1 + self.gamma)
                adj = F.normalize(adj, p=1, dim=2)
            else:
                adj = learned_adj
        else:
            adj = pose_adj if pose_adj is not None else \
                  torch.eye(self.num_keypoints, device=x.device).unsqueeze(0).repeat(x.size(0), 1, 1)
                  
        return x, adj

class AdaptiveGraphBlock(nn.Module):
    def __init__(self, in_features, out_features, num_keypoints=17,
                 dropout=0.1, alpha=0.1, gamma=0.1, learn_graph=True,
                 use_pose=True, dist_method='l2', k=4, temperature=0.1):
        super(AdaptiveGraphBlock, self).__init__()
        
        self.graph_layer = AdaptiveGraphLayer(
            in_features, out_features, num_keypoints,
            learn_graph, use_pose, dist_method,
            gamma, k, temperature
        )
        
        self.bn = nn.BatchNorm1d(num_keypoints * out_features)
        self.dropout = nn.Dropout(dropout)
        self.alpha = alpha
        
    def forward(self, x, pose_adj=None):
        out, adj = self.graph_layer(x, pose_adj)
        
        if x.size(-1) == out.size(-1):
            out = out + self.alpha * x
            
        N, K, C = out.size()
        out = self.bn(out.view(N, -1)).view(N, K, C)
        
        # Dropout
        out = self.dropout(out)
        
        return out, adj 