import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from self_attention import gather_edges, gather_nodes, Normalize


class PositionalEncodings(nn.Module):
    def __init__(self, num_embeddings):
        super(PositionalEncodings, self).__init__()
        self.num_embeddings = num_embeddings

    def forward(self, E_idx):
  
        N_batch = E_idx.size(0)
        N_nodes = E_idx.size(1)
        N_neighbors = E_idx.size(2)
        ii = torch.arange(N_nodes, dtype=torch.float32).view((1, -1, 1)).cuda()
        d = (E_idx.float() - ii).unsqueeze(-1)
     
        frequency = torch.exp(
            torch.arange(0, self.num_embeddings, 2, dtype=torch.float32)
            * -(np.log(10000.0) / self.num_embeddings)).cuda()

        angles = d * frequency.view((1,1,1,-1))
        E = torch.cat((torch.cos(angles), torch.sin(angles)), -1)
        return E 


class EdgeFeatures(nn.Module):
    def __init__(self, edge_features, num_positional_embeddings=16,
        num_rbf=16, top_k=30, augment_eps=0.):
        super(EdgeFeatures, self).__init__()
        self.top_k = top_k
        self.augment_eps = augment_eps 
        self.num_rbf = num_rbf

       
        self.PE = PositionalEncodings(num_positional_embeddings)
        
      
        self.edge_embedding = nn.Linear(num_positional_embeddings + num_rbf + 7, edge_features, bias=True)
        self.norm_edges = Normalize(edge_features)

    def _dist(self, X, mask, eps=1E-6):
        """ Pairwise euclidean distances """
        mask_2D = torch.unsqueeze(mask,1) * torch.unsqueeze(mask,2) 
        dX = torch.unsqueeze(X,1) - torch.unsqueeze(X,2) 
        D = mask_2D * torch.sqrt(torch.sum(dX**2, 3) + eps) 

        D_max, _ = torch.max(D, -1, keepdim=True)
        D_adjust = D + (1. - mask_2D) * D_max
        D_neighbors, E_idx = torch.topk(D_adjust, self.top_k, dim=-1, largest=False) 

        return D_neighbors, E_idx

    def _rbf(self, D):
        
        D_min, D_max, D_count = 0., 20., self.num_rbf
        D_mu = torch.linspace(D_min, D_max, D_count).cuda()  
        D_mu = D_mu.view([1,1,1,-1])
        D_sigma = (D_max - D_min) / D_count
        D_expand = torch.unsqueeze(D, -1)
        RBF = torch.exp(-((D_expand - D_mu) / D_sigma)**2)

        return RBF 

    def _quaternions(self, R):
      
        diag = torch.diagonal(R, dim1=-2, dim2=-1)
        Rxx, Ryy, Rzz = diag.unbind(-1)
        magnitudes = 0.5 * torch.sqrt(torch.abs(1 + torch.stack([
              Rxx - Ryy - Rzz, 
            - Rxx + Ryy - Rzz, 
            - Rxx - Ryy + Rzz
        ], -1)))
        _R = lambda i,j: R[:,:,:,i,j]
        signs = torch.sign(torch.stack([
            _R(2,1) - _R(1,2),
            _R(0,2) - _R(2,0),
            _R(1,0) - _R(0,1)
        ], -1))
        xyz = signs * magnitudes
       
        w = torch.sqrt(F.relu(1 + diag.sum(-1, keepdim=True))) / 2.
        Q = torch.cat((xyz, w), -1)
        Q = F.normalize(Q, dim=-1)

        return Q

    def _orientations(self, X, E_idx, eps=1e-6):
      
        dX = X[:,1:,:] - X[:,:-1,:]
        U = F.normalize(dX, dim=-1) 
        u_2 = U[:,:-2,:] 
        u_1 = U[:,1:-1,:] 
        
        n_2 = F.normalize(torch.cross(u_2, u_1), dim=-1) 

        
        o_1 = F.normalize(u_2 - u_1, dim=-1) 
        O = torch.stack((o_1, n_2, torch.cross(o_1, n_2)), 2)
        O = O.view(list(O.shape[:2]) + [9])
        O = F.pad(O, (0,0,1,2), 'constant', 0)


        O_neighbors = gather_nodes(O, E_idx) 
        X_neighbors = gather_nodes(X, E_idx) 
        
       
        O = O.view(list(O.shape[:2]) + [3,3]) 
        O_neighbors = O_neighbors.view(list(O_neighbors.shape[:3]) + [3,3])

        
        dX = X_neighbors - X.unsqueeze(-2) 
        dU = torch.matmul(O.unsqueeze(2), dX.unsqueeze(-1)).squeeze(-1)
        dU = F.normalize(dU, dim=-1)
        R = torch.matmul(O.unsqueeze(2).transpose(-1,-2), O_neighbors) 
        Q = self._quaternions(R) 

        
        O_features = torch.cat((dU,Q), dim=-1) 

        return O_features


    def forward(self, X, mask): 
      
        if self.training and self.augment_eps > 0:
            X = X + self.augment_eps * torch.randn_like(X)  

        D_neighbors, E_idx = self._dist(X, mask)  
       
        RBF = self._rbf(D_neighbors)
        O_features = self._orientations(X, E_idx)

        E_positional = self.PE(E_idx)

        E = torch.cat((E_positional, RBF, O_features), -1)
        E = self.edge_embedding(E)
        E = self.norm_edges(E)

        return E, E_idx 
