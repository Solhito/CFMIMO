"""
model.py - Deep Unfolding GNN model definition
Complete WMMSE optimization and GNN architecture
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
import dgl.function as fn
import numpy as np
from typing import Dict, Tuple, List, Optional
import math

class BipartiteGraphBuilder:
    """Bipartite graph builder"""
    
    def __init__(self, L: int, K: int):
        self.L = L  # Number of APs
        self.K = K  # Number of UEs
    
    def build_graph(self, device: torch.device) -> dgl.DGLGraph:
        """Build UE-AP bipartite graph"""
        # Create heterogeneous graph
        graph_data = {
            ('UE', 'connects', 'AP'): (torch.tensor([], dtype=torch.long), torch.tensor([], dtype=torch.long)),
            ('AP', 'connected_by', 'UE'): (torch.tensor([], dtype=torch.long), torch.tensor([], dtype=torch.long))
        }
        
        graph = dgl.heterograph(graph_data).to(device)
        
        # Add nodes
        graph.add_nodes(self.K, ntype='UE')
        graph.add_nodes(self.L, ntype='AP')
        
        # Create fully connected edges
        src_nodes = []
        dst_nodes = []
        
        for k in range(self.K):
            for l in range(self.L):
                src_nodes.append(k)
                dst_nodes.append(l)
        
        src_nodes = torch.tensor(src_nodes, device=device)
        dst_nodes = torch.tensor(dst_nodes, device=device)
        
        # Add edges
        graph.add_edges(src_nodes, dst_nodes, etype=('UE', 'connects', 'AP'))
        graph.add_edges(dst_nodes, src_nodes, etype=('AP', 'connected_by', 'UE'))
        
        return graph
    
    def build_batched_graph(self, batch_size: int, device: torch.device) -> dgl.DGLGraph:
        """Build batched graph for multiple independent graphs"""
        graphs = []
        for _ in range(batch_size):
            g = self.build_graph(device)
            graphs.append(g)
        
        # Batch all graphs into one
        batched_graph = dgl.batch(graphs)
        self.batch_size = batch_size
        return batched_graph

class FeatureEncoder(nn.Module):
    """Feature encoder"""
    
    def __init__(self, config: Dict):
        super().__init__()
        
        self.tau_p = config["tau_p"]
        self.L = config["L"]
        self.K = config["K"]
        self.hidden_dim = config["hidden_dim"]
        
        # UE feature encoder [softmax(a), log(ρ), u_real, u_imag, σ²] - Equation (3)
        ue_feat_dim = self.tau_p + 4
        self.ue_encoder = nn.Sequential(
            nn.Linear(ue_feat_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.ReLU()
        )
        
        # AP feature encoder [P_l, C_l, avg_ρ, one-hot(l)] - Equation (4)
        ap_feat_dim = self.L + 3
        self.ap_encoder = nn.Sequential(
            nn.Linear(ap_feat_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.ReLU()
        )
        
        # Edge feature encoder [tr(R_kl), log(ρ_kl), ρ_kl/P_l] - Equation (5)
        edge_feat_dim = 3
        self.edge_encoder = nn.Sequential(
            nn.Linear(edge_feat_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.ReLU()
        )
    
    def forward(self, ue_feats: torch.Tensor, 
                ap_feats: torch.Tensor, 
                edge_feats: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Encode node and edge features"""
        h_ue = self.ue_encoder(ue_feats)
        h_ap = self.ap_encoder(ap_feats)
        e = self.edge_encoder(edge_feats)
        
        return h_ue, h_ap, e

class MessagePassingLayer(nn.Module):
    """Message passing layer"""
    
    def __init__(self, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        
        # Message computation MLPs
        self.msg_ap_to_ue = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        self.msg_ue_to_ap = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Node update GRUs
        self.ue_update = nn.GRUCell(hidden_dim, hidden_dim)
        self.ap_update = nn.GRUCell(hidden_dim, hidden_dim)
        
        # Edge update MLP
        self.edge_update = nn.Sequential(
            nn.Linear(3 * hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim)
        )
    
    def forward(self, batched_graph: dgl.DGLGraph, 
                h_ue: torch.Tensor, 
                h_ap: torch.Tensor, 
                e: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Single-layer message passing"""
        device = h_ue.device
        Batch_Size, K, H = h_ue.shape
        L = h_ap.shape[1]
        total_K = Batch_Size * K
        total_L = Batch_Size * L
        total_edges = Batch_Size * L * K
        
        h_ue_flat = h_ue.view(total_K, H)  # (B*K, H)
        h_ap_flat = h_ap.view(total_L, H)  # (B*L, H)
        e_flat = e.view(total_edges, H)    # (B*L*K, H)

        # Set node and edge features
        batched_graph.nodes['UE'].data['h'] = h_ue_flat
        batched_graph.nodes['AP'].data['h'] = h_ap_flat
        batched_graph.edges[('UE', 'connects', 'AP')].data['e'] = e_flat
        batched_graph.edges[('AP', 'connected_by', 'UE')].data['e'] = e_flat
        
        # AP -> UE message passing
        def ap_to_ue_msg(edges):
            h_src = edges.src['h']  # AP features
            e_data = edges.data['e']  # Edge features
            concat = torch.cat([h_src, e_data], dim=1)
            return {'m': self.msg_ap_to_ue(concat)}
        
        # UE -> AP message passing
        def ue_to_ap_msg(edges):
            h_src = edges.src['h']  # UE features
            e_data = edges.data['e']  # Edge features
            concat = torch.cat([h_src, e_data], dim=1)
            return {'m': self.msg_ue_to_ap(concat)}
        
        # AP -> UE aggregation
        batched_graph.update_all(ap_to_ue_msg, fn.mean('m', 'm_agg'), etype=('AP', 'connected_by', 'UE'))
        m_ue_flat = batched_graph.nodes['UE'].data['m_agg']
        
        # UE -> AP aggregation
        batched_graph.update_all(ue_to_ap_msg, fn.mean('m', 'm_agg'), etype=('UE', 'connects', 'AP'))
        m_ap_flat = batched_graph.nodes['AP'].data['m_agg']
        
        # Node update
        h_ue_new_flat = self.ue_update(m_ue_flat, h_ue_flat)
        h_ap_new_flat = self.ap_update(m_ap_flat, h_ap_flat)
        
        # Set the updated node features back into the graph so that they can be used during updates.
        batched_graph.nodes['UE'].data['h'] = h_ue_new_flat
        batched_graph.nodes['AP'].data['h'] = h_ap_new_flat
        
        # Edge update
        batched_graph.apply_edges(lambda edges: {
            'e': self.edge_update(torch.cat([
                edges.src['h'], edges.dst['h'], edges.data['e']
            ], dim=1))
        }, etype=('UE', 'connects', 'AP'))
        
        e_new_flat = batched_graph.edges[('UE', 'connects', 'AP')].data['e']
        
        h_ue_new = h_ue_new_flat.view(Batch_Size, K, H)
        h_ap_new = h_ap_new_flat.view(Batch_Size, L, H)
        e_new = e_new_flat.view(Batch_Size, L * K, H)
        
        return h_ue_new, h_ap_new, e_new

class UnfoldingLayer(nn.Module):
    """Deep unfolding layer"""
    
    def __init__(self, config: Dict, layer_idx: int):
        super().__init__()
        
        self.layer_idx = layer_idx
        self.config = config
        
        # System parameters
        self.L = config["L"]
        self.K = config["K"]
        self.N = config["N"]
        self.tau_p = config["tau_p"]
        self.tau_c = config["tau_c"]
        self.tau_d = config["tau_d"]
        self.sigma2 = config["sigma2"]
        self.P_max = config["P_max"]
        self.C_max = config["C_max"]
        
        # GNN components
        self.feature_encoder = FeatureEncoder(config)
        
        # Multi-layer message passing
        self.msg_passing_layers = nn.ModuleList([
            MessagePassingLayer(config["hidden_dim"], config.get("dropout", 0.1))
            for _ in range(config["msg_passing_layers"])
        ])
        
        # Output layer (Equations 37-42)
        self.output_layer = nn.ModuleDict({
            'alpha1': nn.Sequential(
                nn.Linear(config["hidden_dim"], config["hidden_dim"] // 2),
                nn.ReLU(),
                nn.Linear(config["hidden_dim"] // 2, 1)
            ),
            'alpha2': nn.Sequential(
                nn.Linear(config["hidden_dim"], config["hidden_dim"] // 2),
                nn.ReLU(),
                nn.Linear(config["hidden_dim"] // 2, 1)
            ),
            'logits': nn.Sequential(
                nn.Linear(config["hidden_dim"], config["hidden_dim"] // 2),
                nn.ReLU(),
                nn.Linear(config["hidden_dim"] // 2, config["tau_p"])
            ),
            'q_vector': nn.Sequential(
                nn.Linear(config["hidden_dim"], config["hidden_dim"] // 2),
                nn.ReLU(),
                nn.Linear(config["hidden_dim"] // 2, 1)
            ),
            'c_correction': nn.Sequential(
                nn.Linear(config["hidden_dim"], config["hidden_dim"] // 2),
                nn.ReLU(),
                nn.Linear(config["hidden_dim"] // 2, 1)
            )
        })
        
        # Temperature parameter (learnable)
        self.temperature = nn.Parameter(torch.tensor(1.0))
    
    def compute_psi_matrices(self, R_all: torch.Tensor, a_probs: torch.Tensor, eta: torch.Tensor) -> torch.Tensor:
        """
        Compute Ψ_{kl} matrices - Equation in Section 4.1
        Ψ_{kl} = ∑_{i=1}^{K} δ_{ki} τ_p η_i R_{il} + σ² I_N
        """
        device = R_all.device
        N = R_all.shape[3]
        
        # Get discrete pilot assignment (hard assignment from probabilities)
        pilot_indices = torch.argmax(a_probs, dim=2)  # (B, K)
        
        # Create delta matrix δ_{ki} = 1 if UE k and i use same pilot
        delta = (pilot_indices.unsqueeze(2) == pilot_indices.unsqueeze(1)).float()  # (B, K, K)
        
        # Compute weights: δ_{ki} * τ_p * η_i
        weights = delta * self.tau_p * eta.unsqueeze(1).to(torch.complex64)  # (B, K, K)
        
        # Compute weighted sum using einsum: ∑_i weights_{b,k,i} * R_all_{b,l,i}
        Psi_all = torch.einsum('bki,blimn->blkmn', weights, R_all)  # (B, L, K, N, N)
        
        # Add noise term
        eye_matrix = torch.eye(N, dtype=torch.complex64, device=device)
        Psi_all = Psi_all + self.sigma2 * eye_matrix.view(1, 1, 1, N, N)
        
        return Psi_all
    
    def compute_B_matrices(self, R_all: torch.Tensor, Psi_all: torch.Tensor) -> torch.Tensor:
        """
        Compute B_{kl} matrices - Equation in Section 4.1
        B_{kl} = τ_p η_k R_{kl} Ψ_{kl}^{-1} R_{kl}
        """
        device = R_all.device
        # _, L, K, N, _ = R_all.shape
        # eta = self.config.get("eta", torch.ones(K, device=device) * 0.1)


        Batch_Size, L, K, N, _ = R_all.shape
        eta = self.config.get("eta", torch.ones(K, device=device) * 0.1)
        
        # Reshape for batch processing: (B*L*K, N, N)
        R_flat = R_all.reshape(Batch_Size * L * K, N, N)
        Psi_flat = Psi_all.reshape(Batch_Size * L * K, N, N)
        
        # Batch inverse of Psi matrices with regularization
        Psi_inv_flat = torch.inverse(Psi_flat + 1e-22 * torch.eye(N, dtype=torch.complex64, device=device))
        
        # Compute B matrices: τ_p η_k R_kl Ψ_kl^{-1} R_kl
        # Using batch matrix multiplication
        B_flat = torch.bmm(torch.bmm(R_flat, Psi_inv_flat), R_flat)
        
        # Multiply by τ_p and η_k
        eta_expanded = eta.view(1, K, 1, 1).expand(Batch_Size, L, K, 1, 1)
        B_flat = B_flat.view(Batch_Size, L, K, N, N)
        B_all = self.tau_p * eta_expanded * B_flat
        
        return B_all
    
    def compute_trB(self, B_all: torch.Tensor) -> torch.Tensor:
        """Compute trace of B matrices - tr(B_{kl})"""
        return torch.diagonal(B_all, dim1=-2, dim2=-1).sum(dim=-1).real
    
    def compute_optimal_u(self, d: torch.Tensor, p: torch.Tensor, 
                         trB: torch.Tensor, R_all: torch.Tensor,
                         a_probs: torch.Tensor, eta: torch.Tensor) -> torch.Tensor:
        """
        Compute optimal receiver coefficients u_k
        
        u_k^opt = (∑_{l=1}^{L} d_{kl} √p_{kl} h_{kl}^H w_{kl}) / 
                  (∑_{i=1}^{K} |∑_{l=1}^{L} d_{il} √p_{il} h_{kl}^H w_{il}|² + σ_k²)
        
        Under MR combining: w_{kl} = \hat{h}_{kl} / √(tr(B_{kl}))
        And using channel statistics: E[h_{kl}^H w_{kl}] = √(tr(B_{kl}))
        """
        device = d.device
        B, L, K = d.shape
        N = self.N
        
        # 1. Compute numerator: Θ₁ = ∑_l d_{kl} √p_{kl} √(tr(B_{kl}))
        mask_n = d > 1e-22  # (B, L, K)
        numerator = torch.sum(mask_n * torch.sqrt(p) * torch.sqrt(trB), dim=1)  # (B, K)
        
        # 2. Compute pilot assignment and delta matrix
        pilot_indices = torch.argmax(a_probs, dim=2)  # (B, K)
        delta = (pilot_indices.unsqueeze(2) == pilot_indices.unsqueeze(1)).float()  # (B, K, K)
        delta_complex = delta.to(torch.complex64)
        
        # 3. Compute Psi inverses (already computed, so reuse)
        Psi_all = self.compute_psi_matrices(R_all, a_probs, eta)
        Psi_inv_all = torch.inverse(Psi_all + 1e-22 * torch.eye(N, dtype=torch.complex64, device=device))
        
        # 4. Precompute masks for served UEs
        mask_served = d > 1e-22  # (B, L, K)
        
        # 5. Compute denominator using vectorized operations
        denominator = torch.zeros(B, K, dtype=torch.complex64, device=device)
        
        for i in range(K):
            # Skip if UE i is not served by any AP
            mask_i = mask_served[:, :, i]  # (B, L)
            if mask_i.sum() == 0:
                continue
                
            # Compute terms for UE i
            trB_i = trB[:, :, i]  # (B, L)
            R_i = R_all[:, :, i]  # (B, L, N, N)
            p_i = p[:, :, i]  # (B, L)
            
            # Compute delta_ki for all k
            delta_ki = delta_complex[:, :, i]  # (B, K)
            
            for k in range(K):
                # Compute Psi inverses for UE k and i
                Psi_inv_k = Psi_inv_all[:, :, k]  # (B, L, N, N)
                Psi_inv_i = Psi_inv_all[:, :, i]  # (B, L, N, N)
                R_k = R_all[:, :, k]  # (B, L, N, N)
                
                # Compute term_l = tr(R_il @ Psi_inv_tk_l @ R_kl) for all l
                term_l = torch.einsum('blmn,blno,blom->bl', R_i, Psi_inv_k, R_k)  # (B, L)
                
                # Compute term_m = tr(R_km @ Psi_inv_ti_m @ R_im) for all l,m
                term_m = torch.einsum('blmn,blno,blom->bl', R_k, Psi_inv_i, R_i)  # (B, L)
                
                # Compute trace_term = tr(Psi_inv_tk_l @ R_il @ R_kl)
                trace_term = torch.einsum('blmn,blno,blom->bl', Psi_inv_k, R_i, R_k)  # (B, L)
                abs_trace_sq = torch.abs(trace_term) ** 2  # (B, L)
                
                # Compute termB = tr(R_kl @ R_il @ Psi_inv_ti_l @ R_il)
                termB = torch.einsum('blmn,blno,blop,blpm->bl', R_k, R_i, Psi_inv_i, R_i)  # (B, L)
                
                # Apply masks and compute contributions
                valid_mask = mask_i & (trB_i > 0)  # (B, L)
                
                if valid_mask.any():
                    # l = m term
                    part1 = delta_ki[:, k:k+1] * (self.tau_p ** 2) * eta[:, k:k+1] * eta[:, i:i+1] * abs_trace_sq / trB_i
                    part2 = self.tau_p * eta[:, i:i+1] * termB / trB_i
                    same_ap_term = torch.sum(p_i * valid_mask.float() * torch.real(part1 + part2), dim=1)
                    
                    # l ≠ m term (vectorized cross products)
                    term_l_expanded = term_l.unsqueeze(2)  # (B, L, 1)
                    term_m_expanded = term_m.unsqueeze(1)  # (B, 1, L)
                    trB_i_expanded_l = trB_i.unsqueeze(2)  # (B, L, 1)
                    trB_i_expanded_m = trB_i.unsqueeze(1)  # (B, 1, L)
                    
                    cross_term = delta_ki[:, k:k+1].unsqueeze(1) * (self.tau_p ** 2) * eta[:, k:k+1].unsqueeze(1) * eta[:, i:i+1].unsqueeze(1)
                    cross_term = cross_term * term_l_expanded * term_m_expanded
                    cross_term = cross_term / torch.sqrt(trB_i_expanded_l * trB_i_expanded_m + 1e-22)
                    
                    p_i_expanded_l = p_i.unsqueeze(2)  # (B, L, 1)
                    p_i_expanded_m = p_i.unsqueeze(1)  # (B, 1, L)
                    cross_weight = torch.sqrt(p_i_expanded_l * p_i_expanded_m)
                    
                    valid_mask_l = valid_mask.unsqueeze(2)  # (B, L, 1)
                    valid_mask_m = valid_mask.unsqueeze(1)  # (B, 1, L)
                    valid_cross_mask = valid_mask_l & valid_mask_m & ~torch.eye(L, device=device).bool().unsqueeze(0)
                    
                    cross_ap_term = torch.sum(cross_weight * valid_cross_mask.float() * torch.real(cross_term), dim=(1, 2))
                    
                    denominator[:, k] += same_ap_term + cross_ap_term
        
        # Add noise term
        denominator += self.sigma2
        
        # Compute optimal u_k
        u_opt = numerator / (denominator + 1e-22)
        
        return u_opt
    
    def compute_optimal_w(self, d: torch.Tensor, p: torch.Tensor, trB: torch.Tensor,
                         u: torch.Tensor, R_all: torch.Tensor,
                         a_probs: torch.Tensor, eta: torch.Tensor) -> torch.Tensor:
        """
        Compute optimal weights w_k for all batches in parallel.
        
        Args:
            d: (B, L, K) Association variables
            p: (B, L, K) Power allocation
            trB: (B, L, K) Trace of B matrices
            u: (B, K) Complex receiver coefficients (not used directly here but from u computation)
            R_all: (B, L, K, N, N) Channel covariance matrices
            a_probs: (B, K, tau_p) Pilot assignment probabilities
            eta: (B, K) Power control coefficients
            
        Returns:
            w_opt: (B, K) Optimal weights
        """
        device = d.device
        Batch_Size, L, K = d.shape
        N = self.N
        
        # 1. Compute Θ₁ = ∑_l d_{kl} √p_{kl} √(tr(B_{kl}))
        mask_n = d > 1e-22  # (B, L, K)
        theta1 = torch.sum(mask_n * torch.sqrt(p) * torch.sqrt(trB), dim=1)  # (B, K,)
        
        # 2. Compute pilot assignment and delta matrix
        pilot_indices = torch.argmax(a_probs, dim=2)  # (B, K)
        delta = (pilot_indices.unsqueeze(2) == pilot_indices.unsqueeze(1)).float()  # (B, K, K)
        delta_complex = delta.to(torch.complex64)
        
        # 3. Compute Psi and its inverses (reuse from compute_optimal_u_vectorized)
        Psi_all = self.compute_psi_matrices(R_all, a_probs, eta)
        Psi_inv_all = torch.inverse(Psi_all + 1e-22 * torch.eye(N, dtype=torch.complex64, device=device))
        
        # 4. Precompute masks for served UEs
        mask_served = d > 1e-22  # (B, L, K)
        
        # 5. Compute Θ₂ = Λ₂ + σ_k² using vectorized operations
        theta2 = torch.zeros(Batch_Size, K, dtype=torch.float32, device=device)
        
        for i in range(K):
            # Skip if UE i is not served by any AP
            mask_i = mask_served[:, :, i]  # (B, L)
            if mask_i.sum() == 0:
                continue
                
            # Compute terms for UE i
            trB_i = trB[:, :, i]  # (B, L)
            R_i = R_all[:, :, i]  # (B, L, N, N)
            p_i = p[:, :, i]  # (B, L)
            
            # Compute delta_ki for all k
            delta_ki = delta_complex[:, :, i]  # (B, K)
            
            for k in range(K):
                # Compute Psi inverses for UE k and i
                Psi_inv_k = Psi_inv_all[:, :, k]  # (B, L, N, N)
                Psi_inv_i = Psi_inv_all[:, :, i]  # (B, L, N, N)
                R_k = R_all[:, :, k]  # (B, L, N, N)
                
                # Compute term_l = tr(R_il @ Psi_inv_tk_l @ R_kl) for all l
                term_l = torch.einsum('blmn,blno,blom->bl', R_i, Psi_inv_k, R_k)  # (B, L)
                
                # Compute term_m = tr(R_km @ Psi_inv_ti_m @ R_im) for all l,m
                term_m = torch.einsum('blmn,blno,blom->bl', R_k, Psi_inv_i, R_i)  # (B, L)
                
                # Compute trace_term = tr(Psi_inv_tk_l @ R_il @ R_kl)
                trace_term = torch.einsum('blmn,blno,blom->bl', Psi_inv_k, R_i, R_k)  # (B, L)
                abs_trace_sq = torch.abs(trace_term) ** 2  # (B, L)
                
                # Compute termB = tr(R_kl @ R_il @ Psi_inv_ti_l @ R_il)
                termB = torch.einsum('blmn,blno,blop,blpm->bl', R_k, R_i, Psi_inv_i, R_i)  # (B, L)
                
                # Apply masks and compute contributions
                valid_mask = mask_i & (trB_i > 0)  # (B, L)
                
                if valid_mask.any():
                    # l = m term
                    part1 = delta_ki[:, k:k+1] * (self.tau_p ** 2) * eta[:, k:k+1] * eta[:, i:i+1] * abs_trace_sq / trB_i
                    part2 = self.tau_p * eta[:, i:i+1] * termB / trB_i
                    same_ap_term = torch.sum(p_i * valid_mask.float() * torch.real(part1 + part2), dim=1)
                    
                    # l ≠ m term (vectorized cross products)
                    term_l_expanded = term_l.unsqueeze(2)  # (B, L, 1)
                    term_m_expanded = term_m.unsqueeze(1)  # (B, 1, L)
                    trB_i_expanded_l = trB_i.unsqueeze(2)  # (B, L, 1)
                    trB_i_expanded_m = trB_i.unsqueeze(1)  # (B, 1, L)
                    
                    cross_term = delta_ki[:, k:k+1].unsqueeze(1) * (self.tau_p ** 2) * eta[:, k:k+1].unsqueeze(1) * eta[:, i:i+1].unsqueeze(1)
                    cross_term = cross_term * term_l_expanded * term_m_expanded
                    cross_term = cross_term / torch.sqrt(trB_i_expanded_l * trB_i_expanded_m + 1e-22)
                    
                    p_i_expanded_l = p_i.unsqueeze(2)  # (B, L, 1)
                    p_i_expanded_m = p_i.unsqueeze(1)  # (B, 1, L)
                    cross_weight = torch.sqrt(p_i_expanded_l * p_i_expanded_m)
                    
                    valid_mask_l = valid_mask.unsqueeze(2)  # (B, L, 1)
                    valid_mask_m = valid_mask.unsqueeze(1)  # (B, 1, L)
                    valid_cross_mask = valid_mask_l & valid_mask_m & ~torch.eye(L, device=device).bool().unsqueeze(0)
                    
                    cross_ap_term = torch.sum(cross_weight * valid_cross_mask.float() * torch.real(cross_term), dim=(1, 2))
                    
                    theta2[:, k] += same_ap_term + cross_ap_term
        
        # Add noise term
        theta2 += self.sigma2
        
        # 6. Compute minimum MSE: e_k^min = 1 - |Θ₁|² / Θ₂
        theta1_abs_sq = torch.abs(theta1) ** 2  # (B, K)
        e_min = 1.0 - theta1_abs_sq / (theta2 + 1e-22)
        
        # 7. Compute optimal weights: w_k^opt = 1 / e_k^min
        w_opt = 1.0 / (e_min + 1e-22)
        
        return w_opt
    
    def compute_node_features(self, rho: torch.Tensor, a_probs: torch.Tensor, 
                            u: torch.Tensor, sigma2: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute node and edge features for all batches in parallel.
        
        Args:
            rho: (B, L, K) Joint association-power variable ρ_{kl} = d_{kl} p_{kl}
            a_probs: (B, K, tau_p) Pilot assignment probabilities
            u: (B, K,) Complex receiver coefficients
            sigma2: (B,) Noise power
            
        Returns:
            ue_feats: (B, K, tau_p+4) UE features
            ap_feats: (B, L, L+3) AP features
            edge_feats: (B, L*K, 3) Edge features
        """
        device = rho.device
        Batch_Size, L, K = rho.shape
        
        # UE features
        # pilot_probs: (B, K, tau_p) already in correct shape
        pilot_probs = a_probs
        
        # log(ρ_k + ε) where ρ_k = ∑_l ρ_{kl}
        rho_k = torch.sum(rho, dim=1)  # (B, K)
        log_rho_k = torch.log(rho_k + 1e-22).unsqueeze(-1)  # (B, K, 1)
        
        # u_k real and imaginary parts
        u_real = u.real.unsqueeze(-1)  # (B, K, 1)
        u_imag = u.imag.unsqueeze(-1)  # (B, K, 1)
        
        # sigma2 expanded to match dimensions
        if sigma2.numel() == 1:
            sigma2_exp = torch.full((Batch_Size, K, 1), sigma2.item(), device=device, dtype=torch.float32)
        else:
            sigma2_exp = sigma2.view(Batch_Size, 1, 1).expand(Batch_Size, K, 1)
        
        # Concatenate all UE features
        ue_feats = torch.cat([
            pilot_probs,  # (B, K, tau_p)
            log_rho_k,    # (B, K, 1)
            u_real,       # (B, K, 1)
            u_imag,       # (B, K, 1)
            sigma2_exp    # (B, K, 1)
        ], dim=2)  # (B, K, tau_p+4)
        
        # AP features
        # Power constraint and fronthaul capacity (scalars)
        P_l = self.P_max
        C_l = self.C_max
        
        # Normalized total association: (1/K) ∑_k ρ_{kl}
        avg_rho = torch.sum(rho, dim=2) / K  # (B, L)
        
        # One-hot encoding of AP index
        one_hot = F.one_hot(torch.arange(L, device=device), L).float()  # (L, L)
        one_hot_batch = one_hot.unsqueeze(0).expand(Batch_Size, L, L)  # (B, L, L)
        
        # Power and capacity as tensors
        P_l_tensor = torch.full((Batch_Size, L, 1), P_l, device=device, dtype=torch.float32)
        C_l_tensor = torch.full((Batch_Size, L, 1), C_l, device=device, dtype=torch.float32)
        avg_rho = avg_rho.unsqueeze(-1)  # (B, L, 1)
        
        # Concatenate all AP features
        ap_feats = torch.cat([
            torch.cat([P_l_tensor, C_l_tensor, avg_rho], dim=2),  # (B, L, 3)
            one_hot_batch  # (B, L, L)
        ], dim=2)  # (B, L, L+3)
        
        # Edge features
        # Reshape to get all edge features
        # tr(R_{kl}) - placeholders (all ones)
        tr_R = torch.ones(Batch_Size, L, K, 1, device=device)  # (B, L, K, 1)
        
        # log(ρ_{kl} + ε)
        log_rho_kl = torch.log(rho + 1e-22).unsqueeze(-1)  # (B, L, K, 1)
        
        # ρ_{kl} / P_l (normalized power allocation)
        normalized_rho = rho.unsqueeze(-1) / self.P_max  # (B, L, K, 1)
        
        # Concatenate and reshape to (B, L*K, 3)
        edge_feats = torch.cat([tr_R, log_rho_kl, normalized_rho], dim=3)  # (B, L, K, 3)
        edge_feats = edge_feats.view(Batch_Size, L * K, 3)
        
        return ue_feats, ap_feats, edge_feats
    
    def compute_rho_closed_form(self, rho_prev: torch.Tensor, d: torch.Tensor, 
                           u: torch.Tensor, R_all: torch.Tensor, 
                           a_probs: torch.Tensor, eta: torch.Tensor) -> torch.Tensor:
        """
        ρ^{(n)}_{kl} = [ Re(u^{(n)}_k) * sqrt(tr(B_{kl})) / 
                    ( |u^{(n)}_k|^2 * D_{kl} + 
                        (|u^{(n)}_k|^2 / (2*(sqrt(ρ^{(n-1)}_{kl})+ε))) * 
                        ∑_{m≠l} d_{km} sqrt(ρ^{(n-1)}_{km}) ) ]^2
        
        Args:
            rho_prev: (B, L, K) Rho of the previous layer
            d: (B, L, K) Correlation variables
            u: (B, K) Complex receiver coefficients
            R_all: (B, L, K, N, N) Spatial correlation matrix
            a_probs: (B, K, tau_p) Pilot assignment probabilities
            eta: (B, K) Pilot power
            
        Returns:
            rho_new: (B, L, K)
        """
        device = rho_prev.device
        Batch_Size, L, K = rho_prev.shape
        N = self.N
        
        Psi_all = self.compute_psi_matrices(R_all, a_probs, eta)
        B_all = self.compute_B_matrices(R_all, Psi_all)
        trB = self.compute_trB(B_all)
        
        # D_{kl}
        D = self.compute_D_matrix(Psi_all, R_all, trB, eta)
        
        # Re(u_k) * sqrt(tr(B_{kl}))
        u_real = torch.real(u)  # (B, K)
        u_abs_sq = torch.abs(u) ** 2  # (B, K)
        
        # (B, L, K)
        u_real_expanded = u_real.unsqueeze(1).unsqueeze(2)  # (B, 1, 1, K)
        u_real_expanded = u_real_expanded.expand(-1, L, 1, K)  # (B, L, 1, K)
        u_real_expanded = u_real_expanded.squeeze(2)  # (B, L, K)
        
        u_abs_sq_expanded = u_abs_sq.unsqueeze(1).unsqueeze(2)  # (B, 1, 1, K)
        u_abs_sq_expanded = u_abs_sq_expanded.expand(-1, L, 1, K)  # (B, L, 1, K)
        u_abs_sq_expanded = u_abs_sq_expanded.squeeze(2)  # (B, L, K)
        sqrt_trB = torch.sqrt(trB)  # (B, L, K)
        
        numerator = u_real_expanded * sqrt_trB
        
        # |u_k|^2 * D_{kl}
        denom_part1 = u_abs_sq_expanded * D
        
        # (|u_k|^2 / (2*(sqrt(ρ^{(n-1)}_{kl})+ε))) * ∑_{m≠l} d_{km} sqrt(ρ^{(n-1)}_{km})
        sqrt_rho_prev = torch.sqrt(rho_prev + 1e-22)
        

        weighted_sum = torch.zeros(Batch_Size, L, K, device=device)
        for b in range(Batch_Size):
            for k in range(K):
                for l in range(L):
                    sum_val = 0.0
                    for m in range(L):
                        if m != l and d[b, m, k] > 1e-22:
                            sum_val += torch.sqrt(rho_prev[b, m, k] + 1e-22)
                    weighted_sum[b, l, k] = sum_val
        

        epsilon = 1e-22
        denom_part2 = u_abs_sq_expanded / (2 * (sqrt_rho_prev + epsilon)) * weighted_sum
        
        denominator = denom_part1 + denom_part2 + 1e-22
        
        rho_new = (numerator / denominator) ** 2
        rho_new = torch.clamp(rho_new, min=0)
        
        return rho_new

    def compute_D_matrix(self, Psi_all: torch.Tensor, R_all: torch.Tensor, 
                        trB: torch.Tensor, eta: torch.Tensor) -> torch.Tensor:
        """
        Calculate the D_{kl} matrix
        
        D_{kl} = (1/tr(B_{kl})) * Re( τ_p^2 η_k^2 |tr(Ψ_{kl}^{-1} R_{kl}^2)|^2 + 
                                    τ_p η_k tr(R_{kl}^2 Ψ_{kl}^{-1} R_{kl}) )
        
        Args:
            Psi_all: (B, L, K, N, N) Psi matrix
            R_all: (B, L, K, N, N) Spatial correlation matrix
            trB: (B, L, K) Trace of the B matrix
            eta: (B, K) Pilot power
            
        Returns:
            D: (B, L, K) D
        """
        device = Psi_all.device
        B, L, K, N, _ = R_all.shape
        
        eta_expanded = eta.unsqueeze(1).unsqueeze(3).unsqueeze(4)  # (B, 1, K, 1, 1)
        eta_expanded = eta_expanded.expand(B, L, K, N, N)
        
        # R_{kl}^2
        R_squared = torch.matmul(R_all, R_all)
        
        # Psi^{-1}
        Psi_inv_all = torch.inverse(Psi_all + 1e-22 * torch.eye(N, dtype=torch.complex64, device=device))
        
        # tr(Ψ_{kl}^{-1} R_{kl}^2)
        trace1 = torch.zeros(B, L, K, dtype=torch.complex64, device=device)
        for b in range(B):
            for l in range(L):
                for k in range(K):
                    mat_prod = torch.matmul(Psi_inv_all[b, l, k], R_squared[b, l, k])
                    trace1[b, l, k] = torch.trace(mat_prod)
        
        # |tr(Ψ_{kl}^{-1} R_{kl}^2)|^2
        abs_trace1_sq = torch.abs(trace1) ** 2
        
        # tr(R_{kl}^2 Ψ_{kl}^{-1} R_{kl})
        trace2 = torch.zeros(B, L, K, dtype=torch.complex64, device=device)
        for b in range(B):
            for l in range(L):
                for k in range(K):
                    # R^2 * Psi^{-1}
                    temp = torch.matmul(R_squared[b, l, k], Psi_inv_all[b, l, k])
                    # (R^2 * Psi^{-1}) * R
                    mat_prod = torch.matmul(temp, R_all[b, l, k])
                    # trace2[b, l, k] = torch.trace(mat_prod)
        
        # D_{kl}
        D = (1.0 / (trB + 1e-22)) * torch.real(
            (self.tau_p ** 2) * (eta.unsqueeze(1) ** 2) * abs_trace1_sq +
            self.tau_p * eta.unsqueeze(1) * trace2
        )
        
        return D
    
    def forward(self, batched_graph: dgl.DGLGraph, rho: torch.Tensor, 
                a_probs: torch.Tensor, u: torch.Tensor, w: torch.Tensor,
                R_all: torch.Tensor, eta: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass of single unfolding layer
        
        Returns:
            Dictionary containing updated variables
        """
        device = rho.device
        Batch_Size, L, K = rho.shape
        
        # Extract d_{kl} and p_{kl} from ρ_{kl} = d_{kl} p_{kl}
        # Use threshold to determine association
        d = (rho > 1e-22)  # Binary association variables
        p = torch.zeros_like(d, dtype=rho.dtype)
        p[d] = rho[d]
        d = d.float()
        
        # Compute B matrices and their traces
        Psi_all = self.compute_psi_matrices(R_all, a_probs, eta)
        B_all = self.compute_B_matrices(R_all, Psi_all)
        trB = self.compute_trB(B_all)
        
        # Step 1: Fixed-point update of u_k
        u_new = self.compute_optimal_u(d, p, trB, R_all, a_probs, eta)
        
        # Step 2: Compute node and edge features
        ue_feats, ap_feats, edge_feats = self.compute_node_features(
            rho, a_probs, u_new, torch.tensor(self.sigma2, device=device)
        )
        
        # Step 3: Feature encoding
        h_ue, h_ap, e = self.feature_encoder(ue_feats, ap_feats, edge_feats)
        
        # Step 4: Message passing
        for msg_layer in self.msg_passing_layers:
            h_ue, h_ap, e = msg_layer(batched_graph, h_ue, h_ap, e)
        
        # Step 5: Output layer
        alpha1 = self.output_layer['alpha1'](h_ue).squeeze(-1)  # (K,)
        alpha2 = self.output_layer['alpha2'](h_ue).squeeze(-1)  # (K,)
        logits = self.output_layer['logits'](h_ue)  # (K, tau_p)
        q_vector = self.output_layer['q_vector'](h_ue).squeeze(-1)  # (K,)
        c_correction = self.output_layer['c_correction'](h_ue).squeeze(-1)  # (K,)
        
        # Step 6: Update pilot assignment (with temperature parameter)
        a_probs_new = F.softmax(logits / self.temperature, dim=-1)
        

        # Step 7: Compute optimal weights (alternative method)
        w_opt = self.compute_optimal_w(d, p, trB, u_new, R_all, a_probs_new, eta)

        # Step 8: Update WMMSE weights
        w_new = w_opt * torch.exp(alpha1) + alpha2
        w_new = torch.clamp(w_new, min=1e-22, max=1e10)
        
        rho_new = self.compute_rho_closed_form(rho, d, u_new, R_all, a_probs_new, eta)

        for l in range(L):
            total_power = torch.sum(rho_new[:, l, :], dim=1)  # (B,)
            for b in range(Batch_Size):
                if total_power[b] > self.P_max:
                    rho_new[b, l, :] = rho_new[b, l, :] * self.P_max / total_power[b]
        
        # # Apply power constraints
        # for l in range(L):
        #     total_power = torch.sum(rho_new[l])
        #     if total_power > self.P_max:
        #         rho_new[l] = rho_new[l] * self.P_max / total_power
        
        # Ensure non-negativity
        rho_new = torch.clamp(rho_new, min=0)
        
        # Step 10: Extract d and p from updated ρ 
        d_new = (rho_new > 1e-22)  # Binary association variables
        p_new = torch.zeros_like(d_new, dtype=rho_new.dtype)
        p_new[d_new] = rho_new[d_new]
        d_new = d_new.float()
        
        return {
            'rho': rho_new,
            'd': d_new,
            'p': p_new,
            'a_probs': a_probs_new,
            'u': u_new,
            'w': w_new,
            'alpha1': alpha1,
            'alpha2': alpha2,
            'logits': logits,
            'q_vector': q_vector,
            'c_correction': c_correction
        }

class DeepUnfoldingGNN(nn.Module):
    """Deep Unfolding GNN model"""
    
    def __init__(self, config: Dict):
        super().__init__()
        
        self.config = config
        
        # System parameters
        self.L = config["L"]
        self.K = config["K"]
        self.N = config["N"]
        self.tau_p = config["tau_p"]
        self.tau_c = config["tau_c"]
        self.tau_d = config["tau_d"]
        self.sigma2 = config["sigma2"]
        self.P_max = config["P_max"]
        self.C_max = config["C_max"]
        self.eta_pilot = config.get("eta_pilot", 0.1)
        self.batch_size = config.get("batch_size")
        
        # Model parameters
        self.hidden_dim = config.get("hidden_dim", 64)
        self.msg_passing_layers = config.get("msg_passing_layers", 3)
        self.unfolding_layers = config.get("unfolding_layers", 5)
        self.dropout = config.get("dropout", 0.1)
        
        # Graph builder
        self.graph_builder = BipartiteGraphBuilder(self.L, self.K)
        
        # Initialize variables
        self._initialize_variables()
        
        # Deep unfolding layers
        self.unfolding_layers_list = nn.ModuleList([
            UnfoldingLayer(config, i) for i in range(self.unfolding_layers)
        ])
    
    def _initialize_variables(self):
        """Initialize optimization variables"""
        # ρ_kl - Joint association and power allocation variable
        self.init_rho = nn.Parameter(torch.rand(self.batch_size, self.L, self.K) * 0.1)
        
        # a_kt - Pilot assignment logits
        self.init_a_logits = nn.Parameter(torch.randn(self.batch_size, self.K, self.tau_p) * 1e-22)
        
        # u_k - WMMSE auxiliary variable (complex)
        self.init_u_real = nn.Parameter(torch.randn(self.batch_size, self.K) * 1e-22)
        self.init_u_imag = nn.Parameter(torch.randn(self.batch_size, self.K) * 1e-22)
        
        # w_k - WMMSE weight
        self.init_w = nn.Parameter(torch.ones(self.batch_size, self.K))
    
    def get_initial_u(self) -> torch.Tensor:
        """Get initial complex u"""
        device = self.init_u_real.device
        return torch.complex(self.init_u_real, self.init_u_imag).to(device)
    
    def compute_se_closed_form(self, d: torch.Tensor, p: torch.Tensor,
                                      a_probs: torch.Tensor, R_all: torch.Tensor,
                                      eta: torch.Tensor) -> torch.Tensor:
        """
        Compute closed-form spectral efficiency for all batches in parallel.
        
        Args:
            d: (B, L, K) Association variables
            p: (B, L, K) Power allocation
            a_probs: (B, K, tau_p) Pilot assignment probabilities
            R_all: (B, L, K, N, N) Channel covariance matrices
            eta: (B, K) Power control coefficients
            
        Returns:
            se: (B, K) Spectral efficiency for each UE
        """
        device = d.device
        B, L, K = d.shape
        N = self.N
        
        # 1. Compute B matrices and their traces
        Psi_all = self.compute_psi_matrices(R_all, a_probs, eta)
        B_all = self.compute_B_matrices(R_all, Psi_all)
        trB = torch.einsum('...ii', B_all).real  # (B, L, K)
        
        # 2. Get pilot assignment indices
        pilot_indices = torch.argmax(a_probs, dim=2)  # (B, K)
        
        # 3. Compute Psi inverses
        Psi_inv_all = torch.inverse(Psi_all + 1e-22 * torch.eye(N, dtype=torch.complex64, device=device))
        
        # 4. Precompute masks for served UEs
        mask_served = d > 1e-22  # (B, L, K)
        
        # 5. Compute numerator: |Θ₁|² = |∑_l d_{kl} √p_{kl} √(tr(B_{kl}))|²
        mask_k = mask_served  # For UE k
        theta1 = torch.sum(mask_k * torch.sqrt(p) * torch.sqrt(trB), dim=1)  # (B, K)
        numerator = torch.abs(theta1) ** 2  # (B, K)
        
        # 6. Compute denominator: Θ₂ = Λ₂ + σ_k² (same as in compute_optimal_u_vectorized)
        theta2 = torch.zeros(B, K, dtype=torch.float32, device=device)
        
        for i in range(K):
            # Skip if UE i is not served by any AP
            mask_i = mask_served[:, :, i]  # (B, L)
            if mask_i.sum() == 0:
                continue
                
            # Compute terms for UE i
            trB_i = trB[:, :, i]  # (B, L)
            R_i = R_all[:, :, i]  # (B, L, N, N)
            p_i = p[:, :, i]  # (B, L)
            
            # Compute delta_ki for all k
            delta = (pilot_indices == pilot_indices[:, i:i+1]).float()  # (B, K)
            
            for k in range(K):
                # Compute Psi inverses for UE k and i
                Psi_inv_k = Psi_inv_all[:, :, k]  # (B, L, N, N)
                Psi_inv_i = Psi_inv_all[:, :, i]  # (B, L, N, N)
                R_k = R_all[:, :, k]  # (B, L, N, N)
                
                # Compute term_l = tr(R_il @ Psi_inv_tk_l @ R_kl) for all l
                term_l = torch.einsum('blmn,blno,blom->bl', R_i, Psi_inv_k, R_k)  # (B, L)
                
                # Compute term_m = tr(R_km @ Psi_inv_ti_m @ R_im) for all l,m
                term_m = torch.einsum('blmn,blno,blom->bl', R_k, Psi_inv_i, R_i)  # (B, L)
                
                # Compute trace_term = tr(Psi_inv_tk_l @ R_il @ R_kl)
                trace_term = torch.einsum('blmn,blno,blom->bl', Psi_inv_k, R_i, R_k)  # (B, L)
                abs_trace_sq = torch.abs(trace_term) ** 2  # (B, L)
                
                # Compute termB = tr(R_kl @ R_il @ Psi_inv_ti_l @ R_il)
                termB = torch.einsum('blmn,blno,blop,blpm->bl', R_k, R_i, Psi_inv_i, R_i)  # (B, L)
                
                # Apply masks and compute contributions
                valid_mask = mask_i & (trB_i > 0)  # (B, L)
                
                if valid_mask.any():
                    # l = m term
                    part1 = delta[:, k:k+1] * (self.tau_p ** 2) * eta[:, k:k+1] * eta[:, i:i+1] * abs_trace_sq / trB_i
                    part2 = self.tau_p * eta[:, i:i+1] * termB / trB_i
                    same_ap_term = torch.sum(p_i * valid_mask.float() * torch.real(part1 + part2), dim=1)
                    
                    # l ≠ m term (vectorized cross products)
                    term_l_expanded = term_l.unsqueeze(2)  # (B, L, 1)
                    term_m_expanded = term_m.unsqueeze(1)  # (B, 1, L)
                    trB_i_expanded_l = trB_i.unsqueeze(2)  # (B, L, 1)
                    trB_i_expanded_m = trB_i.unsqueeze(1)  # (B, 1, L)
                    
                    cross_term = delta[:, k:k+1].unsqueeze(1) * (self.tau_p ** 2) * eta[:, k:k+1].unsqueeze(1) * eta[:, i:i+1].unsqueeze(1)
                    cross_term = cross_term * term_l_expanded * term_m_expanded
                    cross_term = cross_term / torch.sqrt(trB_i_expanded_l * trB_i_expanded_m + 1e-22)
                    
                    p_i_expanded_l = p_i.unsqueeze(2)  # (B, L, 1)
                    p_i_expanded_m = p_i.unsqueeze(1)  # (B, 1, L)
                    cross_weight = torch.sqrt(p_i_expanded_l * p_i_expanded_m)
                    
                    valid_mask_l = valid_mask.unsqueeze(2)  # (B, L, 1)
                    valid_mask_m = valid_mask.unsqueeze(1)  # (B, 1, L)
                    valid_cross_mask = valid_mask_l & valid_mask_m & ~torch.eye(L, device=device).bool().unsqueeze(0)
                    
                    cross_ap_term = torch.sum(cross_weight * valid_cross_mask.float() * torch.real(cross_term), dim=(1, 2))
                    
                    theta2[:, k] += same_ap_term + cross_ap_term
        
        # Add noise term to get Θ₂
        theta2 += self.sigma2
        
        # 7. Compute denominator: Θ₂ - numerator + σ_k²
        denominator = theta2 - numerator + self.sigma2
        
        # 8. Compute SINR
        sinr = numerator / (denominator + 1e-22)
        
        # 9. Compute SE = (τ_d/τ_c) log₂(1 + SINR)
        se = (self.tau_d / self.tau_c) * torch.log2(1.0 + sinr)
        
        return se
    
    def compute_psi_matrices(self, R_all: torch.Tensor, a_probs: torch.Tensor, eta: torch.Tensor) -> torch.Tensor:
        """
        Compute Ψ_{kl} matrices - Equation in Section 4.1
        Ψ_{kl} = ∑_{i=1}^{K} δ_{ki} τ_p η_i R_{il} + σ² I_N
        """
        device = R_all.device
        N = R_all.shape[3]
        
        # Get discrete pilot assignment (hard assignment from probabilities)
        pilot_indices = torch.argmax(a_probs, dim=2)  # (B, K)
        
        # Create delta matrix δ_{ki} = 1 if UE k and i use same pilot
        delta = (pilot_indices.unsqueeze(2) == pilot_indices.unsqueeze(1)).float()  # (B, K, K)
        
        # Compute weights: δ_{ki} * τ_p * η_i
        weights = delta * self.tau_p * eta.unsqueeze(1).to(torch.complex64)  # (B, K, K)
        
        # Compute weighted sum using einsum: ∑_i weights_{b,k,i} * R_all_{b,l,i}
        Psi_all = torch.einsum('bki,blimn->blkmn', weights, R_all)  # (B, L, K, N, N)
        
        # Add noise term
        eye_matrix = torch.eye(N, dtype=torch.complex64, device=device)
        Psi_all = Psi_all + self.sigma2 * eye_matrix.view(1, 1, 1, N, N)
        
        return Psi_all
    
    def compute_B_matrices(self, R_all: torch.Tensor, Psi_all: torch.Tensor) -> torch.Tensor:
        """
        Compute B_{kl} matrices - Equation in Section 4.1
        B_{kl} = τ_p η_k R_{kl} Ψ_{kl}^{-1} R_{kl}
        """
        device = R_all.device
        # _, L, K, N, _ = R_all.shape
        # eta = self.config.get("eta", torch.ones(K, device=device) * 0.1)


        Batch_Size, L, K, N, _ = R_all.shape
        eta = self.config.get("eta", torch.ones(K, device=device) * 0.1)
        
        # Reshape for batch processing: (B*L*K, N, N)
        R_flat = R_all.reshape(Batch_Size * L * K, N, N)
        Psi_flat = Psi_all.reshape(Batch_Size * L * K, N, N)
        
        # Batch inverse of Psi matrices with regularization
        Psi_inv_flat = torch.inverse(Psi_flat + 1e-22 * torch.eye(N, dtype=torch.complex64, device=device))
        
        # Compute B matrices: τ_p η_k R_kl Ψ_kl^{-1} R_kl
        # Using batch matrix multiplication
        B_flat = torch.bmm(torch.bmm(R_flat, Psi_inv_flat), R_flat)
        
        # Multiply by τ_p and η_k
        eta_expanded = eta.view(1, K, 1, 1).expand(Batch_Size, L, K, 1, 1)
        B_flat = B_flat.view(Batch_Size, L, K, N, N)
        B_all = self.tau_p * eta_expanded * B_flat
        
        return B_all
    
    def compute_trB(self, B_all: torch.Tensor) -> torch.Tensor:
        """Compute trace of B matrices - tr(B_{kl})"""
        return torch.diagonal(B_all, dim1=-2, dim2=-1).sum(dim=-1).real
    
    def forward(self, R_all: torch.Tensor, eta: torch.Tensor = None,
                return_history: bool = False) -> Dict[str, torch.Tensor]:
        """
        Forward pass
        
        Args:
            R_all: (L, K, N, N) Spatial correlation matrices
            eta: (K,) Uplink pilot power (if None, use default)
            return_history: Whether to return history
            
        Returns:
            Dictionary containing optimization results
        """
        device = R_all.device
        Batch_Size = R_all.shape[0]
        
        # Use default eta if not provided
        if eta is None:
            eta = torch.ones(self.K, device=device) * self.eta_pilot
        
        # Build graph
        # graph = self.graph_builder.build_graph(device)
        batched_graph = self.graph_builder.build_batched_graph(Batch_Size, device)
        
        # Initialize variables
        rho = torch.sigmoid(self.init_rho) * 0.5  # 0-0.5
        a_logits = self.init_a_logits
        a_probs = F.softmax(a_logits, dim=-1)
        u = self.get_initial_u()
        w = torch.sigmoid(self.init_w)  # 0-1
        
        # Store history
        history = {
            'rho': [],
            'a_probs': [],
            'u': [],
            'w': [],
            'se': [],
            'mse': []
        }
        
        # Deep unfolding iterations
        for layer_idx, unfolding_layer in enumerate(self.unfolding_layers_list):
            # Unfolding layer update
            outputs = unfolding_layer(batched_graph, rho, a_probs, u, w, R_all, eta)
            
            # Update variables
            rho = outputs['rho']
            a_probs = outputs['a_probs']
            u = outputs['u']
            w = outputs['w']
            
            # Extract d and p from ρ
            d = outputs['d']
            p = outputs['p']
            
            # Compute SE
            se = self.compute_se_closed_form(d, p, a_probs, R_all, eta)
            
            # Store history
            if return_history:
                history['rho'].append(rho.detach().clone())
                history['a_probs'].append(a_probs.detach().clone())
                history['u'].append(u.detach().clone())
                history['w'].append(w.detach().clone())
                history['se'].append(se.detach().clone())
            
            # Print progress
            if self.training and layer_idx % 2 == 0:
                print(f"Layer {layer_idx}: Sum SE = {se.sum().item():.4f}")
        
        # Final results
        d_final = (rho > 1e-22)  # Binary association variables
        p_final = torch.zeros_like(d_final, dtype=rho.dtype)
        p_final[d_final] = rho[d_final]
        d_final = d_final.float()
        
        se_final = self.compute_se_closed_form(d_final, p_final, a_probs, R_all, eta)
        
        # 1. Power constraint violation: sum_l max(0, ∑_k rho_{lk} - P_max)^2
        total_power = torch.sum(rho, dim=2)  # (B, L)
        power_violation_per_ap = F.relu(total_power - self.P_max)  # (B, L)
        power_violation = torch.sum(power_violation_per_ap ** 2, dim=1)  # (B,)
        
        # 2. Fronthaul constraint violation: sum_l max(0, ∑_k d_{lk} * SE_k - C_max)^2
        # d_final: (B, L, K), se_final: (B, K)
        fronthaul_load = torch.sum(d_final * se_final.unsqueeze(1), dim=2)  # (B, L)
        fronthaul_violation_per_ap = F.relu(fronthaul_load - self.C_max)  # (B, L)
        fronthaul_violation = torch.sum(fronthaul_violation_per_ap ** 2, dim=1)  # (B,)

        
        result = {
            'rho_final': rho,
            'd_final': d_final,
            'p_final': p_final,
            'a_probs_final': a_probs,
            'u_final': u,
            'w_final': w,
            'pilot_indices_final': torch.argmax(a_probs, dim=1),
            'se_final': se_final,
            'sum_se_final': torch.mean(torch.sum(se_final, dim=1)),
            'power_violation': power_violation,
            'fronthaul_violation': fronthaul_violation
        }
        
        if return_history:
            result['history'] = history
        
        return result
    
    def compute_loss(self, outputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Compute loss function - Equations (43)-(46)
        
        L = L_SE + λ₁ L_power + λ₂ L_capacity
        """
        # SE loss
        se_loss = -outputs['sum_se_final']  # Negative because we maximize SE
        
        # Power constraint penalty
        power_violation = torch.mean(outputs['power_violation'])
        
        # Fronthaul constraint penalty
        fronthaul_violation = torch.mean(outputs['fronthaul_violation'])
        
        # Loss weights
        lambda_power = 0.1
        lambda_fronthaul = 0.1
        
        # Total loss
        total_loss = se_loss + lambda_power * power_violation + lambda_fronthaul * fronthaul_violation
        
        # Record loss components
        self.loss_components = {
            'se_loss': se_loss.item(),
            'power_violation': power_violation.item(),
            'fronthaul_violation': fronthaul_violation.item(),
            'total_loss': total_loss.item()
        }
        
        return total_loss

def create_model(config: Dict, device: torch.device = None) -> DeepUnfoldingGNN:
    """Create model instance"""
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = DeepUnfoldingGNN(config)
    model = model.to(device)
    
    # Print model information
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model created:")
    print(f"  Device: {device}")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Number of APs: {model.L}")
    print(f"  Number of UEs: {model.K}")
    print(f"  Number of antennas per AP: {model.N}")
    print(f"  Unfolding layers: {model.unfolding_layers}")
    print(f"  Message passing layers: {model.msg_passing_layers}")
    print(f"  Hidden dimension: {model.hidden_dim}")
    print(f"  Pilot length: {model.tau_p}")
    print(f"  Coherence block length: {model.tau_c}")
    print(f"  Downlink data length: {model.tau_d}")
    
    return model

if __name__ == "__main__":
    # Test model
    config = {
        "L": 10,
        "K": 20,
        "N": 4,
        "tau_p": 5,
        "tau_c": 200,
        "tau_d": 195,
        "sigma2": 1e-15,
        "P_max": 1.0,
        "C_max": 2.0,
        "eta_pilot": 0.1,
        "hidden_dim": 64,
        "msg_passing_layers": 3,
        "unfolding_layers": 5,
        "dropout": 0.1
    }
    
    model = create_model(config)
    
    # Create test input
    L, K, N = config["L"], config["K"], config["N"]
    R_test = torch.randn(L, K, N, N, dtype=torch.cfloat)
    
    # Forward pass
    outputs = model(R_test, return_history=True)
    
    print(f"\nTest results:")
    print(f"  Final total SE: {outputs['sum_se_final'].item():.4f}")
    print(f"  Power allocation matrix shape: {outputs['rho_final'].shape}")
    print(f"  Pilot assignment matrix shape: {outputs['a_probs_final'].shape}")
    print(f"  Pilot indices: {outputs['pilot_indices_final']}")
    
    # Compute loss
    loss = model.compute_loss(outputs)
    print(f"  Total loss: {loss.item():.4f}")
    print(f"  Loss components: {model.loss_components}")