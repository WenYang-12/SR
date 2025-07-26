import torch
import torch.nn as nn
import torch.nn.functional as F
from gnn import GCN, GraphConv, GAT, Transformer, GIN


class SimpleFFN(nn.Module):
    """简化版FFN，用于全局特征处理"""
    def __init__(self, channels, dropout=0.1):
        super().__init__()
        self.lin1 = nn.Linear(channels, channels)
        self.lin2 = nn.Linear(channels, channels)
        self.norm = nn.LayerNorm(channels)
        self.dropout = nn.Dropout(dropout)
    def forward(self, x):
        residual = x
        x = self.norm(x)
        x = self.dropout(x)
        x = F.relu(self.lin1(x))
        x = self.lin2(x) + residual
        return x


class DummyTransformer(nn.Module):
    """占位的多头自注意力（可替换为更复杂的实现）"""
    def __init__(self, channels, n_head=1, dropout=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(channels, n_head, dropout=dropout, batch_first=True)
    def forward(self, x):
        # x: [B, 1, C] or [B, N, C]
        out, _ = self.attn(x, x, x)
        return out


class FeatureFusionLayer(nn.Module):
    """参考BGALayer的融合方式：节点特征+patch全局特征+transformer+FFN+拼接+线性融合+残差"""
    def __init__(self, channels, dropout=0.1, n_head=1):
        super().__init__()
        self.node_norm = nn.LayerNorm(channels)
        self.dg_norm = nn.LayerNorm(channels)
        self.dg_transformer = DummyTransformer(channels, n_head, dropout)
        self.dg_ffn = SimpleFFN(channels, dropout)
        self.fuse_lin = nn.Linear(2 * channels, channels)
        self.dropout = nn.Dropout(dropout)

    def forward(self, original_features, dg_features, batch):
        # 1. 节点特征归一化
        x = self.node_norm(original_features)
        # 2. 按batch分组，计算每个patch的全局特征
        num_patches = batch.max().item() + 1
        dg_global = []
        for i in range(num_patches):
            mask = (batch == i) # 找到属于分区i的所有节点
            if mask.sum() == 0:
                dg_global.append(torch.zeros((1, x.size(1)), device=x.device, dtype=x.dtype))
            else:
                mean_feat = dg_features[mask].mean(dim=0, keepdim=True)
                dg_global.append(mean_feat)
        dg_global = torch.cat(dg_global, dim=0)  # [num_patches, C]
        dg_global = self.dg_norm(dg_global).unsqueeze(1)   # [num_patches, 1, C]
        dg_global = self.dg_transformer(dg_global)         # [num_patches, 1, C]
        dg_global = self.dg_ffn(dg_global).squeeze(1)      # [num_patches, C]
        # 3. 每个节点拼接自己patch的全局特征
        dg_global_per_node = dg_global[batch]  # 关键步骤：根据batch索引获取对应的全局特征
        z = torch.cat([x, dg_global_per_node], dim=1)  # [N, 2C]
        # 4. 线性融合+残差
        fused = F.relu(self.fuse_lin(z)) + x
        fused = self.dropout(fused)
        return fused


class DualGraphFusionModel(nn.Module):
    """双图融合模型：原图GNN + DG连接图GNN + 特征融合"""
    
    def __init__(self, num_nodes, in_channels, hidden_channels, out_channels,
                 activation, gnn_layers=2, gnn_type='gcn', n_heads=8, 
                 use_bn=False, dropout=0.5, use_fusion=True):
        super(DualGraphFusionModel, self).__init__()
        
        self.num_nodes = num_nodes
        self.use_fusion = use_fusion

        # 原图GNN
        if gnn_type == 'gcn':
            self.original_gnn = GCN(in_channels, hidden_channels, hidden_channels, 
                                   activation, k=gnn_layers, use_bn=use_bn)
        elif gnn_type == 'graphconv':
            self.original_gnn = GraphConv(in_channels, hidden_channels, hidden_channels, 
                                         num_layers=gnn_layers, dropout=dropout, use_bn=use_bn)
        elif gnn_type == 'gat':
            self.original_gnn = GAT(in_channels, hidden_channels, hidden_channels, 
                                   activation, n_heads=n_heads, k=gnn_layers, use_bn=use_bn)
        elif gnn_type == 'transformer':
            self.original_gnn = Transformer(in_channels, hidden_channels, hidden_channels, 
                                          activation, k=gnn_layers, use_bn=use_bn, n_heads=n_heads)
        else:
            raise ValueError(f"Unsupported GNN type: {gnn_type}")
        
        # DG连接图GNN（使用相同的架构）
        if gnn_type == 'gcn':
            self.dg_gnn = GCN(in_channels, hidden_channels, hidden_channels, 
                             activation, k=gnn_layers, use_bn=use_bn)
        elif gnn_type == 'graphconv':
            self.dg_gnn = GraphConv(in_channels, hidden_channels, hidden_channels, 
                                   num_layers=gnn_layers, dropout=dropout, use_bn=use_bn)
        elif gnn_type == 'gat':
            self.dg_gnn = GAT(in_channels, hidden_channels, hidden_channels, 
                             activation, n_heads=n_heads, k=gnn_layers, use_bn=use_bn)
        elif gnn_type == 'transformer':
            self.dg_gnn = Transformer(in_channels, hidden_channels, hidden_channels, 
                                    activation, k=gnn_layers, use_bn=use_bn, n_heads=n_heads)
        
        # 特征融合层
        if use_fusion:
            self.fusion_layer = FeatureFusionLayer(hidden_channels, dropout)
            fusion_input_dim = hidden_channels
        else:
            fusion_input_dim = 2 * hidden_channels
        
        # 最终分类器
        self.classifier = nn.Linear(fusion_input_dim, out_channels)
        
    def forward(self, x, original_edge_index, dg_edge_index, batch=None):
        """
        Args:
            x: 节点特征 [N, in_channels]
            original_edge_index: 原图的边索引 [2, E_orig]
            dg_edge_index: DG连接图的边索引 [2, E_dg]
        Returns:
            output: 分类输出 [N, out_channels]
            original_features: 原图GNN特征 [N, hidden_channels]
            dg_features: DG连接图GNN特征 [N, hidden_channels]
            fused_features: 融合特征 [N, hidden_channels] (如果use_fusion=True)
        """
        # 原图GNN
        original_features = self.original_gnn(x, original_edge_index)
        
        # DG连接图GNN
        dg_features = self.dg_gnn(x, dg_edge_index)
        
        if self.use_fusion:
            # 特征融合
            fused_features = self.fusion_layer(original_features, dg_features, batch)
            output = self.classifier(fused_features)
            return output, original_features, dg_features, fused_features
        else:
            # 直接拼接
            combined_features = torch.cat([original_features, dg_features], dim=1)
            output = self.classifier(combined_features)
            return output, original_features, dg_features, combined_features


class SimpleConcatFusionModel(nn.Module):
    """原图GNN和DG GNN输出简单拼接的融合模型"""
    def __init__(self, num_nodes, in_channels, hidden_channels, out_channels,
                 activation, gnn_layers=2, gnn_type='gcn', n_heads=8, use_bn=False, dropout=0.5):
        super(SimpleConcatFusionModel, self).__init__()
        # 原图GNN
        if gnn_type == 'gcn':
            self.original_gnn = GCN(in_channels, hidden_channels, out_channels, activation, k=gnn_layers, use_bn=use_bn)
        elif gnn_type == 'graphconv':
            self.original_gnn = GraphConv(in_channels, hidden_channels, out_channels, num_layers=gnn_layers, dropout=dropout, use_bn=use_bn)
        elif gnn_type == 'gat':
            self.original_gnn = GAT(in_channels, hidden_channels, out_channels, activation, n_heads=n_heads, k=gnn_layers, use_bn=use_bn)
        elif gnn_type == 'transformer':
            self.original_gnn = Transformer(in_channels, hidden_channels, out_channels, activation, k=gnn_layers, use_bn=use_bn, n_heads=n_heads)
        else:
            raise ValueError(f"Unsupported GNN type: {gnn_type}")
        # DG连接图GNN
        if gnn_type == 'gcn':
            self.dg_gnn = GCN(in_channels, hidden_channels, out_channels, activation, k=gnn_layers, use_bn=use_bn)
        elif gnn_type == 'graphconv':
            self.dg_gnn = GraphConv(in_channels, hidden_channels, out_channels, num_layers=gnn_layers, dropout=dropout, use_bn=use_bn)
        elif gnn_type == 'gat':
            self.dg_gnn = GAT(in_channels, hidden_channels, out_channels, activation, n_heads=n_heads, k=gnn_layers, use_bn=use_bn)
        elif gnn_type == 'transformer':
            self.dg_gnn = Transformer(in_channels, hidden_channels, out_channels, activation, k=gnn_layers, use_bn=use_bn, n_heads=n_heads)
        else:
            raise ValueError(f"Unsupported GNN type: {gnn_type}")
        # 拼接后的线性层
        self.concat_linear = nn.Linear(2 * out_channels, out_channels)
        self.dropout = nn.Dropout(dropout)
    def forward(self, x, original_edge_index, dg_edge_index, batch=None):
        # 原图GNN输出
        original_out = self.original_gnn(x, original_edge_index)
        # DG连接图GNN输出
        dg_out = self.dg_gnn(x, dg_edge_index)
        # 简单拼接
        concat = torch.cat([original_out, dg_out], dim=-1)
        out = self.concat_linear(concat)
        out = self.dropout(F.relu(out))
        # 返回4个值，兼容训练流程
        return out, original_out, dg_out, concat




def get_fusion_model(model_type='dual', **kwargs):
    """获取融合模型的工厂函数"""
    if model_type == 'dual':
        return DualGraphFusionModel(**kwargs)
    elif model_type == 'simple_concat':
        return SimpleConcatFusionModel(**kwargs)
    elif model_type == 'cotraining':
        return CoTrainingFusionModel(**kwargs)
    else:
        raise ValueError(f"Unsupported model type: {model_type}") 