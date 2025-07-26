# -*- coding: utf-8 -*-
import torch
import networkx as nx
import numpy as np
from scipy.spatial import Delaunay
from torch_geometric.utils import to_undirected, to_networkx, from_networkx
import community  # python-louvain
import umap
from torch_geometric.data import Data
import matplotlib.pyplot as plt
from sklearn.metrics import normalized_mutual_info_score
from sklearn.decomposition import PCA
from gnn import GCN

def metis_partition(g, n_patches):
    """使用METIS进行图分区"""
    import networkx as nx
    import numpy as np
    from torch_geometric.utils import to_networkx
    
    # 转换为NetworkX图
    if isinstance(g, Data):
        G = to_networkx(g, to_undirected=True)
    else:
        G = nx.Graph()
        edges = g.graph['edge_index'].t().numpy()
        G.add_edges_from(edges)
    
    # 使用METIS进行分区
    try:
        import metis
        _, parts = metis.part_graph(G, n_patches)
        
        # 检查分区结果
        if len(parts) == 0:
            print("警告：METIS分区失败，使用随机分区")
            parts = np.random.randint(0, n_patches, size=len(G.nodes()))
        
        # 将分区结果转换为列表
        partition = [[] for _ in range(n_patches)]
        for node, part in enumerate(parts):
            partition[part].append(node)
            
        # 打印每个分区的大小
        for i, part in enumerate(partition):
            print(f"  - 分区 {i} 大小: {len(part)}")
            
        return partition
        
    except Exception as e:
        print(f"警告：METIS分区失败: {str(e)}，使用随机分区")
        # 使用随机分区
        parts = np.random.randint(0, n_patches, size=len(G.nodes()))
        partition = [[] for _ in range(n_patches)]
        for node, part in enumerate(parts):
            partition[part].append(node)
        return partition

def louvain_partition(g, n_patches=None):
    """使用Louvain算法进行图分区（保留原生分区数）"""
    import community as community_louvain
    import networkx as nx
    
    # 转换为networkx图
    edge_list = g.edge_index.t().numpy().tolist()
    G = nx.Graph(edge_list)
    
    # 使用Louvain算法进行分区
    partition = community_louvain.best_partition(G)
    
    # 获取所有实际分区ID
    unique_part_ids = sorted(set(partition.values()))
    part_id_map = {pid: idx for idx, pid in enumerate(unique_part_ids)}
    parts = [[] for _ in range(len(unique_part_ids))]
    for node, part_id in partition.items():
        mapped_id = part_id_map[part_id]
        parts[mapped_id].append(node)
    
    return parts

def visualize_graphs(original_graph, new_graph, parts, n_patches):
    """可视化原始图、分区图和重连后的图"""
    import matplotlib.pyplot as plt
    import networkx as nx
    import numpy as np
    
    # 创建图形
    plt.figure(figsize=(20, 6))
    
    # 定义鲜明的颜色方案
    colors = ['#FF0000', '#00FF00', '#0000FF', '#FFA500', '#800080', '#00FFFF', '#FF00FF', '#008000']
    
    # 1. 绘制原始图
    plt.subplot(131)
    G_original = nx.Graph()
    if isinstance(original_graph, Data):
        edges = original_graph.edge_index.t().numpy()
    else:
        edges = original_graph.graph['edge_index'].t().numpy()
    G_original.add_edges_from(edges)
    
    # 添加所有节点
    G_original.add_nodes_from(range(original_graph.num_nodes))
    
    # 使用固定的布局
    pos = nx.spring_layout(G_original, seed=42, k=1.0)
    nx.draw(G_original, pos, node_size=30, node_color='blue', alpha=0.6, width=1.0)
    plt.title('Original Graph', fontsize=14, pad=10)
    
    # 2. 绘制分区图（只显示团内部的连接）
    plt.subplot(132)
    G_partition = nx.Graph()
    G_partition.add_nodes_from(range(original_graph.num_nodes))
    
    # 为每个节点分配颜色
    node_colors = np.zeros(original_graph.num_nodes)
    edge_index = original_graph.edge_index.t().numpy()
    
    # 添加团内部的边
    for i in range(n_patches):
        partition_nodes = parts[i]
        for node in partition_nodes:
            if node < original_graph.num_nodes:
                node_colors[node] = i
                # 添加团内部的边
                for edge in edge_index:
                    if edge[0] == node and edge[1] in partition_nodes:
                        G_partition.add_edge(edge[0], edge[1])
    
    # 使用自定义的颜色映射
    cmap = plt.cm.colors.ListedColormap(colors[:n_patches])
    nx.draw(G_partition, pos, node_size=30, node_color=node_colors, 
            cmap=cmap, alpha=0.6, width=1.0)
    
    # 添加图例
    legend_elements = [plt.Line2D([0], [0], marker='o', color='w', 
                                 markerfacecolor=colors[i], label=f'Partition {i}',
                                 markersize=10) for i in range(n_patches)]
    plt.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1.3, 1))
    plt.title('Partitioned Graph', fontsize=14, pad=10)
    
    # 3. 绘制重连后的图
    plt.subplot(133)
    G_new = nx.Graph()
    if isinstance(new_graph, Data):
        edges = new_graph.edge_index.t().numpy()
    else:
        edges = new_graph.graph['edge_index'].t().numpy()
    G_new.add_edges_from(edges)
    
    # 确保节点顺序一致
    G_new.add_nodes_from(range(original_graph.num_nodes))
    
    # 使用相同的节点颜色
    nx.draw(G_new, pos, node_size=30, node_color=node_colors, 
            cmap=cmap, alpha=0.6, width=1.0)
    plt.title('Rewired Graph', fontsize=14, pad=10)
    
    plt.tight_layout()
    plt.show()

def combine_louvain_delaunay(data, n_patches, return_parts=False, visualize=True):
    """结合Louvain分区和Delaunay重连
    
    Args:
        data: PyG Data对象
        n_patches: 目标分区数量
        return_parts: 是否返回分区信息
        visualize: 是否可视化图结构
    
    Returns:
        new_data: 重连后的PyG Data对象
        parts: 如果return_parts为True，返回分区信息
    """
    print("  - 开始Louvain分区...")
    parts = louvain_partition(data, n_patches)
    
    # 打印每个分区的大小
    for i, part in enumerate(parts):
        print(f"  - 分区 {i} 大小: {len(part)}")
    
    # 计算每个分区的中心点
    partition_centers = []
    for i, part in enumerate(parts):
        print(f"  - 处理分区 {i+1}/{len(parts)}")
        if len(part) > 0:
            print("  - 使用PCA进行降维...")
            # 获取分区内的节点特征
            part_features = data.x[part].numpy()
            
            # 使用PCA降维到2D
            pca = PCA(n_components=2)
            coords_2d = pca.fit_transform(part_features)
            
            # 计算分区的中心点
            center = np.mean(coords_2d, axis=0)
            partition_centers.append(center)
        else:
            print(f"  - 警告：分区 {i} 为空，使用随机位置")
            partition_centers.append(np.random.rand(2))
    
    print("使用Delaunay三角剖分连接不同分区...")
    
    # 使用Delaunay三角剖分连接分区中心点
    partition_centers = np.array(partition_centers)
    tri = Delaunay(partition_centers)
    
    # 创建新的边
    new_edges = []
    for i, j, k in tri.simplices:
        # 选择最相似的节点对
        if i < len(parts) and j < len(parts):
            # 计算两个分区中所有节点对之间的余弦相似度
            part_i_features = data.x[parts[i]].numpy()
            part_j_features = data.x[parts[j]].numpy()

            # 计算余弦相似度矩阵
            similarity = np.dot(part_i_features, part_j_features.T)
            norm_i = np.linalg.norm(part_i_features, axis=1, keepdims=True)
            norm_j = np.linalg.norm(part_j_features, axis=1, keepdims=True)
            similarity = similarity / (norm_i * norm_j.T)

            # 找到最相似的节点对
            max_sim_idx = np.unravel_index(similarity.argmax(), similarity.shape)
            node1 = parts[i][max_sim_idx[0]]
            node2 = parts[j][max_sim_idx[1]]
            new_edges.append([node1, node2])

        if j < len(parts) and k < len(parts):
            part_j_features = data.x[parts[j]].numpy()
            part_k_features = data.x[parts[k]].numpy()
            similarity = np.dot(part_j_features, part_k_features.T)
            norm_j = np.linalg.norm(part_j_features, axis=1, keepdims=True)
            norm_k = np.linalg.norm(part_k_features, axis=1, keepdims=True)
            similarity = similarity / (norm_j * norm_k.T)
            max_sim_idx = np.unravel_index(similarity.argmax(), similarity.shape)
            node1 = parts[j][max_sim_idx[0]]
            node2 = parts[k][max_sim_idx[1]]
            new_edges.append([node1, node2])

        if i < len(parts) and k < len(parts):
            part_i_features = data.x[parts[i]].numpy()
            part_k_features = data.x[parts[k]].numpy()
            similarity = np.dot(part_i_features, part_k_features.T)
            norm_i = np.linalg.norm(part_i_features, axis=1, keepdims=True)
            norm_k = np.linalg.norm(part_k_features, axis=1, keepdims=True)
            similarity = similarity / (norm_i * norm_k.T)
            max_sim_idx = np.unravel_index(similarity.argmax(), similarity.shape)
            node1 = parts[i][max_sim_idx[0]]
            node2 = parts[k][max_sim_idx[1]]
            new_edges.append([node1, node2])

    # 创建新的边索引
    new_edge_index = torch.cat([
        data.edge_index,
        torch.tensor(new_edges).t().contiguous()
    ], dim=1)

    #------------------------------------去除相似度连接---------------------------
    # # 计算每个节点属于哪个patch
    # node2patch = {}
    # for patch_id, part in enumerate(parts):
    #     for node in part:
    #         node2patch[node] = patch_id
    #
    # # 遍历原始边，找出所有跨patch的边
    # edge_index_np = data.edge_index.cpu().numpy()
    # cross_patch_edges = []
    # for idx in range(edge_index_np.shape[1]):
    #     u, v = edge_index_np[0, idx], edge_index_np[1, idx]
    #     if node2patch.get(u) is not None and node2patch.get(v) is not None:
    #         if node2patch[u] != node2patch[v]:
    #             cross_patch_edges.append([u, v])
    #
    # # 只保留原始边
    # new_edge_index = data.edge_index

    # 创建新的数据对象
    new_data = Data(
        x=data.x,
        edge_index=new_edge_index,
        y=data.y
    )
    
    # 可视化图结构
    if visualize and data.num_nodes <= 100:  # 只对小型图进行可视化
        print("\n可视化图结构...")
        visualize_graphs(data, new_data, parts, n_patches)
    
    if return_parts:
        return new_data, parts
    return new_data

def combine_louvain_random_rewire(data, dg_edge_index=None, num_random_edges=None, return_parts=False, visualize=True):
    """Louvain分区后，随机连接分区间边，边数可自定义，默认与DG连接图一致"""
    print("  - 开始Louvain分区...")
    parts = louvain_partition(data)
    for i, part in enumerate(parts):
        print(f"  - 分区 {i} 大小: {len(part)}")

    # 统计所有分区对（只考虑不同分区）
    partition_pairs = []
    for i in range(len(parts)):
        for j in range(i+1, len(parts)):
            if len(parts[i]) > 0 and len(parts[j]) > 0:
                partition_pairs.append((i, j))

    # 决定随机边数
    if num_random_edges is not None:
        num_inter_edges = num_random_edges
    elif dg_edge_index is not None:
        dg_edges = dg_edge_index.t().cpu().numpy()
        # 计算每个节点属于哪个分区
        node2part = {}
        for idx, nodes in enumerate(parts):
            for node in nodes:
                node2part[node] = idx
        inter_edges = []
        for u, v in dg_edges:
            if node2part.get(u, -1) != node2part.get(v, -1):
                inter_edges.append((u, v))
        num_inter_edges = len(inter_edges)
    else:
        num_inter_edges = len(parts) * 2

    print(f"  - 目标随机分区间边数: {num_inter_edges}")

    # 随机生成分区间边
    rng = np.random.default_rng()
    random_edges = []
    for _ in range(num_inter_edges):
        # 随机选两个不同分区
        i, j = rng.choice(len(parts), size=2, replace=False)
        node_i = rng.choice(parts[i])
        node_j = rng.choice(parts[j])
        random_edges.append([node_i, node_j])

    # 创建新的边索引
    new_edge_index = torch.cat([
        data.edge_index,
        torch.tensor(random_edges, dtype=torch.long).t().contiguous()
    ], dim=1)

    new_data = Data(
        x=data.x,
        edge_index=new_edge_index,
        y=data.y
    )
    if hasattr(data, 'train_mask'):
        new_data.train_mask = data.train_mask
        new_data.val_mask = data.val_mask
        new_data.test_mask = data.test_mask

    if visualize:
        print("可视化随机重连图结构...（略）")
        # 可选：调用visualize_graphs(new_data, ...)

    if return_parts:
        return new_data, parts
    else:
        return new_data

def visualize_degree_distribution(edge_index, title, ax=None):
    """可视化图的度分布
    
    Args:
        edge_index: 边索引
        title: 图表标题
        ax: 可选的子图对象
    """
    import matplotlib.pyplot as plt
    import numpy as np
    
    # 计算每个节点的度
    degrees = {}
    for edge in edge_index.t().numpy():
        degrees[edge[0]] = degrees.get(edge[0], 0) + 1
        degrees[edge[1]] = degrees.get(edge[1], 0) + 1
    
    # 转换为列表
    degree_list = list(degrees.values())
    
    # 计算度分布
    max_degree = max(degree_list)
    degree_counts = np.zeros(max_degree + 1)
    for degree in degree_list:
        degree_counts[degree] += 1
    
    # 计算概率
    degree_probs = degree_counts / len(degree_list)
    
    # 绘制度分布
    if ax is None:
        plt.figure(figsize=(10, 6))
        ax = plt.gca()
    
    ax.bar(range(len(degree_probs)), degree_probs, alpha=0.6)
    ax.set_xlabel('Degree')
    ax.set_ylabel('Probability')
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    
    if ax is None:
        plt.tight_layout()
        plt.show()

def compare_graphs(original_edge_index, new_edge_index, original_parts=None, new_parts=None):
    """比较两个图的指标"""
    # 创建临时的Data对象来计算指标
    original_data = Data(edge_index=original_edge_index)
    new_data = Data(edge_index=new_edge_index)
    
    # 计算指标
    original_metrics = calculate_graph_metrics(original_data, original_parts)
    new_metrics = calculate_graph_metrics(new_data, new_parts)
    
    print("\n图结构比较:")
    print("-" * 50)
    for metric in original_metrics:
        if metric in ['num_nodes', 'is_connected']:  # 跳过这些指标
            continue
            
        print(f"{metric}:")
        print(f"  原始图: {original_metrics[metric]:.4f}")
        print(f"  重连图: {new_metrics[metric]:.4f}")
        # 处理除零情况
        if original_metrics[metric] == 0:
            if new_metrics[metric] == 0:
                print("  变化率: 0.00% (无变化)")
            else:
                print("  变化率: ∞ (从0开始增加)")
        else:
            change_rate = ((new_metrics[metric] - original_metrics[metric]) / original_metrics[metric] * 100)
            print(f"  变化率: {change_rate:.2f}%")
        print("-" * 50)
    
    # 可视化比较
    plt.figure(figsize=(15, 5))
    
    # 度分布比较
    plt.subplot(1, 2, 1)
    original_degrees = [val for (node, val) in nx.Graph(original_edge_index.t().numpy().tolist()).degree()]
    new_degrees = [val for (node, val) in nx.Graph(new_edge_index.t().numpy().tolist()).degree()]
    
    plt.hist(original_degrees, bins=range(min(original_degrees), max(original_degrees) + 1),
             alpha=0.5, label='Original', color='blue')
    plt.hist(new_degrees, bins=range(min(new_degrees), max(new_degrees) + 1),
             alpha=0.5, label='Rewired', color='red')
    plt.xlabel('Degree')
    plt.ylabel('Number of Nodes')
    plt.title('Degree Distribution Comparison')
    plt.legend()
    
    # 指标比较条形图
    plt.subplot(1, 2, 2)
    metrics = [m for m in original_metrics.keys() if m not in ['num_nodes', 'is_connected']]
    x = np.arange(len(metrics))
    width = 0.35
    
    plt.bar(x - width/2, [original_metrics[m] for m in metrics], width, label='Original')
    plt.bar(x + width/2, [new_metrics[m] for m in metrics], width, label='Rewired')
    plt.xlabel('Metrics')
    plt.ylabel('Value')
    plt.title('Metrics Comparison')
    plt.xticks(x, metrics, rotation=45)
    plt.legend()
    
    plt.tight_layout()
    plt.show()
    
    return original_metrics, new_metrics

def gnn_on_rewired_graph(new_data, parts=None, gcn_params=None, return_subgraph_rep=False):
    """
    对重连后的图整体使用GCN，并可选输出每个分区的池化表示
    Args:
        new_data: PyG Data对象，重连后的图
        parts: 分区节点索引列表（可选），如需输出每个分区的表示
        gcn_params: dict，GCN参数（in_channels, hidden_channels, out_channels, activation, k, use_bn）
        return_subgraph_rep: 是否返回每个分区的池化表示
    Returns:
        out: 所有节点的GNN输出
        sub_g_reps: [n_patches, out_channels]，每个分区的池化表示（如果return_subgraph_rep为True）
    """
    if gcn_params is None:
        # 默认参数
        in_channels = new_data.x.shape[1]
        hidden_channels = 64
        out_channels = 32
        activation = None
        k = 2
        use_bn = True
    else:
        in_channels = gcn_params.get('in_channels', new_data.x.shape[1])
        hidden_channels = gcn_params.get('hidden_channels', 64)
        out_channels = gcn_params.get('out_channels', 32)
        activation = gcn_params.get('activation', None)
        k = gcn_params.get('k', 2)
        use_bn = gcn_params.get('use_bn', True)
    
    gcn = GCN(in_channels, hidden_channels, out_channels, activation, k=k, use_bn=use_bn)
    gcn.eval()  # 只做特征提取
    with torch.no_grad():
        out = gcn(new_data.x, new_data.edge_index)  # [num_nodes, out_channels]
    
    if return_subgraph_rep and parts is not None:
        sub_g_reps = torch.stack([out[torch.tensor(part)].mean(dim=0) if len(part) > 0 else torch.zeros(out.shape[1]) for part in parts])
        return out, sub_g_reps
    return out 

def get_gnn_node_embeddings(data, gcn_params=None):
    """
    对任意PyG Data对象做GCN，返回节点嵌入
    Args:
        data: PyG Data对象
        gcn_params: dict，GCN参数（可选）
    Returns:
        out: [num_nodes, out_channels] 节点嵌入
    """
    from .gnn import GCN
    if gcn_params is None:
        in_channels = data.x.shape[1]
        hidden_channels = 64
        out_channels = 32
        activation = None
        k = 2
        use_bn = True
    else:
        in_channels = gcn_params.get('in_channels', data.x.shape[1])
        hidden_channels = gcn_params.get('hidden_channels', 64)
        out_channels = gcn_params.get('out_channels', 32)
        activation = gcn_params.get('activation', None)
        k = gcn_params.get('k', 2)
        use_bn = gcn_params.get('use_bn', True)
    gcn = GCN(in_channels, hidden_channels, out_channels, activation, k=k, use_bn=use_bn)
    gcn.eval()
    with torch.no_grad():
        out = gcn(data.x, data.edge_index)
    return out 