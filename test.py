import torch
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from torch_geometric.data import Data
from SR import combine_louvain_delaunay

def create_test_graph():
    """创建一个包含多个明显团结构的测试图，团的大小和结构各不相同"""
    # 创建节点特征（使用2D坐标，便于可视化）
    num_nodes = 20  # 5 + 7 + 3 + 5 = 20
    x = torch.zeros(num_nodes, 2)
    
    # 团1：左下角，5个节点，结构紧密
    x[0:5, 0] = torch.tensor([0.1, 0.15, 0.2, 0.15, 0.1])
    x[0:5, 1] = torch.tensor([0.1, 0.1, 0.15, 0.2, 0.15])
    
    # 团2：右下角，7个节点，大团
    x[5:12, 0] = torch.tensor([0.7, 0.75, 0.8, 0.85, 0.8, 0.75, 0.7])
    x[5:12, 1] = torch.tensor([0.1, 0.1, 0.15, 0.2, 0.25, 0.2, 0.15])
    
    # 团3：左上角，3个节点，小团
    x[12:15, 0] = torch.tensor([0.1, 0.15, 0.1])
    x[12:15, 1] = torch.tensor([0.7, 0.75, 0.8])
    
    # 团4：右上角，5个节点，结构松散
    x[15:20, 0] = torch.tensor([0.7, 0.8, 0.9, 0.8, 0.7])
    x[15:20, 1] = torch.tensor([0.7, 0.7, 0.8, 0.9, 0.8])
    
    # 创建边（四个明显的团）
    edge_index = []
    
    # 团1：紧密连接
    for i in range(5):
        for j in range(i+1, 5):
            edge_index.append([i, j])
            edge_index.append([j, i])
    
    # 团2：大团，较多连接
    for i in range(5, 12):
        for j in range(i+1, 12):
            edge_index.append([i, j])
            edge_index.append([j, i])
    
    # 团3：小团，完全连接
    for i in range(12, 15):
        for j in range(i+1, 15):
            edge_index.append([i, j])
            edge_index.append([j, i])
    
    # 团4：松散连接
    for i in range(15, 20):
        for j in range(i+1, 20):
            if torch.rand(1) < 0.6:  # 60%的概率添加边
                edge_index.append([i, j])
                edge_index.append([j, i])
    
    # 添加一些跨团的边
    # 团1和团2之间的连接
    edge_index.append([0, 5])
    edge_index.append([5, 0])
    
    # 团2和团3之间的连接
    edge_index.append([5, 12])
    edge_index.append([12, 5])
    
    # 团1和团3之间的连接
    edge_index.append([0, 12])
    edge_index.append([12, 0])
    
    edge_index = torch.tensor(edge_index).t().contiguous()
    
    # 创建标签（四个类别）
    y = torch.zeros(num_nodes, dtype=torch.long)
    y[5:12] = 1  # 团2
    y[12:15] = 2  # 团3
    y[15:] = 3  # 团4
    
    return Data(x=x, edge_index=edge_index, y=y)

def visualize_graphs(original_graph, new_graph, parts, n_patches):
    """可视化原始图、分区图和重连后的图"""
    plt.figure(figsize=(20, 6))
    
    # 定义颜色方案
    colors = ['#FF0000', '#00FF00', '#0000FF', '#FFA500']
    
    # 1. 绘制原始图
    plt.subplot(131)
    G_original = nx.Graph()
    edges = original_graph.edge_index.t().numpy()
    G_original.add_edges_from(edges)
    G_original.add_nodes_from(range(original_graph.num_nodes))
    
    # 使用节点特征作为布局
    pos = {i: (original_graph.x[i, 0].item(), original_graph.x[i, 1].item()) 
           for i in range(original_graph.num_nodes)}
    
    nx.draw(G_original, pos, node_size=100, node_color='blue', alpha=0.6, width=1.0)
    plt.title('Original Graph', fontsize=14, pad=10)
    
    # 2. 绘制分区图
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
                for edge in edge_index:
                    if edge[0] == node and edge[1] in partition_nodes:
                        G_partition.add_edge(edge[0], edge[1])
    
    cmap = plt.cm.colors.ListedColormap(colors[:n_patches])
    nx.draw(G_partition, pos, node_size=100, node_color=node_colors, 
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
    edges = new_graph.edge_index.t().numpy()
    G_new.add_edges_from(edges)
    G_new.add_nodes_from(range(original_graph.num_nodes))
    
    nx.draw(G_new, pos, node_size=100, node_color=node_colors, 
            cmap=cmap, alpha=0.6, width=1.0)
    plt.title('Rewired Graph', fontsize=14, pad=10)
    
    plt.tight_layout()
    plt.show()

def test_simple_graph():
    # 创建测试图
    print("创建测试图...")
    data = create_test_graph()
    
    # 打印图的基本信息
    print(f"\n图信息:")
    print(f"节点数: {data.num_nodes}")
    print(f"边数: {data.edge_index.shape[1]}")
    print(f"特征维度: {data.x.shape[1]}")
    print(f"类别数: 4")
    
    # 应用Louvain分区和Delaunay重连
    print("\n应用Louvain分区和Delaunay重连...")
    n_patches = 4  # 分成4个部分
    print(f"目标分区数量: {n_patches}")
    
    try:
        # 执行分区和重连
        new_graph, parts = combine_louvain_delaunay(data, n_patches, return_parts=True, visualize=False)  # 不在这里可视化
        
        # 打印结果
        print("\n重连后的图信息:")
        print(f"节点数: {new_graph.num_nodes}")
        print(f"边数: {new_graph.edge_index.shape[1]}")
        
        # 可视化图
        print("\n可视化图结构...")
        visualize_graphs(data, new_graph, parts, n_patches)
        
    except Exception as e:
        print(f"\n处理失败: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_simple_graph()