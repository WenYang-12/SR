# -*- coding: utf-8 -*-
import torch
import torch.nn.functional as F
import numpy as np
from torch_geometric.datasets import Planetoid, WikipediaNetwork, WebKB, Amazon, Actor, DeezerEurope
from torch_geometric.transforms import NormalizeFeatures
from torch_geometric.utils import train_test_split_edges
from sklearn.metrics import accuracy_score, f1_score
import matplotlib.pyplot as plt
import argparse
from ogb.nodeproppred import PygNodePropPredDataset
import os

from SR import combine_louvain_delaunay, combine_louvain_random_rewire
from fusion_model import DualGraphFusionModel, get_fusion_model, SimpleConcatFusionModel
from gnn import GCN, GAT, GIN, GraphConv, GraphSAGE


def evaluate_single_model(model, data, mask, device):
    """评估单个GNN模型性能"""
    model.eval()
    with torch.no_grad():
        pred = model(data.x.to(device), data.edge_index.to(device))
        pred = pred.argmax(dim=1)
        accuracy = accuracy_score(data.y[mask].cpu(), pred[mask].cpu())
        f1_micro = f1_score(data.y[mask].cpu(), pred[mask].cpu(), average='micro')
        f1_macro = f1_score(data.y[mask].cpu(), pred[mask].cpu(), average='macro')
    return accuracy, f1_micro, f1_macro


def run_multiple_experiments(model_class, model_params, data, device, n_runs=5, epochs=200):
    """运行多次实验并计算统计结果"""
    all_results = []
    
    for run in range(n_runs):
        print(f"运行实验 {run + 1}/{n_runs}...")
        
        # 重新初始化模型
        model = model_class(**model_params).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
        
        # 训练模型
        result = train_single_model(model, data, optimizer, device, epochs=epochs, seed=42+run)
        all_results.append({
            'best_val_acc': result['best_val_acc'],
            'test_acc': result['test_acc'],
            'test_f1_micro': result['test_f1_micro'],
            'test_f1_macro': result['test_f1_macro']
        })
        
        print(f"  实验 {run + 1} 测试准确率: {result['test_acc']:.4f}")
    
    # 计算统计结果
    metrics = ['best_val_acc', 'test_acc', 'test_f1_micro', 'test_f1_macro']
    stats = {}
    
    for metric in metrics:
        values = [result[metric] for result in all_results]
        mean_val = np.mean(values)
        std_val = np.std(values)
        stats[metric] = {'mean': mean_val, 'std': std_val, 'values': values}
    
    return stats


def test_gnn_models(data, device, n_runs=5):
    """测试各种GNN模型的性能（多次实验）"""
    print("=" * 60)
    print("开始GNN模型多次实验测试")
    print("=" * 60)
    
    # 模型配置
    in_channels = data.x.size(1)
    hidden_channels = 64
    out_channels = data.y.max().item() + 1
    activation = F.relu
    
    results = {}
    
    # 1. GCN模型
    print("\n1. 训练GCN模型...")
    gcn_params = {
        'in_channels': in_channels,
        'hidden_channels': hidden_channels,
        'out_channels': out_channels,
        'activation': activation,
        'k': 2,
        'use_bn': True
    }
    gcn_stats = run_multiple_experiments(GCN, gcn_params, data, device, n_runs)
    results['GCN'] = gcn_stats
    
    # 2. GAT模型
    print("\n2. 训练GAT模型...")
    gat_params = {
        'in_channels': in_channels,
        'hidden_channels': hidden_channels,
        'out_channels': out_channels,
        'activation': activation,
        'n_heads': 8,
        'k': 2,
        'use_bn': True
    }
    gat_stats = run_multiple_experiments(GAT, gat_params, data, device, n_runs)
    results['GAT'] = gat_stats
    
    # 3. GIN模型
    print("\n3. 训练GIN模型...")
    gin_params = {
        'in_channels': in_channels,
        'hidden_channels': hidden_channels,
        'out_channels': out_channels,
        'activation': activation,
        'k': 2,
        'use_bn': True,
        'eps': 0.0
    }
    gin_stats = run_multiple_experiments(GIN, gin_params, data, device, n_runs)
    results['GIN'] = gin_stats
    
    # 4. GraphSAGE模型
    print("\n4. 训练GraphSAGE模型...")
    sage_params = {
        'in_channels': in_channels,
        'hidden_channels': hidden_channels,
        'out_channels': out_channels,
        'activation': activation,
        'k': 2,
        'use_bn': True,
        'aggr': 'mean'
    }
    sage_stats = run_multiple_experiments(GraphSAGE, sage_params, data, device, n_runs)
    results['GraphSAGE'] = sage_stats
    
    # 5. GraphConv模型
    print("\n5. 训练GraphConv模型...")
    graphconv_params = {
        'in_channels': in_channels,
        'hidden_channels': hidden_channels,
        'out_channels': out_channels,
        'num_layers': 2,
        'dropout': 0.5,
        'use_bn': True,
        'use_residual': True,
        'use_weight': True,
        'use_init': False,
        'use_act': True
    }
    graphconv_stats = run_multiple_experiments(GraphConv, graphconv_params, data, device, n_runs)
    results['GraphConv'] = graphconv_stats
    
    return results


def print_results_table(results, dataset_name):
    """按照论文格式打印结果表格（百分比形式）"""
    print("\n" + "=" * 100)
    print(f"数据集: {dataset_name}")
    print("=" * 100)
    print(f"{'模型':<12} {'最佳验证准确率':<20} {'测试准确率':<20} {'F1-Micro':<20} {'F1-Macro':<20}")
    print("-" * 100)
    
    for model_name, result in results.items():
        # 转换为百分比形式
        best_val = f"{result['best_val_acc']['mean']*100:.2f}% ± {result['best_val_acc']['std']*100:.2f}"
        test_acc = f"{result['test_acc']['mean']*100:.2f}% ± {result['test_acc']['std']*100:.2f}"
        f1_micro = f"{result['test_f1_micro']['mean']*100:.2f}% ± {result['test_f1_micro']['std']*100:.2f}"
        f1_macro = f"{result['test_f1_macro']['mean']*100:.2f}% ± {result['test_f1_macro']['std']*100:.2f}"
        
        print(f"{model_name:<12} {best_val:<20} {test_acc:<20} {f1_micro:<20} {f1_macro:<20}")
    
    print("=" * 100)
    
    # 打印详细结果
    print("\n详细结果:")
    for model_name, result in results.items():
        print(f"\n{model_name}:")
        print(f"  最佳验证准确率: {result['best_val_acc']['values']}")
        print(f"    平均值: {result['best_val_acc']['mean']*100:.2f}% ± {result['best_val_acc']['std']*100:.2f}")
        print(f"  测试准确率: {result['test_acc']['values']}")
        print(f"    平均值: {result['test_acc']['mean']*100:.2f}% ± {result['test_acc']['std']*100:.2f}")
        print(f"  F1-Micro: {result['test_f1_micro']['values']}")
        print(f"    平均值: {result['test_f1_micro']['mean']*100:.2f}% ± {result['test_f1_micro']['std']*100:.2f}")
        print(f"  F1-Macro: {result['test_f1_macro']['values']}")
        print(f"    平均值: {result['test_f1_macro']['mean']*100:.2f}% ± {result['test_f1_macro']['std']*100:.2f}")


def train_single_model(model, data, optimizer, device, epochs=200, seed=None):
    """训练单个GNN模型"""
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)
    
    model.train()
    
    # 创建训练掩码
    num_nodes = data.x.size(0)
    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)

    # 随机划分训练/验证/测试集
    indices = torch.randperm(num_nodes)
    train_size = int(0.6 * num_nodes)
    val_size = int(0.2 * num_nodes)

    train_mask[indices[:train_size]] = True
    val_mask[indices[train_size:train_size + val_size]] = True
    test_mask[indices[train_size + val_size:]] = True

    best_val_acc = 0
    train_losses = []
    val_accuracies = []
    
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        
        pred = model(data.x.to(device), data.edge_index.to(device))
        loss = F.cross_entropy(pred[train_mask.to(device)], data.y[train_mask.to(device)])
        
        loss.backward()
        optimizer.step()
        
        # 评估
        if epoch % 10 == 0:
            val_acc, val_f1_micro, val_f1_macro = evaluate_single_model(model, data, val_mask, device)
            train_losses.append(loss.item())
            val_accuracies.append(val_acc)
            
            if val_acc > best_val_acc:
                best_val_acc = val_acc
    
    # 最终测试
    test_acc, test_f1_micro, test_f1_macro = evaluate_single_model(model, data, test_mask, device)
    
    return {
        'best_val_acc': best_val_acc,
        'test_acc': test_acc,
        'test_f1_micro': test_f1_micro,
        'test_f1_macro': test_f1_macro,
        'train_losses': train_losses,
        'val_accuracies': val_accuracies
    }


def evaluate_model(model, data, dg_graph, mask, device, batch=None):
    """评估模型性能"""
    model.eval()
    with torch.no_grad():
        if hasattr(model, 'use_fusion') and model.use_fusion:
            pred, _, _, _ = model(data.x.to(device),
                                  data.edge_index.to(device),
                                  dg_graph.edge_index.to(device),
                                  batch=batch)
        elif hasattr(model, 'fusion_layer'):
            try:
                pred, _, _, _ = model(data.x.to(device),
                                      data.edge_index.to(device),
                                      dg_graph.edge_index.to(device),
                                      batch=batch)
            except ValueError:
                pred1, pred2 = model(data.x.to(device),
                                     data.edge_index.to(device),
                                     dg_graph.edge_index.to(device),
                                     batch=batch)
                pred = (pred1 + pred2) / 2
        else:
            pred, _, _, _ = model(data.x.to(device),
                                  data.edge_index.to(device),
                                  dg_graph.edge_index.to(device))
        
        pred = pred.argmax(dim=1)
        accuracy = accuracy_score(data.y[mask].cpu(), pred[mask].cpu())
        f1_micro = f1_score(data.y[mask].cpu(), pred[mask].cpu(), average='micro')
        f1_macro = f1_score(data.y[mask].cpu(), pred[mask].cpu(), average='macro')
        
    return accuracy, f1_micro, f1_macro


def parts_to_batch(parts, num_nodes):
    batch = torch.zeros(num_nodes, dtype=torch.long)
    for i, node_list in enumerate(parts):
        batch[torch.tensor(node_list)] = i
    return batch


def run_fusion_multiple_experiments(model_class, model_params, data, dg_graph, device, n_runs=5, epochs=200, dg_parts=None):
    """运行多次融合模型实验并计算统计结果"""
    all_results = []
    
    for run in range(n_runs):
        print(f"运行融合实验 {run + 1}/{n_runs}...")
        
        # 重新初始化模型
        model = model_class(**model_params).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
        
        # 训练模型
        result = train_fusion_model(model, data, dg_graph, optimizer, device, epochs=epochs, dg_parts=dg_parts, seed=42+run)
        
        # 最终测试
        test_mask = torch.ones(data.x.size(0), dtype=torch.bool)
        test_acc, test_f1_micro, test_f1_macro = evaluate_model(model, data, dg_graph, test_mask, device, batch=parts_to_batch(dg_parts, data.num_nodes).to(device) if dg_parts else None)
        
        all_results.append({
            'best_val_acc': result['best_val_acc'],
            'test_acc': test_acc,
            'test_f1_micro': test_f1_micro,
            'test_f1_macro': test_f1_macro
        })
        
        print(f"  实验 {run + 1} 测试准确率: {test_acc:.4f}")
    
    # 计算统计结果
    metrics = ['best_val_acc', 'test_acc', 'test_f1_micro', 'test_f1_macro']
    stats = {}
    
    for metric in metrics:
        values = [result[metric] for result in all_results]
        mean_val = np.mean(values)
        std_val = np.std(values)
        stats[metric] = {'mean': mean_val, 'std': std_val, 'values': values}
    
    return stats


def compare_models(data, device, n_patches=4, n_runs=5, gnn_type='gcn', only_simple_concat=False, test_random_rewire=False, random_rewire_edges=None):
    """比较不同模型的性能（多次实验）"""
    print("=" * 60)
    print(f"开始双图融合模型多次实验测试 (Backbone: {gnn_type.upper()})")
    print("=" * 60)
    
    # 获取DG连接图
    print("生成DG连接图...")
    dg_graph, parts = combine_louvain_delaunay(data, n_patches, visualize=False, return_parts=True)
    
    # 模型配置
    in_channels = data.x.size(1)
    hidden_channels = 64
    out_channels = data.y.max().item() + 1
    activation = F.relu
    
    results = {}
    
    if not only_simple_concat and not test_random_rewire:
        # 双图融合模型
        print(f"\n训练双图融合模型 ({gnn_type.upper()} backbone)...")
        dual_params = {
            'num_nodes': data.x.size(0),
            'in_channels': in_channels,
            'hidden_channels': hidden_channels,
            'out_channels': out_channels,
            'activation': activation,
            'gnn_layers': 2,
            'gnn_type': gnn_type,
            'use_fusion': True
        }
        dual_stats = run_fusion_multiple_experiments(DualGraphFusionModel, dual_params, data, dg_graph, device, n_runs, dg_parts=parts)
        results[f'Dual-Fusion-{gnn_type.upper()}'] = dual_stats

    # 简单拼接模型
    if not test_random_rewire:
        print(f"\n训练简单拼接融合模型 ({gnn_type.upper()} backbone)...")
        concat_params = {
            'num_nodes': data.x.size(0),
            'in_channels': in_channels,
            'hidden_channels': hidden_channels,
            'out_channels': out_channels,
            'activation': activation,
            'gnn_layers': 2,
            'gnn_type': gnn_type
        }
        concat_stats = run_fusion_multiple_experiments(SimpleConcatFusionModel, concat_params, data, dg_graph, device, n_runs, dg_parts=parts)
        results[f'SimpleConcatFusion-{gnn_type.upper()}'] = concat_stats

    # Louvain分区+随机重连拼接模型
    if test_random_rewire:
        print(f"\n训练Louvain分区+随机重连拼接融合模型 ({gnn_type.upper()} backbone)...")
        # 生成随机重连图
        if random_rewire_edges is None:
            num_edges = dg_graph.edge_index.size(1)
        else:
            num_edges = random_rewire_edges
        random_graph, random_parts = combine_louvain_random_rewire(data, dg_edge_index=dg_graph.edge_index, num_random_edges=random_rewire_edges, return_parts=True)
        random_concat_params = {
            'num_nodes': data.x.size(0),
            'in_channels': in_channels,
            'hidden_channels': hidden_channels,
            'out_channels': out_channels,
            'activation': activation,
            'gnn_layers': 2,
            'gnn_type': gnn_type
        }
        random_concat_stats = run_fusion_multiple_experiments(SimpleConcatFusionModel, random_concat_params, data, random_graph, device, n_runs, dg_parts=random_parts)
        results[f'LouvainRandomConcat-{gnn_type.upper()}'] = random_concat_stats

    return results


def train_fusion_model(model, data, dg_graph, optimizer, device, epochs=200, dg_parts=None, seed=None):
    """训练融合模型（支持随机种子）"""
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)
    
    model.train()
    
    # 创建训练掩码（这里简化处理，实际应该使用数据集提供的掩码）
    num_nodes = data.x.size(0)
    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)

    # 随机划分训练/验证/测试集
    indices = torch.randperm(num_nodes)
    train_size = int(0.6 * num_nodes)
    val_size = int(0.2 * num_nodes)

    train_mask[indices[:train_size]] = True
    val_mask[indices[train_size:train_size + val_size]] = True
    test_mask[indices[train_size + val_size:]] = True

    # 构造batch
    if dg_parts is not None:
        batch = parts_to_batch(dg_parts, num_nodes).to(device)
    else:
        batch = None

    best_val_acc = 0
    train_losses = []
    val_accuracies = []
    
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        
        if hasattr(model, 'use_fusion') and model.use_fusion:
            pred, _, _, _ = model(data.x.to(device),
                                  data.edge_index.to(device),
                                  dg_graph.edge_index.to(device),
                                  batch=batch)
            loss = F.cross_entropy(pred[train_mask.to(device)], data.y[train_mask.to(device)])
        elif hasattr(model, 'fusion_layer'):
            # 兼容CoTrainingFusionModel
            try:
                pred, _, _, _ = model(data.x.to(device),
                                      data.edge_index.to(device),
                                      dg_graph.edge_index.to(device),
                                      batch=batch)
            except ValueError:
                pred1, pred2 = model(data.x.to(device),
                                     data.edge_index.to(device),
                                     dg_graph.edge_index.to(device),
                                     batch=batch)
                # 这里可选用pred1或pred2，或平均
                pred = (pred1 + pred2) / 2
            loss = F.cross_entropy(pred[train_mask.to(device)], data.y[train_mask.to(device)])
        else:
            pred, _, _, _ = model(data.x.to(device),
                                  data.edge_index.to(device),
                                  dg_graph.edge_index.to(device))
            loss = F.cross_entropy(pred[train_mask.to(device)], data.y[train_mask.to(device)])
        
        loss.backward()
        optimizer.step()
        
        # 评估
        if epoch % 10 == 0:
            val_acc, val_f1_micro, val_f1_macro = evaluate_model(model, data, dg_graph, val_mask, device, batch=batch)
            train_losses.append(loss.item())
            val_accuracies.append(val_acc)
            
            if val_acc > best_val_acc:
                best_val_acc = val_acc
    
    return {
        'best_val_acc': best_val_acc,
        'train_losses': train_losses,
        'val_accuracies': val_accuracies
    }


def visualize_results(results):
    """可视化实验结果"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # 1. 训练损失
    ax1 = axes[0, 0]
    for model_name, result in results.items():
        ax1.plot(result['train_losses'], label=model_name)
    ax1.set_title('Training Loss')
    ax1.set_xlabel('Epoch (x10)')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)
    
    # 2. 验证准确率
    ax2 = axes[0, 1]
    for model_name, result in results.items():
        ax2.plot(result['val_accuracies'], label=model_name)
    ax2.set_title('Validation Accuracy')
    ax2.set_xlabel('Epoch (x10)')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    ax2.grid(True)
    
    # 3. 测试性能比较
    ax3 = axes[1, 0]
    models = list(results.keys())
    test_accs = [results[model]['test_acc'] for model in models]
    test_f1_micro = [results[model]['test_f1_micro'] for model in models]
    test_f1_macro = [results[model]['test_f1_macro'] for model in models]
    
    x = np.arange(len(models))
    width = 0.25
    
    ax3.bar(x - width, test_accs, width, label='Accuracy')
    ax3.bar(x, test_f1_micro, width, label='F1-Micro')
    ax3.bar(x + width, test_f1_macro, width, label='F1-Macro')
    
    ax3.set_title('Test Performance Comparison')
    ax3.set_xlabel('Models')
    ax3.set_ylabel('Score')
    ax3.set_xticks(x)
    ax3.set_xticklabels(models)
    ax3.legend()
    ax3.grid(True, axis='y')
    
    # 4. 最佳验证准确率
    ax4 = axes[1, 1]
    best_val_accs = [results[model]['best_val_acc'] for model in models]
    ax4.bar(models, best_val_accs, color=['skyblue', 'lightgreen', 'lightcoral'])
    ax4.set_title('Best Validation Accuracy')
    ax4.set_ylabel('Accuracy')
    ax4.grid(True, axis='y')
    
    plt.tight_layout()
    plt.show()
    
    # 打印详细结果
    print("\n" + "=" * 60)
    print("详细实验结果")
    print("=" * 60)
    print(f"{'模型':<15} {'最佳验证准确率':<15} {'测试准确率':<12} {'F1-Micro':<12} {'F1-Macro':<12}")
    print("-" * 60)
    for model_name, result in results.items():
        print(f"{model_name:<15} {result['best_val_acc']:<15.4f} {result['test_acc']:<12.4f} "
              f"{result['test_f1_micro']:<12.4f} {result['test_f1_macro']:<12.4f}")


def main():
    """主函数"""
    # 命令行参数解析
    parser = argparse.ArgumentParser(description='Dual Graph Fusion Model Experiment')
    parser.add_argument('--dataset', type=str, default='Cora',
                        help='Dataset to use (e.g. Cora, Citeseer, Pubmed, cornell, texas, wisconsin, photo, computers,film )')
    parser.add_argument('--test_type', type=str, default='gnn', choices=['gnn', 'fusion', 'simple_concat', 'random_rewire'],
                        help='Test type: gnn (direct GNN models) or fusion (dual graph fusion models) or simple_concat (simple concat fusion) or random_rewire (Louvain partition + random rewire)')
    parser.add_argument('--gnn_type', type=str, default='gcn',
                       choices=['gcn', 'gat', 'transformer', 'graphconv'],
                       help='Backbone type for fusion model')
    parser.add_argument('--n_runs', type=int, default=5, 
                       help='Number of experimental runs')
    parser.add_argument('--num_random_edges', type=int, default=None, help='Number of random inter-partition edges for random rewire')
    args = parser.parse_args()

    # 设置随机种子
    torch.manual_seed(42)
    np.random.seed(42)
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    # 数据集加载逻辑
    name = args.dataset
    print(f"加载{name}数据集...")
    data_path = 'data/Planetoid'  # 默认路径
    if name in ['Cora', 'Citeseer', 'Pubmed']:
        dataset = Planetoid(root=data_path, name=name, transform=NormalizeFeatures())
        data = dataset[0]
    elif name in ['chameleon', 'squirrel']:
        dataset = WikipediaNetwork(root='data/WikipediaNetwork', name=name, transform=NormalizeFeatures())
        data = dataset[0]
    elif name in ['cornell', 'texas', 'wisconsin']:
        dataset = WebKB(root='data/WebKB', name=name.capitalize(), transform=NormalizeFeatures())
        data = dataset[0]
    elif name in ['photo', 'computers']:
        dataset = Amazon(root='data/Amazon', name=name, transform=NormalizeFeatures())
        data = dataset[0]
    elif name in ['film']:
        dataset = Actor(root='data/Actor', transform=NormalizeFeatures())
        data = dataset[0]
    elif name in ['deezer']:
        dataset = DeezerEurope(root='data/DeezerEurope')
        data = dataset[0]
    elif name in ['ogbn-arxiv', 'ogbn-products']:
        dataset = PygNodePropPredDataset(name=name, root='data/OGB')
        data = dataset[0]
        # OGB数据集y为二维，需 squeeze
        if data.y.dim() > 1:
            data.y = data.y.squeeze()
    else:
        raise ValueError(f"Unknown dataset: {name}")

    print(f"数据集信息:")
    print(f"  节点数量: {data.num_nodes}")
    print(f"  边数量: {data.edge_index.shape[1]}")
    print(f"  特征维度: {data.x.shape[1]}")
    if hasattr(dataset, 'num_classes'):
        print(f"  类别数量: {dataset.num_classes}")
    elif hasattr(data, 'y'):
        print(f"  类别数量: {int(data.y.max().item()) + 1}")
    
    # 根据测试类型运行不同的实验
    if args.test_type == 'gnn':
        # 直接测试GNN模型（多次实验）
        results = test_gnn_models(data, device, n_runs=args.n_runs)
        print_results_table(results, name)
    elif args.test_type == 'simple_concat':
        # 只测试简单拼接融合模型
        results = compare_models(data, device, n_patches=4, n_runs=args.n_runs, gnn_type=args.gnn_type, only_simple_concat=True)
        print_results_table(results, name)
    elif args.test_type == 'random_rewire':
        # 只测试Louvain分区+随机重连拼接融合模型
        results = compare_models(data, device, n_patches=4, n_runs=args.n_runs, gnn_type=args.gnn_type, test_random_rewire=True, random_rewire_edges=args.num_random_edges)
        print_results_table(results, name)
    else:
        # 测试融合模型（多次实验）
        results = compare_models(data, device, n_patches=4, n_runs=args.n_runs, gnn_type=args.gnn_type)
        print_results_table(results, name)


if __name__ == "__main__":
    main() 