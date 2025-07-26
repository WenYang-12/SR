# Dual Graph Fusion for Node Classification

本项目核心实现了基于 **GraphConv** 的**双图融合**（Dual Graph Fusion）节点分类方法，并支持多种经典和最新的GNN模型作为对照组，在多个同质/异质图数据集上进行了系统评测和对比实验。

## 核心思想 Core Idea
- **双图融合（Dual Graph Fusion）**：以GraphConv为主干，分别在原始图和结构增强图（如Louvain分区+Delaunay重连、Louvain分区+随机重连）上提取节点特征，并融合两者信息进行节点分类。
- **对照实验**：支持GCN、GAT、GIN、GraphSAGE等多种GNN模型，以及不同的图结构增强方式（如简单拼接、随机重连等）作为对照组，系统评估融合策略的有效性。

## 特性 Features
- 支持多种GNN模型（GraphConv为主，GCN, GAT, GIN, GraphSAGE等为对照）
- 实现了多种双图融合方式（如特征拼接、BGA等）
- 支持Louvain分区+Delaunay重连、Louvain分区+随机重连等结构增强
- 可在多种同质/异质图数据集上复现实验
- 实验结果自动统计与可视化

## 依赖 Dependencies
请先安装以下依赖（建议使用虚拟环境）：
```bash
pip install -r requirements.txt
```

## 数据集 Datasets
本项目支持以下公开数据集：
- **同质图**：Cora, Citeseer, Pubmed
- **异质图**：Cornell, Texas, Wisconsin, Film

数据集将自动下载并保存在 `data/` 目录下。
- 同质图采用 PyG 官方划分，
- 异质图采用 5 次随机划分（60%/20%/20%）。

## 用法 Usage
### 1. 运行GraphConv为主干的双图融合模型（推荐核心实验）
```bash
python fusion_example.py --dataset Cora --test_type fusion --gnn_type graphconv --n_runs 5
```
- `--test_type fusion` 运行双图融合模型
- `--gnn_type graphconv` 以GraphConv为主干

### 2. 运行其它GNN模型（对照组）
```bash
python fusion_example.py --dataset Cora --test_type fusion --gnn_type gcn --n_runs 5
```
- `--gnn_type` 可选 gcn, gat, transformer, graphsage, gin

### 3. 运行单一GNN模型（无融合，对照组）
```bash
python fusion_example.py --dataset Cora --test_type gnn --gnn_type graphconv --n_runs 5
```

### 4. 运行Louvain分区+随机重连拼接融合（对照组）
```bash
python fusion_example.py --dataset Cora --test_type random_rewire --gnn_type graphconv --n_runs 5
```
- 可用 `--num_random_edges` 指定随机分区间边数（默认与DG连接图一致）

## 文件结构 File Structure
- `fusion_example.py`：主实验脚本，包含模型训练与评测
- `fusion_model.py`：双图融合与对照模型实现
- `gnn.py`：各类GNN模型实现
- `SR.py`：Louvain分区、Delaunay重连、随机重连等方法
- `data_utils.py`：数据集加载与处理工具
- `data/`：数据集目录

## 结果展示 Results
实验结果将在终端输出，并以表格形式展示各模型在不同数据集上的准确率与F1分数。

## 核心推荐实验
- **GraphConv主干的双图融合**是本项目的核心创新，建议优先关注。
- 其它GNN主干、单图模型、随机重连等为对照组，便于系统评估融合策略的有效性。

## 引用 Citation
如本项目对您的研究有帮助，请引用相关论文和本仓库。

```
@misc{dualgraphfusion,
  title={Dual Graph Fusion for Node Classification},
  author={Your Name},
  year={2024},
  howpublished={\url{https://github.com/your-repo/dual-graph-fusion}}
}
```

---
