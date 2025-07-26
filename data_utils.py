import torch
import torch.nn.functional as F
import numpy as np
import networkx as nx
from torch_geometric.utils import to_undirected
from torch_geometric.data import Data
from torch_geometric.datasets import Planetoid, WikipediaNetwork, WebKB, Actor, DeezerEurope
from ogb.nodeproppred import PygNodePropPredDataset
import os
import ssl
import urllib.request
import time

# 禁用SSL验证
ssl._create_default_https_context = ssl._create_unverified_context

class GraphDataset:
    """图数据集类"""
    def __init__(self, name):
        self.name = name
        self.graph = None
        self.label = None
        self.train_mask = None
        self.val_mask = None
        self.test_mask = None
        self.split_idx = None

    def load_pyg_data(self, data):
        """从PyG数据加载"""
        self.graph = {
            'edge_index': data.edge_index,
            'edge_feat': None,
            'node_feat': data.x,
            'num_nodes': data.num_nodes
        }
        self.label = data.y
        if hasattr(data, 'train_mask'):
            self.train_mask = data.train_mask
            self.val_mask = data.val_mask
            self.test_mask = data.test_mask

    def to_pyg_data(self):
        """转换为PyG数据格式"""
        data = Data(
            x=self.graph['node_feat'],
            edge_index=self.graph['edge_index'],
            y=self.label
        )
        if self.train_mask is not None:
            data.train_mask = self.train_mask
            data.val_mask = self.val_mask
            data.test_mask = self.test_mask
        return data

    def get_idx_split(self, split_type='random', train_prop=.5, valid_prop=.25):
        """获取数据集划分"""
        if split_type == 'random':
            # 随机划分
            num_nodes = self.graph['num_nodes']
            indices = torch.randperm(num_nodes)
            train_size = int(num_nodes * train_prop)
            valid_size = int(num_nodes * valid_prop)
            
            self.split_idx = {
                'train': indices[:train_size],
                'valid': indices[train_size:train_size + valid_size],
                'test': indices[train_size + valid_size:]
            }
        return self.split_idx

def download_with_retry(url, max_retries=3, delay=1):
    """带重试机制的文件下载"""
    for i in range(max_retries):
        try:
            response = urllib.request.urlopen(url)
            return response.read()
        except Exception as e:
            if i == max_retries - 1:
                raise e
            print(f"Download attempt {i+1} failed: {str(e)}")
            print(f"Retrying in {delay} seconds...")
            time.sleep(delay)
            delay *= 2  # 指数退避

def load_dataset(data_path, name):
    """加载数据集"""
    try:
        # 确保数据目录存在
        os.makedirs(data_path, exist_ok=True)
        
        # 尝试加载数据集
        if name in ['Cora', 'CiteSeer', 'PubMed']:
            try:
                # 首先尝试直接加载
                pyg_dataset = Planetoid(root=data_path, name=name)
            except Exception as e:
                print(f"Error loading {name} dataset: {str(e)}")
                print("Attempting to clean and reload dataset...")
                # 清理可能损坏的文件
                raw_dir = os.path.join(data_path, name, 'raw')
                if os.path.exists(raw_dir):
                    for file in os.listdir(raw_dir):
                        try:
                            os.remove(os.path.join(raw_dir, file))
                        except:
                            pass
                # 重新加载
                pyg_dataset = Planetoid(root=data_path, name=name)
        elif name in ['ogbn-arxiv', 'ogbn-products']:
            pyg_dataset = PygNodePropPredDataset(name=name, root=data_path)
        elif name in ['chameleon', 'squirrel']:
            pyg_dataset = WikipediaNetwork(root=data_path, name=name)
        elif name in ['cornell', 'texas', 'wisconsin']:
            pyg_dataset = WebKB(root=data_path, name=name)
        elif name in ['film']:
            pyg_dataset = Actor(root=data_path)
        elif name in ['deezer']:
            pyg_dataset = DeezerEurope(root=data_path)
        else:
            raise ValueError(f"Unknown dataset: {name}")
        
        # 转换为GraphDataset格式
        dataset = GraphDataset(name)
        dataset.graph = {
            'edge_index': pyg_dataset[0].edge_index,
            'edge_feat': None,
            'node_feat': pyg_dataset[0].x,
            'num_nodes': pyg_dataset[0].num_nodes
        }
        dataset.label = pyg_dataset[0].y
        
        # 处理数据集划分
        if hasattr(pyg_dataset[0], 'train_mask'):
            dataset.train_mask = pyg_dataset[0].train_mask
            dataset.val_mask = pyg_dataset[0].val_mask
            dataset.test_mask = pyg_dataset[0].test_mask
        else:
            # 如果没有预定义的划分，创建随机划分
            dataset.split_idx = dataset.get_idx_split()
        
        return dataset
        
    except Exception as e:
        print(f"Error loading dataset {name}: {str(e)}")
        print("Please check your internet connection and try again.")
        raise 