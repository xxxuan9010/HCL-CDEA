# -*- coding: utf-8 -*-
# @Author  : hanyx9010@163.com
# @Time    : 2025/4/20 17:51
# @File    : test.py
# @Description    :
import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import HypergraphConv, global_mean_pool
import data_pre
import CLloss
from torch_geometric.loader import DataLoader
import copy
import torch.optim as optim
from torch.nn import LayerNorm
import pandas as pd
from sklearn.cluster import KMeans, SpectralClustering, AgglomerativeClustering
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
import matplotlib.pyplot as plt
import random
from torch_geometric.data import Data
from perturbation_strategy import find_valid_perturbations_edges, apply_noise_to_unperturbed

# ========== GNN 模型定义 ==========
class HGNN_Encoder(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, projection_dim=None):
        super().__init__()
        self.hgc1 = HypergraphConv(in_dim, hidden_dim)
        self.hgc2 = HypergraphConv(hidden_dim, out_dim)
        self.norm1 = LayerNorm(hidden_dim)
        self.norm2 = LayerNorm(out_dim)

        if projection_dim is not None:
            self.projection = nn.Sequential(
                nn.Linear(out_dim, projection_dim),
                nn.ReLU(),
                nn.LayerNorm(projection_dim)
            )

    def forward(self, x, edge_index, edge_weight=None, batch=None):
        x = self.hgc1(x, edge_index, edge_weight)
        x = self.norm1(x)  # 第一层正则化
        x = F.relu(x)

        x = self.hgc2(x, edge_index, edge_weight)
        x = self.norm2(x)  # 第二层正则化

        if batch is not None:
            x = global_mean_pool(x, batch)  # 图级表示

        if self.projection is not None:
            x = self.projection(x)

        return x

# ========== 图扰动函数 ==========
def perturb_graph_list(graph_list, valid_perturb_sets):
    """
    对每个图，找到其权重最小的超边，向其中添加一个新的参考节点
    （节点从 valid_perturb_sets 中随机抽取且不能在原超边中）
    """
    perturbed_list = []

    for data in graph_list:
        edge_index = data.edge_index.clone()
        edge_attr = data.edge_attr.clone()
        center_id = data.graph_id  # 当前 DMU 对应图


        all_hyperedge_ids = edge_index[1].unique()
        min_edge_id = None
        min_weight = float('inf')

        for he_id in all_hyperedge_ids:
            he_mask = (edge_index[1] == he_id)
            he_weights = edge_attr[he_mask]
            he_weight = he_weights[0].item()  # 每条超边的权重相同

            if he_weight < min_weight:
                min_weight = he_weight
                min_edge_id = he_id.item()

        mask = (edge_index[1] == min_edge_id)
        original_nodes = edge_index[0][mask].tolist()

        candidates = valid_perturb_sets.get(center_id + 1, [])
        if not candidates:
            perturbed_list.append(data)
            continue

        # 随机选择1组扰动边,并保留新增的节点
        perturb_ref_set = random.choice(candidates)
        perturb_ref_set = [i - 1 for i in perturb_ref_set if (i - 1) not in original_nodes]

        if not perturb_ref_set:
            perturbed_list.append(data)
            continue

        # 构造新的边 (new_node, min_edge_id), 权重与原边一致
        new_edge = torch.tensor([[nid, min_edge_id] for nid in perturb_ref_set], dtype=torch.long, device=edge_index.device).T
        new_weight = torch.tensor([min_weight] * len(perturb_ref_set), dtype=torch.float, device=edge_attr.device)

        new_edge_index = torch.cat([edge_index, new_edge], dim=1)
        new_edge_attr = torch.cat([edge_attr, new_weight], dim=0)

        # 构造新的图
        new_data = Data(
            x=data.x.clone(),
            edge_index=new_edge_index,
            edge_attr=new_edge_attr
        )
        new_data.graph_id = center_id
        perturbed_list.append(new_data)

    return perturbed_list


# ========== 主训练函数 ==========
def train(model, graph_list, valid_perturb_sets, epochs=200, batch_size=24, lr=1e-3):
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # 初始化记录器
    best_loss = float('inf')
    loss_history = []
    counter = 0
    best_weights = None

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        perturbed_graph_list = perturb_graph_list(graph_list, valid_perturb_sets)
        final_graph_list = apply_noise_to_unperturbed(perturbed_graph_list, inputs, outputs, theta_list, delta=0.0001)

        loader_orig = DataLoader(graph_list, batch_size=batch_size, shuffle=True)
        loader_perturb = DataLoader(final_graph_list, batch_size=batch_size, shuffle=True)

        for data_orig, data_perturb in zip(loader_orig, loader_perturb):
            data_orig = data_orig.to(device)
            data_perturb = data_perturb.to(device)

            z1 = model(data_orig.x, data_orig.edge_index, data_orig.edge_attr, data_orig.batch)
            z2 = model(data_perturb.x, data_perturb.edge_index, data_perturb.edge_attr, data_perturb.batch)

            loss = CLloss.graph_contrastive_loss(z1, z2)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(loader_orig)
        loss_history.append(avg_loss)  # 记录损失
        print(f"[Epoch {epoch+1}] Loss: {avg_loss:.4f}")

        if avg_loss < best_loss - 0.001:
            best_loss = avg_loss
            best_weights = copy.deepcopy(model.state_dict())
            counter = 0
        else:
            counter += 1

    # 恢复最佳模型
    model.load_state_dict(best_weights)

    # 新增可视化部分
    plt.figure(figsize=(10, 5))
    plt.plot(loss_history, 'b-o', markersize=4)
    plt.title("Contrastive Loss Curve")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True, alpha=0.3)
    plt.savefig('./loss_curve.png', bbox_inches='tight', dpi=300)
    plt.close()

    return counter


def extract_graph_embeddings(model, graph_list):
    model.eval()
    model = model.to(device)
    embeddings = []

    loader = DataLoader(graph_list, batch_size=1, shuffle=False)

    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            emb = model(data.x, data.edge_index, data.edge_attr, data.batch)
            embeddings.append(emb.cpu())  # 保存在 CPU 上，方便聚类

    return torch.vstack(embeddings)

def extract_and_save_embeddings(model, graph_list, save_path='embeddings.pt'):
    embeddings = extract_graph_embeddings(model, graph_list)
    torch.save(embeddings, save_path)
    return embeddings


if __name__ == "__main__":
    # 读取DEA特征
    X_weights = pd.read_excel('./data/data.xlsx', sheet_name='DEA_weight')
    X_envelop = pd.read_excel('./data/data.xlsx', sheet_name='DEA_lambda')
    X_raw = pd.read_excel('./data/data.xlsx', sheet_name='features')
    X_raw = X_raw.iloc[:, 2:8]

    # 读取图数据
    graph_list, graph_list_eff, node_features = data_pre.load_multiple_hypergraphs_from_excel("./data/completEnvelope24.xlsx")
    all_graphs = graph_list + graph_list_eff
    inputs, outputs, theta_list, reference_sets = data_pre.load_perturbations_views_from_excel("./data/completEnvelope24.xlsx")
    valid_perturb_sets = find_valid_perturbations_edges(
        inputs=inputs,
        outputs=outputs,
        theta_list=theta_list,
        reference_sets=reference_sets,
        delta=0.05
    )

    # 训练HGNN
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    input = node_features.shape[1]
    num_of_nodes = node_features.shape[0]
    HGNN_model_input = HGNN_Encoder(in_dim=input, hidden_dim=64, out_dim=32, projection_dim=input).to(device)     # HCL-CDEA-6
    HGNN_model_DMU = HGNN_Encoder(in_dim=input, hidden_dim=64, out_dim=32, projection_dim=num_of_nodes).to(device)   # HCL-CDEA-24

    train(HGNN_model_input, all_graphs, valid_perturb_sets)
    train(HGNN_model_DMU, all_graphs, valid_perturb_sets)

    # 提取嵌入
    all_graphs_sorted = sorted(all_graphs, key=lambda g: g.graph_id)
    embeddings_input = extract_graph_embeddings(HGNN_model_input, all_graphs_sorted)
    embeddings_DMU = extract_graph_embeddings(HGNN_model_DMU, all_graphs_sorted)

    # 聚类性能比较
    X_list = [X_raw, X_weights, X_envelop, embeddings_input, embeddings_DMU]

    # 聚类方法
    clustering_methods = {
        'KMeans-3': KMeans(n_clusters=3, random_state=42),
        'KMeans-4': KMeans(n_clusters=4, random_state=42),
        'KMeans-5': KMeans(n_clusters=5, random_state=42),
        'Spectral-3': SpectralClustering(n_clusters=3, assign_labels='discretize', affinity='nearest_neighbors',random_state=42),
        'Spectral-4': SpectralClustering(n_clusters=4, assign_labels='discretize', affinity='nearest_neighbors',random_state=42),
        'Spectral-5': SpectralClustering(n_clusters=5, assign_labels='discretize', affinity='nearest_neighbors',random_state=42),
        'Agglomerative-3': AgglomerativeClustering(n_clusters=3),
        'Agglomerative-4': AgglomerativeClustering(n_clusters=4),
        'Agglomerative-5': AgglomerativeClustering(n_clusters=5)
    }

    # 存储结果
    results = {}
    results_validity = {f'X_{i + 1}': {} for i in range(len(X_list))}
    labels_dict = {f'X_{i + 1}': {} for i in range(len(X_list))}    

    # 进行聚类并计算指标
    for i, X in enumerate(X_list):
        for method_name, method in clustering_methods.items():
            labels = method.fit_predict(X)
            labels_dict[f'X_{i + 1}'][method_name] = labels
            
            silhouette = silhouette_score(X, labels)
            calinski_harabasz = calinski_harabasz_score(X, labels)
            davies_bouldin = davies_bouldin_score(X, labels)

            results_validity[f'X_{i + 1}'][method_name] = {
                'Silhouette Score': silhouette,
                'Calinski-Harabasz Score': calinski_harabasz,
                'Davies-Bouldin Score': davies_bouldin
            }

    #         # 计算 ARI 和 NMI
    #         for other_method_name, other_labels in labels_dict[f'X_{i + 1}'].items():
    #             if method_name != other_method_name:
    #                 ari = adjusted_rand_score(labels, other_labels)
    #                 nmi = normalized_mutual_info_score(labels, other_labels)
    #                 results[(f'X_{i + 1}', method_name, other_method_name)] = {
    #                     'ARI': ari,
    #                     'NMI': nmi
    #                 }
    #
    # # 计算同一聚类方法下不同特征表示之间的一致性
    # for method_name, _ in clustering_methods.items():
    #     for i in range(len(X_list)):
    #         for j in range(i + 1, len(X_list)):
    #             labels_i = labels_dict[f'X_{i + 1}'][method_name]
    #             labels_j = labels_dict[f'X_{j + 1}'][method_name]
    #             ari = adjusted_rand_score(labels_i, labels_j)
    #             nmi = normalized_mutual_info_score(labels_i, labels_j)
    #             results[(method_name, f'X_{i + 1}', f'X_{j + 1}')] = {
    #                 'ARI': ari,
    #                 'NMI': nmi
    #             }
    #
    # # 将结果转换为 DataFrame
    # results_df = pd.DataFrame.from_dict(results, orient='index')
    # results_df.index.names = ['Feature Representation', 'Method 1', 'Method 2']

    validity_rows = []
    for dataset, methods in results_validity.items():
        for method, scores in methods.items():
            validity_rows.append([dataset, method, scores['Silhouette Score'], scores['Calinski-Harabasz Score'], scores['Davies-Bouldin Score']])

    validity_df = pd.DataFrame(validity_rows, columns=['Dataset', 'Method', 'Silhouette Score', 'Calinski-Harabasz Score', 'Davies-Bouldin Score'])

    # 将 labels_dict 转换为 DataFrame
    labels_rows = []
    for dataset, methods in labels_dict.items():
        for method, labels in methods.items():
            # 将标签展平并拼接到方法后面
            labels_flat = labels.flatten()  # 确保标签是一个一维数组
            labels_rows.append([dataset, method] + labels_flat.tolist())  # 拼接

    # 创建 DataFrame，假设 labels 有 24 个元素
    column_names = ['Dataset', 'Method'] + [f'Label_{i + 1}' for i in range(24)]
    labels_df = pd.DataFrame(labels_rows, columns=column_names)


    # 保存嵌入向量
    # embeddings_input = extract_and_save_embeddings(HGNN_model_DMU, all_graphs_sorted, save_path='embeddings_24.pt')
    # # 加载PyTorch格式
    # loaded_pt = torch.load('embeddings.pt')
















