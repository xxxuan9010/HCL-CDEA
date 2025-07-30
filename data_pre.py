# -*- coding: utf-8 -*-
# @Author  : hanyx9010@163.com
# @Time    : 2025/4/20 17:52
# @File    : data_pre.py
# @Description    : preprocess of data to construct Hypergraph

import pandas as pd
import torch
from torch_geometric.data import Data

def load_multiple_hypergraphs_from_excel(filepath):
    xls = pd.read_excel(filepath, header=None, sheet_name=None)

    sheet_names = list(xls.keys())

    # 第一张表为节点特征
    node_features_df = xls[sheet_names[0]]
    x_all = torch.tensor(node_features_df.values, dtype=torch.float)
    n_nodes = x_all.size(0)

    graph_list = []
    nodes_with_edges = set()

    # 每个后续 sheet 构成一个图
    for sheet_name in sheet_names[1:]:
        sheet = xls[sheet_name]
        edge_index = []
        edge_weights = []
        num_nodes_per_edge = []

        for _, row in sheet.iterrows():
            weight = row.iloc[0]
            node_ids = row.iloc[1:]
            node_ids = node_ids[node_ids != 0].astype(int).tolist()
            node_ids = [i - 1 for i in node_ids]  # 和x = x_all.clone()中的节点编号对齐

            if len(node_ids) == 0:
                continue

            hyperedge_id = len(edge_weights)

            for node_id in node_ids:
                edge_index.append([node_id, hyperedge_id])

            edge_weights.append(weight)
            num_nodes_per_edge.append(len(node_ids))

        if len(edge_index) == 0:
            continue  # 避免空图

        edge_index = torch.tensor(edge_index, dtype=torch.long).t()  # [2, num_edges]
        edge_weights_tensor = torch.tensor(edge_weights, dtype=torch.float)
        repeat_counts = torch.tensor(num_nodes_per_edge, dtype=torch.long)
        edge_attr = edge_weights_tensor.repeat_interleave(repeat_counts)

        data = Data(x=x_all.clone(), edge_index=edge_index, edge_attr=edge_attr)
        data.graph_id = int(sheet_name)-1
        nodes_with_edges.add(int(sheet_name)-1)
        graph_list.append(data)

    graph_list_eff = []
    # 对于未被包络的有效 DMU，只添加一个仅包含自身的超边
    all_node_indices = set(range(n_nodes))
    nodes_without_edges = all_node_indices - nodes_with_edges

    for node_id in nodes_without_edges:
        # 构造 edge_index，只包含该节点本身的超边
        edge_index = torch.tensor([[node_id], [0]], dtype=torch.long)
        edge_attr = torch.tensor([1.0], dtype=torch.float)

        # 全部节点特征仍然保持，edge 只连接目标节点
        data = Data(x=x_all.clone(), edge_index=edge_index, edge_attr=edge_attr)
        data.graph_id = node_id
        graph_list_eff.append(data)
    return graph_list, graph_list_eff, node_features_df


def load_perturbations_views_from_excel(filepath):
    xls = pd.read_excel(filepath, header=None, sheet_name=None)

    features = xls['features']
    data = features.values  # shape: [24, 6]
    outputs = data[:, :2].astype(float)
    inputs = data[:, 2:].astype(float)

    n_dmu = inputs.shape[0]
    theta_list = []
    reference_sets = []

    for i in range(n_dmu):
        sheet_name = str(i + 1)  # Excel 中用的是1-based
        if sheet_name in xls:
            sheet = xls[sheet_name]
            theta = float(sheet.iloc[0, 0])
            refs = sheet.iloc[0, 1:]
            refs = refs[refs.notna() & (refs != 0)].astype(int).tolist()  # 等于0的ref删掉(占位符)
            ref_indices = [r - 1 for r in refs]  # 转换成Python的0-based索引

        else:
            # 有效DMU，假设theta为1，参考集为空
            theta = 1.0
            ref_indices = []
        theta_list.append(theta)
        reference_sets.append(ref_indices)

    return inputs, outputs, theta_list, reference_sets


if __name__ == "__main__":
    # 示例路径
    graph_list = load_multiple_hypergraphs_from_excel("demo_20.xlsx")
    print(f"一共加载了 {len(graph_list)} 张图")
    print(graph_list[0])

    def check_graph_validity(graph_list):
        for data in graph_list:
            num_nodes = data.x.size(0)
            if data.edge_index[0].max().item() >= num_nodes:
                print(
                    f"图 {data.graph_id} 越界！x.size(0) = {num_nodes}, max(edge_index) = {data.edge_index[0].max().item()}")
            else:
                print(f"图 {data.graph_id}  合法")


    check_graph_validity(graph_list[0])


