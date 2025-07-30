# -*- coding: utf-8 -*-
# @Author  : hanyx9010@163.com
# @Time    : 2025/4/21 20:26
# @File    : perturbation_strategy.py
# @Description    :
import copy
import torch
from torch_geometric.data import Data
from pyomo.environ import ConcreteModel, Var, Constraint, Objective, NonNegativeReals, SolverFactory, minimize, value, summation, ConstraintList, Param, Set


def solve_theta(dmu_index, inputs, outputs, reference_indices, solver_name="mosek"):
    """
    输入：
        dmu_index: 当前DMU编号（int）
        inputs: 所有DMU的投入（numpy array, shape: [n_dmu, n_inputs]）
        outputs: 所有DMU的产出（numpy array, shape: [n_dmu, n_outputs]）
        reference_indices: 用于参考集的DMU编号（list）
    返回：
        theta值（float）
    """
    import pyomo.environ as pyo

    m = inputs.shape[1]  # 输入维度
    s = outputs.shape[1]  # 输出维度
    R = reference_indices  # 参考集（已是0-based索引）

    model = pyo.ConcreteModel()

    model.J = Set(initialize=R)  # 参考集索引
    model.m = m
    model.s = s

    # 决策变量：θ, λ_j
    model.theta = Var(domain=NonNegativeReals)
    model.lambda_ = Var(model.J, domain=NonNegativeReals)

    # 目标函数：min θ
    model.obj = Objective(expr=model.theta, sense=minimize)

    # 输入约束：∑λ_j x_{jm} ≤ θ x_{im}
    def input_constraint_rule(model, i):
        return sum(model.lambda_[j] * inputs[j, i] for j in model.J) <= model.theta * inputs[dmu_index, i]
    model.input_constraints = Constraint(range(m), rule=input_constraint_rule)

    # 输出约束：∑λ_j y_{js} ≥ y_{is}
    def output_constraint_rule(model, r):
        return sum(model.lambda_[j] * outputs[j, r] for j in model.J) >= outputs[dmu_index, r]
    model.output_constraints = Constraint(range(s), rule=output_constraint_rule)

    # 凸性约束：∑λ_j = 1
    model.convexity = Constraint(expr=sum(model.lambda_[j] for j in model.J) == 1)

    # 求解
    solver = SolverFactory(solver_name)
    result = solver.solve(model, tee=False)

    if (result.solver.status == pyo.SolverStatus.ok) and (result.solver.termination_condition == pyo.TerminationCondition.optimal):
        return value(model.theta)
    else:
        return float('inf')  # 不可行或未找到最优解

def find_valid_perturbations_edges(inputs, outputs, theta_list, reference_sets, delta=0.01):
    """
    遍历找出所有可接受的扰动参考集（仅添加一个DMU，且效率变化≤delta）
    返回：字典，key为DMU编号，value为列表，每项是新的参考集索引
    """
    n_dmu = len(inputs)
    results = {}

    for i in range(n_dmu):
        original_ref = set(reference_sets[i])
        original_theta = theta_list[i]
        if original_theta < 1:
            candidates = [j for j in range(n_dmu) if j not in original_ref]
            valid_perturbations = []
            for j in candidates:
                new_ref = list(original_ref | {j})
                new_theta = solve_theta(i, inputs, outputs, new_ref)
                if abs(original_theta - new_theta) <= delta:
                    valid_perturbations.append([r + 1 for r in new_ref])  # 输出时转为1-based
            results[i + 1] = valid_perturbations  # key也变为1-based
        else:
            continue

    return results

def find_single_hyperedge_graphs(graph_list):
    """
    返回只包含一条超边的图的索引列表
    """
    single_edge_indices = []
    for idx, data in enumerate(graph_list):
        num_hyperedges = len(torch.unique(data.edge_index[1]))
        if num_hyperedges == 1:
            single_edge_indices.append(idx)
    return single_edge_indices

def add_noise_to_dmu_node(data, noise_std=0.1):
    """
    只对 data.graph_id 对应的节点特征添加高斯噪声，其余节点不变
    """
    x_noisy = data.x.clone()
    center_id = data.graph_id  # 当前图的 DMU 编号

    noise = torch.randn_like(x_noisy[center_id]) * noise_std
    x_noisy[center_id] += noise

    # 构造新图对象
    new_data = Data(
        x=x_noisy,
        edge_index=data.edge_index.clone(),
        edge_attr=data.edge_attr.clone()
    )
    new_data.graph_id = center_id
    return new_data

def check_noise_validity(data_noisy, inputs, outputs, theta_old, delta=0.01):
    """
    判断噪声扰动是否合理（效率变化不大）
    - data_noisy.graph_id 是当前DMU
    - inputs, outputs 是所有DMU的数据
    """
    dmu_index = data_noisy.graph_id

    outputs_new = data_noisy.x[:, :2].cpu().numpy()
    inputs_new = data_noisy.x[:, 2:].cpu().numpy()

    theta_new = solve_theta(dmu_index, inputs_new, outputs_new, reference_indices=list(range(len(inputs))))
    return abs(theta_new - theta_old[dmu_index]) <= delta


def apply_noise_to_unperturbed(graph_list, inputs, outputs, theta_list, delta=0.0001, noise_std=0.1, max_attempts=5, decay_rate=0.5):
    """
    对只包含一条超边的图添加噪声，且确保效率几乎不变，返回一个增强后的图列表
    """
    single_edge_ids = find_single_hyperedge_graphs(graph_list)
    new_graph_list = []

    for idx, data in enumerate(graph_list):
        if idx not in single_edge_ids:
            new_graph_list.append(data)  # 已扰动
            continue

        success = False
        current_std = noise_std

        for attempt in range(max_attempts):
            data_noisy = add_noise_to_dmu_node(data, noise_std=current_std)
            is_valid = check_noise_validity(data_noisy, inputs, outputs, theta_list, delta=delta)

            if is_valid:
                new_graph_list.append(data_noisy)
                success = True
                break
            else:
                current_std *= decay_rate  # 衰减噪声强度

        if not success:
            new_graph_list.append(data)

    return new_graph_list