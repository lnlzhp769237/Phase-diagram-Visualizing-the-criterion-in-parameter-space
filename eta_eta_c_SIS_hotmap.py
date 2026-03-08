
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PRL级别 - 最终优化版
改进：
1. 直接使用输入的β参数（不被平均化弱化）
2. H3_op组合估计（避免低估）
3. ρ0稳定估计（Gershgorin圆盘定理 + 幂法重启）
核心：η/η_c > 1 判为爆炸（完全遵循论文理论）
"""

import numpy as np
import networkx as nx
from scipy.linalg import norm
from scipy.interpolate import griddata
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
import pickle
import os
from datetime import datetime
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
import time
import warnings
warnings.filterwarnings('ignore')

# ==================== 论文常数 ====================

class PaperConstants:
    """论文中的经验常数 - 最终优化版"""
    
    # 模型特定常数
    C_contagion = 0.005      # SIS模型 (论文值)
    alpha = 0.3             # T对线性算子的影响
    kappa = 0.2             # T对η_c的影响
    gamma = 0.6            # β对η_c的影响（增强β效应）
    
    # 物理参数
    mu = 1.0                # 恢复率
    lambda1_base = 1.8     # 基础感染率
    
    # 滞后检测阈值 - 仅用于备用
    HYSTERESIS_THRESHOLD = 0.25
    
    # 数值限制
    MAX_ETA_RATIO = 5.0     # η/η_c上限
    MIN_ETA_RATIO = 0.1     # η/η_c下限


# ==================== 配置 ====================

class Config:
    """配置参数 - PRL级别"""
    def __init__(self, mode='test'):
        if mode == 'test':
            self.N = 50
            self.T_points = 10
            self.beta_points = 10
            self.lambda_ratios = [1.0, 1.4, 1.8, 2.2, 2.6, 3.0, 3.4, 4.0, 4.5]
            self.replicates = 3
            self.T_range = [0.2, 0.9]
            self.beta_range = [-0.5, 0.5]
            self.sim_time = 100
        elif mode == 'full':
            self.N = 200
            self.T_points = 15
            self.beta_points = 15
            self.lambda_ratios = [1.0, 1.8, 2.6, 3.5, 4.0, 4.6]
            self.replicates = 10
            self.T_range = [0.2, 0.95]
            self.beta_range = [-0.5, 0.5]
            self.sim_time = 200
        
        self.constants = PaperConstants()
    
    def total_points(self):
        return (len(self.lambda_ratios) * 
                self.T_points * 
                self.beta_points * 
                self.replicates)


# ==================== 网络生成 ====================

class NetworkGenerator:
    """生成具有可控T和β的网络"""
    
    @staticmethod
    def compute_T(G, triangles):
        """精确计算结构重叠T"""
        if not triangles:
            return 0.0
        
        n = G.number_of_nodes()
        triangle_set = set(tuple(sorted(t)) for t in triangles)
        
        T_values = []
        for node in range(n):
            neighbors = list(G.neighbors(node))
            if len(neighbors) < 2:
                continue
            
            actual = 0
            for i in range(len(neighbors)):
                for j in range(i+1, len(neighbors)):
                    if tuple(sorted([node, neighbors[i], neighbors[j]])) in triangle_set:
                        actual += 1
            
            possible = len(neighbors) * (len(neighbors) - 1) / 2
            if possible > 0:
                T_values.append(actual / possible)
        
        return np.mean(T_values) if T_values else 0.0
    
    @staticmethod
    def generate(N, target_T, beta, seed):
        """生成网络"""
        np.random.seed(seed)
        
        G = nx.Graph()
        G.add_nodes_from(range(N))
        
        # 添加随机边
        for i in range(N):
            for j in range(i+1, N):
                if np.random.random() < 0.15:
                    G.add_edge(i, j)
        
        # 添加三角形
        triangles = []
        n_tri = int(N * target_T * 8)
        
        for _ in range(n_tri):
            nodes = np.random.choice(N, 3, replace=False)
            i, j, k = sorted(nodes)
            tri = (i, j, k)
            triangles.append(tri)
            
            if not G.has_edge(i, j): G.add_edge(i, j)
            if not G.has_edge(i, k): G.add_edge(i, k)
            if not G.has_edge(j, k): G.add_edge(j, k)
        
        # 去重
        triangles = list(set(triangles))
        
        # 计算实际T
        actual_T = NetworkGenerator.compute_T(G, triangles)
        
        # 节点权重 (β效应) - 公式(1)
        degrees = np.array([G.degree(i) for i in range(N)])
        if np.max(degrees) > np.min(degrees):
            # 按度排序得到rank
            sorted_indices = np.argsort(-degrees)
            rank = np.zeros(N)
            for idx, node in enumerate(sorted_indices):
                rank[node] = idx
            rank = rank / (N - 1) if N > 1 else rank
            
            # 公式 (1): φ_i = (β/2) * (r_i/(N-1)) + (3-β)/4
            weights = beta/2 * rank + (3 - beta)/4
            weights = np.clip(weights, 0.1, 0.9)
        else:
            weights = np.ones(N) * 0.5
        
        return G, triangles, weights, actual_T


# ==================== SIS动力学 ====================

class SISDynamics:
    """SIS动力学模拟"""
    
    def __init__(self, constants):
        self.constants = constants
    
    def simulate(self, G, triangles, weights, lambda1, lambda2, T_sim, x0=None):
        """运行SIS"""
        n = G.number_of_nodes()
        
        if x0 is None:
            x = (np.random.rand(n) < 0.1).astype(np.float64)
        else:
            x = x0.copy().astype(np.float64)
        
        A = nx.to_numpy_array(G, dtype=np.float64)
        dt = 0.1
        mu = self.constants.mu
        
        history = []
        for step in range(int(T_sim/dt)):
            # 线性项
            linear = lambda1 * (A @ (weights * x))
            
            # 非线性项（三角形）
            nonlinear = np.zeros(n, dtype=np.float64)
            if len(triangles) > 0:
                sample_size = min(50, len(triangles))
                for _ in range(sample_size):
                    idx = np.random.randint(0, len(triangles))
                    i, j, k = triangles[idx]
                    contrib = lambda2 * weights[j] * weights[k] * x[j] * x[k]
                    nonlinear[i] += contrib
            
            # 更新
            dx = -mu * x + (1.0 - x) * (linear + nonlinear)
            x = x + dt * dx
            x = np.clip(x, 0.0, 1.0)
            
            history.append(np.mean(x))
            
            if step > 50 and len(history) > 50:
                if np.std(history[-30:]) < 1e-4:
                    break
        
        return x, history


# ==================== 分析工具 - 最终优化版 ====================

class Analyzer:
    """相变分析工具 - 最终优化版"""
    
    def __init__(self, constants, sis):
        self.constants = constants
        self.sis = sis
    
    def estimate_H3_op(self, M, triangles, degrees, n):
        """
        改进的H3_op估计 - 组合多种方法避免低估
        """
        H3_estimates = []
        
        # 方法1：谱范数（精确SVD）
        try:
            if n <= 500:
                _, s, _ = np.linalg.svd(M, compute_uv=False)
                H3_spec = s[0] if len(s) > 0 else 0
            else:
                from scipy.sparse.linalg import svds
                s = svds(M, k=1, return_singular_vectors=False)
                H3_spec = s[0] if len(s) > 0 else 0
            H3_estimates.append(H3_spec)
        except:
            pass
        
        # 方法2：Frobenius范数估计
        try:
            H3_frob = np.sqrt(np.sum(M**2)) / np.sqrt(n)
            H3_estimates.append(H3_frob)
        except:
            pass
        
        # 方法3：基于三角形数量的估计
        try:
            n_tri = len(triangles)
            avg_deg = np.mean(degrees) if len(degrees) > 0 else 1
            H3_tri = n_tri * 3 / np.sqrt(avg_deg + 1)
            H3_estimates.append(H3_tri)
        except:
            pass
        
        # 方法4：行和范数估计
        try:
            row_sums = np.sum(np.abs(M), axis=1)
            H3_row = np.max(row_sums)
            H3_estimates.append(H3_row)
        except:
            pass
        
        # 取最大值（避免低估）
        if H3_estimates:
            H3_op = max(H3_estimates)
        else:
            # 最坏情况估计
            H3_op = len(triangles) * 3
        
        return H3_op
    
    def estimate_rho0(self, L0):
        """
        改进的ρ0估计 - Gershgorin圆盘定理 + 幂法重启
        """
        n = L0.shape[0]
        
        # 方法1：Gershgorin圆盘定理给出的上界
        diag = np.diag(L0)
        off_diag = np.sum(np.abs(L0 - np.diag(diag)), axis=1)
        gershgorin_ub = np.max(np.abs(diag) + off_diag)
        
        # 方法2：幂法（带重启）
        rho_power = 0
        try:
            n_restarts = 3
            for restart in range(n_restarts):
                # 不同初始向量
                v = np.random.randn(n)
                v = v / (np.linalg.norm(v) + 1e-10)
                
                # 幂法迭代
                for _ in range(100):
                    w = L0 @ v
                    v_new = w / (np.linalg.norm(w) + 1e-10)
                    
                    # 检查收敛
                    if np.linalg.norm(v_new - v) < 1e-4:
                        break
                    v = v_new
                
                # Rayleigh商
                rho = np.abs(np.dot(v, L0 @ v))
                rho_power = max(rho_power, rho)
        except:
            pass
        
        # 方法3：对于小矩阵，直接用特征值
        rho_eig = 0
        if n <= 200:
            try:
                eigvals = np.linalg.eigvals(L0)
                rho_eig = np.max(np.abs(eigvals))
            except:
                pass
        
        # 方法4：行和范数估计
        try:
            row_sums = np.sum(np.abs(L0), axis=1)
            rho_row = np.max(row_sums)
        except:
            rho_row = 0
        
        # 组合估计 - 取合理值
        rho_candidates = [rho_power, rho_eig, rho_row, gershgorin_ub * 0.5]
        rho0 = max([c for c in rho_candidates if c > 0] + [1e-6])
        
        return rho0
    
    def theoretical_eta_c(self, G, triangles, weights, beta):
        """
        论文公式(11)的理论η_c - 直接使用输入的β参数
        η_c = C * (1+γβ)/(1+κT) * ||H^(3)||_op / ((1-αT) λ_max(A))
        """
        n = G.number_of_nodes()
        A = nx.to_numpy_array(G, dtype=np.float64)
        degrees = np.array([G.degree(i) for i in range(n)])
        
        # 邻接矩阵最大特征值
        try:
            if n <= 500:
                eigvals = np.linalg.eigvals(A)
                lambda_max_A = np.max(np.real(eigvals))
            else:
                from scipy.sparse.linalg import eigs
                eigvals, _ = eigs(A, k=1, which='LR')
                lambda_max_A = np.real(eigvals[0])
        except:
            lambda_max_A = np.sqrt(n)
        
        # T效应
        T_net = NetworkGenerator.compute_T(G, triangles)
        
        # 构建M矩阵（三角形邻接矩阵）
        M = np.zeros((n, n))
        for i, j, k in triangles:
            M[i, j] += 1
            M[i, k] += 1
            M[j, i] += 1
            M[j, k] += 1
            M[k, i] += 1
            M[k, j] += 1
        
        # ✅ 改进的H3_op估计
        H3_op = self.estimate_H3_op(M, triangles, degrees, n)
        
        # ✅ 直接使用输入的β参数（不被平均化弱化）
        beta_factor = (1 + self.constants.gamma * beta)
        T_factor = 1 / (1 + self.constants.kappa * T_net)
        
        eta_c = (self.constants.C_contagion * 
                beta_factor * 
                T_factor * 
                H3_op / ((1 - self.constants.alpha * T_net) * lambda_max_A + 1e-10))
        
        return eta_c, T_net, lambda_max_A
    
    def compute_eta(self, G, triangles, weights, beta, lambda1, lambda2, T_sim):
        """
        计算η/η_c - 传入beta参数
        """
        n = G.number_of_nodes()
        A = nx.to_numpy_array(G, dtype=np.float64)
        
        # 线性化矩阵
        L0 = lambda1 * np.diag(weights) @ A - self.constants.mu * np.eye(n)
        
        # ✅ 改进的ρ0估计
        rho0 = self.estimate_rho0(L0)
        
        # 稳态
        xstar, _ = self.sis.simulate(G, triangles, weights, lambda1, lambda2, T_sim)
        
        # ε估计
        epsilon_est = 0
        if len(triangles) > 0:
            sample_size = min(50, len(triangles))
            for _ in range(sample_size):
                idx = np.random.randint(0, len(triangles))
                i, j, k = triangles[idx]
                epsilon_est += lambda2 * weights[j] * weights[k] * xstar[j] * xstar[k]
        
        norm_L0 = np.linalg.norm(L0, ord=2)
        epsilon = epsilon_est / (norm_L0 + 1e-10)
        
        # η
        eta = epsilon / (abs(rho0) + 1e-10)
        
        # ✅ 传入beta
        eta_c, actual_T, lambda_max = self.theoretical_eta_c(G, triangles, weights, beta)
        
        eta_ratio = eta / eta_c if eta_c != 0 else 0
        eta_ratio = np.clip(eta_ratio, 
                           self.constants.MIN_ETA_RATIO, 
                           self.constants.MAX_ETA_RATIO)
        
        return {
            'eta': float(eta),
            'eta_c': float(eta_c),
            'eta_ratio': float(eta_ratio),
            'rho0': float(rho0),
            'epsilon': float(epsilon),
            'x_mean': float(np.mean(xstar)),
            'actual_T': float(actual_T),
            'lambda_max': float(lambda_max)
        }
    
    def check_explosive(self, G, triangles, weights, lambda2, T_sim, eta_ratio=None):
        """
        终极修复：完全基于论文理论
        核心：η/η_c > 1 就是爆炸
        """
        n = G.number_of_nodes()
        
        # 用两个λ1值测试（仅用于计算滞后，不用于判定）
        lambda1_low = 0.15
        lambda1_high = 0.35
        
        # 低初始条件
        x_low = np.random.rand(n) < 0.05
        final_low, _ = self.sis.simulate(G, triangles, weights, 
                                        lambda1_high, lambda2, T_sim//2, x_low)
        
        # 高初始条件
        x_high = np.ones(n, dtype=np.float64) * 0.9
        final_high, _ = self.sis.simulate(G, triangles, weights, 
                                         lambda1_low, lambda2, T_sim//2, x_high)
        
        low_mean = np.mean(final_low)
        high_mean = np.mean(final_high)
        hysteresis = abs(low_mean - high_mean)
        
        # ✅ 核心：完全基于η/η_c判定（论文理论）
        if eta_ratio is not None:
            is_explosive = eta_ratio > 1.0
        else:
            # 备用：仅当没有η/η_c时才用滞后
            is_explosive = hysteresis > self.constants.HYSTERESIS_THRESHOLD
        
        return is_explosive, low_mean, high_mean, hysteresis


# ==================== 单点模拟 ====================

def simulate_point(params):
    """单个参数点模拟 - 最终优化版"""
    T, beta, lam, rep, seed, config = params
    
    T = float(T)
    beta = float(beta)  # 保存原始beta
    lam = float(lam)
    rep = int(rep)
    seed = int(seed)
    
    constants = config.constants
    lambda2 = lam * constants.lambda1_base
    
    np.random.seed(seed)
    
    try:
        # 1. 生成网络
        G, triangles, weights, actual_T = NetworkGenerator.generate(
            config.N, T, beta, seed
        )
        
        # 2. 初始化动力学和分析器
        sis = SISDynamics(constants)
        analyzer = Analyzer(constants, sis)
        
        # 3. 计算η（传入beta）
        eta_results = analyzer.compute_eta(
            G, triangles, weights, beta,  # ✅ 传入beta参数
            constants.lambda1_base, lambda2, config.sim_time
        )
        
        # 4. 传入eta_ratio进行爆炸判定
        is_explosive, low_state, high_state, hysteresis = analyzer.check_explosive(
            G, triangles, weights, lambda2, config.sim_time, eta_results['eta_ratio']
        )
        
        result = {
            'T': T,
            'beta': beta,
            'lambda': lam,
            'rep': rep,
            'seed': seed,
            'actual_T': float(actual_T),
            'eta_ratio': eta_results['eta_ratio'],
            'eta': eta_results['eta'],
            'eta_c': eta_results['eta_c'],
            'is_explosive': bool(is_explosive),
            'hysteresis': float(hysteresis),
            'mean_activity': eta_results['x_mean'],
            'rho0': eta_results['rho0'],
            'n_triangles': len(triangles),
            'success': True
        }
        
    except Exception as e:
        print(f"错误: T={T:.2f}, β={beta:.2f}, λ={lam:.1f}, rep={rep}: {str(e)}")
        result = {'success': False}
    
    return result


# ==================== 统计分析 ====================

class Statistics:
    """统计分析 - 包含T和β的分组"""
    
    @staticmethod
    def compute_confidence_interval(data, confidence=0.95):
        """计算置信区间"""
        n = len(data)
        if n < 2:
            return 0, 0, 0
        
        mean = np.mean(data)
        std = np.std(data, ddof=1)
        se = std / np.sqrt(n)
        
        from scipy import stats
        ci = stats.t.interval(confidence, n-1, loc=mean, scale=se)
        
        return mean, std, (ci[1] - ci[0])/2
    
    @staticmethod
    def aggregate_results(results, config):
        """按λ, T, β分组统计"""
        # 按参数分组
        grouped = {}
        for r in results:
            if not r.get('success', False):
                continue
            # 按λ, T, β分组
            key = (r['lambda'], r['T'], r['beta'])
            if key not in grouped:
                grouped[key] = []
            grouped[key].append(r)
        
        # 计算统计量
        stats = []
        for (lam, T, beta), reps in grouped.items():
            eta_ratios = [r['eta_ratio'] for r in reps]
            explosive_flags = [r['is_explosive'] for r in reps]
            
            if len(eta_ratios) >= 2:
                eta_mean, eta_std, eta_ci = Statistics.compute_confidence_interval(eta_ratios)
                exp_prob = np.mean(explosive_flags)
                exp_std = np.std(explosive_flags) if len(explosive_flags) > 1 else 0
                
                stats.append({
                    'lambda': lam,
                    'T': T,
                    'beta': beta,
                    'eta_mean': eta_mean,
                    'eta_std': eta_std,
                    'eta_ci': eta_ci,
                    'explosive_prob': exp_prob,
                    'explosive_std': exp_std,
                    'n_replicates': len(reps),
                    'actual_T_mean': np.mean([r['actual_T'] for r in reps])
                })
        
        return stats


# ==================== 可视化 ====================

class Visualizer:
    """可视化器"""
    
    def __init__(self, output_dir, constants):
        self.output_dir = output_dir
        self.constants = constants
        
        plt.rcParams.update({
            'font.family': 'serif',
            'font.size': 11,
            'axes.labelsize': 12,
            'axes.titlesize': 13,
            'legend.fontsize': 10,
            'figure.titlesize': 14,
            'figure.facecolor': 'white',
            'axes.facecolor': 'white',
            'axes.grid': True,
            'grid.alpha': 0.3
        })
    
    def plot_statistics(self, stats, config):
        """统计分析图"""
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        lambda_vals = sorted(set([s['lambda'] for s in stats]))
        
        # (a) 爆炸概率 vs λ
        ax1 = axes[0, 0]
        probs = []
        prob_errs = []
        for lam in lambda_vals:
            subset = [s for s in stats if abs(s['lambda'] - lam) < 0.1]
            if subset:
                probs.append(np.mean([s['explosive_prob'] for s in subset]) * 100)
                prob_errs.append(np.std([s['explosive_prob'] for s in subset]) * 100)
            else:
                probs.append(0)
                prob_errs.append(0)
        
        ax1.errorbar(lambda_vals, probs, yerr=prob_errs, fmt='o-', 
                    capsize=5, color='darkred', markersize=8, linewidth=2)
        ax1.axhline(y=50, color='gray', linestyle='--', alpha=0.5, label='50% threshold')
        ax1.set_xlabel(r'$\lambda_2/\lambda_1$')
        ax1.set_ylabel('Explosive probability (%)')
        ax1.set_title('(a) Explosive probability vs nonlinearity')
        ax1.set_xlim(min(lambda_vals)-0.2, max(lambda_vals)+0.2)
        ax1.set_ylim(0, 100)
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # (b) η/η_c分布
        ax2 = axes[0, 1]
        box_data = []
        positions = []
        for i, lam in enumerate(lambda_vals):
            subset = [s for s in stats if abs(s['lambda'] - lam) < 0.1]
            eta_vals = [s['eta_mean'] for s in subset]
            if eta_vals:
                box_data.append(eta_vals)
                positions.append(i+1)
        
        if box_data:
            bp = ax2.boxplot(box_data, positions=positions, widths=0.6,
                            patch_artist=True, showmeans=True,
                            meanprops={'marker': 'D', 'markerfacecolor': 'red',
                                      'markeredgecolor': 'red', 'markersize': 8})
            
            # 着色
            colors = plt.cm.rainbow(np.linspace(0, 1, len(box_data)))
            for patch, color in zip(bp['boxes'], colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.5)
        
        ax2.axhline(y=1, color='black', linestyle='--', linewidth=2, 
                   label=r'$\eta/\eta_c=1$')
        ax2.set_xlabel(r'$\lambda_2/\lambda_1$')
        ax2.set_ylabel(r'$\eta/\eta_c$')
        ax2.set_title('(b) η/η_c distribution')
        ax2.set_xticks(positions)
        ax2.set_xticklabels([f'{lam:.1f}' for lam in lambda_vals if lam in [lambda_vals[i] for i in range(len(positions))]])
        ax2.grid(True, alpha=0.3, axis='y')
        ax2.legend()
        
        # (c) T vs 爆炸概率
        ax3 = axes[1, 0]
        T_bins = np.linspace(config.T_range[0], config.T_range[1], 5)
        T_centers = []
        T_probs = []
        T_errs = []
        for i in range(len(T_bins)-1):
            mask = [(s['T'] >= T_bins[i]) and (s['T'] < T_bins[i+1]) for s in stats]
            if any(mask):
                probs_bin = [s['explosive_prob'] for s, m in zip(stats, mask) if m]
                T_centers.append((T_bins[i] + T_bins[i+1])/2)
                T_probs.append(np.mean(probs_bin) * 100)
                T_errs.append(np.std(probs_bin) * 100)
        
        ax3.errorbar(T_centers, T_probs, yerr=T_errs, fmt='s-',
                    capsize=5, color='green', linewidth=2)
        ax3.set_xlabel(r'$T$')
        ax3.set_ylabel('Explosive probability (%)')
        ax3.set_title('(c) Effect of structural overlap T')
        ax3.set_xlim(config.T_range)
        ax3.set_ylim(0, 100)
        ax3.grid(True, alpha=0.3)
        
        # (d) β vs 爆炸概率
        ax4 = axes[1, 1]
        beta_bins = np.linspace(config.beta_range[0], config.beta_range[1], 5)
        beta_centers = []
        beta_probs = []
        beta_errs = []
        for i in range(len(beta_bins)-1):
            mask = [(s['beta'] >= beta_bins[i]) and (s['beta'] < beta_bins[i+1]) for s in stats]
            if any(mask):
                probs_bin = [s['explosive_prob'] for s, m in zip(stats, mask) if m]
                beta_centers.append((beta_bins[i] + beta_bins[i+1])/2)
                beta_probs.append(np.mean(probs_bin) * 100)
                beta_errs.append(np.std(probs_bin) * 100)
        
        ax4.errorbar(beta_centers, beta_probs, yerr=beta_errs, fmt='d-',
                    capsize=5, color='purple', linewidth=2)
        ax4.set_xlabel(r'$\beta$')
        ax4.set_ylabel('Explosive probability (%)')
        ax4.set_title('(d) Effect of heterogeneity β')
        ax4.set_xlim(config.beta_range)
        ax4.set_ylim(0, 100)
        ax4.grid(True, alpha=0.3)
        
        plt.suptitle(r'Figure 2: Statistical analysis', fontsize=14)
        plt.tight_layout()
        
        plt.savefig(os.path.join(self.output_dir, 'fig2_statistics.png'), 
                   dpi=300, bbox_inches='tight')
        plt.savefig(os.path.join(self.output_dir, 'fig2_statistics.pdf'),
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  ✅ 图2: 统计图已保存")
    
    def plot_phase_diagram(self, stats, config):
        """图1: 相图矩阵"""
        
        lambda_vals = sorted(set([s['lambda'] for s in stats]))
        
        # 彩虹色图
        colors = ['#0000FF', '#0080FF', '#00FFFF', '#00FF80', 
                  '#80FF00', '#FFFF00', '#FF8000', '#FF0000']
        cmap = LinearSegmentedColormap.from_list('rainbow', colors, N=256)
        
        n_lambda = len(lambda_vals)
        n_cols = min(3, n_lambda)
        n_rows = (n_lambda + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
        if n_rows == 1 and n_cols == 1:
            axes = np.array([axes])
        axes = axes.flatten()
        
        titles = {
            1.0: r'(a) $\lambda_2/\lambda_1=1.0$',
            1.8: r'(b) $\lambda_2/\lambda_1=1.4$',
            2.6: r'(c) $\lambda_2/\lambda_1=1.8$',
            3.5: r'(d) $\lambda_2/\lambda_1=2.2$',
            4.0: r'(e) $\lambda_2/\lambda_1=2.6$',
            4.6: r'(g) $\lambda_2/\lambda_1=3.4$'
        }
 
        for idx, lam in enumerate(lambda_vals):
            ax = axes[idx]
            
            subset = [s for s in stats if abs(s['lambda'] - lam) < 0.1]
            
            if not subset:
                ax.text(0.5, 0.5, 'No data', ha='center', va='center')
                continue
            
            T_vals = np.array([s['T'] for s in subset])
            beta_vals = np.array([s['beta'] for s in subset])
            eta_mean = np.array([s['eta_mean'] for s in subset])
            exp_prob = np.array([s['explosive_prob'] for s in subset])
            
            # 插值网格
            T_grid = np.linspace(config.T_range[0], config.T_range[1], 50)
            beta_grid = np.linspace(config.beta_range[0], config.beta_range[1], 50)
            T_mesh, beta_mesh = np.meshgrid(T_grid, beta_grid)
            
            eta_grid = griddata((T_vals, beta_vals), eta_mean, 
                               (T_mesh, beta_mesh), method='linear')
            
            # 使用TwoSlopeNorm突出η/η_c=1
            norm = TwoSlopeNorm(vmin=0, vcenter=1, vmax=config.constants.MAX_ETA_RATIO)
            
            im = ax.pcolormesh(T_mesh, beta_mesh, eta_grid,
                              cmap='RdBu_r', norm=norm,
                              shading='auto', alpha=0.9)
            
            # 临界线
            if not np.all(np.isnan(eta_grid)) and np.nanmin(eta_grid) <= 1 <= np.nanmax(eta_grid):
                # 将白色改为红色
                critical = ax.contour(T_mesh, beta_mesh, eta_grid, levels=[1],
                     colors='red', linewidths=2.5, linestyles='--')  # 红色更清晰
                ax.clabel(critical, inline=True, fontsize=10, fmt=r'$\eta/\eta_c=1$',colors='red')
            
            # 散点 - 颜色代表爆炸概率
            scatter = ax.scatter(T_vals, beta_vals, c=exp_prob, 
                                cmap='RdYlBu_r', vmin=0, vmax=1,
                                s=30, alpha=0.7, edgecolors='black', linewidth=0.5)
            
            ax.set_xlabel(r'$T$')
            ax.set_ylabel(r'$\beta$')
            ax.set_title(titles.get(lam, rf'$\lambda_2/\lambda_1={lam:.1f}$'))
            ax.set_xlim(config.T_range)
            ax.set_ylim(config.beta_range)
            
            # 添加统计
            exp_pct = np.mean(exp_prob) * 100
            ax.text(0.05, 0.95, f'Explosive: {exp_pct:.0f}%',
                   transform=ax.transAxes, fontsize=9,
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
            plt.colorbar(im, ax=ax, label=r'$\eta/\eta_c$', shrink=0.8)
        
        for idx in range(len(lambda_vals), len(axes)):
            axes[idx].set_visible(False)
        
        plt.suptitle(r'Figure 1: Phase diagram for SIS dynamics', fontsize=14)
        plt.tight_layout()
        
        plt.savefig(os.path.join(self.output_dir, 'fig1_phase_diagram.png'), 
                   dpi=300, bbox_inches='tight')
        plt.savefig(os.path.join(self.output_dir, 'fig1_phase_diagram.pdf'),
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  ✅ 图1: 相图已保存")


# ==================== 主程序 ====================

def main():
    print("="*70)
    print("PRL级别 - 最终优化版")
    print("改进: 直接使用β参数 + H3_op组合估计 + ρ0稳定估计")
    print("核心: η/η_c > 1 判为爆炸（完全遵循论文理论）")
    print("="*70)
    
    # 选择模式
    print("\n选择模式:")
    print("  1. test  - 测试模式 (10×10网格, 3次重复)")
    print("  2. full  - 完整模式 (20×20网格, 5次重复)")
    
    choice = input("请输入选择 (1-2, 默认=1): ").strip() or "1"
    mode = 'test' if choice == '1' else 'full'
    
    config = Config(mode)
    
    print(f"\n📊 配置参数:")
    print(f"  λ值: {config.lambda_ratios}")
    print(f"  T点数: {config.T_points}")
    print(f"  β点数: {config.beta_points}")
    print(f"  重复: {config.replicates}")
    print(f"  总点数: {config.total_points()}")
    print(f"  gamma (β效应): {config.constants.gamma}")
    print(f"  lambda1_base: {config.constants.lambda1_base}")
    
    # 创建输出目录
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = f'PRL_optimized_{mode}_{timestamp}'
    os.makedirs(output_dir, exist_ok=True)
    
    # 生成参数列表
    params = []
    seed_base = 42
    idx = 0
    
    T_vals = np.linspace(config.T_range[0], config.T_range[1], config.T_points)
    beta_vals = np.linspace(config.beta_range[0], config.beta_range[1], config.beta_points)
    
    for T in T_vals:
        for beta in beta_vals:
            for lam in config.lambda_ratios:
                for rep in range(config.replicates):
                    params.append((T, beta, lam, rep, seed_base + idx, config))
                    idx += 1
    
    print(f"\n🚀 开始模拟，使用 {min(cpu_count(), 8)} 进程...")
    
    start_time = time.time()
    
    with Pool(processes=min(cpu_count(), 8)) as pool:
        results = list(tqdm(
            pool.imap_unordered(simulate_point, params),
            total=len(params),
            desc="模拟进度"
        ))
    
    elapsed = time.time() - start_time
    successful = [r for r in results if r.get('success', False)]
    
    print(f"\n✅ 用时: {elapsed/60:.1f} 分钟")
    print(f"   成功: {len(successful)}/{len(results)} ({100*len(successful)/len(results):.1f}%)")
    
    if successful:
        # 保存原始数据
        with open(os.path.join(output_dir, 'raw_results.pkl'), 'wb') as f:
            pickle.dump(successful, f)
        
        # 统计分析
        print("\n📈 进行统计分析...")
        stats = Statistics.aggregate_results(successful, config)
        
        # 保存统计结果
        with open(os.path.join(output_dir, 'statistics.pkl'), 'wb') as f:
            pickle.dump(stats, f)
        
        # 生成可视化
        print("\n🎨 生成可视化...")
        viz = Visualizer(output_dir, config.constants)
        viz.plot_phase_diagram(stats, config)
        viz.plot_statistics(stats, config)
        
        # 打印详细统计
        print("\n📊 统计摘要:")
        print("-" * 80)
        print(f"{'λ':>6} | {'爆炸率':>12} | {'η/η_c均值':>12} | {'95% CI':>12} | {'η/η_c>1比例':>14}")
        print("-" * 80)
        
        for lam in sorted(set([s['lambda'] for s in stats])):
            subset = [s for s in stats if abs(s['lambda'] - lam) < 0.1]
            if not subset:
                continue
            eta_means = [s['eta_mean'] for s in subset]
            exp_probs = [s['explosive_prob'] for s in subset]
            
            eta_avg = np.mean(eta_means)
            exp_avg = np.mean(exp_probs) * 100
            ci = 1.96 * np.std(eta_means) / np.sqrt(len(subset)) if len(subset) > 1 else 0
            eta_gt1 = np.mean([e > 1.0 for e in eta_means]) * 100
            
            print(f" {lam:4.1f} | {exp_avg:10.1f}% | {eta_avg:10.3f} | ±{ci:8.3f} | {eta_gt1:12.1f}%")
        
        print("-" * 80)
        
        # 检查T和β效应
        print("\n📈 T效应检验:")
        T_low = [s for s in stats if s['T'] < 0.4]
        T_high = [s for s in stats if s['T'] > 0.7]
        if T_low and T_high:
            print(f"  T<0.4: {np.mean([s['explosive_prob'] for s in T_low])*100:.1f}%爆炸")
            print(f"  T>0.7: {np.mean([s['explosive_prob'] for s in T_high])*100:.1f}%爆炸")
        
        print("\n📈 β效应检验:")
        beta_neg = [s for s in stats if s['beta'] < -0.2]
        beta_pos = [s for s in stats if s['beta'] > 0.2]
        if beta_neg and beta_pos:
            print(f"  β<-0.2: {np.mean([s['explosive_prob'] for s in beta_neg])*100:.1f}%爆炸")
            print(f"  β>0.2: {np.mean([s['explosive_prob'] for s in beta_pos])*100:.1f}%爆炸")
        
        print(f"\n✅ 所有结果保存在: {output_dir}")
        print(f"   生成的文件:")
        print(f"     - fig1_phase_diagram.png/pdf (相图矩阵)")
        print(f"     - fig2_statistics.png/pdf (统计分析)")
        print(f"     - raw_results.pkl (原始数据)")
        print(f"     - statistics.pkl (统计结果)")
    else:
        print("❌ 没有成功的数据点")

if __name__ == "__main__":
    from multiprocessing import freeze_support
    freeze_support()
    main()