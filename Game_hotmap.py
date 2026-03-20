"""
Generate heatmap of η/η_c in (T, β) plane for Evolutionary Game dynamics (Public Goods Game).
Version with tunable r_ratio and subcritical factor to control nonlinear strength and distance from critical point.
Outputs high-quality PDF and EPS figures.
"""

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from scipy.linalg import eig, svd
from scipy.integrate import solve_ivp
from scipy.interpolate import griddata
from scipy.sparse import csr_matrix
from dataclasses import dataclass, replace
from multiprocessing import Pool, cpu_count
import time
import warnings
warnings.filterwarnings('ignore')

# ==================== Global Parameters ====================
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

ODE_T_MAX = 50
ODE_RTOL = 1e-3
ODE_ATOL = 1e-4
MAX_STEP = 2.0
STEADY_MEAN_LIMIT = 0.05          # threshold for low-cooperation steady state (x < 0.05)
TARGET_RHO = -0.01
P_SUB_REDUCTION_FACTOR = 0.8
MAX_RETRIES = 2
DEFAULT_COST = 0.1                 # cost parameter c
DEFAULT_ALPHA_MODEL = 0.8
DEFAULT_GAMMA_MODEL = 0.3
DEFAULT_KAPPA_MODEL = 1.0

# ==================== Helper Functions ====================
def h_beta(beta, delta):
    """Heterogeneity factor h(β,δ) = 4/(3-β+4δ)"""
    return 4.0 / (3.0 - beta + 4.0 * delta)

def compute_spectral_norm_W3(W3_sparse, N, exact_threshold=50):
    """Spectral norm of the three‑body tensor (Appendix A)"""
    if N <= exact_threshold:
        W_dense = np.zeros((N, N * N))
        for i in range(N):
            for j, k in W3_sparse[i]:
                col = j * N + k
                W_dense[i, col] += 1.0
        s = np.linalg.svd(W_dense, compute_uv=False)
        return s[0]
    else:
        x = np.random.randn(N)
        x /= np.linalg.norm(x)
        for _ in range(100):
            Wx = np.zeros(N)
            for i in range(N):
                s_val = 0.0
                for j, k in W3_sparse[i]:
                    s_val += x[j] * x[k]
                Wx[i] = s_val
            norm_new = np.linalg.norm(Wx)
            if norm_new < 1e-12:
                break
            x_new = Wx / norm_new
            if np.linalg.norm(x_new - x) < 1e-10:
                break
            x = x_new
        return norm_new

# ==================== Hypergraph Class ====================
@dataclass
class Hypergraph:
    N: int
    A_proj_sparse: csr_matrix
    W3_sparse: list
    T_mean: float
    avg_k: float
    avg_k2: float
    lambda_max: float
    W3_norm: float

    @staticmethod
    def generate_ER(N: int, p_edge: float, seed: int = None) -> 'Hypergraph':
        if seed is not None:
            np.random.seed(seed)
        G = nx.erdos_renyi_graph(N, p_edge)
        return Hypergraph._from_nx_graph(G, N)

    @staticmethod
    def _from_nx_graph(G, N):
        A_base = nx.to_numpy_array(G)
        # extract triangles
        triplets = []
        for i in range(N):
            for j in range(i+1, N):
                if A_base[i, j] > 0:
                    for k in range(j+1, N):
                        if A_base[i, k] > 0 and A_base[j, k] > 0:
                            triplets.append((i, j, k))
        # projected adjacency matrix
        A_proj_dense = np.zeros((N, N))
        for i, j, k in triplets:
            A_proj_dense[i, j] += 0.5
            A_proj_dense[i, k] += 0.5
            A_proj_dense[j, i] += 0.5
            A_proj_dense[j, k] += 0.5
            A_proj_dense[k, i] += 0.5
            A_proj_dense[k, j] += 0.5
        A_proj_sparse = csr_matrix(A_proj_dense)

        # three‑body sparse representation
        W3_sparse = [[] for _ in range(N)]
        for a, b, c in triplets:
            W3_sparse[a].extend([(b, c), (c, b)])
            W3_sparse[b].extend([(a, c), (c, a)])
            W3_sparse[c].extend([(a, b), (b, a)])

        max_possible = N * (N - 1) * (N - 2) // 6
        T_mean = len(triplets) / max_possible if max_possible else 0
        degrees = np.sum(A_base > 0, axis=1)
        avg_k = np.mean(degrees)
        avg_k2 = np.mean(degrees ** 2)
        lambda_max = np.max(np.real(eig(A_proj_dense)[0]))
        W3_norm = compute_spectral_norm_W3(W3_sparse, N, exact_threshold=50)
        return Hypergraph(N, A_proj_sparse, W3_sparse, T_mean, avg_k, avg_k2, lambda_max, W3_norm)


# ==================== Evolutionary Game Dynamics ====================
@dataclass
class GameParams:
    beta: float
    r: float          # benefit (multiplication factor)
    c: float = DEFAULT_COST
    delta: float = DEFAULT_COST   # used for h_beta
    alpha_model: float = DEFAULT_ALPHA_MODEL
    gamma_model: float = DEFAULT_GAMMA_MODEL
    kappa_model: float = DEFAULT_KAPPA_MODEL

class GameDynamics:
    """Evolutionary game (public goods) dynamics, Eqs. (18)-(20)"""

    @staticmethod
    def ode(t, s, hg, params):
        N = hg.N
        factor = (1 - params.alpha_model * hg.T_mean) * h_beta(params.beta, params.delta)
        r_eff = params.r * factor
        c_eff = params.c * factor

        ds = np.zeros(N)
        Pi = np.zeros(N)  # payoff

        # Calculate payoffs
        for i in range(N):
            Pi[i] -= c_eff * s[i]  # cost of cooperating
            # pairwise interactions
            for j in range(N):
                if hg.A_proj_sparse[i, j] > 0:
                    Pi[i] += r_eff / 2 * hg.A_proj_sparse[i, j] * s[j]
            # three-body interactions
            for j, k in hg.W3_sparse[i]:
                Pi[i] += r_eff / 3 * s[j] * s[k]

        # Average payoff of neighbors
        Pi_bar = np.zeros(N)
        for i in range(N):
            neighbors = hg.A_proj_sparse[i].indices
            if len(neighbors) > 0:
                Pi_bar[i] = np.mean(Pi[neighbors])

        # Replicator dynamics
        for i in range(N):
            ds[i] = s[i] * (1 - s[i]) * (Pi[i] - Pi_bar[i])

        return ds

    @staticmethod
    def steady_state(hg, params, t_max=None, x0=None):
        if t_max is None:
            t_max = ODE_T_MAX
        if x0 is None:
            x0 = np.random.rand(hg.N) * 1e-4  # start near zero cooperation
        try:
            sol = solve_ivp(GameDynamics.ode, [0, t_max], x0,
                            args=(hg, params), method='LSODA',
                            rtol=ODE_RTOL, atol=ODE_ATOL, max_step=MAX_STEP)
            s_star = sol.y[:, -1]
            if np.any(np.isnan(s_star)) or np.any(np.isinf(s_star)):
                return None
            return s_star
        except Exception:
            return None

    @staticmethod
    def linear_operator(s_star, hg, params):
        N = hg.N
        factor = (1 - params.alpha_model * hg.T_mean) * h_beta(params.beta, params.delta)
        r_eff = params.r * factor
        c_eff = params.c * factor

        L0 = np.zeros((N, N))
        A_dense = hg.A_proj_sparse.toarray()

        # Compute payoffs and their derivatives
        Pi = np.zeros(N)
        for i in range(N):
            Pi[i] = -c_eff * s_star[i]
            for j in range(N):
                if A_dense[i, j] > 0:
                    Pi[i] += r_eff / 2 * A_dense[i, j] * s_star[j]
            for j, k in hg.W3_sparse[i]:
                Pi[i] += r_eff / 3 * s_star[j] * s_star[k]

        # Average neighbor payoff
        Pi_bar = np.zeros(N)
        deg = np.array(hg.A_proj_sparse.sum(axis=1)).flatten()
        for i in range(N):
            neighbors = hg.A_proj_sparse[i].indices
            if len(neighbors) > 0:
                Pi_bar[i] = np.mean(Pi[neighbors])

        # Derivative of payoff w.r.t. s
        Pderiv = np.zeros((N, N))
        for i in range(N):
            Pderiv[i, i] -= c_eff
            for j in range(N):
                if A_dense[i, j] > 0:
                    Pderiv[i, j] += r_eff / 2 * A_dense[i, j]
            for j, k in hg.W3_sparse[i]:
                Pderiv[i, j] += r_eff / 3 * s_star[k]
                Pderiv[i, k] += r_eff / 3 * s_star[j]

        # Derivative of average neighbor payoff
        Pbar_deriv = np.zeros((N, N))
        for i in range(N):
            d = deg[i]
            if d > 0:
                for k in hg.A_proj_sparse[i].indices:
                    for j in range(N):
                        Pbar_deriv[i, j] += Pderiv[k, j] / d

        g = Pi - Pi_bar
        for i in range(N):
            for j in range(N):
                if i == j:
                    L0[i, i] += (1 - 2 * s_star[i]) * g[i]
                L0[i, j] += s_star[i] * (1 - s_star[i]) * (Pderiv[i, j] - Pbar_deriv[i, j])

        return L0

    @staticmethod
    def nonlinear_jacobian(s_star, hg, params):
        N = hg.N
        factor = (1 - params.alpha_model * hg.T_mean) * h_beta(params.beta, params.delta)
        r_eff = params.r * factor
        Jnl = np.zeros((N, N))
        for i in range(N):
            for j, k in hg.W3_sparse[i]:
                term = r_eff / 3 * s_star[i] * (1 - s_star[i])
                Jnl[i, j] += term * s_star[k]
                Jnl[i, k] += term * s_star[j]
        return Jnl

    @staticmethod
    def estimate_critical_r(hg, params):
        """
        Estimate critical r (benefit) using mean‑field approximation.
        For evolutionary game, linear stability of zero state gives:
            maximum eigenvalue ≈ -c + (r/2) * factor * λ_max
        Setting to zero yields r_c = 2c / (factor * λ_max)
        """
        factor = (1 - params.alpha_model * hg.T_mean) * h_beta(params.beta, params.delta)
        r_c = (2 * params.c) / (factor * hg.lambda_max)
        return r_c

    @staticmethod
    def compute_eta_ratio(hg, params, r_ratio, subcritical_factor):
        """
        Compute η/η_c for evolutionary game.
        
        Parameters:
        - r_ratio: scaling factor for three-body interactions (smaller = weaker nonlinearity)
        - subcritical_factor: how far below critical r to operate (e.g., 0.5 = 50% of critical)
        """
        try:
            # 1. Estimate critical r based on linear part only
            r_c = GameDynamics.estimate_critical_r(hg, params)
            
            # 2. Choose subcritical r for linear part (conservative, away from critical point)
            r_linear = subcritical_factor * r_c
            
            # 3. Create parameter set with linear r
            params_sub = replace(params, r=r_linear)
            
            # 4. Find low‑cooperation steady state
            s_star = GameDynamics.steady_state(hg, params_sub)
            if s_star is None:
                return np.nan
            mean_x = np.mean(s_star)
            print(f"      mean cooperation = {mean_x:.6f}")

            # If still too cooperative, reduce further
            retries = 0
            current_factor = subcritical_factor
            while mean_x > STEADY_MEAN_LIMIT and retries < MAX_RETRIES:
                current_factor *= P_SUB_REDUCTION_FACTOR
                r_linear = current_factor * r_c
                params_sub = replace(params, r=r_linear)
                s_star = GameDynamics.steady_state(hg, params_sub)
                if s_star is None:
                    return np.nan
                mean_x = np.mean(s_star)
                print(f"      retry {retries+1}: mean_x = {mean_x:.6f}, r = {r_linear:.6f}")
                retries += 1

            if mean_x > STEADY_MEAN_LIMIT:
                print(f"      No low‑cooperation state found after {MAX_RETRIES} retries.")
                return np.nan

            # 5. Compute L0 (linear operator)
            L0 = GameDynamics.linear_operator(s_star, hg, params_sub)
            
            # 6. Compute Jnl (nonlinear Jacobian) and scale by r_ratio
            Jnl = GameDynamics.nonlinear_jacobian(s_star, hg, params_sub)
            Jnl = Jnl * r_ratio   # scale the nonlinear Jacobian

            # 7. Compute eigenvalues and spectral abscissa
            eigvals = eig(L0)[0]
            rho0 = np.max(np.real(eigvals))
            rho0_abs = max(abs(rho0), 1e-12)

            # 8. Compute norms and eta
            norm_L0 = svd(L0, compute_uv=False)[0]
            norm_Jnl = svd(Jnl, compute_uv=False)[0]
            epsilon = norm_Jnl / max(norm_L0, 1e-12)
            eta = epsilon / rho0_abs

            # 9. theoretical η_c for evolutionary game (Eq. 11)
            C = 2.0 / 3.0
            numerator = (1 + params.gamma_model * params.beta) * hg.W3_norm
            denominator = (1 + params.kappa_model * hg.T_mean) * (1 - params.alpha_model * hg.T_mean) * hg.lambda_max
            eta_c = C * numerator / denominator if denominator > 1e-12 else np.inf
            eta_ratio = eta / eta_c if eta_c > 0 else np.inf

            # 10. Detailed output
            print(f"      r_c = {r_c:.6f}, r_linear = {r_linear:.6f}, mean_x = {mean_x:.6f}")
            print(f"      η = {eta:.6f}, η_c = {eta_c:.6f}, η/η_c = {eta_ratio:.6f}")

            return eta_ratio

        except Exception as e:
            print(f"      Exception in compute_eta_ratio: {e}")
            return np.nan


# ==================== Generate Hypergraph with Target T ====================
def generate_hypergraph_with_target_T(N, target_T, tol=0.01, max_trials=20, seed=None):
    """
    Generate ER hypergraph whose T_mean is as close as possible to target_T.
    Returns a Hypergraph object (best approximation) or None if all attempts fail.
    """
    low_p, high_p = 0.01, 0.8
    best_hg = None
    best_err = np.inf
    for i in range(max_trials):
        current_seed = seed + i if seed is not None else None
        p = (low_p + high_p) / 2
        hg = Hypergraph.generate_ER(N, p, seed=current_seed)
        err = abs(hg.T_mean - target_T)
        if err < best_err:
            best_err = err
            best_hg = hg
        if err < tol:
            return hg
        if hg.T_mean < target_T:
            low_p = p
        else:
            high_p = p
        if high_p - low_p < 1e-6:
            break
    return best_hg


# ==================== Worker for Parallel Execution ====================
def worker(args):
    T, beta, r_ratio, subcritical_factor, case_id = args
    t_start = time.time()
    print(f"[{time.strftime('%H:%M:%S')}] Start T={T:.3f}, β={beta:.3f}, case={case_id}")
    try:
        # generate hypergraph with target T
        print(f"[{time.strftime('%H:%M:%S')}]   Generating hypergraph...")
        seed = RANDOM_SEED + case_id
        hg = generate_hypergraph_with_target_T(N=150, target_T=T, max_trials=10, seed=seed)
        if hg is None:
            print(f"[{time.strftime('%H:%M:%S')}]   Failed to generate hypergraph")
            return {'T': T, 'beta': beta, 'eta_ratio': np.nan}

        # set base parameters (β only, r determined dynamically)
        params = GameParams(beta=beta, r=0.0)  # placeholder

        # compute η/η_c
        print(f"[{time.strftime('%H:%M:%S')}]   Computing η/η_c...")
        eta_ratio = GameDynamics.compute_eta_ratio(hg, params, r_ratio, subcritical_factor)
        elapsed = time.time() - t_start
        print(f"[{time.strftime('%H:%M:%S')}]   Done T={T:.3f}, β={beta:.3f}, η/η_c={eta_ratio:.4f}, time={elapsed:.1f}s")
        return {'T': T, 'beta': beta, 'eta_ratio': eta_ratio}
    except Exception as e:
        print(f"[{time.strftime('%H:%M:%S')}]   Error T={T:.3f}, β={beta:.3f}: {e}")
        return {'T': T, 'beta': beta, 'eta_ratio': np.nan}


# ==================== Main ====================
def main():
    # ===== 可调参数 =====
    r_ratio = 0.012             # 非线性强度缩放因子 (尝试: 0.02, 0.005, 0.001, 0.0005)
    subcritical_factor = 0.5      # 亚临界系数 (尝试: 0.5, 0.3, 0.2)
    # ===================
    
    N = 150
    # Use a coarse grid for quick testing; increase to 15×15 for final figure
    T_vals = np.linspace(0.2, 0.95, 5)
    beta_vals = np.linspace(-0.5, 0.5, 5)

    print(f"Running with r_ratio = {r_ratio}, subcritical_factor = {subcritical_factor}")
    print(f"Using cost c = {DEFAULT_COST}")

    # build job list
    jobs = []
    case_id = 0
    for T in T_vals:
        for beta in beta_vals:
            jobs.append((T, beta, r_ratio, subcritical_factor, case_id))
            case_id += 1

    print(f"Starting {len(jobs)} jobs with {min(cpu_count(), 8)} processes...")
    # For debugging, set processes=1
    # with Pool(processes=1) as pool:
    with Pool(processes=min(cpu_count(), 8)) as pool:
        results = pool.map(worker, jobs)

    # collect results into grid
    eta_grid = np.full((len(T_vals), len(beta_vals)), np.nan)
    for res in results:
        if res is not None:
            i = np.where(np.abs(T_vals - res['T']) < 1e-6)[0]
            j = np.where(np.abs(beta_vals - res['beta']) < 1e-6)[0]
            if len(i) > 0 and len(j) > 0:
                eta_grid[i[0], j[0]] = res['eta_ratio']

    # check if any valid points exist
    if np.all(np.isnan(eta_grid)):
        print("No valid data points. Exiting.")
        return

    # prepare data for interpolation
    points = np.array([(T, beta) for T in T_vals for beta in beta_vals])
    values = eta_grid.flatten()
    mask = ~np.isnan(values)
    if np.sum(mask) < 3:
        print("Too few valid points for interpolation. Skipping contour.")
        # still plot raw points
        plt.scatter(points[mask,0], points[mask,1], c=values[mask], cmap='RdBu_r')
        plt.colorbar()
        plt.xlabel('$T$')
        plt.ylabel('$\\beta$')
        plt.title(f'Raw data points (η/η_c), r_ratio={r_ratio}')
        plt.savefig(f'game_raw_points_r{r_ratio}.pdf', bbox_inches='tight')
        plt.savefig(f'game_raw_points_r{r_ratio}.eps', bbox_inches='tight')
        plt.show()
        return

    # interpolate for smooth plot
    grid_T, grid_beta = np.mgrid[0.2:0.95:100j, -0.5:0.5:100j]
    grid_eta = griddata(points[mask], values[mask], (grid_T, grid_beta), method='cubic')

    # plot
    plt.figure(figsize=(8, 6))
    plt.contourf(grid_T, grid_beta, grid_eta, levels=20, cmap='RdBu_r', alpha=0.8)
    plt.colorbar(label=r'$\langle \eta/\eta_c \rangle$')
    # contour line at η/η_c = 1
    cs = plt.contour(grid_T, grid_beta, grid_eta, levels=[1], colors='yellow', linewidths=2, linestyles='dashed')
    plt.clabel(cs, inline=True, fontsize=10, fmt='%.1f')

    plt.xlabel('Structural overlap $T$')
    plt.ylabel('Nodal heterogeneity $\\beta$')
    plt.title(f'Evolutionary Game, r_ratio={r_ratio}, subcritical={subcritical_factor}')
    plt.tight_layout()
    # Save high-quality vector graphics with descriptive filename
    filename = f'game_heatmap_r{r_ratio}_sub{subcritical_factor}'
    plt.savefig(f'{filename}.pdf', bbox_inches='tight')
    plt.savefig(f'{filename}.eps', bbox_inches='tight')
    plt.show()
    print(f"Heatmap saved as {filename}.pdf and .eps")


if __name__ == "__main__":
    main()