"""
Generate heatmap of η/η_c in (T, β) plane for social contagion dynamics.
Improved version with:
- Higher λ₂/λ₁ ratio (4.0)
- Closer-to-critical λ₁ (0.95 * λ₁_c)
- Detailed output of intermediate quantities (λ₁_c, λ₁_sub, mean_x, eta, eta_c)
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
STEADY_MEAN_LIMIT = 0.05          # threshold for low-activity steady state
TARGET_RHO = -0.01
P_SUB_REDUCTION_FACTOR = 0.8
MAX_RETRIES = 2
DEFAULT_DELTA = 0.1
DEFAULT_ALPHA_MODEL = 1.0
DEFAULT_GAMMA_MODEL = 0.75
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


# ==================== Contagion Dynamics ====================
@dataclass
class ContagionParams:
    beta: float
    lambda1: float
    lambda2: float
    delta: float = DEFAULT_DELTA
    alpha_model: float = DEFAULT_ALPHA_MODEL
    gamma_model: float = DEFAULT_GAMMA_MODEL
    kappa_model: float = DEFAULT_KAPPA_MODEL

class ContagionDynamics:
    """Social contagion dynamics (SIS-type), Eq. (16)"""

    @staticmethod
    def ode(t, x, hg, params):
        factor = (1 - params.alpha_model * hg.T_mean) * h_beta(params.beta, params.delta)
        lambda1_eff = params.lambda1 * factor
        lambda2_eff = params.lambda2 * factor

        pair = lambda1_eff * (hg.A_proj_sparse @ x)
        triple = np.zeros(hg.N)
        weight = 1.0 / 6.0
        for i in range(hg.N):
            s = 0.0
            for j, k in hg.W3_sparse[i]:
                s += x[j] * x[k]
            triple[i] = lambda2_eff * weight * s
        return -params.delta * x + (1 - x) * (pair + triple)

    @staticmethod
    def steady_state(hg, params, t_max=None, x0=None):
        if t_max is None:
            t_max = ODE_T_MAX
        if x0 is None:
            x0 = np.random.rand(hg.N) * 1e-3
        sol = solve_ivp(ContagionDynamics.ode, [0, t_max], x0,
                        args=(hg, params), method='LSODA',
                        rtol=ODE_RTOL, atol=ODE_ATOL, max_step=MAX_STEP)
        return sol.y[:, -1]

    @staticmethod
    def linear_operator(x_star, hg, params):
        N = hg.N
        factor = (1 - params.alpha_model * hg.T_mean) * h_beta(params.beta, params.delta)
        lambda1_eff = params.lambda1 * factor
        lambda2_eff = params.lambda2 * factor

        L0 = -params.delta * np.eye(N)
        A = hg.A_proj_sparse.toarray()
        for i in range(N):
            for j in range(N):
                if i == j:
                    L0[i, i] -= lambda1_eff * (A[i, :] @ x_star)
                L0[i, j] += (1 - x_star[i]) * lambda1_eff * A[i, j]

        weight = 1.0 / 6.0
        for i in range(N):
            diag_term = 0.0
            for j, k in hg.W3_sparse[i]:
                diag_term += x_star[j] * x_star[k]
                L0[i, j] += (1 - x_star[i]) * lambda2_eff * weight * x_star[k]
                L0[i, k] += (1 - x_star[i]) * lambda2_eff * weight * x_star[j]
            L0[i, i] -= lambda2_eff * weight * diag_term
        return L0

    @staticmethod
    def nonlinear_jacobian(x_star, hg, params):
        N = hg.N
        factor = (1 - params.alpha_model * hg.T_mean) * h_beta(params.beta, params.delta)
        lambda2_eff = params.lambda2 * factor
        Jnl = np.zeros((N, N))
        weight = 1.0 / 6.0
        for i in range(N):
            for j, k in hg.W3_sparse[i]:
                Jnl[i, j] += (1 - x_star[i]) * lambda2_eff * weight * x_star[k]
                Jnl[i, k] += (1 - x_star[i]) * lambda2_eff * weight * x_star[j]
        return Jnl

    @staticmethod
    def estimate_critical_lambda1(hg, params):
        """
        Estimate critical λ1 via mean‑field approximation.
        Returns λ1_c such that the maximum eigenvalue of the linearized system ≈ 0.
        """
        factor = (1 - params.alpha_model * hg.T_mean) * h_beta(params.beta, params.delta)
        # approximate linear operator at zero: L0 ≈ -δ I + λ1 * factor * A_proj
        # maximum eigenvalue: -δ + λ1 * factor * λ_max(A_proj)
        # set to zero → λ1_c = δ / (factor * λ_max)
        lambda1_c = params.delta / (factor * hg.lambda_max)
        return lambda1_c

    @staticmethod
    def compute_eta_ratio(hg, params, lambda_ratio):
        """
        Compute η/η_c for given hypergraph and parameters.
        Returns a float (η/η_c) or NaN if calculation fails.
        """
        try:
            # 1. Estimate critical λ1 and choose a subcritical value
            lambda1_c = ContagionDynamics.estimate_critical_lambda1(hg, params)
            # Use 0.95 * lambda1_c to be much closer to criticality
            lambda1_sub = 0.95 * lambda1_c
            lambda2 = lambda1_sub * lambda_ratio
            params_sub = replace(params, lambda1=lambda1_sub, lambda2=lambda2)

            # 2. Find low‑activity steady state
            x_star = ContagionDynamics.steady_state(hg, params_sub)
            mean_x = np.mean(x_star)

            # If still too active, try reducing further (should not happen often)
            retries = 0
            while mean_x > STEADY_MEAN_LIMIT and retries < MAX_RETRIES:
                lambda1_sub *= P_SUB_REDUCTION_FACTOR
                lambda2 = lambda1_sub * lambda_ratio
                params_sub = replace(params, lambda1=lambda1_sub, lambda2=lambda2)
                x_star = ContagionDynamics.steady_state(hg, params_sub)
                mean_x = np.mean(x_star)
                retries += 1

            if mean_x > STEADY_MEAN_LIMIT:
                print(f"      No low‑activity state found after {MAX_RETRIES} retries.")
                return np.nan

            # 3. Compute L0 and Jnl at this state
            L0 = ContagionDynamics.linear_operator(x_star, hg, params_sub)
            Jnl = ContagionDynamics.nonlinear_jacobian(x_star, hg, params_sub)

            eigvals = eig(L0)[0]
            rho0 = np.max(np.real(eigvals))
            rho0_abs = max(abs(rho0), 1e-12)

            norm_L0 = svd(L0, compute_uv=False)[0]
            norm_Jnl = svd(Jnl, compute_uv=False)[0]
            epsilon = norm_Jnl / max(norm_L0, 1e-12)
            eta = epsilon / rho0_abs

            # theoretical η_c (Eq. 11)
            C = 0.5
            numerator = (1 + params.gamma_model * params.beta) * hg.W3_norm
            denominator = (1 + params.kappa_model * hg.T_mean) * (1 - params.alpha_model * hg.T_mean) * hg.lambda_max
            eta_c = C * numerator / denominator if denominator > 1e-12 else np.inf
            eta_ratio = eta / eta_c if eta_c > 0 else np.inf

            # Detailed output for debugging (unconditional for all points)
            print(f"      λ1_c = {lambda1_c:.6f}, λ1_sub = {lambda1_sub:.6f}, mean_x = {mean_x:.6f}")
            print(f"      η = {eta:.6f}, η_c = {eta_c:.6f}, η/η_c = {eta_ratio:.6f}")

            return eta_ratio

        except Exception as e:
            print(f"      Exception in compute_eta_ratio: {e}")
            return np.nan


# ==================== Generate Hypergraph with Target T ====================
def generate_hypergraph_with_target_T(N, target_T, tol=0.01, max_trials=20):
    """
    Generate ER hypergraph whose T_mean is as close as possible to target_T.
    Returns a Hypergraph object (best approximation) or None if all attempts fail.
    """
    low_p, high_p = 0.01, 0.8
    best_hg = None
    best_err = np.inf
    for i in range(max_trials):
        p = (low_p + high_p) / 2
        hg = Hypergraph.generate_ER(N, p, seed=np.random.randint(10000))
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
    return best_hg   # may be None if even the first trial failed


# ==================== Worker for Parallel Execution ====================
def worker(args):
    T, beta, lambda_ratio, case_id = args
    t_start = time.time()
    print(f"[{time.strftime('%H:%M:%S')}] Start T={T:.3f}, β={beta:.3f}, case={case_id}")
    try:
        # generate hypergraph with target T
        print(f"[{time.strftime('%H:%M:%S')}]   Generating hypergraph...")
        hg = generate_hypergraph_with_target_T(N=150, target_T=T, max_trials=10)
        if hg is None:
            print(f"[{time.strftime('%H:%M:%S')}]   Failed to generate hypergraph")
            return {'T': T, 'beta': beta, 'eta_ratio': np.nan}

        # set base parameters (β only, λ1 will be determined dynamically)
        params = ContagionParams(beta=beta, lambda1=0.0, lambda2=0.0)  # placeholder

        # compute η/η_c
        print(f"[{time.strftime('%H:%M:%S')}]   Computing η/η_c...")
        eta_ratio = ContagionDynamics.compute_eta_ratio(hg, params, lambda_ratio)
        elapsed = time.time() - t_start
        print(f"[{time.strftime('%H:%M:%S')}]   Done T={T:.3f}, β={beta:.3f}, η/η_c={eta_ratio:.4f}, time={elapsed:.1f}s")
        return {'T': T, 'beta': beta, 'eta_ratio': eta_ratio}
    except Exception as e:
        print(f"[{time.strftime('%H:%M:%S')}]   Error T={T:.3f}, β={beta:.3f}: {e}")
        return {'T': T, 'beta': beta, 'eta_ratio': np.nan}


# ==================== Main ====================
def main():
    # Set a larger nonlinearity ratio to possibly push η/η_c above 1
    lambda_ratio=2.5     # try 4.0 or 5.0; you can modify this
    N = 150    # Us a coarse grid for quick testing; increase to 15×15 for final figure
    T_vals = np.linspace(0.2, 0.95, 5)
    beta_vals = np.linspace(-0.5, 0.5, 5)

    # build job list
    jobs = []
    case_id = 0
    for T in T_vals:
        for beta in beta_vals:
            jobs.append((T, beta, lambda_ratio, case_id))
            case_id += 1

    print(f"Starting {len(jobs)} jobs with {min(cpu_count(), 8)} processes...")
    # For debugging, you may set processes=1 to see sequential output
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
        plt.title('Raw data points (η/η_c)')
        plt.savefig('coupling_raw_points.png')
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
    plt.title(f'$\\lambda_2/\\lambda_1 = {lambda_ratio}$')
    plt.tight_layout()
    plt.savefig('coupling_effect_heatmap_v2.pdf', bbox_inches='tight')
    plt.savefig('coupling_effect_heatmap_v2.eps', bbox_inches='tight')
    plt.show()
    print("Heatmap saved as coupling_effect_heatmap_v2.png")


if __name__ == "__main__":
    main()