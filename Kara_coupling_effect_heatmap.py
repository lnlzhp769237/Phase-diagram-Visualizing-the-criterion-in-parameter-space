"""
Generate heatmap of η/η_c in (T, β) plane for Kuramoto synchronization dynamics.
Based on the social contagion heatmap code, adapted to Kuramoto model.
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
STEADY_MEAN_LIMIT = 0.1         # threshold for low-synchronization steady state (R < 0.05)
TARGET_RHO = -0.01
P_SUB_REDUCTION_FACTOR = 0.7
MAX_RETRIES = 5
DEFAULT_DELTA = 0.1
DEFAULT_OMEGA_STD = 0.5            # frequency heterogeneity
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

# ==================== Hypergraph Class (same as before) ====================
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


# ==================== Kuramoto Dynamics ====================
@dataclass
class KuramotoParams:
    beta: float
    sigma1: float
    sigma2: float
    omega_std: float = DEFAULT_OMEGA_STD
    delta: float = DEFAULT_DELTA
    alpha_model: float = DEFAULT_ALPHA_MODEL
    gamma_model: float = DEFAULT_GAMMA_MODEL
    kappa_model: float = DEFAULT_KAPPA_MODEL

class KuramotoDynamics:
    """Kuramoto synchronization dynamics with three‑body interactions (Eq. 17)"""

    @staticmethod
    def ode(t, theta, hg, params):
        N = hg.N
        factor = (1 - params.alpha_model * hg.T_mean) * h_beta(params.beta, params.delta)
        sigma1_eff = params.sigma1 * factor
        sigma2_eff = params.sigma2 * factor

        dtheta = np.zeros(N)
        # intrinsic frequencies (fixed seed for reproducibility)
        np.random.seed(42 + int(params.sigma1 * 100))
        omega = np.random.uniform(-params.omega_std, params.omega_std, N)
        dtheta += omega

        # pairwise coupling
        pref1 = sigma1_eff / max(hg.avg_k, 1e-10)
        for i in range(N):
            sin_sum = 0.0
            for j in range(N):
                if hg.A_proj_sparse[i, j] > 0:
                    sin_sum += hg.A_proj_sparse[i, j] * np.sin(theta[j] - theta[i])
            dtheta[i] += pref1 * sin_sum

        # three‑body coupling
        pref2 = sigma2_eff / (2 * max(hg.avg_k2, 1e-10))
        for i in range(N):
            sin_sum = 0.0
            for j, k in hg.W3_sparse[i]:
                sin_sum += np.sin(theta[j] + theta[k] - 2 * theta[i])
            dtheta[i] += pref2 * sin_sum
        return dtheta

    @staticmethod
    def steady_state(hg, params, t_max=None, x0=None):
        if t_max is None:
            t_max = ODE_T_MAX
        if x0 is None:
            x0 = np.random.uniform(-np.pi, np.pi, hg.N)
        sol = solve_ivp(KuramotoDynamics.ode, [0, t_max], x0,
                        args=(hg, params), method='LSODA',
                        rtol=ODE_RTOL, atol=ODE_ATOL, max_step=MAX_STEP)
        theta = sol.y[:, -1]
        theta -= np.mean(theta)   # remove global phase
        return theta

    @staticmethod
    def linear_operator(theta_star, hg, params):
        N = hg.N
        factor = (1 - params.alpha_model * hg.T_mean) * h_beta(params.beta, params.delta)
        sigma1_eff = params.sigma1 * factor
        sigma2_eff = params.sigma2 * factor

        L0 = np.zeros((N, N))
        pref1 = sigma1_eff / max(hg.avg_k, 1e-10)
        for i in range(N):
            cos_sum = 0.0
            for j in range(N):
                if hg.A_proj_sparse[i, j] > 0:
                    cos_val = np.cos(theta_star[j] - theta_star[i])
                    cos_sum += hg.A_proj_sparse[i, j] * cos_val
                    L0[i, j] += pref1 * hg.A_proj_sparse[i, j] * cos_val
            L0[i, i] -= pref1 * cos_sum

        pref2 = sigma2_eff / (2 * max(hg.avg_k2, 1e-10))
        for i in range(N):
            for j, k in hg.W3_sparse[i]:
                cos_val = np.cos(theta_star[j] + theta_star[k] - 2 * theta_star[i])
                L0[i, j] += pref2 * cos_val
                L0[i, k] += pref2 * cos_val
            diag_sum = 0.0
            for j, k in hg.W3_sparse[i]:
                diag_sum -= 2 * np.cos(theta_star[j] + theta_star[k] - 2 * theta_star[i])
            L0[i, i] += pref2 * diag_sum
        return L0

    @staticmethod
    def nonlinear_jacobian(theta_star, hg, params):
        N = hg.N
        factor = (1 - params.alpha_model * hg.T_mean) * h_beta(params.beta, params.delta)
        sigma2_eff = params.sigma2 * factor
        Jnl = np.zeros((N, N))
        pref2 = sigma2_eff / (2 * max(hg.avg_k2, 1e-10))
        for i in range(N):
            for j, k in hg.W3_sparse[i]:
                cos_val = np.cos(theta_star[j] + theta_star[k] - 2 * theta_star[i])
                Jnl[i, j] += pref2 * cos_val
                Jnl[i, k] += pref2 * cos_val
        return Jnl

    @staticmethod
    def estimate_critical_sigma1(hg, params):
        """
        Estimate critical σ₁ using mean‑field approximation.
        For Kuramoto, linear stability of incoherent state gives:
            maximum eigenvalue ≈ -δ + σ₁ * factor / (2*<k>) * λ_max
        Setting to zero yields σ₁_c = δ * (2*<k>) / (factor * λ_max)
        """
        factor = (1 - params.alpha_model * hg.T_mean) * h_beta(params.beta, params.delta)
        sigma1_c = params.delta * (2 * hg.avg_k) / (factor * hg.lambda_max)
        return sigma1_c

    @staticmethod
    def compute_eta_ratio(hg, params, sigma_ratio):
        """
        Compute η/η_c for Kuramoto.
        Returns η/η_c or NaN on failure.
        """
        try:
            # 1. Estimate critical σ₁ and choose a subcritical value
            sigma1_c = KuramotoDynamics.estimate_critical_sigma1(hg, params)
            sigma1_sub = 0.8 * sigma1_c   # 5% below critical
            sigma2 = sigma1_sub * sigma_ratio
            params_sub = replace(params, sigma1=sigma1_sub, sigma2=sigma2)

            # 2. Find low‑synchronization steady state
            theta_star = KuramotoDynamics.steady_state(hg, params_sub)
            # compute order parameter R
            R = np.abs(np.sum(np.exp(1j * theta_star))) / hg.N
            print(f"      R = {R:.6f}")

            # If still too synchronized, reduce further
            retries = 0
            while R > STEADY_MEAN_LIMIT and retries < MAX_RETRIES:
                sigma1_sub *= P_SUB_REDUCTION_FACTOR
                sigma2 = sigma1_sub * sigma_ratio
                params_sub = replace(params, sigma1=sigma1_sub, sigma2=sigma2)
                theta_star = KuramotoDynamics.steady_state(hg, params_sub)
                R = np.abs(np.sum(np.exp(1j * theta_star))) / hg.N
                print(f"      retry {retries+1}: R = {R:.6f}, σ₁ = {sigma1_sub:.6f}")
                retries += 1

            if R > STEADY_MEAN_LIMIT:
                print(f"      No low‑sync state found after {MAX_RETRIES} retries.")
                return np.nan

            # 3. Compute L0 and Jnl at this state
            L0 = KuramotoDynamics.linear_operator(theta_star, hg, params_sub)
            Jnl = KuramotoDynamics.nonlinear_jacobian(theta_star, hg, params_sub)

            eigvals = eig(L0)[0]
            rho0 = np.max(np.real(eigvals))
            rho0_abs = max(abs(rho0), 1e-12)

            norm_L0 = svd(L0, compute_uv=False)[0]
            norm_Jnl = svd(Jnl, compute_uv=False)[0]
            epsilon = norm_Jnl / max(norm_L0, 1e-12)
            eta = epsilon / rho0_abs

            # theoretical η_c for Kuramoto (Eq. 11 with appropriate C)
            C = hg.avg_k / max(hg.avg_k2, 1e-10)   # from paper, for Kuramoto
            numerator = (1 + params.gamma_model * params.beta) * hg.W3_norm
            denominator = (1 + params.kappa_model * hg.T_mean) * (1 - params.alpha_model * hg.T_mean) * hg.lambda_max
            eta_c = C * numerator / denominator if denominator > 1e-12 else np.inf
            eta_ratio = eta / eta_c if eta_c > 0 else np.inf

            # Detailed output
            print(f"      σ₁_c = {sigma1_c:.6f}, σ₁_sub = {sigma1_sub:.6f}, R = {R:.6f}")
            print(f"      η = {eta:.6f}, η_c = {eta_c:.6f}, η/η_c = {eta_ratio:.6f}")

            return eta_ratio

        except Exception as e:
            print(f"      Exception in compute_eta_ratio: {e}")
            return np.nan


# ==================== Generate Hypergraph with Target T (same as before) ====================
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
    return best_hg


# ==================== Worker for Parallel Execution ====================
def worker(args):
    T, beta, sigma_ratio, case_id = args
    t_start = time.time()
    print(f"[{time.strftime('%H:%M:%S')}] Start T={T:.3f}, β={beta:.3f}, case={case_id}")
    try:
        # generate hypergraph with target T
        print(f"[{time.strftime('%H:%M:%S')}]   Generating hypergraph...")
        hg = generate_hypergraph_with_target_T(N=150, target_T=T, max_trials=10)
        if hg is None:
            print(f"[{time.strftime('%H:%M:%S')}]   Failed to generate hypergraph")
            return {'T': T, 'beta': beta, 'eta_ratio': np.nan}

        # set base parameters (β only, σ₁ determined dynamically)
        params = KuramotoParams(beta=beta, sigma1=0.0, sigma2=0.0)  # placeholder

        # compute η/η_c
        print(f"[{time.strftime('%H:%M:%S')}]   Computing η/η_c...")
        eta_ratio = KuramotoDynamics.compute_eta_ratio(hg, params, sigma_ratio)
        elapsed = time.time() - t_start
        print(f"[{time.strftime('%H:%M:%S')}]   Done T={T:.3f}, β={beta:.3f}, η/η_c={eta_ratio:.4f}, time={elapsed:.1f}s")
        return {'T': T, 'beta': beta, 'eta_ratio': eta_ratio}
    except Exception as e:
        print(f"[{time.strftime('%H:%M:%S')}]   Error T={T:.3f}, β={beta:.3f}: {e}")
        return {'T': T, 'beta': beta, 'eta_ratio': np.nan}


# ==================== Main ====================
def main():
    # Choose a suitable σ₂/σ₁ ratio to get both continuous and explosive regions.
    # For Kuramoto, typical values might be around 2.0–4.0; adjust based on results.
    sigma_ratio = 0.2  # modify to explore
    N = 150
    # Use a coarse grid for quick testing; increase to 15×15 for final figure
    T_vals = np.linspace(0.2, 0.95, 5)
    beta_vals = np.linspace(-0.5, 0.5, 5)

    # build job list
    jobs = []
    case_id = 0
    for T in T_vals:
        for beta in beta_vals:
            jobs.append((T, beta, sigma_ratio, case_id))
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
        plt.title('Raw data points (η/η_c)')
        plt.savefig('kuramoto_raw_points.pdf', bbox_inches='tight')
        plt.savefig('kuramoto_raw_points.eps', bbox_inches='tight')
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
    plt.title(f'$\\sigma_2/\\sigma_1 = {sigma_ratio}$')
    plt.tight_layout()
    # Save high-quality vector graphics
    plt.savefig('kuramoto_heatmap.pdf', bbox_inches='tight')
    plt.savefig('kuramoto_heatmap.eps', bbox_inches='tight')
    plt.show()
    print("Heatmap saved as kuramoto_heatmap.pdf and .eps")


if __name__ == "__main__":
    main()