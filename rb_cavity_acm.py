"""
2D natural convection in a tall vertical cavity (Rayleigh-Benard / differentially heated cavity)
Artificial Compressibility Method (ACM) + FTCS explicit scheme, Boussinesq approximation.

Python (NumPy + Numba) port of the MATLAB solver in:
https://github.com/wajeehasiddiqui831/Natural-Convection-Cavity-Matlab
Base MATLAB framework: Prof. Imran Akhtar (NUST). Python port: W. Siddiqui.

Indexing and update order intentionally follow the MATLAB code (i = x index, j = y index,
in-place sweep with j outer loop and i inner loop), so results can be compared one-to-one.
"""
import time
import numpy as np

try:
    from numba import njit
    HAVE_NUMBA = True
except ImportError:                      # slow pure-Python fallback (fine for tiny test grids only)
    HAVE_NUMBA = False
    def njit(*args, **kwargs):
        if len(args) == 1 and callable(args[0]):
            return args[0]
        return lambda f: f


@njit(cache=True)
def _march(U, V, P, T, Tprev, dx, dy, dt, beta2, nuc, alc, tol, max_iter, hist, avg_from, acc, series):
    Nx, Ny = T.shape
    it = 1
    err = 1.0
    while err > tol and it < max_iter:
        for j in range(1, Ny - 1):
            for i in range(1, Nx - 1):
                P[i, j] = (P[i, j]
                           - (0.5 * dt / (dx * beta2)) * (U[i + 1, j] - U[i - 1, j])
                           - (0.5 * dt / (dy * beta2)) * (V[i, j + 1] - V[i, j - 1]))

                U[i, j] = (U[i, j]
                           - (0.5 * dt / dx) * (U[i + 1, j] ** 2 - U[i - 1, j] ** 2)
                           - (0.5 * dt / dy) * (U[i, j + 1] * V[i, j + 1] - U[i, j - 1] * V[i, j - 1])
                           - (0.5 * dt / dx) * (P[i + 1, j] - P[i - 1, j])
                           + (nuc * dt / dx ** 2) * (U[i + 1, j] - 2.0 * U[i, j] + U[i - 1, j])
                           + (nuc * dt / dy ** 2) * (U[i, j + 1] - 2.0 * U[i, j] + U[i, j - 1]))

                V[i, j] = (V[i, j]
                           - (0.5 * dt / dy) * (V[i, j + 1] ** 2 - V[i, j - 1] ** 2)
                           - (0.5 * dt / dx) * (U[i + 1, j] * V[i + 1, j] - U[i - 1, j] * V[i - 1, j])
                           - (0.5 * dt / dy) * (P[i, j + 1] - P[i, j - 1])
                           + (nuc * dt / dy ** 2) * (V[i, j + 1] - 2.0 * V[i, j] + V[i, j - 1])
                           + (nuc * dt / dx ** 2) * (V[i + 1, j] - 2.0 * V[i, j] + V[i - 1, j])
                           + dt * T[i, j])

                T[i, j] = (T[i, j]
                           - (0.5 * dt / dx) * (T[i + 1, j] * U[i + 1, j] - T[i - 1, j] * U[i - 1, j])
                           - (0.5 * dt / dy) * (T[i, j + 1] * V[i, j + 1] - T[i, j - 1] * V[i, j - 1])
                           + (alc * dt / dx ** 2) * (T[i + 1, j] - 2.0 * T[i, j] + T[i - 1, j])
                           + (alc * dt / dy ** 2) * (T[i, j + 1] - 2.0 * T[i, j] + T[i, j - 1]))

        # convergence measure on temperature (before boundary update, as in the MATLAB code)
        err = 0.0
        for j in range(Ny):
            for i in range(Nx):
                d = abs(T[i, j] - Tprev[i, j])
                if d > err:
                    err = d
        hist[it - 1] = err

        # Neumann conditions: adiabatic top/bottom for T, zero-gradient pressure on all walls
        for i in range(Nx):
            T[i, 0] = T[i, 1]
            T[i, Ny - 1] = T[i, Ny - 2]
            P[i, 0] = P[i, 1]
            P[i, Ny - 1] = P[i, Ny - 2]
        for j in range(Ny):
            P[0, j] = P[1, j]
            P[Nx - 1, j] = P[Nx - 2, j]

        for j in range(Ny):
            for i in range(Nx):
                Tprev[i, j] = T[i, j]

        # running time-average of Nu over the last part of the run (useful when the flow is unsteady)
        if it % 100 == 0:
            s = 0.0
            Tw = T[Nx - 1, 1]
            Tc = T[0, 1]
            for j in range(Ny):
                s += (Tw - T[Nx - 2, j]) / (Tw - Tc)
            s /= Ny
            series[it // 100 - 1] = (Nx - 1) * s          # Nu(t) every 100 iterations (Nx-1 scaling)
            if it > avg_from:
                acc[0] += Nx * s
                acc[1] += (Nx - 1) * s
                acc[2] += 1.0
        it += 1
    return it - 1, err


def stability_numbers(Ra, Pr, dx, dy, dt):
    """Diffusion numbers for the explicit scheme; each should be <= 0.5."""
    nuc = np.sqrt(Pr / Ra)
    alc = 1.0 / np.sqrt(Ra * Pr)
    return {"viscous": nuc * dt * (1 / dx**2 + 1 / dy**2),
            "thermal": alc * dt * (1 / dx**2 + 1 / dy**2)}


def run_case(Ra, AR, Pr=0.71, dt=0.005, pts_per_unit=40, beta=np.sqrt(0.6),
             tol=1e-8, max_iter=40000, verbose=True):
    Lx, Ly = 1.0, float(AR)
    Nx = int(Lx * pts_per_unit) + 1
    Ny = int(Ly * pts_per_unit) + 1
    dx, dy = Lx / (Nx - 1), Ly / (Ny - 1)

    stab = stability_numbers(Ra, Pr, dx, dy, dt)
    if max(stab.values()) > 0.5:
        print(f"WARNING Ra={Ra}, AR={AR}: diffusion number {max(stab.values()):.2f} > 0.5; "
              f"reduce dt (currently {dt}).")

    U = np.zeros((Nx, Ny)); V = np.zeros((Nx, Ny)); P = np.zeros((Nx, Ny))
    T = np.zeros((Nx, Ny)); Tprev = np.zeros((Nx, Ny))
    T[0, :] = -0.5          # cold left wall
    T[-1, :] = 0.5          # hot right wall
    hist = np.zeros(max_iter)
    acc = np.zeros(3)
    series = np.zeros(max_iter // 100 + 1)

    t0 = time.time()
    n_iter, err = _march(U, V, P, T, Tprev, dx, dy, dt, beta ** 2,
                         np.sqrt(Pr / Ra), 1.0 / np.sqrt(Ra * Pr), tol, max_iter, hist,
                         0.9 * max_iter, acc, series)
    elapsed = time.time() - t0

    Nu_local = local_nusselt(T, Nx)
    out = dict(Ra=Ra, AR=AR, Pr=Pr, Nx=Nx, Ny=Ny, dt=dt, U=U, V=V, P=P, T=T,
               iters=n_iter, final_error=err, converged=bool(err <= tol),
               seconds=elapsed, hist=hist[:n_iter],
               Nu_local=Nu_local, Nu_avg=float(np.mean(Nu_local)),
               Nu_avg_alt=float(np.mean(local_nusselt(T, Nx, variant="nx_minus_1"))),
               # time-averaged over the last 10% of max_iter (only filled if the run did not stop earlier)
               Nu_series=series[: n_iter // 100], dt_series=100 * dt,
               Nu_tavg=float(acc[0] / acc[2]) if acc[2] > 0 else None,
               Nu_tavg_alt=float(acc[1] / acc[2]) if acc[2] > 0 else None)
    if verbose:
        print(f"Ra={Ra:>6}  AR={AR:>3}  grid={Nx}x{Ny}  iters={n_iter:>5}  "
              f"converged={out['converged']}  Nu_avg={out['Nu_avg']:.4f}  time={elapsed:.1f}s")
    return out


def local_nusselt(T, Nx, variant="matlab"):
    """Local Nu on the hot wall.
    'matlab'     : Nx*(Tw - T[Nx-2, j])/(Tw - Tc)        (exactly as in the MATLAB code)
    'nx_minus_1' : (Nx-1)*(Tw - T[Nx-2, j])/(Tw - Tc)   (= 1/dx for Lx = 1; first-order wall gradient)
    """
    k = Nx if variant == "matlab" else (Nx - 1)
    Tw, Tc = T[-1, 1], T[0, 1]
    return k * (Tw - T[-2, :]) / (Tw - Tc)


def vorticity(U, V, dx, dy):
    Wz = np.zeros_like(U)
    Wz[1:-1, 1:-1] = (0.5 * (V[2:, 1:-1] - V[:-2, 1:-1]) / dx
                      - 0.5 * (U[1:-1, 2:] - U[1:-1, :-2]) / dy)
    return Wz


def plot_case(res, save_prefix=None):
    import matplotlib.pyplot as plt
    Ra, AR = res["Ra"], res["AR"]
    Nx, Ny = res["Nx"], res["Ny"]
    y = np.linspace(0, 1, Ny)
    fig, ax = plt.subplots(1, 4, figsize=(15, max(4, 1.2 * AR) if AR <= 8 else 8))
    ax[0].contourf(res["V"].T, 40); ax[0].set_title(f"V velocity\nAR={AR}, Ra={Ra}")
    ax[1].contourf(res["T"].T, 40); ax[1].set_title("Temperature")
    ax[2].plot(res["U"][(Nx - 1) // 2, :], y); ax[2].set_title("U at x = 0.5 (centre)")
    ax[3].plot(y, res["Nu_local"]); ax[3].set_title("Local Nu (hot wall)")
    for a in ax[:2]:
        a.set_xticks([]); a.set_yticks([])
    plt.tight_layout()
    if save_prefix:
        plt.savefig(f"{save_prefix}_AR{AR}_Ra{Ra}.png", dpi=150)
    plt.show()
