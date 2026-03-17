import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as tri
from numba import njit

# =========================
# Antoine parameters
# =========================
A = np.array([9.3251, 9.2876, 9.3240])
B = np.array([1348.18, 1388.10, 1467.44])
C = np.array([-51.1717, -56.8825, -59.2760])

P_total = 101325.0


# =========================
# Thermodynamics
# =========================
@njit
def p_sat(T):
    out = np.zeros(3)
    for i in range(3):
        out[i] = 10 ** (A[i] - B[i] / (C[i] + T))
    return out


@njit
def bubble_temperature(x):
    T = x[0]*363 + x[1]*390 + x[2]*410

    for _ in range(40):
        ps = p_sat(T)
        f = P_total - (x[0]*ps[0] + x[1]*ps[1] + x[2]*ps[2])

        if abs(f) < 1e-5:
            return T

        df = 0.0
        for i in range(3):
            dps = ps[i] * np.log(10.0) * (B[i] / (C[i] + T)**2)
            df -= x[i] * dps

        if abs(df) < 1e-10:
            return -1.0

        T = T - f / df

    return -1.0


@njit
def vapor_y(x, T):
    ps = p_sat(T)

    y = np.zeros(3)
    total = 0.0

    for i in range(3):
        y[i] = x[i] * ps[i] / P_total
        total += y[i]

    for i in range(3):
        y[i] /= total

    return y


# =========================
# Geometry
# =========================
@njit
def to_xy(x):
    return 0.5*(2*x[1] + x[2]), (np.sqrt(3)/2)*x[2]


# =========================
# Line tracing (robust)
# =========================
@njit
def trace_line(x, y):
    pts = np.zeros((50, 3))
    count = 0

    for t in np.linspace(-0.5, 1.5, 50):
        p = x + t*(y - x)

        if np.min(p) >= -0.02:
            for k in range(3):
                if abs(p[k]) < 0.02:
                    pts[count] = p
                    count += 1
                    break

    return pts, count


# =========================
# PATHS
# =========================
def find_ac(x, y):
    pts, n = trace_line(x, y)
    if n < 2:
        return None, None

    pts = pts[:n]
    idx = np.argsort(pts[:,2])
    return pts[idx[0]], pts[idx[-1]]


def find_a(x, y):
    pts, n = trace_line(x, y)
    if n == 0:
        return None, None

    for i in range(n):
        if abs(pts[i][0]) < 0.05:
            return np.array([1.0,0.0,0.0]), pts[i]
    return None, None


def find_c(x, y):
    pts, n = trace_line(x, y)
    if n == 0:
        return None, None

    for i in range(n):
        if abs(pts[i][2]) < 0.05:
            return pts[i], np.array([0.0,0.0,1.0])
    return None, None


# =========================
# Energy
# =========================
@njit
def compute_energy(z, xF, yF, xd, xb):

    eps = 1e-6

    RG = 0.0
    RL = 0.0

    for i in range(3):
        denom = yF[i] - xF[i]
        if abs(denom) < eps:
            denom = eps

        RG += (xF[i] - xb[i]) / denom
        RL += (xd[i] - yF[i]) / denom

    RG /= 3
    RL /= 3

    Qc = 0.0
    Qb = 0.0

    for i in range(3):
        denom = xd[i] - xb[i]
        if abs(denom) < eps:
            denom = eps

        Qc += (z[i] - xb[i]) / denom
        Qb += (xd[i] - z[i]) / denom

    Qc = -(RG + 1)*Qc/3
    Qb = RL*Qb/3

    return Qc, Qb


# =========================
# Generate grid
# =========================
def generate(path_func, N=35):

    pts, Qc_vals, Qb_vals = [], [], []

    for i in range(1, N):
        for j in range(1, N-i):

            x = np.array([i/N, j/N, 1 - i/N - j/N])

            T = bubble_temperature(x)
            if T < 0:
                continue

            y = vapor_y(x, T)

            xd, xb = path_func(x, y)
            if xd is None:
                continue

            Qc, Qb = compute_energy(x, x, y, xd, xb)

            if np.isnan(Qc):
                continue

            X, Y = to_xy(x)

            pts.append([X, Y])
            Qc_vals.append(Qc)
            Qb_vals.append(Qb)

    return np.array(pts), np.array(Qc_vals), np.array(Qb_vals)


# =========================
# Proper ternary plot
# =========================
def plot_ternary(pts, values, title):

    if len(pts) == 0:
        print("No data:", title)
        return

    triang = tri.Triangulation(pts[:,0], pts[:,1])

    fig, ax = plt.subplots(figsize=(7,6))

    tcf = ax.tricontourf(triang, values, levels=40, cmap='plasma')

    # triangle
    triangle = np.array([
        to_xy([1,0,0]),
        to_xy([0,1,0]),
        to_xy([0,0,1]),
        to_xy([1,0,0])
    ])
    ax.plot(triangle[:,0], triangle[:,1], 'k')

    # labels
    ax.text(*to_xy([1,0,0]), "DMC", ha='right')
    ax.text(*to_xy([0,1,0]), "EMC", ha='left')
    ax.text(*to_xy([0,0,1]), "DEC", ha='center', va='bottom')

    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])

    fig.colorbar(tcf)
    plt.show()


# =========================
# MAIN
# =========================
if __name__ == "__main__":

    print("a-c path...")
    pts, Qc, Qb = generate(find_ac)
    plot_ternary(pts, Qc, "a-c path: Qc")
    plot_ternary(pts, Qb, "a-c path: Qb")

    print("a path...")
    pts, Qc, Qb = generate(find_a)
    plot_ternary(pts, Qc, "a path: Qc")
    plot_ternary(pts, Qb, "a path: Qb")

    print("c path...")
    pts, Qc, Qb = generate(find_c)
    plot_ternary(pts, Qc, "c path: Qc")
    plot_ternary(pts, Qb, "c path: Qb")