import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as tri

# =========================
# Antoine Parameters
# =========================
A = np.array([9.3251, 9.2876, 9.3240])
B = np.array([1348.18, 1388.10, 1467.44])
C = np.array([-51.1717, -56.8825, -59.2760])

P_total = 101325


# =========================
# Thermodynamics
# =========================
def p_sat(T):
    return 10 ** (A - B / (C + T))


def bubble_temperature(x):
    T = np.dot(x, [363, 390, 410])

    for _ in range(50):
        ps = p_sat(T)
        f = P_total - np.sum(x * ps)

        if abs(f) < 1e-5:
            return T

        dps = ps * np.log(10) * (B / (C + T)**2)
        df = -np.sum(x * dps)

        if abs(df) < 1e-10:
            return None

        T_new = T - f / df

        if abs(T_new - T) > 20:
            T_new = T + np.sign(T_new - T) * 20

        T = T_new

    return None


def vapor_y(x, T):
    ps = p_sat(T)
    y = x * ps / P_total
    return y / np.sum(y)


# =========================
# Geometry
# =========================
def to_xy(x):
    x1, x2, x3 = x
    X = 0.5 * (2*x2 + x3)
    Y = (np.sqrt(3)/2) * x3
    return X, Y


def from_xy(X, Y):
    x3 = Y / (np.sqrt(3)/2) #60 degrees
    x2 = X - 0.5*x3
    x1 = 1 - x2 - x3
    return np.array([x1, x2, x3])


# =========================
# Find edge intersections
# =========================
def find_products(x, y):
    p1 = np.array(to_xy(x))
    p2 = np.array(to_xy(y))

    pts = []

    for t in np.linspace(-0.5, 1.5, 300):  # extend line
        p = p1 + t * (p2 - p1)
        comp = from_xy(p[0], p[1])

        if np.all(comp >= -0.01):
            if np.any(np.abs(comp) < 0.01):
                pts.append(comp)

    if len(pts) < 2:
        return None, None

    pts = sorted(pts, key=lambda c: c[2])  # DEC sorting

    return pts[0], pts[-1]


# =========================
# Energy
# =========================
def safe_mean(arr):
    arr = arr[np.isfinite(arr)]
    if len(arr) == 0:
        return None
    return np.mean(arr)


def compute_energy(z, xF, yF, xd, xb):
    eps = 1e-8

    denom = (yF - xF)
    if np.any(np.abs(denom) < 1e-6):
        return None, None

    RG = safe_mean((xF - xb) / (denom + eps))
    RL = safe_mean((xd - yF) / (denom + eps))

    if RG is None or RL is None:
        return None, None

    Qc = -(RG + 1) * safe_mean((z - xb) / (xd - xb + eps))
    Qb = RL * safe_mean((xd - z) / (xd - xb + eps))

    return Qc, Qb


# =========================
# FULL GRID
# =========================
def generate(N=40):
    pts = []
    Qc_vals = []
    Qb_vals = []

    for i in range(1, N):
        for j in range(1, N - i):
            x1 = i / N
            x2 = j / N
            x3 = 1 - x1 - x2

            x = np.array([x1, x2, x3])

            try:
                T = bubble_temperature(x)
                if T is None:
                    continue

                y = vapor_y(x, T)

                xd, xb = find_products(x, y)
                if xd is None:
                    continue

                Qc, Qb = compute_energy(x, x, y, xd, xb)
                if Qc is None:
                    continue

                X, Y = to_xy(x)

                pts.append([X, Y])
                Qc_vals.append(Qc)
                Qb_vals.append(Qb)

            except:
                continue

    return np.array(pts), np.array(Qc_vals), np.array(Qb_vals)


# =========================
# Plot
# =========================
def plot_map(pts, values, title):
    if len(pts) == 0:
        print("No data to plot")
        return

    x = pts[:,0]
    y = pts[:,1]

    triang = tri.Triangulation(x, y)

    plt.figure(figsize=(7,6))
    plt.tricontourf(triang, values, levels=40, cmap='plasma')

    triangle = np.array([
        to_xy([1,0,0]),
        to_xy([0,1,0]),
        to_xy([0,0,1]),
        to_xy([1,0,0])
    ])
    plt.plot(triangle[:,0], triangle[:,1], 'k')

    plt.title(title)
    plt.xticks([])
    plt.yticks([])
    plt.colorbar()
    plt.show()


# =========================
# MAIN
# =========================
if __name__ == "__main__":
    print("Generating grid...")
    pts, Qc_vals, Qb_vals = generate(N=35)

    print("Points computed:", len(pts))

    plot_map(pts, Qc_vals, "Qc,min / (F*r)")
    plot_map(pts, Qb_vals, "Qb,min / (F*r)")