import numpy as np
import matplotlib.pyplot as plt

# =========================
# Antoine Parameters
# =========================
# Order: DMC, EMC, DEC
A = np.array([9.3251, 9.2876, 9.3240])
B = np.array([1348.18, 1388.10, 1467.44])
C = np.array([-51.1717, -56.8825, -59.2760])

P_total = 101325  # Pa
T_init=360
tol=1e-6

# =========================
# Antoine Equation
# =========================
def p_sat(T):
    return 10 ** (A - B / (C + T))

# =========================
# Functions for Newton
# =========================
def f(T, x):
    return P_total - np.sum(x * p_sat(T))


def df_dT(T, x):
    psat = p_sat(T)
    dpsat = psat * np.log(10) * (B / (C + T)**2)
    return -np.sum(x * dpsat)

# =========================
# Newton Solver
# =========================
def solve_temperature(x, T_init, tol, max_iter=100):
    T = T_init

    for _ in range(max_iter):
        f_val = f(T, x)
        df_val = df_dT(T, x)

        if abs(f_val) < tol:
            return T

        if abs(df_val) < 1e-12:
            raise ValueError("Derivative too small")

        T_new = T - f_val / df_val

        # Stabilization
        if abs(T_new - T) > 20:
            T_new = T + np.sign(T_new - T) * 20

        T = T_new

    raise ValueError("Newton did not converge")


# =========================
# Vapor composition
# =========================
def vapor_composition(x, T):
    psat = p_sat(T)
    y = x * psat / P_total
    return y


# =========================
# Generate ternary data
# =========================
def generate_data(n_points=25):
    results = []

    for x1 in np.linspace(0.01, 0.98, n_points):
        for x2 in np.linspace(0.01, 0.98 - x1, n_points):
            x3 = 1 - x1 - x2

            if x3 <= 0:
                continue

            x = np.array([x1, x2, x3])

            # Initial guess (weighted boiling points approx)
            T_init = np.dot(x, [363, 390, 410])

            try:
                T = solve_temperature(x, T_init)
                y = vapor_composition(x, T)

                results.append({
                    "x": x,
                    "y": y,
                    "T": T
                })

            except:
                continue

    return results


# =========================
# Convert to ternary coordinates
# =========================
def to_ternary_coords(comp):
    x1, x2, x3 = comp
    X = 0.5 * (2*x2 + x3)
    Y = (np.sqrt(3)/2) * x3
    return X, Y


# =========================
# Plot ternary diagram
# =========================
def plot_ternary(results):
    fig, ax = plt.subplots(figsize=(8, 7))

    # Triangle border
    triangle = np.array([
        to_ternary_coords([1,0,0]),
        to_ternary_coords([0,1,0]),
        to_ternary_coords([0,0,1]),
        to_ternary_coords([1,0,0])
    ])
    ax.plot(triangle[:,0], triangle[:,1])

    # Plot points (colored by temperature)
    for r in results:
        X, Y = to_ternary_coords(r["x"])
        ax.scatter(X, Y, c=r["T"], cmap='viridis', s=10)

    ax.set_title("Ternary Diagram (Temperature Field)")
    ax.set_xticks([])
    ax.set_yticks([])

    plt.show()


# =========================
# MAIN EXECUTION
# =========================
if __name__ == "__main__":

    # Example single calculation
    x_test = np.array([0.4, 0.4, 0.2])

    T = solve_temperature(x_test)
    y = vapor_composition(x_test, T)

    print("=== Single Point ===")
    print("x =", x_test)
    print(f"T = {T:.2f} K")
    print("y =", y)
    print("sum(y) =", np.sum(y))

    # Generate full ternary dataset
    print("\nGenerating ternary data...")
    data = generate_data(n_points=20)

    # Plot
    plot_ternary(data)