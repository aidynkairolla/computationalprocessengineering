import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp


cin = 0.0073
eps_bed = 0.4
Rp = 1.4e-3
eps_p = 0.58
rho_s = 1100.0
Dax = 2.436e-4
u = 0.1
D = 1.63e-6
L = 1.0

Nz = 100
z = np.linspace(0.0, L, Nz + 1)
dz = z[1] - z[0]

n = Nz

t_start = 0.0
t_end = 30000.0
t_eval = np.linspace(t_start, t_end, 400)


def calculate_model_coefficients(isotherm, c):
    if isotherm == "linear":
        dXdc = 4.715 * np.ones_like(c)

    elif isotherm == "freundlich":
        dXdc = 82.68 * np.sqrt(cin) / cin * c

    else:
        raise ValueError("Unknown isotherm. Use 'linear' or 'freundlich'.")

    K = rho_s / eps_p * dXdc
    Deff = D / (1.0 + K)

    k1 = (1.0 - eps_bed) / eps_bed * rho_s
    k2 = 15.0 * Deff / (Rp**2) * dXdc
    k3 = 15.0 * Deff / (Rp**2)

    return dXdc, Deff, k1, k2, k3


def build_matrices(a1, b1):
    M1 = np.zeros((n, n))
    M2 = np.zeros((n, n))
    d1 = np.zeros(n)
    d2 = np.zeros(n)

    for i in range(n):
        if i == 0:
            M1[i, i] = a1 / dz
            d1[i] = -a1 / dz * cin
        else:
            M1[i, i - 1] = -a1 / dz
            M1[i, i] = a1 / dz

    for i in range(n):
        if i == 0:
            M2[i, i] = -2.0 * b1 / dz**2
            M2[i, i + 1] = b1 / dz**2
            d2[i] = b1 / dz**2 * cin

        elif i == n - 1:
            M2[i, :] = 0.0

        else:
            M2[i, i - 1] = b1 / dz**2
            M2[i, i] = -2.0 * b1 / dz**2
            M2[i, i + 1] = b1 / dz**2

    return M1, M2, d1, d2

def ode_system(t, y, M1, M2, d1, d2, isotherm):
    c = y[:n]
    X = y[n:]

    dXdc, Deff, k1, k2, k3 = calculate_model_coefficients(isotherm, c)

    dXdt = k2 * c - k3 * X

    source_c = -k1 * dXdt

    dcdt = M1 @ c + M2 @ c + d1 + d2 + source_c

    return np.concatenate([dcdt, dXdt])

def solve_fixed_bed(isotherm):
    a1 = -u / eps_bed
    b1 = Dax

    M1, M2, d1, d2 = build_matrices(a1, b1)

    c_initial = np.zeros(n)
    X_initial = np.zeros(n)
    y_initial = np.concatenate([c_initial, X_initial])

    solution = solve_ivp(
        ode_system,
        (t_start, t_end),
        y_initial,
        args=(M1, M2, d1, d2, isotherm),
        method="BDF",
        t_eval=t_eval,
        rtol=1e-6,
        atol=1e-10
    )

    if not solution.success:
        raise RuntimeError(solution.message)

    c_solution = solution.y[:n, :]
    X_solution = solution.y[n:, :]

    c_full = np.vstack([cin * np.ones(solution.t.size), c_solution])
    X_full = np.vstack([np.zeros(solution.t.size), X_solution])

    return {
        "isotherm": isotherm,
        "t": solution.t,
        "z": z,
        "c": c_full,
        "X": X_full,
        "M1": M1,
        "M2": M2,
        "d1": d1,
        "d2": d2
    }

results = {}

for isotherm in ["linear", "freundlich"]:
    results[isotherm] = solve_fixed_bed(isotherm)


for isotherm, result in results.items():
    print(f"\nIsotherm: {isotherm}")
    print("Calculation finished.")

times_to_plot = np.arange(t_start, t_end + 1.0, 3000.0)

for isotherm, result in results.items():
    t = result["t"]
    z_plot = result["z"]
    c = result["c"]
    X = result["X"]

    plt.figure()

    for time_value in times_to_plot:
        time_index = np.argmin(np.abs(t - time_value))
        plt.plot(z_plot, c[:, time_index], label=f"{t[time_index]:.0f} s")

    plt.xlabel("z (m)")
    plt.ylabel("c (kg/m³)")
    plt.title(f"Concentration profiles over time, {isotherm} isotherm")
    plt.grid(True)
    plt.legend()
    plt.savefig(f"concentration_profiles_{isotherm}.png", dpi=300)
    plt.show()

    plt.figure()

    for time_value in times_to_plot:
        time_index = np.argmin(np.abs(t - time_value))
        plt.plot(z_plot, X[:, time_index], label=f"{t[time_index]:.0f} s")

    plt.xlabel("z (m)")
    plt.ylabel("X (kg/kg)")
    plt.title(f"Loading profiles over time, {isotherm} isotherm")
    plt.grid(True)
    plt.legend()
    plt.savefig(f"loading_profiles_{isotherm}.png", dpi=300)
    plt.show()


for isotherm, result in results.items():
    t = result["t"]
    c = result["c"]
    X = result["X"]

    output_rows = []

    for time_index in range(len(t)):
        for z_index in range(len(z)):
            output_rows.append([
                t[time_index],
                z[z_index],
                c[z_index, time_index],
                X[z_index, time_index]
            ])

    output = np.array(output_rows)

    np.savetxt(
        f"fixed_bed_all_results_{isotherm}.txt",
        output,
        header="t_s z_m c_kg_per_m3 X_kg_per_kg",
        comments=""
    )


plt.figure()

for isotherm, result in results.items():
    outlet_c = result["c"][-1, :]
    plt.plot(result["t"], outlet_c / cin, label=isotherm)

plt.xlabel("t (s)")
plt.ylabel("c_out / c_in")
plt.title("Breakthrough curve")
plt.grid(True)
plt.legend()
plt.savefig("breakthrough_curve.png", dpi=300)
plt.show()