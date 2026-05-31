import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp


eps_p = 0.7
rho_p = 450.0
D_eff = 4.4e-7
Y_CO2 = 1.0
p_total = 40.0e5
A = 5.83e-4
E = 93900.0
m = 0.22
S0 = 184.0 * 1000.0
mu_f = 4.6e-5
D_p = 2.0e-3
R_particle = D_p / 2.0
v_in = 1.0

R_u = 8.314462618
M_CO2 = 44.01e-3
R_CO2 = R_u / M_CO2


def calculate_parameters(T):
    c_inf = p_total / (R_CO2 * T)
    rho_f = c_inf

    Re = rho_f * v_in * D_p / mu_f
    Sc = mu_f / (rho_f * D_eff)
    Sh = 2.0 + 0.6 * Re**0.5 * Sc**(1.0 / 3.0)
    beta = Sh * D_eff / D_p

    k = S0 * A * np.exp(-E / (R_u * T)) * (p_total**m) / c_inf

    return c_inf, Re, Sc, Sh, beta, k


def ode_system(t, y, N, dr, c_inf, beta, k):
    c = y[:N+1]
    X = y[N+1:]

    dc_dt = np.zeros(N+1)
    dX_dt = np.zeros(N+1)

    reaction = k * (1.0 - X) * c

    dc_dt[0] = (
        6.0 * D_eff / (eps_p * dr**2) * (c[1] - c[0])
        - rho_p * reaction[0]
    )

    for j in range(1, N):
        diffusion_spherical = (
            2.0 * D_eff / (j * dr**2 * eps_p) * (c[j] - c[j-1])
        )

        diffusion_second = (
            D_eff / (eps_p * dr**2) * (c[j+1] - 2.0*c[j] + c[j-1])
        )

        reaction_term = rho_p * reaction[j]

        dc_dt[j] = diffusion_spherical + diffusion_second - reaction_term

    dc_dt[N] = (
        2.0 * beta / (R_particle * eps_p) * (c_inf - c[N])
        + D_eff / (eps_p * dr**2)
        * (c[N-1] - c[N] - dr * beta / D_eff * (c[N] - c_inf))
        - rho_p * reaction[N]
    )

    dX_dt = reaction

    return np.concatenate([dc_dt, dX_dt])


def solve_case(T, N=100):
    r = np.linspace(0.0, R_particle, N+1)
    dr = r[1] - r[0]

    c_inf, Re, Sc, Sh, beta, k = calculate_parameters(T)

    c_initial = np.zeros(N+1)
    X_initial = np.zeros(N+1)
    y_initial = np.concatenate([c_initial, X_initial])

    t_span = (0.0, 20.0)
    t_eval = np.linspace(0.0, 20.0, 200)

    solution = solve_ivp(
        ode_system,
        t_span,
        y_initial,
        args=(N, dr, c_inf, beta, k),
        method="BDF",
        t_eval=t_eval,
        rtol=1e-6,
        atol=1e-9
    )

    if not solution.success:
        raise RuntimeError("ODE solver failed: " + solution.message)

    c_solution = solution.y[:N+1, :]
    X_solution = solution.y[N+1:, :]

    return r, solution.t, c_solution, X_solution, c_inf, Re, Sc, Sh, beta, k


temperatures = [1273.0, 1773.0]
results = {}

for T in temperatures:
    results[T] = solve_case(T)


for T in temperatures:
    r, t, c, X, c_inf, Re, Sc, Sh, beta, k = results[T]

    print(f"\nTemperature: {T:.0f} K")
    print(f"c_inf = {c_inf:.4f} kg/m^3")
    print(f"Re    = {Re:.4f}")
    print(f"Sc    = {Sc:.4f}")
    print(f"Sh    = {Sh:.4f}")
    print(f"beta  = {beta:.6e} m/s")
    print(f"k     = {k:.6e}")


time_to_plot = 5.0

for T in temperatures:
    r, t, c, X, c_inf, Re, Sc, Sh, beta, k = results[T]

    time_index = np.argmin(np.abs(t - time_to_plot))

    plt.figure()
    plt.plot(r, c[:, time_index])
    plt.xlabel("r (m)")
    plt.ylabel("c (kg/m^3)")
    plt.title(f"CO2 concentration profile at T = {T:.0f} K, t = {t[time_index]:.2f} s")
    plt.grid(True)
    plt.savefig(f"concentration_{int(T)}K.png", dpi=300)
    plt.show()

    plt.figure()
    plt.plot(r, X[:, time_index])
    plt.xlabel("r (m)")
    plt.ylabel("X (kg/kg)")
    plt.title(f"Conversion profile at T = {T:.0f} K, t = {t[time_index]:.2f} s")
    plt.grid(True)
    plt.savefig(f"conversion_{int(T)}K.png", dpi=300)
    plt.show()


for T in temperatures:
    r, t, c, X, c_inf, Re, Sc, Sh, beta, k = results[T]

    final_c = c[:, -1]
    final_X = X[:, -1]

    output = np.column_stack([r, final_c, final_X])

    np.savetxt(
        f"result_{int(T)}K.txt",
        output,
        header="r_m c_kg_per_m3 X_kg_per_kg",
        comments=""
    )