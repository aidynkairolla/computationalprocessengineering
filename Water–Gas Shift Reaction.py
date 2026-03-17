"""
===========================================================
COMPUTATIONAL PROCESS ENGINEERING – PFR WITH REACTION
Water–Gas Shift Reaction:
CO + H2O -> CO2 + H2

Group 1 solved all 7 TASKS:

1. Set up material balances
2. Compute concentration profiles:
   a) Stationary, no diffusion
   b) Transient, no diffusion
   c) Transient, with diffusion
3. Compare transient vs stationary (CO, no diffusion)
4. Stationary solution with diffusion (pseudo-transient)
5. Reactor length & residence time for equilibrium
6. Grid and timestep study
7. Discussion of results

Additional task is skipped!

===========================================================
"""

import numpy as np
import matplotlib.pyplot as plt

# =========================================================
# ------------------ TASK 1 -------------------------------
# MATERIAL BALANCES
# =========================================================
"""
General 1D convection–diffusion–reaction equation:

∂ci/∂t = -w ∂ci/∂z + D ∂²ci/∂z² + νi R

Reaction rate:
R = k * c_CO * c_H2O

Stoichiometry:
CO  : ν = -1
H2O : ν = -1
CO2 : ν = +1
H2  : ν = +1
"""

# =========================================================
# ---------------- PARAMETERS -----------------------------
# =========================================================
k = 0.1      # m3/(kmol*s)
w = 0.1      # m/s
D = 0.01     # m2/s

L = 10.0     # m
dz = 1.0     # m
dt = 1.0     # s
t_max = 100  # s

# inlet concentrations (kmol/m3)
c_CO0  = 0.4
c_H2O0 = 0.6
c_CO20 = 0.0
c_H20  = 0.0

# grid
z = np.arange(0, L + dz, dz)
Nz = len(z) # Number of length points
Nt = int(t_max / dt) # Number of time points

# =========================================================
# UTILITY FUNCTION: PLOT PROFILES
# =========================================================
def plot_profiles(z, profiles, title):
    plt.figure(figsize=(8,5))
    for label, data in profiles.items():
        plt.plot(z, data, label=label)
    plt.xlabel("Axial position z [m]")
    plt.ylabel("Concentration [kmol/m³]")
    plt.title(title)
    plt.legend()
    plt.grid()
    plt.show()

# =========================================================
# ------------------ TASK 2a ------------------------------
# STATIONARY FLOW PIPE WITHOUT DIFFUSION
# =========================================================
"""
Steady-state PFR without diffusion:
w dc/dz = ν R

Solved numerically as an ODE along z.
"""
# Arrays with zeros because the reactor is empty initially
#ss - stationary state
c_CO_ss = np.zeros(Nz)
c_H2O_ss = np.zeros(Nz)
c_CO2_ss = np.zeros(Nz)
c_H2_ss = np.zeros(Nz)

# inlet
c_CO_ss[0]  = c_CO0
c_H2O_ss[0] = c_H2O0

for j in range(1, Nz):
    R = k * c_CO_ss[j-1] * c_H2O_ss[j-1]

    c_CO_ss[j]  = c_CO_ss[j-1]  - (R / w) * dz
    c_H2O_ss[j] = c_H2O_ss[j-1] - (R / w) * dz
    c_CO2_ss[j] = c_CO2_ss[j-1] + (R / w) * dz
    c_H2_ss[j]  = c_H2_ss[j-1]  + (R / w) * dz

plot_profiles(
    z,
    {"CO": c_CO_ss, "H2O": c_H2O_ss, "CO2": c_CO2_ss, "H2": c_H2_ss},
    "Task 2a: Stationary PFR without diffusion"
)

# =========================================================
# ---------------- TASK 5 ---------------------------------
# EXTENDED REACTOR LENGTH FOR CHEMICAL EQUILIBRIUM
# (Based on Task 2a: stationary, no diffusion)
# =========================================================
"""
The reactor length is extended beyond 10 m to determine
at which axial position the CO concentration becomes stable.
Chemical equilibrium is assumed when |dc_CO/dz| ≈ 0.
"""

# ----- extended reactor settings -----
L_ext = 50.0      # extended reactor length [m]
dz_ext = dz       # use same spatial step as Task 2a
z_ext = np.arange(0, L_ext + dz_ext, dz_ext)
Nz_ext = len(z_ext)

# ----- recompute stationary PFR (no diffusion) on extended domain -----
c_CO_ext  = np.zeros(Nz_ext)
c_H2O_ext = np.zeros(Nz_ext)
c_CO2_ext = np.zeros(Nz_ext)

# inlet conditions
c_CO_ext[0]  = c_CO0
c_H2O_ext[0] = c_H2O0

for j in range(1, Nz_ext):
    R = k * c_CO_ext[j-1] * c_H2O_ext[j-1]

    c_CO_ext[j]  = c_CO_ext[j-1]  - (R / w) * dz_ext
    c_H2O_ext[j] = c_H2O_ext[j-1] - (R / w) * dz_ext
    c_CO2_ext[j] = c_CO2_ext[j-1] + (R / w) * dz_ext

# ----- equilibrium criterion -----
tolerance = 1e-4  # kmol/m3 per meter

L_eq = None
for j in range(1, Nz_ext):
    if abs(c_CO_ext[j] - c_CO_ext[j-1]) / dz_ext < tolerance:
        L_eq = z_ext[j]
        break

# ----- residence time -----
if L_eq is not None:
    tau_eq = L_eq / w
    print("\nTask 5 – Chemical equilibrium (extended reactor):")
    print(f"Equilibrium reached at L ≈ {L_eq:.2f} m")
    print(f"Corresponding residence time τ ≈ {tau_eq:.2f} s")
else:
    print("\nTask 5 – Equilibrium not reached within 50 m reactor")

# ----- plot -----
plt.figure(figsize=(8,5))
plt.plot(z_ext, c_CO_ext, label="CO")
plt.plot(z_ext, c_CO2_ext, label="CO2")

if L_eq is not None:
    plt.axvline(L_eq, color="k", linestyle="--", label="Equilibrium length")

plt.xlabel("Axial position z [m]")
plt.ylabel("Concentration [kmol/m³]")
plt.title("Task 5: Extended reactor – equilibrium detection (Task 2a)")
plt.legend()
plt.grid()
plt.show()

#------------------TASK 6-----------------------------------------
# ---------------- Reference solution (fine grid) ----------------
dz_ref = 0.1
z_ref = np.arange(0, L + dz_ref, dz_ref)
Nz_ref = len(z_ref)

c_CO_ref  = np.zeros(Nz_ref)
c_H2O_ref = np.zeros(Nz_ref)

c_CO_ref[0]  = c_CO0
c_H2O_ref[0] = c_H2O0

for j in range(1, Nz_ref):
    R = k * c_CO_ref[j-1] * c_H2O_ref[j-1]
    c_CO_ref[j]  = c_CO_ref[j-1]  - (R / w) * dz_ref
    c_H2O_ref[j] = c_H2O_ref[j-1] - (R / w) * dz_ref

# ---------------- Test grid sizes ----------------
dz_list = [2.0, 1.0, 0.5]
results = {}

for dz_test in dz_list:
    z_test = np.arange(0, L + dz_test, dz_test)
    Nz_test = len(z_test)

    c_CO_test  = np.zeros(Nz_test)
    c_H2O_test = np.zeros(Nz_test)

    c_CO_test[0]  = c_CO0
    c_H2O_test[0] = c_H2O0

    for j in range(1, Nz_test):
        R = k * c_CO_test[j-1] * c_H2O_test[j-1]
        c_CO_test[j]  = c_CO_test[j-1]  - (R / w) * dz_test
        c_H2O_test[j] = c_H2O_test[j-1] - (R / w) * dz_test

    results[dz_test] = (z_test, c_CO_test)

# ---------------- Plot comparison ----------------
plt.figure(figsize=(9,5))
plt.plot(z_ref, c_CO_ref, "k--", linewidth=3, label="Reference (Δz = 0.1 m)")

for dz_test, (z_test, c_CO_test) in results.items():
    plt.plot(z_test, c_CO_test, marker="o", label=f"Δz = {dz_test} m")

plt.xlabel("Axial position z [m]")
plt.ylabel("CO concentration [kmol/m³]")
plt.title("Task 6: Grid size study based on Task 2a (stationary, no diffusion)")
plt.legend()
plt.grid()
plt.show()


# =========================================================
# ------------------ TASK 2b ------------------------------
# TRANSIENT FLOW PIPE WITHOUT DIFFUSION
# =========================================================
"""
∂c/∂t = -w ∂c/∂z + νR
Upwind scheme, explicit in time
"""

c_CO = np.zeros(Nz)
c_H2O = np.zeros(Nz)
c_CO2 = np.zeros(Nz)
c_H2 = np.zeros(Nz)

snapshots_no_diff = {}

for n in range(Nt):
    # inlet
    c_CO[0]  = c_CO0
    c_H2O[0] = c_H2O0

    CO_old  = c_CO.copy()
    H2O_old = c_H2O.copy()
    CO2_old = c_CO2.copy()
    H2_old  = c_H2.copy()

    for j in range(1, Nz):
        R = k * CO_old[j] * H2O_old[j]

        c_CO[j]  = CO_old[j]  - dt*w*(CO_old[j]-CO_old[j-1])/dz - dt*R
        c_H2O[j] = H2O_old[j] - dt*w*(H2O_old[j]-H2O_old[j-1])/dz - dt*R
        c_CO2[j] = CO2_old[j] - dt*w*(CO2_old[j]-CO2_old[j-1])/dz + dt*R
        c_H2[j]  = H2_old[j]  - dt*w*(H2_old[j]-H2_old[j-1])/dz + dt*R

    # store selected times for Task 3
    if n in range(0, Nt, 20):
        snapshots_no_diff[n*dt] = c_CO.copy()

plot_profiles(
    z,
    {
        "CO":  c_CO,
        "H2O": c_H2O,
        "CO2": c_CO2,
        "H2":  c_H2
    },
    "Task 2b: Transient PFR without diffusion (final time)"
)

# =========================================================
# ------------------ TASK 2c ------------------------------
# TRANSIENT FLOW PIPE WITH DIFFUSION
# =========================================================
"""
Full convection–diffusion–reaction equation
Central difference for diffusion
"""
#with diffusion
c_CO_d = np.zeros(Nz)
c_H2O_d = np.zeros(Nz)
c_CO2_d = np.zeros(Nz)
c_H2_d = np.zeros(Nz)

for n in range(Nt):

    c_CO_d[0]  = c_CO0
    c_H2O_d[0] = c_H2O0

    CO_old  = c_CO_d.copy()
    H2O_old = c_H2O_d.copy()
    CO2_old = c_CO2_d.copy()
    H2_old  = c_H2_d.copy()

    for j in range(1, Nz-1):
        R = k * CO_old[j] * H2O_old[j]

        c_CO_d[j] = CO_old[j] + dt * (
            -w*(CO_old[j]-CO_old[j-1])/dz
            + D*(CO_old[j+1]-2*CO_old[j]+CO_old[j-1])/dz**2
            - R
        )

        c_H2O_d[j] = H2O_old[j] + dt * (
            -w*(H2O_old[j]-H2O_old[j-1])/dz
            + D*(H2O_old[j+1]-2*H2O_old[j]+H2O_old[j-1])/dz**2
            - R
        )

        c_CO2_d[j] = CO2_old[j] + dt * (
            -w*(CO2_old[j]-CO2_old[j-1])/dz
            + D*(CO2_old[j+1]-2*CO2_old[j]+CO2_old[j-1])/dz**2
            + R
        )

        c_H2_d[j] = H2_old[j] + dt * (
            -w*(H2_old[j]-H2_old[j-1])/dz
            + D*(H2_old[j+1]-2*H2_old[j]+H2_old[j-1])/dz**2
            + R
        )

    # outlet zero-gradient
    c_CO_d[-1]  = c_CO_d[-2]
    c_H2O_d[-1] = c_H2O_d[-2]
    c_CO2_d[-1] = c_CO2_d[-2]
    c_H2_d[-1]  = c_H2_d[-2]

plot_profiles(
    z,
    {"CO": c_CO_d, "H2O": c_H2O_d, "CO2": c_CO2_d, "H2": c_H2_d},
    "Task 2c: Transient PFR with diffusion (final time)"
)


# =========================================================
# ------------------ TASK 3 -------------------------------
# TRANSIENT vs STATIONARY (NO DIFFUSION)
# =========================================================
plt.figure(figsize=(8,5))
plt.plot(z, c_CO_ss, "k--", label="Stationary")
for t, profile in snapshots_no_diff.items():
    plt.plot(z, profile, label=f"t = {t} s")
plt.xlabel("z [m]")
plt.ylabel("CO concentration [kmol/m³]")
plt.title("Task 3: Transient vs Stationary CO (no diffusion)")
plt.legend()
plt.grid()
plt.show()

# =========================================================
# ------------------ TASK 4  ------------------------------
# STATIONARY SOLUTION WITH DIFFUSION
# PSEUDO-TRANSIENT METHOD

# tolerance for steady-state detection
tol = 1e-6
max_iterations = 20000

# re-initialize concentrations
c_CO_d  = np.zeros(Nz)
c_H2O_d = np.zeros(Nz)
c_CO2_d = np.zeros(Nz)
c_H2_d  = np.zeros(Nz)

for n in range(max_iterations):

    # inlet boundary
    c_CO_d[0]  = c_CO0
    c_H2O_d[0] = c_H2O0

    CO_old  = c_CO_d.copy()
    H2O_old = c_H2O_d.copy()
    CO2_old = c_CO2_d.copy()
    H2_old  = c_H2_d.copy()

    for j in range(1, Nz - 1):

        R = k * CO_old[j] * H2O_old[j]

        c_CO_d[j] = CO_old[j] + dt * (
            -w * (CO_old[j] - CO_old[j-1]) / dz
            + D * (CO_old[j+1] - 2*CO_old[j] + CO_old[j-1]) / dz**2
            - R
        )

        c_H2O_d[j] = H2O_old[j] + dt * (
            -w * (H2O_old[j] - H2O_old[j-1]) / dz
            + D * (H2O_old[j+1] - 2*H2O_old[j] + H2O_old[j-1]) / dz**2
            - R
        )

        c_CO2_d[j] = CO2_old[j] + dt * (
            -w * (CO2_old[j] - CO2_old[j-1]) / dz
            + D * (CO2_old[j+1] - 2*CO2_old[j] + CO2_old[j-1]) / dz**2
            + R
        )

        c_H2_d[j] = H2_old[j] + dt * (
            -w * (H2_old[j] - H2_old[j-1]) / dz
            + D * (H2_old[j+1] - 2*H2_old[j] + H2_old[j-1]) / dz**2
            + R
        )

    # outlet zero-gradient
    c_CO_d[-1]  = c_CO_d[-2]
    c_H2O_d[-1] = c_H2O_d[-2]
    c_CO2_d[-1] = c_CO2_d[-2]
    c_H2_d[-1]  = c_H2_d[-2]

    # steady-state check (CO is sufficient)
    max_change = np.max(np.abs(c_CO_d - CO_old))
'''
    if max_change < tol:
        print(f"Task 4: Stationary solution with diffusion reached after {n} iterations")
        break
'''
# =========================================================
# TASK 4 – GRAPHICAL COMPARISON
# =========================================================
plt.figure(figsize=(8,5))
plt.plot(z, c_CO_ss, "k--", label="Stationary without diffusion")
plt.plot(z, c_CO_d, "b", label="Stationary with diffusion (pseudo-transient)")
plt.xlabel("Axial position z [m]")
plt.ylabel("CO concentration [kmol/m³]")
plt.title("Task 4: Stationary CO profiles with and without diffusion")
plt.legend()
plt.grid()
plt.show()

