import numpy as np
import matplotlib.pyplot as plt

# Reaction rate constants [1/s]
k1 = 0.002    # Rate constant for reaction A → B
k2 = 0.001    # Rate constant for reaction B → C
k3 = 0.003    # Rate constant for reaction B → D
k4 = 0.0005   # Rate constant for reaction D → B
k5 = 0.0002   # Rate constant for reaction D → E

# Time discretization settings
dt = 150.0        # Time step size in seconds
t_max = 50000.0   # Maximum simulation time in seconds
eps = 1e-6        # Tolerance used to detect equilibrium

# Initial concentrations [kmol/m³] as a NumPy array
c = np.array([1.0, 0.0, 0.0, 0.0, 0.0])
# Concentration vector:
# c[0] = c_A, c[1] = c_B, c[2] = c_C, c[3] = c_D, c[4] = c_E

# Right-hand side of ODE system
# dc/dt = f(t, c)
def f(t, c):
    # Unpack concentration vector
    cA, cB, cC, cD, cE = c

    # Define reaction rates based on first-order kinetics
    r1 = k1 * cA
    r2 = k2 * cB
    r3 = k3 * cB
    r4 = k4 * cD
    r5 = k5 * cD

    # Material balances for each component
    dcA = -r1
    dcB = r1 - r2 - r3 + r4
    dcC = r2
    dcD = r3 - r4 - r5
    dcE = r5

    # Return derivatives as a NumPy array
    return np.array([dcA, dcB, dcC, dcD, dcE])

# Lists to store time and results
time = [0.0]        # List to store time values
C = [c.copy()]      # List to store concentration vectors

t = 0.0             # Initialize time
equilibrium_reached = False  # Flag to indicate equilibrium

# Main RK4 time integration loop
while t < t_max:

    x1 = f(t, c)
    x2 = f(t + dt/2, c + dt/2 * x1)
    x3 = f(t + dt/2, c + dt/2 * x2)
    x4 = f(t + dt, c + dt * x3)

    c_new = c + (dt / 6.0) * (x1 + 2*x2 + 2*x3 + x4)

    # Check if concentrations have stopped changing significantly
    if np.max(np.abs(c_new - c)) < eps:
        equilibrium_reached = True  # Equilibrium detected
        c = c_new                  # Update concentration
        t += dt                    # Advance time
        time.append(t)             # Store time
        C.append(c.copy())         # Store concentrations
        break                      # Stop simulation

    # Update concentration vector
    c = c_new

    # Advance time by one time step
    t += dt

    # Store current results
    time.append(t)
    C.append(c.copy())

# Convert stored results to arrays
time = np.array(time)    # Convert time list to NumPy array
C = np.array(C)          # Convert concentration history to NumPy array

# Extract individual concentration profiles
cA, cB, cC, cD, cE = C.T

# Plot concentration profiles
plt.figure(figsize=(9, 5))      # Create figure
plt.plot(time, cA, label="A")   # Plot concentration of A
plt.plot(time, cB, label="B")   # Plot concentration of B
plt.plot(time, cC, label="C")   # Plot concentration of C
plt.plot(time, cD, label="D")   # Plot concentration of D
plt.plot(time, cE, label="E")   # Plot concentration of E

plt.xlabel("Time [s]")
plt.ylabel("Concentration [kmol/m³]")
plt.title("Concentration Profiles – Classical RK4")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("concentration_profiles_RK4.png", dpi=300)
plt.show()

print("===== RESULTS (RK4) =====")
print(f"Simulation terminated at t = {t:.1f} s")

# Print whether equilibrium was reached
if equilibrium_reached:
    print("Equilibrium reached (concentrations constant)")
else:
    print("Equilibrium not reached within maximum simulation time")

# Print final concentrations
print("\nFinal concentrations:")
print(f"A = {cA[-1]:.6f}")
print(f"B = {cB[-1]:.6f}")
print(f"C = {cC[-1]:.6f}")
print(f"D = {cD[-1]:.6f}")
print(f"E = {cE[-1]:.6f}")

# Check mass balance by summing all concentrations
print(f"\nMass balance check: {np.sum(C[-1]):.6f}")

# =====================================================
# Time step sensitivity analysis
# =====================================================

dt_values = [10.0, 50.0, 100.0, 150.0, 300.0]   # different time step sizes [s]
t_max = 50000.0
eps = 1e-6

final_C = []     # final concentration of C (accuracy indicator)
steps_used = []  # number of steps (computational cost)

for dt_test in dt_values:

    c = np.array([1.0, 0.0, 0.0, 0.0, 0.0])
    t = 0.0
    steps = 0

    while t < t_max:

        x1 = f(t, c)
        x2 = f(t + dt_test/2, c + dt_test/2 * x1)
        x3 = f(t + dt_test/2, c + dt_test/2 * x2)
        x4 = f(t + dt_test,   c + dt_test * x3)

        c_new = c + (dt_test / 6.0) * (x1 + 2*x2 + 2*x3 + x4)

        if np.max(np.abs(c_new - c)) < eps:
            c = c_new
            break

        c = c_new
        t += dt_test
        steps += 1

    final_C.append(c[2])   # concentration of C at equilibrium
    steps_used.append(steps)

# -----------------------------------------------------
# Plot: accuracy vs time step
# -----------------------------------------------------
plt.figure(figsize=(8, 5))
plt.plot(dt_values, final_C, marker='o')
plt.xlabel("Time step Δt [s]")
plt.ylabel("Final concentration of C [kmol/m³]")
plt.title("Effect of Time Step Size on Accuracy")
plt.grid(True)
plt.tight_layout()
plt.show()

# -----------------------------------------------------
# Plot: computational cost vs time step
# -----------------------------------------------------
plt.figure(figsize=(8, 5))
plt.plot(dt_values, steps_used, marker='o')
plt.xlabel("Time step Δt [s]")
plt.ylabel("Number of interations")
plt.title("Computational Cost vs Time Step Size")
plt.grid(True)
plt.tight_layout()
plt.show()
