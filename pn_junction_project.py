import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# ============================================================
# 1. Output setting
# ============================================================

FIG_DIR = Path("figures")
FIG_DIR.mkdir(exist_ok=True)

SAVE_FIGURES = True
SHOW_FIGURES = True


# ============================================================
# 2. Physical constants
# ============================================================

q = 1.602176634e-19          # elementary charge, C
k_B = 1.380649e-23           # Boltzmann constant, J/K
T = 300.0                    # temperature, K

epsilon_0 = 8.8541878128e-12 # vacuum permittivity, F/m
epsilon_r_si = 11.7          # relative permittivity of silicon
epsilon_si = epsilon_r_si * epsilon_0

V_T = k_B * T / q            # thermal voltage, V

# Silicon parameters
E_g = 1.12                   # silicon band gap, eV
n_i_cm3 = 1.0e10             # intrinsic carrier concentration, cm^-3

# Doping concentration
N_A_cm3 = 1.0e17             # acceptor concentration, cm^-3
N_D_cm3 = 1.0e16             # donor concentration, cm^-3

# Convert cm^-3 to m^-3 for electrostatics
n_i_m3 = n_i_cm3 * 1e6
N_A_m3 = N_A_cm3 * 1e6
N_D_m3 = N_D_cm3 * 1e6


# ============================================================
# 3. Built-in potential and depletion width
# ============================================================

V_bi = V_T * np.log(N_A_cm3 * N_D_cm3 / n_i_cm3**2)

def depletion_widths(V_app=0.0):
    """
    Calculate depletion width under applied bias.

    V_app > 0: forward bias
    V_app < 0: reverse bias

    The junction barrier is V_bi - V_app.
    """

    V_barrier = V_bi - V_app

    if V_barrier <= 0:
        raise ValueError(
            "V_bi - V_app must be positive. "
            "Forward bias is too large for depletion approximation."
        )

    W = np.sqrt(
        2 * epsilon_si / q
        * (1 / N_A_m3 + 1 / N_D_m3)
        * V_barrier
    )

    x_p = N_D_m3 / (N_A_m3 + N_D_m3) * W
    x_n = N_A_m3 / (N_A_m3 + N_D_m3) * W

    E_max = q * N_A_m3 * x_p / epsilon_si

    return W, x_p, x_n, E_max, V_barrier


W0, x_p0, x_n0, E_max0, V_barrier0 = depletion_widths(V_app=0.0)

print("============================================================")
print("Equilibrium p-n junction parameters")
print("============================================================")
print(f"Thermal voltage V_T        = {V_T:.5f} V")
print(f"Built-in potential V_bi    = {V_bi:.5f} V")
print(f"x_p                       = {x_p0 * 1e6:.5f} um")
print(f"x_n                       = {x_n0 * 1e6:.5f} um")
print(f"W                         = {W0 * 1e6:.5f} um")
print(f"E_max                     = {E_max0:.5e} V/m")
print(f"E_max                     = {E_max0 / 100:.5e} V/cm")
print("============================================================")


# ============================================================
# 4. Electrostatic profiles under depletion approximation
# ============================================================

def electrostatic_profiles(x, V_app=0.0):
    """
    Return charge density rho(x), electric field E(x),
    and electrostatic potential psi(x).

    x must be in meters.
    rho: C/m^3
    E: V/m
    psi: V

    Convention:
    p-side is x < 0
    n-side is x > 0
    """

    W, x_p, x_n, E_max, V_barrier = depletion_widths(V_app)

    rho = np.zeros_like(x)
    E = np.zeros_like(x)
    psi = np.zeros_like(x)

    # Regions
    p_depletion = (x >= -x_p) & (x < 0)
    n_depletion = (x >= 0) & (x <= x_n)
    right_neutral = x > x_n

    # Charge density
    rho[p_depletion] = -q * N_A_m3
    rho[n_depletion] = q * N_D_m3

    # Electric field
    # Boundary condition: E(-x_p) = 0 and E(x_n) = 0
    E[p_depletion] = -q * N_A_m3 * (x[p_depletion] + x_p) / epsilon_si
    E[n_depletion] = q * N_D_m3 * (x[n_depletion] - x_n) / epsilon_si

    # Electrostatic potential
    # Boundary condition: psi(-x_p) = 0
    psi[p_depletion] = (
        q * N_A_m3 / (2 * epsilon_si)
        * (x[p_depletion] + x_p) ** 2
    )

    psi_0 = q * N_A_m3 * x_p**2 / (2 * epsilon_si)

    psi[n_depletion] = (
        psi_0
        + q * N_D_m3 / epsilon_si
        * (x_n * x[n_depletion] - 0.5 * x[n_depletion] ** 2)
    )

    psi[right_neutral] = V_barrier

    return rho, E, psi, W, x_p, x_n, E_max, V_barrier


# ============================================================
# 5. Energy band diagram
# ============================================================

def band_edges(x, V_app=0.0):
    """
    Calculate E_C(x), E_V(x), E_i(x).

    Energies are in eV.

    For equilibrium:
    E_F is flat and set to 0 eV.
    The intrinsic energy level on p-side is:
        E_i - E_F = V_T ln(N_A / n_i)
    """

    rho, E, psi, W, x_p, x_n, E_max, V_barrier = electrostatic_profiles(x, V_app)

    E_F = np.zeros_like(x)

    # Intrinsic level on the p-side relative to E_F
    E_i_left = V_T * np.log(N_A_cm3 / n_i_cm3)

    # Electron energy decreases when electrostatic potential increases
    E_i = E_i_left - psi

    E_C = E_i + E_g / 2
    E_V = E_i - E_g / 2

    return E_C, E_V, E_i, E_F, W, x_p, x_n, V_barrier


# ============================================================
# 6. Plot equilibrium electrostatics
# ============================================================

# Use a domain slightly larger than the depletion region
x_min = -0.25e-6
x_max = 0.80e-6
x = np.linspace(x_min, x_max, 2000)

rho, E, psi, W, x_p, x_n, E_max, V_barrier = electrostatic_profiles(x, V_app=0.0)

x_um = x * 1e6

fig, axes = plt.subplots(3, 1, figsize=(8, 9), sharex=True)

axes[0].plot(x_um, rho)
axes[0].axvline(-x_p * 1e6, linestyle="--")
axes[0].axvline(0, linestyle="--")
axes[0].axvline(x_n * 1e6, linestyle="--")
axes[0].set_ylabel(r"$\rho(x)$ (C/m$^3$)")
axes[0].set_title("Equilibrium charge density")
axes[0].grid(True)

axes[1].plot(x_um, E)
axes[1].axvline(-x_p * 1e6, linestyle="--")
axes[1].axvline(0, linestyle="--")
axes[1].axvline(x_n * 1e6, linestyle="--")
axes[1].set_ylabel(r"$E(x)$ (V/m)")
axes[1].set_title("Equilibrium electric field")
axes[1].grid(True)

axes[2].plot(x_um, psi)
axes[2].axvline(-x_p * 1e6, linestyle="--")
axes[2].axvline(0, linestyle="--")
axes[2].axvline(x_n * 1e6, linestyle="--")
axes[2].set_xlabel(r"Position $x$ ($\mu$m)")
axes[2].set_ylabel(r"$\psi(x)$ (V)")
axes[2].set_title("Equilibrium electrostatic potential")
axes[2].grid(True)

plt.tight_layout()

if SAVE_FIGURES:
    plt.savefig(FIG_DIR / "01_equilibrium_electrostatics.png", dpi=300)

if SHOW_FIGURES:
    plt.show()
else:
    plt.close()


# ============================================================
# 7. Plot equilibrium energy band diagram
# ============================================================

E_C, E_V, E_i, E_F, W, x_p, x_n, V_barrier = band_edges(x, V_app=0.0)

plt.figure(figsize=(8, 5))
plt.plot(x_um, E_C, label=r"$E_C$")
plt.plot(x_um, E_V, label=r"$E_V$")
plt.plot(x_um, E_i, label=r"$E_i$")
plt.plot(x_um, E_F, linestyle="--", label=r"$E_F$")

plt.axvline(-x_p * 1e6, linestyle="--")
plt.axvline(0, linestyle="--")
plt.axvline(x_n * 1e6, linestyle="--")

plt.xlabel(r"Position $x$ ($\mu$m)")
plt.ylabel("Energy (eV)")
plt.title("Equilibrium energy band diagram")
plt.legend()
plt.grid(True)

# Mark qV_bi. Since energy is in eV, qV_bi corresponds to V_bi eV.
x_arrow = 0.65
plt.annotate(
    r"$qV_{bi}$",
    xy=(x_arrow, E_i[-1]),
    xytext=(x_arrow, E_i[0]),
    arrowprops=dict(arrowstyle="<->")
)

plt.tight_layout()

if SAVE_FIGURES:
    plt.savefig(FIG_DIR / "02_equilibrium_band_diagram.png", dpi=300)

if SHOW_FIGURES:
    plt.show()
else:
    plt.close()


# ============================================================
# 8. Ideal Shockley current-voltage characteristic
# ============================================================

# Diode parameters for current calculation
# These are assumptions and must be listed in the report.
A_cm2 = 1.0e-4       # diode area, cm^2
D_n = 35.0           # electron diffusion coefficient, cm^2/s
D_p = 12.0           # hole diffusion coefficient, cm^2/s
L_n = 10e-4          # electron diffusion length in p-side, cm
L_p = 10e-4          # hole diffusion length in n-side, cm

# Minority carrier concentrations
p_n0 = n_i_cm3**2 / N_D_cm3     # hole concentration on n-side, cm^-3
n_p0 = n_i_cm3**2 / N_A_cm3     # electron concentration on p-side, cm^-3

I_S = q * A_cm2 * (
    D_p * p_n0 / L_p
    + D_n * n_p0 / L_n
)

print("============================================================")
print("Ideal diode current parameters")
print("============================================================")
print(f"Area A                    = {A_cm2:.3e} cm^2")
print(f"D_n                       = {D_n:.3e} cm^2/s")
print(f"D_p                       = {D_p:.3e} cm^2/s")
print(f"L_n                       = {L_n:.3e} cm")
print(f"L_p                       = {L_p:.3e} cm")
print(f"p_n0                      = {p_n0:.3e} cm^-3")
print(f"n_p0                      = {n_p0:.3e} cm^-3")
print(f"Saturation current I_S    = {I_S:.3e} A")
print("============================================================")


def safe_exp(y):
    """
    Prevent numerical overflow in exponential.
    """
    return np.exp(np.clip(y, -100, 100))


def ideal_diode_current(V):
    """
    Ideal Shockley diode current.
    """
    return I_S * (safe_exp(V / V_T) - 1)


V = np.linspace(-5.0, 0.8, 1000)
I_ideal = ideal_diode_current(V)

fig, axes = plt.subplots(2, 1, figsize=(8, 8), sharex=True)

axes[0].plot(V, I_ideal)
axes[0].axhline(I_S, linestyle="--", label=r"$I_S$")
axes[0].axhline(-I_S, linestyle="--", label=r"$-I_S$")
axes[0].axvspan(0.55, 0.75, alpha=0.15, label="turn-on region")
axes[0].set_ylabel("Current I (A)")
axes[0].set_title("Ideal Shockley diode current-voltage characteristic")
axes[0].legend()
axes[0].grid(True)

axes[1].semilogy(V, np.abs(I_ideal) + 1e-300)
axes[1].axhline(I_S, linestyle="--", label=r"$I_S$")
axes[1].axvspan(0.55, 0.75, alpha=0.15, label="turn-on region")
axes[1].set_xlabel("Voltage V (V)")
axes[1].set_ylabel(r"$|I|$ (A)")
axes[1].set_title(r"Semilog plot of $|I|$")
axes[1].legend()
axes[1].grid(True)

plt.tight_layout()

if SAVE_FIGURES:
    plt.savefig(FIG_DIR / "03_ideal_IV.png", dpi=300)

if SHOW_FIGURES:
    plt.show()
else:
    plt.close()


# ============================================================
# 9. Bias-dependent band diagrams
# ============================================================

# Example biases:
# V_app = +0.5 V: forward bias
# V_app = -2.0 V: reverse bias
bias_list = [0.0, 0.5, -2.0]

# Use a wider x range because reverse bias increases depletion width.
W_reverse, x_p_reverse, x_n_reverse, _, _ = depletion_widths(V_app=-2.0)
x_bias = np.linspace(-0.2e-6, 1.0e-6, 2500)
x_bias_um = x_bias * 1e6

fig, axes = plt.subplots(len(bias_list), 1, figsize=(8, 11), sharex=True)

for ax, V_app in zip(axes, bias_list):
    E_C, E_V, E_i, E_F, W, x_p, x_n, V_barrier = band_edges(x_bias, V_app=V_app)

    ax.plot(x_bias_um, E_C, label=r"$E_C$")
    ax.plot(x_bias_um, E_V, label=r"$E_V$")
    ax.plot(x_bias_um, E_i, label=r"$E_i$")

    ax.axvline(-x_p * 1e6, linestyle="--")
    ax.axvline(0, linestyle="--")
    ax.axvline(x_n * 1e6, linestyle="--")

    ax.set_ylabel("Energy (eV)")
    ax.set_title(
        f"Band diagram under V = {V_app:+.2f} V, "
        f"W = {W * 1e6:.3f} um, barrier = {V_barrier:.3f} eV"
    )
    ax.grid(True)
    ax.legend()

axes[-1].set_xlabel(r"Position $x$ ($\mu$m)")

plt.tight_layout()

if SAVE_FIGURES:
    plt.savefig(FIG_DIR / "04_bias_dependent_band_diagrams.png", dpi=300)

if SHOW_FIGURES:
    plt.show()
else:
    plt.close()


# ============================================================
# 10. Suggested extension:
#     Recombination-generation current
#     Series resistance
#     Reverse breakdown
# ============================================================

def depletion_width_cm_for_current(V_app):
    """
    Depletion width in cm for recombination-generation current model.

    For very large forward bias, V_bi - V_app may become non-positive.
    We clamp the barrier to a small positive value to avoid numerical error.
    """
    V_barrier = np.maximum(V_bi - V_app, 1e-6)

    W_m = np.sqrt(
        2 * epsilon_si / q
        * (1 / N_A_m3 + 1 / N_D_m3)
        * V_barrier
    )

    return W_m * 100.0  # meter to cm


# Shockley-Read-Hall lifetime assumption
tau_SRH = 1e-6  # s


def recombination_generation_prefactor(V):
    """
    Calculate the bias-dependent Shockley-Read-Hall
    recombination-generation current scale:

        I_RG0(V) = q A n_i W(V) / (2 tau_SRH)
    """

    W_cm = depletion_width_cm_for_current(V)

    I_RG0 = (
        q
        * A_cm2
        * n_i_cm3
        * W_cm
        / (2.0 * tau_SRH)
    )

    return I_RG0


def recombination_generation_current(V):
    """
    Simplified depletion-region recombination-generation current:

        I_RG(V) = I_RG0(V) * [exp(V / (2 V_T)) - 1]
    """

    I_RG0 = recombination_generation_prefactor(V)

    return I_RG0 * (
        safe_exp(V / (2.0 * V_T)) - 1.0
    )


def diode_current_with_RG(V):
    """
    Ideal diffusion current plus recombination-generation current.
    """
    return ideal_diode_current(V) + recombination_generation_current(V)


# ============================================================
# 10A. Section 2.1:
#      Recombination-generation current comparison
# ============================================================

# Do not include V = 0 because total current is exactly zero there,
# and zero cannot be displayed properly on a logarithmic scale.

# Forward-bias range
# Keep the maximum voltage below V_bi because the depletion
# approximation becomes inaccurate when V approaches V_bi.
V_forward_RG = np.linspace(0.02, 0.70, 800)

I_diffusion_forward = ideal_diode_current(V_forward_RG)
I_RG_forward = recombination_generation_current(V_forward_RG)
I_total_forward = diode_current_with_RG(V_forward_RG)


# Reverse-bias range
V_reverse_RG = np.linspace(-5.0, -0.02, 800)

I_diffusion_reverse = ideal_diode_current(V_reverse_RG)
I_RG_reverse = recombination_generation_current(V_reverse_RG)
I_total_reverse = diode_current_with_RG(V_reverse_RG)


# Plot forward- and reverse-bias effects
fig, axes = plt.subplots(2, 1, figsize=(8, 9))

# ------------------------------------------------------------
# Forward bias
# ------------------------------------------------------------

axes[0].semilogy(
    V_forward_RG,
    np.abs(I_diffusion_forward),
    label="Ideal diffusion current"
)

axes[0].semilogy(
    V_forward_RG,
    np.abs(I_RG_forward),
    label="Depletion-region recombination current"
)

axes[0].semilogy(
    V_forward_RG,
    np.abs(I_total_forward),
    linestyle="--",
    label="Total current"
)

axes[0].set_xlabel("Forward voltage V (V)")
axes[0].set_ylabel(r"$|I|$ (A)")
axes[0].set_title(
    "Effect of depletion-region recombination under forward bias"
)
axes[0].grid(True)
axes[0].legend()


# ------------------------------------------------------------
# Reverse bias
# ------------------------------------------------------------

axes[1].semilogy(
    V_reverse_RG,
    np.abs(I_diffusion_reverse),
    label="Ideal diffusion current"
)

axes[1].semilogy(
    V_reverse_RG,
    np.abs(I_RG_reverse),
    label="Depletion-region generation current"
)

axes[1].semilogy(
    V_reverse_RG,
    np.abs(I_total_reverse),
    linestyle="--",
    label="Total current"
)

axes[1].set_xlabel("Reverse voltage V (V)")
axes[1].set_ylabel(r"$|I|$ (A)")
axes[1].set_title(
    "Effect of depletion-region generation under reverse bias"
)
axes[1].grid(True)
axes[1].legend()

plt.tight_layout()

if SAVE_FIGURES:
    plt.savefig(
        FIG_DIR / "05_recombination_generation_current.png",
        dpi=300,
        bbox_inches="tight"
    )

if SHOW_FIGURES:
    plt.show()
else:
    plt.close()

    # ============================================================
# Numerical results for Section 2.1
# ============================================================

I_RG0_equilibrium = recombination_generation_prefactor(0.0)

print("\n============================================================")
print("Section 2.1: Recombination-generation current")
print("============================================================")
print(f"Shockley-Read-Hall lifetime = {tau_SRH:.3e} s")
print(f"I_RG0 at equilibrium        = {I_RG0_equilibrium:.3e} A")
print(f"Ideal saturation current    = {I_S:.3e} A")
print(
    f"I_RG0 / I_S                 = "
    f"{I_RG0_equilibrium / I_S:.3e}"
)


test_voltages = [-5.0, -2.0, 0.2, 0.4]

for V_test in test_voltages:

    I_diff_test = ideal_diode_current(V_test)
    I_RG_test = recombination_generation_current(V_test)
    I_total_test = diode_current_with_RG(V_test)

    # depletion_width_cm_for_current returns cm
    # 1 cm = 10^4 um
    W_test_um = depletion_width_cm_for_current(V_test) * 1e4

    print("------------------------------------------------------------")
    print(f"Applied voltage V           = {V_test:+.2f} V")
    print(f"Depletion width W           = {W_test_um:.4f} um")
    print(f"I_diffusion                 = {I_diff_test:.3e} A")
    print(f"I_RG                        = {I_RG_test:.3e} A")
    print(f"I_total                     = {I_total_test:.3e} A")


# Find approximate crossover voltage:
# |I_diffusion| = |I_RG|
V_scan = np.linspace(0.02, 0.70, 5000)

I_diff_scan = np.abs(ideal_diode_current(V_scan))
I_RG_scan = np.abs(recombination_generation_current(V_scan))

log_difference = np.abs(
    np.log10(I_diff_scan)
    - np.log10(I_RG_scan)
)

crossover_index = np.argmin(log_difference)
V_crossover = V_scan[crossover_index]

print("------------------------------------------------------------")
print(
    f"Approximate crossover voltage = "
    f"{V_crossover:.3f} V"
)
print("============================================================")


# ------------------------------------------------------------
# Series resistance model
# ------------------------------------------------------------

R_s = 5.0  # ohm, assumed lumped series resistance

def solve_current_with_series_resistance(V_terminal):
    """
    Solve:
        I = I_diode(V_terminal - I R_s)

    This equation is implicit because I appears on both sides.
    We solve it by bisection.
    """

    def f(I):
        V_junction = V_terminal - I * R_s
        return I - diode_current_with_RG(V_junction)

    # A wide bracket for current
    low = -1.0
    high = 1.0

    f_low = f(low)
    f_high = f(high)

    # Expand bracket if needed
    for _ in range(50):
        if f_low * f_high < 0:
            break
        low *= 2
        high *= 2
        f_low = f(low)
        f_high = f(high)

    # Bisection
    for _ in range(200):
        mid = 0.5 * (low + high)
        f_mid = f(mid)

        if f_low * f_mid <= 0:
            high = mid
            f_high = f_mid
        else:
            low = mid
            f_low = f_mid

    return 0.5 * (low + high)


def current_with_series_resistance(V_array):
    """
    Apply series resistance solver to an array of terminal voltages.
    """
    I_out = np.zeros_like(V_array)

    for i, V_terminal in enumerate(V_array):
        I_out[i] = solve_current_with_series_resistance(V_terminal)

    return I_out


# ============================================================
# 10B. Section 2.2:
#      Series resistance effect
# ============================================================

# Use a forward-bias range below or close to V_bi.
# This keeps the depletion-region model physically meaningful.
V_forward_Rs = np.linspace(0.02, 0.77, 800)

# Current without series resistance:
# diffusion current + recombination-generation current
I_without_Rs = diode_current_with_RG(V_forward_Rs)

# Current with series resistance
I_with_Rs_section = current_with_series_resistance(V_forward_Rs)

# Actual voltage across the p-n junction
V_junction = V_forward_Rs - I_with_Rs_section * R_s

# Voltage drop across the series resistance
V_drop_Rs = I_with_Rs_section * R_s


# ============================================================
# Plot the effect of series resistance
# ============================================================

fig, axes = plt.subplots(2, 1, figsize=(8, 9))

# ------------------------------------------------------------
# Linear-scale plot
# ------------------------------------------------------------

axes[0].plot(
    V_forward_Rs,
    I_without_Rs,
    label="Without series resistance"
)

axes[0].plot(
    V_forward_Rs,
    I_with_Rs_section,
    label=f"With series resistance, Rs = {R_s:.1f} ohm"
)

axes[0].set_xlabel("Terminal voltage V (V)")
axes[0].set_ylabel("Current I (A)")
axes[0].set_title(
    "Effect of series resistance on forward current"
)
axes[0].grid(True)
axes[0].legend()


# ------------------------------------------------------------
# Semilogarithmic plot
# ------------------------------------------------------------

small_number = 1e-300

axes[1].semilogy(
    V_forward_Rs,
    np.abs(I_without_Rs) + small_number,
    label="Without series resistance"
)

axes[1].semilogy(
    V_forward_Rs,
    np.abs(I_with_Rs_section) + small_number,
    label=f"With series resistance, Rs = {R_s:.1f} ohm"
)

axes[1].set_xlabel("Terminal voltage V (V)")
axes[1].set_ylabel(r"$|I|$ (A)")
axes[1].set_title(
    "Semilogarithmic comparison of series-resistance effect"
)
axes[1].grid(True)
axes[1].legend()

plt.tight_layout()

if SAVE_FIGURES:
    plt.savefig(
        FIG_DIR / "06_series_resistance.png",
        dpi=300,
        bbox_inches="tight"
    )

if SHOW_FIGURES:
    plt.show()
else:
    plt.close()

    # ============================================================
# Numerical results for Section 2.2
# ============================================================

print("\n============================================================")
print("Section 2.2: Series resistance")
print("============================================================")
print(f"Assumed series resistance Rs = {R_s:.3f} ohm")

test_voltages_Rs = [0.60, 0.70, 0.75, 0.77]

for V_terminal_test in test_voltages_Rs:

    I_no_Rs_test = float(
        diode_current_with_RG(V_terminal_test)
    )

    I_with_Rs_test = solve_current_with_series_resistance(
        V_terminal_test
    )

    voltage_drop_test = I_with_Rs_test * R_s

    V_junction_test = (
        V_terminal_test - voltage_drop_test
    )

    current_reduction_percent = (
        1.0 - I_with_Rs_test / I_no_Rs_test
    ) * 100.0

    print("------------------------------------------------------------")
    print(
        f"Terminal voltage             = "
        f"{V_terminal_test:.2f} V"
    )
    print(
        f"Current without Rs           = "
        f"{I_no_Rs_test:.3e} A"
    )
    print(
        f"Current with Rs              = "
        f"{I_with_Rs_test:.3e} A"
    )
    print(
        f"Voltage drop I*Rs            = "
        f"{voltage_drop_test:.3e} V"
    )
    print(
        f"Junction voltage             = "
        f"{V_junction_test:.5f} V"
    )
    print(
        f"Current reduction            = "
        f"{current_reduction_percent:.2f} %"
    )

print("============================================================")


# ============================================================
# Section 2.4: Metal-semiconductor contact
# ============================================================

# Assume one non-ideal metal / n-type silicon Schottky contact.
# The other contact is assumed to be an ideal ohmic contact.

A_Richardson = 112.0   # effective Richardson constant for n-Si,
                       # A/(cm^2 K^2)

phi_Bn = 0.55          # assumed Schottky barrier height, eV
n_MS = 1.10            # Schottky-contact ideality factor

# Thermionic-emission saturation current
# phi_Bn / V_T is dimensionless because phi_Bn is expressed in eV.
I_MS0 = (
    A_cm2
    * A_Richardson
    * T**2
    * np.exp(-phi_Bn / V_T)
)


def schottky_contact_voltage(I):
    """
    Voltage drop across the metal-semiconductor Schottky contact.

        V_MS = n_MS V_T ln(1 + I / I_MS0)

    This simplified function is used only for forward current.
    """

    I = np.asarray(I, dtype=float)

    # The present extension only compares the forward characteristic.
    I_forward = np.maximum(I, 0.0)

    return (
        n_MS
        * V_T
        * np.log1p(I_forward / I_MS0)
    )


def solve_current_with_MS_contact(V_terminal):
    """
    Solve the terminal current when a Schottky contact is connected
    in series with the p-n junction.

    Terminal-voltage relation:

        V_terminal = V_PN + I*R_s + V_MS

    Therefore:

        V_PN = V_terminal - I*R_s - V_MS(I)

    The resulting implicit equation is solved using bisection.
    """

    if V_terminal <= 0.0:
        return 0.0

    # The current with an ideal ohmic contact provides a reasonable
    # upper bound because the Schottky contact can only reduce current.
    I_upper = solve_current_with_series_resistance(V_terminal)

    if I_upper <= 0.0:
        return 0.0

    low = 0.0
    high = I_upper

    def residual(I):
        V_MS = float(schottky_contact_voltage(I))

        V_PN = (
            V_terminal
            - I * R_s
            - V_MS
        )

        I_PN = float(diode_current_with_RG(V_PN))

        return I - I_PN

    f_low = residual(low)
    f_high = residual(high)

    # The upper bound should normally bracket the solution.
    # Expand it if numerical roundoff prevents bracketing.
    for _ in range(30):
        if f_low * f_high <= 0:
            break

        high *= 2.0
        f_high = residual(high)

    # Bisection method
    for _ in range(200):
        mid = 0.5 * (low + high)
        f_mid = residual(mid)

        if f_low * f_mid <= 0:
            high = mid
            f_high = f_mid
        else:
            low = mid
            f_low = f_mid

    return 0.5 * (low + high)


def current_with_MS_contact(V_array):
    """
    Calculate current for an array of forward terminal voltages.
    """

    V_array = np.asarray(V_array, dtype=float)
    I_array = np.zeros_like(V_array)

    for i, V_terminal in enumerate(V_array):
        I_array[i] = solve_current_with_MS_contact(V_terminal)

    return I_array

# ============================================================
# Section 2.4 M-S contact comparison plot
# ============================================================

V_forward_MS = np.linspace(0.02, 0.85, 800)

# Ideal ohmic-contact case:
# p-n junction plus series resistance
I_ohmic_contact = current_with_series_resistance(V_forward_MS)

# Non-ideal Schottky-contact case
I_schottky_contact = current_with_MS_contact(V_forward_MS)


fig, axes = plt.subplots(2, 1, figsize=(8, 9))

# Linear-scale plot
axes[0].plot(
    V_forward_MS,
    I_ohmic_contact,
    label="Ideal ohmic contact"
)

axes[0].plot(
    V_forward_MS,
    I_schottky_contact,
    label=(
        "Schottky contact, "
        rf"$\Phi_{{Bn}}={phi_Bn:.2f}$ eV"
    )
)

axes[0].set_xlabel("Terminal voltage V (V)")
axes[0].set_ylabel("Current I (A)")
axes[0].set_title(
    "Effect of metal-semiconductor contact on forward current"
)
axes[0].grid(True)
axes[0].legend()


# Semilogarithmic plot
plot_floor = 1e-300

axes[1].semilogy(
    V_forward_MS,
    np.abs(I_ohmic_contact) + plot_floor,
    label="Ideal ohmic contact"
)

axes[1].semilogy(
    V_forward_MS,
    np.abs(I_schottky_contact) + plot_floor,
    label=(
        "Schottky contact, "
        rf"$\Phi_{{Bn}}={phi_Bn:.2f}$ eV"
    )
)

axes[1].set_xlabel("Terminal voltage V (V)")
axes[1].set_ylabel(r"$|I|$ (A)")
axes[1].set_title(
    "Semilogarithmic M-S contact comparison"
)
axes[1].grid(True)
axes[1].legend()

plt.tight_layout()

if SAVE_FIGURES:
    plt.savefig(
        FIG_DIR / "08_MS_contact.png",
        dpi=300,
        bbox_inches="tight"
    )

if SHOW_FIGURES:
    plt.show()
else:
    plt.close()

    # ============================================================
# Numerical results for Section 2.4
# ============================================================

print("\n============================================================")
print("Section 2.4: Metal-semiconductor contact")
print("============================================================")
print(f"Richardson constant          = {A_Richardson:.3f} A/cm^2/K^2")
print(f"Schottky barrier height      = {phi_Bn:.3f} eV")
print(f"M-S contact ideality factor  = {n_MS:.3f}")
print(f"M-S saturation current       = {I_MS0:.3e} A")

test_voltages_MS = [0.60, 0.70, 0.80, 0.85]

for V_test_MS in test_voltages_MS:

    I_ohmic_test = solve_current_with_series_resistance(
        V_test_MS
    )

    I_MS_test = solve_current_with_MS_contact(
        V_test_MS
    )

    V_contact_test = float(
        schottky_contact_voltage(I_MS_test)
    )

    V_PN_test = (
        V_test_MS
        - I_MS_test * R_s
        - V_contact_test
    )

    print("------------------------------------------------------------")
    print(f"Terminal voltage             = {V_test_MS:.2f} V")
    print(f"Current with ohmic contact   = {I_ohmic_test:.3e} A")
    print(f"Current with Schottky contact= {I_MS_test:.3e} A")
    print(f"M-S contact voltage drop     = {V_contact_test:.3e} V")
    print(f"Actual p-n junction voltage  = {V_PN_test:.5f} V")

print("============================================================")




# ------------------------------------------------------------
# Reverse breakdown model
# ------------------------------------------------------------

V_BR = 20.0       # breakdown voltage magnitude, V
I_BR0 = 1e-6      # breakdown current scale, A
V_avalanche = 1.0 # controls how sharp breakdown is, V

def reverse_breakdown_current(V):
    """
    Simple empirical reverse breakdown model.

    When V < -V_BR, a large negative current is added.
    """
    I_BD = np.zeros_like(V)

    mask = V < -V_BR

    I_BD[mask] = -I_BR0 * (
        safe_exp((-V[mask] - V_BR) / V_avalanche) - 1
    )

    return I_BD

def reverse_breakdown_current(V):
    """
    Simple empirical reverse breakdown model.
    """
    I_BD = np.zeros_like(V)

    mask = V < -V_BR

    I_BD[mask] = -I_BR0 * (
        safe_exp((-V[mask] - V_BR) / V_avalanche) - 1
    )

    return I_BD


# ============================================================
# Section 2.3: Reverse breakdown
# 這裡插入專用圖片程式
# ============================================================

V_reverse_BD = np.linspace(-30.0, 0.0, 1200)

I_before_BD = current_with_series_resistance(V_reverse_BD)
I_BD = reverse_breakdown_current(V_reverse_BD)
I_after_BD = I_before_BD + I_BD

fig, axes = plt.subplots(2, 1, figsize=(8, 9), sharex=True)

axes[0].plot(
    V_reverse_BD,
    I_before_BD,
    label="Before reverse breakdown"
)

axes[0].plot(
    V_reverse_BD,
    I_after_BD,
    label="With reverse breakdown"
)

axes[0].axvline(
    -V_BR,
    linestyle="--",
    label=r"$-V_{\mathrm{BR}}$"
)

axes[0].set_ylabel("Current I (A)")
axes[0].set_title("Effect of reverse breakdown on diode current")
axes[0].grid(True)
axes[0].legend()

small_number = 1e-300

axes[1].semilogy(
    V_reverse_BD,
    np.abs(I_before_BD) + small_number,
    label="Before reverse breakdown"
)

axes[1].semilogy(
    V_reverse_BD,
    np.abs(I_after_BD) + small_number,
    label="With reverse breakdown"
)

axes[1].axvline(
    -V_BR,
    linestyle="--",
    label=r"$-V_{\mathrm{BR}}$"
)

axes[1].set_xlabel("Applied voltage V (V)")
axes[1].set_ylabel(r"$|I|$ (A)")
axes[1].set_title("Semilogarithmic reverse-breakdown characteristic")
axes[1].grid(True)
axes[1].legend()

plt.tight_layout()

if SAVE_FIGURES:
    plt.savefig(
        FIG_DIR / "07_reverse_breakdown.png",
        dpi=300,
        bbox_inches="tight"
    )

if SHOW_FIGURES:
    plt.show()
else:
    plt.close()


# ============================================================
# Section 2.3 numerical results
# 數值輸出接在專用圖片後面
# ============================================================

print("\n============================================================")
print("Section 2.3: Reverse breakdown")
print("============================================================")
print(f"Breakdown voltage magnitude = {V_BR:.2f} V")
print(f"Breakdown current scale      = {I_BR0:.3e} A")
print(f"Breakdown sharpness voltage  = {V_avalanche:.3f} V")

test_voltages_BD = [-18.0, -20.0, -22.0, -25.0, -30.0]

for V_test_BD in test_voltages_BD:

    V_array = np.array([V_test_BD])

    I_base = current_with_series_resistance(V_array)[0]
    I_breakdown = reverse_breakdown_current(V_array)[0]
    I_total = I_base + I_breakdown

    print("------------------------------------------------------------")
    print(f"Applied voltage             = {V_test_BD:+.2f} V")
    print(f"Current before breakdown    = {I_base:.3e} A")
    print(f"Breakdown current component = {I_breakdown:.3e} A")
    print(f"Total current               = {I_total:.3e} A")

print("============================================================")


# ============================================================
# 原本的總比較圖
# 這一段保持在 2.3 後面
# ============================================================

V_ext = np.linspace(-30.0, 0.85, 1600)

I_diffusion = ideal_diode_current(V_ext)
I_diffusion_RG = diode_current_with_RG(V_ext)
I_with_Rs = current_with_series_resistance(V_ext)
I_breakdown = reverse_breakdown_current(V_ext)
I_real = I_with_Rs + I_breakdown


# Use a wider reverse-bias range to show breakdown.
V_ext = np.linspace(-30.0, 0.85, 1600)

I_diffusion = ideal_diode_current(V_ext)
I_diffusion_RG = diode_current_with_RG(V_ext)
I_with_Rs = current_with_series_resistance(V_ext)
I_breakdown = reverse_breakdown_current(V_ext)
I_real = I_with_Rs + I_breakdown

fig, axes = plt.subplots(2, 1, figsize=(8, 8), sharex=True)

axes[0].plot(V_ext, I_diffusion, label="Ideal diffusion current")
axes[0].plot(V_ext, I_diffusion_RG, label="With recombination-generation")
axes[0].plot(V_ext, I_with_Rs, label="With recombination-generation and series resistance")
axes[0].plot(V_ext, I_real, label="With reverse breakdown")
axes[0].set_ylabel("Current I (A)")
axes[0].set_title("Real-device effects on current-voltage characteristic")
axes[0].legend()
axes[0].grid(True)

axes[1].semilogy(V_ext, np.abs(I_diffusion) + 1e-300, label="Ideal diffusion current")
axes[1].semilogy(V_ext, np.abs(I_diffusion_RG) + 1e-300, label="With recombination-generation")
axes[1].semilogy(V_ext, np.abs(I_with_Rs) + 1e-300, label="With recombination-generation and series resistance")
axes[1].semilogy(V_ext, np.abs(I_real) + 1e-300, label="With reverse breakdown")
axes[1].axvline(-V_BR, linestyle="--", label=r"$-V_{BR}$")
axes[1].set_xlabel("Voltage V (V)")
axes[1].set_ylabel(r"$|I|$ (A)")
axes[1].set_title("Semilog plot with real-device effects")
axes[1].legend()
axes[1].grid(True)

plt.tight_layout()

if SAVE_FIGURES:
    plt.savefig(FIG_DIR / "05_real_device_effects_IV.png", dpi=300)

if SHOW_FIGURES:
    plt.show()
else:
    plt.close()


# ============================================================
# 11. Print parameter table for report
# ============================================================

print("\n============================================================")
print("Parameter table for report")
print("============================================================")
print(f"T                         = {T:.1f} K")
print(f"V_T                       = {V_T:.5f} V")
print(f"epsilon_r,Si              = {epsilon_r_si:.2f}")
print(f"E_g                       = {E_g:.2f} eV")
print(f"n_i                       = {n_i_cm3:.3e} cm^-3")
print(f"N_A                       = {N_A_cm3:.3e} cm^-3")
print(f"N_D                       = {N_D_cm3:.3e} cm^-3")
print(f"V_bi                      = {V_bi:.5f} V")
print(f"x_p                       = {x_p0 * 1e6:.5f} um")
print(f"x_n                       = {x_n0 * 1e6:.5f} um")
print(f"W                         = {W0 * 1e6:.5f} um")
print(f"E_max                     = {E_max0 / 100:.5e} V/cm")
print(f"A                         = {A_cm2:.3e} cm^2")
print(f"D_n                       = {D_n:.3e} cm^2/s")
print(f"D_p                       = {D_p:.3e} cm^2/s")
print(f"L_n                       = {L_n:.3e} cm")
print(f"L_p                       = {L_p:.3e} cm")
print(f"I_S                       = {I_S:.3e} A")
print(f"tau_SRH                   = {tau_SRH:.3e} s")
print(f"R_s                       = {R_s:.3e} ohm")
print(f"V_BR                      = {V_BR:.3e} V")
print("============================================================")
