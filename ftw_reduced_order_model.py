"""
Reduced-order 3-DOF Surge-Sway-Yaw dynamic model for floating treatment wetland (FTW)
laboratory wave-tank testing.

Solves the coupled planar equations of motion:
    (M + A) x_ddot + B x_dot + K x = F_exc(t) + F_drift

where:
    x = [surge x (m), sway y (m), yaw psi (rad)]^T
    Specimen: 29.0 cm regular hexagonal mat (circumscribed diameter D = 0.58 m)
    Wave forcing: Monochromatic regular waves (H = 0.04 m, a = 0.02 m, f = 1.0 Hz)
"""

import numpy as np
from scipy.integrate import quad

def run_model(config_name="Distributed-Modular", verbose=True):
    # ---------------------------------------------------------
    # 1. Specimen Geometry & Inertia
    # ---------------------------------------------------------
    s = 0.29              # side length of regular hexagon (m)
    D = 2.0 * s           # circumscribed diameter = 0.58 m
    W = np.sqrt(3.0) * s  # width across flats = 0.502 m
    A_wp = (3.0 * np.sqrt(3.0) / 2.0) * s**2  # planform area = 0.2185 m^2
    m = 1.25              # total operating mass in tank (kg)
    rho_w = 1000.0        # freshwater density (kg/m^3)
    d = m / (rho_w * A_wp)  # equilibrium static draft ≈ 0.0057 m (5.7 mm)

    # Mass moment of inertia about vertical centroidal axis
    # For a regular hexagon of uniform mass: I_zz = (5/12) * m * s^2
    I_zz = (5.0 / 12.0) * m * s**2  # 0.0438 kg*m^2
    R_g = np.sqrt(5.0 / 12.0) * s   # radius of gyration = 0.1872 m

    # Hydrodynamic added mass (potential flow for shallow floating disk/plate)
    # A_11 = C_a * rho_w * A_wp * d
    C_a_trans = 0.28
    A_11 = C_a_trans * rho_w * A_wp * d  # ~0.35 kg
    A_22 = A_11                          # symmetric in plane
    
    # Yaw added moment of inertia (units: kg*m^2)
    C_a_yaw = 0.25
    A_66 = C_a_yaw * rho_w * A_wp * d * (R_g**2)  # ~0.012 kg*m^2

    M_eff = np.array([m + A_11, m + A_22, I_zz + A_66])

    # ---------------------------------------------------------
    # 2. Wave Environment & Hydrodynamic Forcing
    # ---------------------------------------------------------
    f = 1.0               # wave frequency (Hz)
    omega = 2.0 * np.pi * f  # 6.283 rad/s
    T_wave = 1.0 / f      # 1.0 s
    H = 0.04              # wave height (m)
    a = H / 2.0           # wave amplitude = 0.02 m
    g = 9.81              # gravitational acceleration (m/s^2)

    # Wave reflection coefficient (perforated mat leading edge)
    R = 0.18
    # Second-order mean horizontal wave drift force (Maruo, 1960):
    # F_drift = 0.5 * rho_w * g * R^2 * a^2 * W
    F_drift = 0.5 * rho_w * g * (R**2) * (a**2) * W  # ~0.0319 N

    # First-order oscillatory wave excitation:
    # F_exc(t) = [F_x0 * cos(omega*t), F_y0 * sin(omega*t), M_psi0 * sin(omega*t)]^T
    F_x0 = 0.30           # longitudinal wave excitation force amplitude (N)
    F_y0 = 0.08           # transverse excitation from small tank reflections (N)
    M_psi0 = 0.012        # yaw excitation moment from wave crest encounter asymmetry (N*m)

    # ---------------------------------------------------------
    # 3. Mooring Restoring Stiffness & Damping by Configuration
    # ---------------------------------------------------------
    if config_name == "Distributed-Modular":
        # Multi-point compliant bottom tethers:
        # High, isotropic horizontal stiffness and large yaw rotational moment arm
        K_xx = 48.0       # N/m
        K_yy = 48.0       # N/m
        K_psi = 2.10      # N*m/rad
        B_xx = 5.80       # N*s/m
        B_yy = 5.80       # N*s/m
        B_psi = 0.18      # N*m*s/rad
    elif config_name == "Tension-Guide":
        # Spanning wire guide with sliding edge tethers:
        # Softer longitudinal surge compliance, lower rotational restraint
        K_xx = 18.0       # N/m
        K_yy = 28.0       # N/m
        K_psi = 0.65      # N*m/rad
        B_xx = 4.80       # N*s/m
        B_yy = 4.80       # N*s/m
        B_psi = 0.12      # N*m*s/rad
    else:
        raise ValueError(f"Unknown config: {config_name}")

    # ---------------------------------------------------------
    # 4. Steady-State Solution of Coupled Equations
    # ---------------------------------------------------------
    # Steady 1D surge drift offset from mean wave drift force:
    x_0 = F_drift / K_xx

    # First-order dynamic response amplitudes:
    # X_0 = F_x0 / sqrt( (K_xx - M_11 * omega^2)^2 + (B_xx * omega)^2 )
    denom_x = np.sqrt((K_xx - M_eff[0] * omega**2)**2 + (B_xx * omega)**2)
    X_0 = F_x0 / denom_x

    denom_y = np.sqrt((K_yy - M_eff[1] * omega**2)**2 + (B_yy * omega)**2)
    Y_0 = F_y0 / denom_y

    denom_psi = np.sqrt((K_psi - M_eff[2] * omega**2)**2 + (B_psi * omega)**2)
    Psi_0 = M_psi0 / denom_psi  # rad

    # ---------------------------------------------------------
    # 5. Radial Drift Integration & Statistical Metrics
    # ---------------------------------------------------------
    # Exact time-integral definition of mean radial drift:
    # r_mean = (1/T) * \int_0^T sqrt( x(t)^2 + y(t)^2 ) dt
    def instantaneous_r(t):
        xt = x_0 + X_0 * np.cos(omega * t)
        yt = Y_0 * np.sin(omega * t)
        return np.sqrt(xt**2 + yt**2)

    val_integral, _ = quad(instantaneous_r, 0.0, T_wave)
    mean_radial_drift_cm = (val_integral / T_wave) * 100.0

    # Approximate root-mean-square (RMS) radial excursion:
    # r_rms = sqrt( x_0^2 + 0.5*(X_0^2 + Y_0^2) )
    rms_radial_drift_cm = np.sqrt(x_0**2 + 0.5 * (X_0**2 + Y_0**2)) * 100.0

    # Peak excursion:
    t_fine = np.linspace(0.0, T_wave, 500)
    peak_radial_drift_cm = np.max([instantaneous_r(t) for t in t_fine]) * 100.0

    # RMS heading deviation:
    psi_rms_deg = (Psi_0 / np.sqrt(2.0)) * (180.0 / np.pi)

    if verbose:
        print(f"=== {config_name} ===")
        print(f"  Steady 1D surge offset (x_0): {x_0*100.0:.3f} cm")
        print(f"  First-order surge amplitude (X_0): {X_0*100.0:.3f} cm")
        print(f"  First-order sway amplitude (Y_0): {Y_0*100.0:.3f} cm")
        print(f"  Integrated Mean Radial Drift (r_mean): {mean_radial_drift_cm:.2f} cm")
        print(f"  RMS Radial Excursion (r_rms): {rms_radial_drift_cm:.2f} cm")
        print(f"  Peak Radial Excursion (r_peak): {peak_radial_drift_cm:.2f} cm")
        print(f"  RMS Heading Deviation (psi_rms): {psi_rms_deg:.2f} deg")

    return {
        "x_0_cm": x_0 * 100.0,
        "X_0_cm": X_0 * 100.0,
        "Y_0_cm": Y_0 * 100.0,
        "mean_radial_drift_cm": mean_radial_drift_cm,
        "rms_radial_drift_cm": rms_radial_drift_cm,
        "peak_radial_drift_cm": peak_radial_drift_cm,
        "psi_rms_deg": psi_rms_deg
    }

if __name__ == "__main__":
    print("----------------------------------------------------------------------")
    print("REDUCED-ORDER DYNAMIC MODEL VERIFICATION (Faltinsen 1990 / Newman 1977)")
    print("----------------------------------------------------------------------")
    res_dm = run_model("Distributed-Modular")
    print()
    res_tg = run_model("Tension-Guide")
