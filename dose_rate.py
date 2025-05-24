#!/usr/bin/env python3
# ---------------------------------------------------------------------------
# dose_rate.py  (v1.2, 2025-05-17)
#
# First-order dose-rate model for PHYS 250 “Fermi Probe #5” group project.
# Given cruise β, hull areal density σ (g cm⁻²), and magnetic-field strength B,
# the script outputs:
#   • daily effective dose rate   [µSv day⁻¹]
#   • cumulative dose curve       [Sv] over an arbitrary mission length
#   • sensitivity grid            years to 1 Sv for a range of σ and B
#
# Copyright © 2025  Team 4, PHYS 250, Embry-Riddle Aeronautical University
# MIT License
# ---------------------------------------------------------------------------
# Version history
#   1.0   rough model + cumulative curve
#   1.1   added magnetic-shield attenuation
#   1.2   sensitivity sweep + CLI
# ---------------------------------------------------------------------------

import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ----------------------------- constants -----------------------------------
C_LIGHT = 2.998e8            # m s⁻¹
SV_PER_MSV = 1e-3
USV_PER_MSV = 1e3
DAYS_PER_YEAR = 365.25
# Dose-rate coefficients (µSv day⁻¹) — tuned to match HZETRN tables
K_UNSHIELDED = 1_095.0        # 1.095 mSv day⁻¹
K_HULL_REF   = 500.0          # at σ = 5 g cm⁻²
K_BFIELD_REF =  100.0         # at B = 3 T  (hull + magnet)

# Scaling exponents (empirical, from table curve-fits)
ALPHA_SIGMA  = 0.85           # dose ∝ σ^-α
ALPHA_B      = 1.4            # dose ∝ B^-α

# NASA career limit
SV_LIMIT = 1.0                # Sv


# ----------------------------- helper functions ----------------------------
def gamma_from_beta(beta: float) -> float:
    """Return Lorentz γ given β = v / c."""
    return 1.0 / np.sqrt(1.0 - beta**2)


def dose_rate(beta: float, sigma: float, B: float) -> float:
    """
    Return daily effective dose rate (µSv day⁻¹).

    Parameters
    ----------
    beta   : cruise speed as fraction of c
    sigma  : hull areal density (g cm⁻²)
    B      : magnetic-shield field strength (T)
    """
    γ = gamma_from_beta(beta)          # relativistic time factor

    # Passive hull attenuation (power-law fit)
    hull_factor = (sigma / 5.0) ** (-ALPHA_SIGMA)

    # Magnetic-field attenuation (power-law fit)
    if B > 0.0:
        mag_factor = (B / 3.0) ** (-ALPHA_B)
    else:
        mag_factor = 1.0

    # Combine reference dose with scaling factors
    daily_dose_usv = K_HULL_REF * hull_factor * mag_factor * γ
    return daily_dose_usv


def cumulative_curve(beta: float, sigma: float, B: float,
                     years: float = 35.0) -> pd.DataFrame:
    """Return a DataFrame with mission years and cumulative dose (Sv)."""
    daily = dose_rate(beta, sigma, B)        # µSv day⁻¹
    annual_sv = daily * DAYS_PER_YEAR * SV_PER_MSV * SV_PER_MSV
    t = np.linspace(0.0, years, int(years) + 1)
    dose = annual_sv * t
    return pd.DataFrame({'years': t, 'cum_sv': dose})


def years_to_limit(beta: float, sigma: float, B: float) -> float:
    """Return years until cumulative dose reaches 1 Sv."""
    daily = dose_rate(beta, sigma, B)        # µSv day⁻¹
    annual_sv = daily * DAYS_PER_YEAR * SV_PER_MSV * SV_PER_MSV
    if annual_sv == 0.0:
        return np.inf
    return SV_LIMIT / annual_sv


# ----------------------------- CLI & main ----------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="First order dose model for relativistic starship cruise")
    p.add_argument('--beta', type=float, default=0.9,
                   help='cruise speed as fraction of c (default 0.9)')
    p.add_argument('--sigma', type=float, default=5.0,
                   help='hull areal density in g cm⁻² (default 5)')
    p.add_argument('--bfield', type=float, default=3.0,
                   help='magnetic field strength in tesla (default 3)')
    p.add_argument('--plot', action='store_true',
                   help='generate cumulative-dose plot')
    p.add_argument('--sweep', action='store_true',
                   help='run sensitivity grid and save CSV')
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # 1. Report single-case dose rate
    d_usv = dose_rate(args.beta, args.sigma, args.bfield)
    yrs   = years_to_limit(args.beta, args.sigma, args.bfield)
    print(f"Daily effective dose  : {d_usv:8.2f} µSv day⁻¹")
    print(f"Years to 1 Sv career limit : {yrs:6.2f} yr")

    # 2. Optional cumulative-dose plot
    if args.plot:
        df = cumulative_curve(args.beta, args.sigma, args.bfield)
        plt.figure(figsize=(6, 4))
        plt.plot(df.years, df.cum_sv)
        plt.axhline(SV_LIMIT, color='red', ls='--')
        plt.xlabel('Mission duration (yr)')
        plt.ylabel('Cumulative dose (Sv)')
        plt.title('Cumulative Dose vs Time')
        plt.tight_layout()
        plt.savefig('cum_dose_single.png', dpi=300)
        print("Saved figure: cum_dose_single.png")

    # 3. Optional sensitivity sweep
    if args.sweep:
        sigma_range = np.arange(2.0, 10.1, 1.0)
        B_range = np.arange(2.0, 6.1, 0.5)
        rows = []
        for sig in sigma_range:
            for B in B_range:
                yrs = years_to_limit(args.beta, sig, B)
                rows.append({'sigma': sig, 'B': B, 'years': yrs})
        out = pd.DataFrame(rows)
        out.to_csv('sensitivity_grid.csv', index=False)
        print("Saved grid: sensitivity_grid.csv")


if __name__ == '__main__':
    main()
