"""Water-chemistry conversions used by the generator and the data adapter.

Purpose
    Three physical formulas the pipeline needs in more than one place:
      * do_saturation(): how much oxygen water can hold at a given temperature
        and salinity (Benson & Krause 1984, the equation tabulated in APHA
        Standard Methods 4500-O). Used by 01 for the DO mass balance and by 00
        to turn "% saturation" readings into mg/L.
      * salinity_from_conductivity(): Practical Salinity Scale 1978 (PSS-78,
        UNESCO 1983 / Fofonoff & Millard), with the Hill et al. (1986)
        low-salinity extension, so files that log conductivity can be used.
      * fahrenheit_to_celsius().

Paper link
    None of the three review papers; these are standard-method formulas. They
    are here so unit handling is tested once instead of re-typed per script.

Outputs
    Nothing when imported. `python water_chem.py` runs the checks: DO
    saturation against the APHA 4500-O table at 0 ppt, PSS-78 against its
    defining point (42.914 mS/cm at 15 °C is salinity 35), plus sanity checks.
"""
from __future__ import annotations

import numpy as np

# Conductivity of standard seawater (S = 35, 15 °C, surface pressure), mS/cm.
C_35_15_0 = 42.914


def fahrenheit_to_celsius(temp_f):
    return (np.asarray(temp_f, dtype=float) - 32.0) * 5.0 / 9.0


def do_saturation(temp_c, salinity_ppt=0.0):
    """Oxygen solubility in mg/L at 1 atm, moist air (Benson & Krause 1984).

    ln C* = fresh-water polynomial in 1/T  -  S * (salinity correction),
    T in kelvin. Warmer or saltier water holds less oxygen.
    """
    t = np.asarray(temp_c, dtype=float) + 273.15
    s = np.asarray(salinity_ppt, dtype=float)
    ln_c = (-139.34411 + 1.575701e5 / t - 6.642308e7 / t**2
            + 1.243800e10 / t**3 - 8.621949e11 / t**4
            - s * (1.7674e-2 - 1.0754e1 / t + 2.1407e3 / t**2))
    return np.exp(ln_c)


def do_percent_to_mgl(do_pct, temp_c, salinity_ppt=0.0):
    """Convert DO in % air saturation to mg/L."""
    return np.asarray(do_pct, dtype=float) / 100.0 * do_saturation(temp_c, salinity_ppt)


def do_mgl_to_percent(do_mgl, temp_c, salinity_ppt=0.0):
    return np.asarray(do_mgl, dtype=float) / do_saturation(temp_c, salinity_ppt) * 100.0


# PSS-78 coefficients (UNESCO Technical Papers in Marine Science 44).
_A = (0.0080, -0.1692, 25.3851, 14.0941, -7.0261, 2.7081)
_B = (0.0005, -0.0056, -0.0066, -0.0375, 0.0636, -0.0144)
_C = (0.6766097, 2.00564e-2, 1.104259e-4, -6.9698e-7, 1.0031e-9)
_K = 0.0162


def _rt35(temp_c):
    """Temperature dependence of standard seawater's conductivity (1.0 at 15 °C)."""
    t = np.asarray(temp_c, dtype=float)
    return _C[0] + _C[1] * t + _C[2] * t**2 + _C[3] * t**3 + _C[4] * t**4


def salinity_from_conductivity(cond_ms_cm, temp_c):
    """Practical salinity from in-situ conductivity (mS/cm) and temperature (°C).

    PSS-78 at surface pressure (ponds are shallow, so no pressure term), valid
    for salinity 2-42; below 2 the Hill et al. (1986) correction is applied so
    the result falls smoothly to 0 for fresh water.
    """
    t = np.asarray(temp_c, dtype=float)
    ratio = np.clip(np.asarray(cond_ms_cm, dtype=float), 0.0, None) / C_35_15_0
    rt = ratio / _rt35(t)                       # conductivity ratio at temperature t
    root = np.sqrt(rt)
    f_t = (t - 15.0) / (1.0 + _K * (t - 15.0))
    sal = (sum(a * root**i for i, a in enumerate(_A))
           + f_t * sum(b * root**i for i, b in enumerate(_B)))
    # Hill et al. (1986): removes the a0/b0 offsets that PSS-78 leaves at Rt -> 0.
    x, y = 400.0 * rt, 100.0 * rt
    sal = sal - _A[0] / (1.0 + 1.5 * x + x**2) - _B[0] * f_t / (1.0 + np.sqrt(y) + y + y**1.5)
    return np.clip(sal, 0.0, None)


def salinity_from_specific_conductance(ec25_ms_cm):
    """Salinity from specific conductance, i.e. conductivity the logger has
    already temperature-compensated to 25 °C (what most handheld meters show)."""
    ec = np.asarray(ec25_ms_cm, dtype=float)
    return salinity_from_conductivity(ec, np.full_like(ec, 25.0))


def conductivity_from_salinity(salinity_ppt, temp_c):
    """Inverse of salinity_from_conductivity by bisection (PSS-78 has no closed
    inverse). Only used to build the messy round-trip test file for 00."""
    sal = np.asarray(salinity_ppt, dtype=float)
    t = np.broadcast_to(np.asarray(temp_c, dtype=float), sal.shape)
    lo, hi = np.zeros_like(sal), np.full_like(sal, 120.0)
    for _ in range(60):
        mid = (lo + hi) / 2.0
        too_low = salinity_from_conductivity(mid, t) < sal
        lo, hi = np.where(too_low, mid, lo), np.where(too_low, hi, mid)
    return (lo + hi) / 2.0


def _run_checks() -> None:
    from config import section

    section("Check 1: DO saturation at 0 ppt vs APHA 4500-O table")
    apha = {0: 14.621, 5: 12.770, 10: 11.288, 15: 10.084, 20: 9.092,
            25: 8.263, 30: 7.559, 35: 6.950}        # mg/L, chlorinity 0, 1 atm
    print("  T (°C)   computed    APHA    diff")
    for temp, ref in apha.items():
        got = float(do_saturation(temp, 0.0))
        print(f"  {temp:>5}   {got:8.3f}  {ref:6.3f}  {got - ref:+.4f}")
        assert abs(got - ref) < 0.002, f"DO saturation at {temp} °C: {got:.4f} vs APHA {ref}"
    print("  -> all within 0.002 mg/L of the table")

    section("Check 2: salinity lowers DO saturation")
    for temp in (25, 30):
        row = [float(do_saturation(temp, s)) for s in (0, 10, 20, 35)]
        print(f"  {temp} °C at 0/10/20/35 ppt: " + "  ".join(f"{v:.3f}" for v in row))
        assert all(a > b for a, b in zip(row, row[1:])), "saturation must fall as salinity rises"
    back = float(do_mgl_to_percent(do_percent_to_mgl(87.5, 29.0, 18.0), 29.0, 18.0))
    print(f"  87.5 % -> mg/L -> % at 29 °C, 18 ppt: {back:.6f} %")
    assert abs(back - 87.5) < 1e-9

    section("Check 3: PSS-78 conductivity -> salinity")
    s_def = float(salinity_from_conductivity(C_35_15_0, 15.0))
    print(f"  {C_35_15_0} mS/cm at 15 °C -> {s_def:.3f} (definition point: 35.000)")
    assert abs(s_def - 35.0) < 5e-4
    s_zero = float(salinity_from_conductivity(0.0, 28.0))
    print(f"  0 mS/cm at 28 °C       -> {s_zero:.4f} (fresh water must give 0)")
    assert abs(s_zero) < 1e-9
    temps = np.array([15.0, 22.0, 28.0, 33.0])
    for target in (1.0, 5.0, 15.0, 25.0, 35.0):
        cond = conductivity_from_salinity(np.full(4, target), temps)
        back = salinity_from_conductivity(cond, temps)
        print(f"  S = {target:>4} ppt -> {cond[2]:7.3f} mS/cm at 28 °C -> back to "
              f"{back[2]:.4f} (max round-trip error {np.abs(back - target).max():.1e})")
        assert np.abs(back - target).max() < 1e-6
        assert np.all(np.diff(cond) > 0), "same salinity must conduct more when warmer"
    s25 = float(salinity_from_specific_conductance(conductivity_from_salinity(20.0, 25.0)))
    print(f"  specific-conductance path (25 °C) round-trips 20 ppt -> {s25:.4f}")
    assert abs(s25 - 20.0) < 1e-6

    section("Check 4: temperature units")
    print(f"  86 °F -> {float(fahrenheit_to_celsius(86.0)):.2f} °C")
    assert abs(float(fahrenheit_to_celsius(86.0)) - 30.0) < 1e-12
    print("\nAll water_chem checks passed.")


if __name__ == "__main__":
    _run_checks()
