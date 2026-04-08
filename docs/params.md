---
layout: default
title: Parameters
nav_order: 2
---


## Parameter Reference

### Physical inputs

| Symbol | Code name | Unit | Description |
|--------|-----------|------|-------------|
| `Nshells` | `self.Nshells` | — | Number of shells |
| `g` | `self.g` | — | Yukawa coupling constant |
| `m_phi` | `self.m_phi` | eV | Mediator (scalar field) mass |
| `m_nu` | `self.m_nu` | eV | Neutrino mass |
| `T_nu` | `self.T_nu` | eV | Present-day neutrino temperature |
| `H0` | `self.H0` | eV | Hubble constant |
| `r_phi` | `self.r_phi` | eV⁻¹ | Yukawa range `= 1/m_phi` |

### Dimensionless ratios

| Symbol | Code name | Definition | Description |
|--------|-----------|------------|-------------|
| `alpha` | `self.alpha` | $$g^2 T_{\nu,0}^2 / m_\phi^2$$ | Yukawa force strength |
| `beta` | `self.beta` | $$T_{\nu, 0}^4 / (m_\phi^2 m_{\rm pl}^2)$$ | Gravity force strength |
| `eta` | `self.eta` | $$\beta/\alpha$$ | Gravity-to-Yukawa ratio |
| `m0` | `self.m0` | $$m_\nu / T_{\nu, 0}$$ | Dimensionless bare neutrino mass |
| `m_phi_hat` | `self.m_phi_hat` | $$m_\phi / H_0$$ | Dimensionless mediator mass; sets scale factor evolution |

### Scale factors & time

| Symbol | Code name | Definition | Description |
|--------|-----------|------------|-------------|
| `a` | `self.a` | — | Current scale factor |
| `a_ini` | `self.a_ini` | `kappa * a_NR` | Initial scale factor |
| `a_NR` | — | `1 / m0` | Scale factor at NR transition |
| `a_end` | `self.a_end` | `kappa2 / Rmin` | Final scale factor; Yukawa range shrinks below `Rmin` |
| `dt` | `self.dt` | — | Current timestep (adaptive, Courant condition) |
| `dt_frac` | `self.dt_frac` | — | Courant safety factor multiplying `sqrt(soft/F)` |
| `kappa` | — | — | Sets `a_ini` as fraction of `a_NR` (default 0.75) |
| `kappa2` | — | — | Sets `a_end` as fraction of `1/Rmin` (default 0.75) |

### Grid & force

| Symbol | Code name | Unit | Description |
|--------|-----------|------|-------------|
| `Rmin` | `self.Rmin` | `1/m_phi` | Inner radial boundary `= 0.01 * lambda_phi(a_ini)` |
| `Rmax` | `self.Rmax` | `1/m_phi` | Outer radial boundary `= 50 * max(lambda_phi, lambda_FS)` |
| `soft` | `self.soft` | `1/m_phi` | Softening length; used as reflecting boundary threshold |
| `lambda_phi` | — | `1/m_phi` | Yukawa range `= 1/a` at a given scale factor |
| `lambda_FS` | — | `1/m_phi` | Free-streaming scale; sets minimum resolved clustering scale |

### Per-shell fields

| Symbol | Code name | Definition | Description |
|--------|-----------|------------|-------------|
| `R` | `self.R` | `1/m_phi` | Dimensionless radial position `= r * m_phi` |
| `q` | `self.q` | — | Radial momentum `= q_r / T_nu` |
| `ell` | `self.ell` | — | Angular momentum `= r * q_perp / T_nu * H0` |
| `w` | `self.w` | — | Phase-space weight (dimensionless) |
| `phi` | `self.phi` | — | Scalar field `= hat_phi / T_nu` |
| `m` | `self.m` | — | Effective mass `= m0 + phi` |
| `eps` | `self.eps` | — | Energy `= sqrt(q^2 + ell^2/R^2 + a^2 m^2)` |
| `F_fs` | `self.F_fs` | — | Free-streaming acceleration `= ell^2 / (eps * R^3)` |
| `F_lr` | `self.F_lr` | — | Yukawa (long-range) acceleration |
| `F_g` | `self.F_g` | — | Gravitational acceleration |

### Initial condition parameters

| Symbol | Code name | Description |
|--------|-----------|-------------|
| `ic_type` | `self.ic_type` | Initial perturbation profile |
| `Psi0` | — | Amplitude of initial perturbation |
| `R0` | `self.R0` | Scale of initial perturbation (default: `lambda_phi` at `a_ini`) |
| `w_min` | — | Floor on shell weights after normalisation (default: `1e-3`) |

### Solver parameters

| Symbol | Code name | Description |
|--------|-----------|-------------|
| `iter_m` | `self.iter_m` | Method for `solvePhi`: `'anderson'` or `'naive'` |
| `iter_tol` | `self.iter_tol` | Convergence tolerance for `solvePhi` |
