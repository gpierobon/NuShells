---
layout: default
title: Overview
nav_order: 1
---


## Overview

A spherical shell N-body code for simulating collisionless neutrino dynamics with long-range Yukawa interactions mediated by a scalar field.  Shells evolve under free-streaming and long-range forces using a kick-drift-kick (KDK) leapfrog integrator.

[View it on GitHub][github repo]{: .btn} 

---

## Implemented Features

### Grid and ICs

- ✅ **Geometric radial grid** — shells placed on `geomspace(Rmin, Rmax)`
- ✅ **Reflecting boundary conditions at infinity** — shells softened at `soft` and bouncing at `Rmax`
- ✅ **Fermi-Dirac initial conditions** — momentum samples drawn from FD distribution, angular momentum assigned via random sampling
- ✅ **Generic perturbation profile** — Gaussian, top-hat, poly, exponential and compensated

### Evolution

- ✅ **KDK leapfrog integrator** — symplectic kick-drift-kick time stepping 
- ✅ **Adaptive timestep**: condition `dt` on acceleration magnitude 
- ✅ **Effective mass update** — consistent mass/energy at each potential update
- ✅ **Scale factor evolution** — matter-dominated Universe only
- ✅ **Enclosed mass tracking** — cumulative weighted sum for gravity

### Forces

- ✅ **Neutrino free-streaming force** — angular momentum term `ell^2 / (eps * r^3)`
- ✅ **Yukawa force solver** — inner and outer shells contribtions at each `R`
- ✅ **Gravity force solver**  — Newtonian gravity force using `_cumMass` and `beta`
- ✅ **Scalar field Poisson solver** — iterative solve for `phi(r)` at each timestep: `naive` and `anderson` methods
- ✅ **Phi interpolation** — interpolates `phi` from pre-drift positions as initial guess for iteration

### I/O

- ✅ **Text output** — shell state saved with `numpy.savetxt`
- ✅ **HDF5 output** — shell state saved with header metadata using `h5py`
- ✅ **Measurements** — shell state saved with linear or log-spacing in a(t)

### Utilities

- ✅ **Driver** — driver script to set params and run the simulation
- ✅ **Profiler** — wall-clock profiling decorator (`@timed`)
- ✅ **Logger** — info and debug verbosity levels
- ✅ **Radial density** — bins shells by weight into radial number density `n(r)`
- ✅ **Visualisation** — scripts to visualise circles or neutrino delta


## To do list

- **Convergence tests in the iteration**: Naive vs anderson check
- **Boundary term**: Add contribution from infinity for phi and Yukawa force
- **Alternative IC sampling**: Explore PSD-proportional radial sampling


[github repo]: https://github.com/gpierobon/NuShells.git
