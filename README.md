# Heston Monte Carlo Simulation Suite

A C++17 implementation of Euler-discretized and Broadie–Kaya exact Monte Carlo methods for pricing European call options under the Heston stochastic volatility model.

---

## Overview

This suite replicates and extends the simulation tables from:

> Broadie, M. and Kaya, Ö. (2006). *Exact Simulation of Stochastic Volatility and Other Affine Jump Diffusion Processes*. Operations Research, 54(2), 217–231.

Three programs are included:

| Program | Description |
|---|---|
| `heston_euler.cpp` | Euler-discretized Heston Monte Carlo pricing table |
| `heston_exact.cpp` | Broadie–Kaya exact Heston Monte Carlo pricing table |
| `heston_euler_table3.cpp` | Table 3-style Euler bias comparison across three time-step designs |

---

## Requirements

- C++17 compiler (e.g. `clang++` or `g++`)
- Standard library only — no external dependencies

---

## Compilation

```bash
clang++ -std=c++17 heston_euler.cpp         -o heston_euler
clang++ -std=c++17 heston_exact.cpp         -o heston_exact
clang++ -std=c++17 heston_euler_table3.cpp  -o heston_table3
```

Output CSV files are written to a `Tables/` subdirectory, which is created automatically.

---

## Usage

All three programs can be run with default parameters (reproducing the Broadie–Kaya baseline experiment) or with fully custom parameters supplied via the command line.

### Program 1 — `heston_euler`

Runs Euler-discretized Heston Monte Carlo pricing for a European call option.

**Default run (Broadie–Kaya Table 1 parameters):**
```bash
./heston_euler
```

**Custom parameters and experiment grid:**
```bash
./heston_euler S_0 K V_0 kappa theta sigma rho r T C_true output_filename seed M_vals time_steps_vals
```

| Argument | Description |
|---|---|
| `S_0` | Initial stock price |
| `K` | Strike price |
| `V_0` | Initial variance |
| `kappa` | Speed of variance mean reversion |
| `theta` | Long-run variance |
| `sigma` | Volatility of variance (vol-of-vol) |
| `rho` | Correlation between stock and variance shocks |
| `r` | Risk-free rate |
| `T` | Maturity in years |
| `C_true` | Benchmark call price (used for bias and RMSE) |
| `output_filename` | Name of the output CSV file |
| `seed` | Random seed |
| `M_vals` | Comma-separated Monte Carlo path counts (no spaces) |
| `time_steps_vals` | Comma-separated Euler time-step counts (no spaces) |

`M_vals` and `time_steps_vals` must have the same number of entries; the *i*-th path count is paired with the *i*-th time-step count.

**Example:**
```bash
./heston_euler 120 115 0.04 3.50 0.06 0.45 -0.40 0.025 2.0 18.50 euler_custom.csv 12345 5000,20000,80000 100,250,500
```

Output: `Tables/euler_custom.csv`

---

### Program 2 — `heston_exact`

Runs the Broadie–Kaya exact simulation method. No Euler time-step grid is required.

**Default run:**
```bash
./heston_exact
```

**Custom parameters and experiment grid:**
```bash
./heston_exact S K V0 kappa theta sigma rho r T benchmark_price output_filename seed M_vals
```

Arguments are the same as `heston_euler`, minus `time_steps_vals`. The optional final argument accepts either a comma-separated `M_vals` list or the keyword `test` to run a small diagnostic grid (`100,500,1000` paths).

**Example:**
```bash
./heston_exact 120 115 0.04 3.50 0.06 0.45 -0.40 0.025 2.0 18.50 exact_custom.csv 98765 5000,20000,80000
```

Output: `Tables/exact_custom.csv`

---

### Program 3 — `heston_euler_table3`

Constructs a Table 3-style Euler bias comparison. For each supplied *N*, the program automatically runs three time-step designs: `0.1√N`, `√N`, and `10√N`.

**Default run:**
```bash
./heston_table3
```

**Custom parameters and N grid:**
```bash
./heston_table3 scheme_name S_0 K V_0 kappa theta sigma rho r T C_true output_filename seed N_vals
```

`scheme_name` is a descriptive label for the parameter set; use underscores instead of spaces.

**Example:**
```bash
./heston_table3 Custom_Scheme 120 115 0.04 3.50 0.06 0.45 -0.40 0.025 2.0 18.50 table3_custom.csv 98765 5000,20000,80000
```

Output: `Tables/table3_custom.csv`

---

## Output Format

Each program writes a CSV file to the `Tables/` directory. Columns vary by program but include some or all of:

| Column | Description |
|---|---|
| `M` | Number of Monte Carlo paths |
| `time_steps` | Number of Euler time steps (Euler programs only) |
| `price` | Estimated discounted European call price |
| `bias` | `price − C_true` |
| `std_error` | Monte Carlo standard error |
| `rmse` | `√(bias² + std_error²)` |
| `seconds` | Wall-clock runtime for that row |

---

## Default Parameters (Broadie–Kaya Table 1)

| Parameter | Value |
|---|---|
| S₀ | 100.0 |
| K | 100.0 |
| V₀ | 0.010201 |
| κ | 6.21 |
| θ | 0.019 |
| σ | 0.61 |
| ρ | −0.70 |
| r | 0.0319 |
| T | 1.0 |
| C_true | 6.8061 |

---

## License

Copyright (c) 2026 Your Name. All rights reserved.

Permission is granted to use this code for personal, educational, and research purposes only.
Reproduction, modification, and commercial use are strictly prohibited without the express written permission of the author.

See [LICENSE](LICENSE) for full terms.
