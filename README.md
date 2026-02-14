# ic_lattice

Monte Carlo lattice simulation for A-B molecule interactions on a 2D lattice.

## Description

`ic_lattice` is a Python package that simulates the behavior of two types of molecules (A and B) on a 2D square lattice using Monte Carlo methods. The simulation uses the Metropolis algorithm to sample configurations according to their Boltzmann probabilities.

## Features

- Efficient numba-accelerated Monte Carlo simulation
- Periodic boundary conditions
- Configurable interaction energies (A-A, A-B, B-B)
- Real-time monitoring of energy and neighbor statistics
- Both library and command-line interfaces

## Installation

### Using pixi

```bash
pixi add ic_lattice
```

Or add to your `pixi.toml`:

```toml
[dependencies]
ic_lattice = "*"
```

### Using pip

```bash
pip install ic_lattice
```

### From source

```bash
git clone https://github.com/yourusername/ic-lattice.git
cd ic-lattice
pip install -e .
```

## Usage

### Command-line interface

```bash
ic-lattice --N 50 --n_A 1250 --E_AA -1.0 --E_AB 0.0 --E_BB -1.0 --beta 1.0 --steps 10000 --print_interval 1000
```

Parameters:
- `--N`: Lattice size (creates N x N grid)
- `--n_A`: Number of A molecules
- `--E_AA`: A-A interaction energy
- `--E_AB`: A-B interaction energy
- `--E_BB`: B-B interaction energy
- `--beta`: Inverse temperature (1/kT)
- `--steps`: Number of Monte Carlo steps
- `--print_interval`: Print statistics every N steps (0 for no printing)

### Python API

```python
from ic_lattice import initialize_lattice, run_simulation

# Initialize lattice
N = 50
n_A = 1250
lattice = initialize_lattice(N, n_A)

# Set parameters
E_AA = -1.0
E_AB = 0.0
E_BB = -1.0
beta = 1.0
n_steps = 10000

# Run simulation
results = run_simulation(
    lattice, N, beta, E_AA, E_AB, E_BB,
    n_steps, print_interval=1000
)

# Access results
print(f"Final energy: {results['final_energy']}")
print(f"Acceptance rate: {results['acceptance_rate']}")
print(f"A-B fraction: {results['final_ab_fraction']}")
```

## Physics

The simulation models a canonical ensemble where:
- The lattice has N×N sites
- Each site contains either an A or B molecule
- Molecules can swap positions via Monte Carlo moves
- Energy depends on nearest-neighbor interactions with periodic boundary conditions

The acceptance probability for a swap follows the Metropolis criterion:
```
P(accept) = min(1, exp(-β * ΔE))
```

where β = 1/(kT) is the inverse temperature and ΔE is the energy change.

## Development

### Setting up development environment with pixi

```bash
pixi install
pixi run pytest
```

### Running tests

```bash
pytest tests/
```

### Code formatting

```bash
black src/
ruff check src/
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citation

If you use this package in your research, please cite:

```bibtex
@software{ic_lattice,
  author = {Your Name},
  title = {ic_lattice: Monte Carlo lattice simulation},
  year = {2024},
  url = {https://github.com/yourusername/ic-lattice}
}
```
