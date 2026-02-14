# ic_lattice

Monte Carlo lattice simulation for A-B molecule interactions on a 2D lattice.

## Description

`ic_lattice` is a Python package that simulatesa regular solution of two types of molecules (A and B) on a 2D square lattice with periodic boundary conditions using Monte Carlo methods. The simulation uses the Metropolis algorithm to sample configurations according to their Boltzmann probabilities.

## Features

- Efficient numba-accelerated Monte Carlo simulation
- Periodic boundary conditions
- Configurable interaction energies (A-A, A-B, B-B)
- Real-time monitoring of energy and neighbor statistics

## Installation

### Using pixi

```bash
pixi add --pypi ic_lattice
```

Or add to your `pixi.toml`:

```toml
[pypi-dependencies]
ic_lattice = "*"
```

### Using pip

```bash
pip install ic_lattice
```

### From source

```bash
git clone https://github.com/justinbois/ic-lattice.git
cd ic-lattice
pip install -e .
```


## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
