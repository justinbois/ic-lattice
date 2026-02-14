"""Core simulation functions for the ic_lattice package."""
import numpy as np
import numba


def initialize_lattice(lattice_dim, n_A):
    """
    Initialize the lattice with A and B molecules.

    Parameters:
    -----------
    lattice_dim : int
        Lattice dimension (lattice_dim x lattice_dim grid)
    n_A : int
        Number of type A molecules

    Returns:
    --------
    output : 2D Numpy array
        lattice_dim x lattice_dim lattice (1 for A, 0 for B)
    """
    lattice = np.zeros((lattice_dim, lattice_dim), dtype=int)
    positions = np.random.choice(lattice_dim * lattice_dim, n_A, replace=False)
    lattice.flat[positions] = 1
    return lattice


@numba.njit
def _get_neighbors(lattice_dim, i, j):
    """
    Get indices of four nearest neighbors (periodic boundary conditions).

    Parameters:
    -----------
    lattice_dim : int
        Lattice dimension
    i, j : int
        Site coordinates

    Returns:
    --------
    list : List of (i, j) tuples for the four neighbors
    """
    neighbors = [
        ((i - 1) % lattice_dim, j),  # top
        ((i + 1) % lattice_dim, j),  # bottom
        (i, (j - 1) % lattice_dim),  # left
        (i, (j + 1) % lattice_dim)   # right
    ]
    return neighbors


@numba.njit
def _get_site_energy(lattice, lattice_dim, i, j, E_AA, E_BB, E_AB):
    """
    Calculate interaction energy for a single site with its neighbors.

    Parameters:
    -----------
    lattice : np.ndarray
        The lattice_dim x lattice_dim lattice
    lattice_dim : int
        Lattice dimension
    i, j : int
        Site coordinates
    E_AA, E_BB, E_AB : float
        Interaction energies

    Returns:
    --------
    float : Total interaction energy for this site
    """
    site_type = lattice[i, j]
    energy = 0.0

    for ni, nj in _get_neighbors(lattice_dim, i, j):
        neighbor_type = lattice[ni, nj]

        if site_type == 1 and neighbor_type == 1:    # A-A
            energy += E_AA
        elif site_type == 0 and neighbor_type == 0:  # B-B
            energy += E_BB
        else:  # A-B
            energy += E_AB

    return energy


@numba.njit
def count_neighbor_pairs(lattice, lattice_dim):
    """
    Count the number of A-A, B-B, and A-B nearest neighbor pairs.

    Parameters:
    -----------
    lattice : np.ndarray
        The lattice_dim x lattice_dim lattice
    lattice_dim : int
        Lattice dimension

    Returns:
    --------
    tuple : (n_AA, n_BB, n_AB) number of each type of neighbor pair
    """
    n_AA = 0
    n_BB = 0
    n_AB = 0

    for i in range(lattice_dim):
        for j in range(lattice_dim):
            site_type = lattice[i, j]

            for ni, nj in _get_neighbors(lattice_dim, i, j):
                neighbor_type = lattice[ni, nj]

                if site_type == 1 and neighbor_type == 1:  # A-A
                    n_AA += 1
                elif site_type == 0 and neighbor_type == 0:  # B-B
                    n_BB += 1
                else:  # A-B
                    n_AB += 1

    # Divide by 2 since each bond is counted twice
    return n_AA // 2, n_BB // 2, n_AB // 2


@numba.njit
def calculate_total_energy(lattice, lattice_dim, E_AA, E_BB, E_AB):
    """
    Calculate total energy of the lattice.

    Parameters:
    -----------
    lattice : np.ndarray
        The lattice_dim x lattice_dim lattice
    lattice_dim : int
        Lattice dimension
    E_AA, E_BB, E_AB : float
        Interaction energies

    Returns:
    --------
    float : Total energy of the system
    """
    total_energy = 0.0
    for i in range(lattice_dim):
        for j in range(lattice_dim):
            total_energy += _get_site_energy(lattice, lattice_dim, i, j, E_AA, E_BB, E_AB)

    # Divide by 2 since each bond is counted twice
    return total_energy / 2.0


@numba.njit
def _swap_delta_pairs(lattice, lattice_dim, pos_A, pos_B):
    """
    Calculate the change in neighbor pair counts if pos_A (A site) and
    pos_B (B site) are swapped. Must be called before the swap is applied.

    Parameters:
    -----------
    lattice : np.ndarray
        The lattice_dim x lattice_dim lattice (unmodified)
    lattice_dim : int
        Lattice dimension
    pos_A : array-like (i, j)
        Position of the A molecule
    pos_B : array-like (i, j)
        Position of the B molecule

    Returns:
    --------
    tuple : (d_AA, d_BB, d_AB) changes in pair counts
    """
    i_A, j_A = pos_A
    i_B, j_B = pos_B

    d_AA = 0
    d_BB = 0
    d_AB = 0

    # Neighbors of the A site: A→B after swap.
    # Skip the shared bond with pos_B (A-B → B-A, still AB, net zero).
    for ni, nj in _get_neighbors(lattice_dim, i_A, j_A):
        if ni == i_B and nj == j_B:
            continue

        t = lattice[ni, nj]
        if t == 1:   # A-A → B-A: lose AA, gain AB
            d_AA -= 1
            d_AB += 1
        else:        # A-B → B-B: lose AB, gain BB
            d_AB -= 1
            d_BB += 1

    # Neighbors of the B site: B→A after swap.
    for ni, nj in _get_neighbors(lattice_dim, i_B, j_B):
        if ni == i_A and nj == j_A:
            continue

        t = lattice[ni, nj]
        if t == 1:   # B-A → A-A: lose AB, gain AA
            d_AB -= 1
            d_AA += 1
        else:        # B-B → A-B: lose BB, gain AB
            d_BB -= 1
            d_AB += 1

    return d_AA, d_BB, d_AB


@numba.njit
def _swap_delta_E(lattice, lattice_dim, pos_A, pos_B, E_AA, E_BB, E_AB):
    """
    Calculate energy change if molecules at pos_A and pos_B are swapped.

    Parameters:
    -----------
    lattice : np.ndarray
        The lattice_dim x lattice_dim lattice
    lattice_dim : int
        Lattice dimension
    pos_A : tuple (i, j)
        Position of A molecule
    pos_B : tuple (i, j)
        Position of B molecule
    E_AA, E_BB, E_AB : float
        Interaction energies

    Returns:
    --------
    float : Change in energy (E_new - E_old)
    """
    i_A, j_A = pos_A
    i_B, j_B = pos_B

    # Calculate current energy contribution from both sites
    E_old = (
        _get_site_energy(lattice, lattice_dim, i_A, j_A, E_AA, E_BB, E_AB)
        + _get_site_energy(lattice, lattice_dim, i_B, j_B, E_AA, E_BB, E_AB)
    )

    # If sites are neighbors, we need to adjust for double counting
    are_neighbors = (i_B, j_B) in _get_neighbors(lattice_dim, i_A, j_A)
    if are_neighbors:
        # Subtract the interaction between the two sites (counted in both)
        E_old -= E_AB

    # Perform temporary swap
    lattice[i_A, j_A] = 0
    lattice[i_B, j_B] = 1

    # Calculate new energy contribution
    E_new = (
        _get_site_energy(lattice, lattice_dim, i_A, j_A, E_AA, E_BB, E_AB)
        + _get_site_energy(lattice, lattice_dim, i_B, j_B, E_AA, E_BB, E_AB)
    )

    if are_neighbors:
        # Subtract the interaction between the two sites (counted in both)
        E_new -= E_AB

    # Swap back
    lattice[i_A, j_A] = 1
    lattice[i_B, j_B] = 0

    return E_new - E_old


@numba.njit
def _mc_step(lattice, flat_A, flat_B, lattice_dim, nA, nB, E_AA, E_BB, E_AB):
    """
    Perform one Monte Carlo step.

    Parameters:
    -----------
    lattice : 2D array
        The lattice_dim x lattice_dim lattice with ones and zeros denoting A and B
        molecules, respectively.
    flat_A : 1D Numpy array
        Flat indices of A molecule positions in the lattice.
        Modified in place if the swap is accepted.
    flat_B : 1D Numpy array
        Flat indices of B molecule positions in the lattice.
        Modified in place if the swap is accepted.
    lattice_dim : int
        Lattice dimension.
    nA : int
        Number of A molecules on the lattice.
    nB : int
        Number of B molecules on the lattice.
    E_AA : float
        A-A interaction energy in units of kB T.
    E_BB : float
        B-B interaction energy in units of kB T.
    E_AB : float
        A-B interaction energy in units of kB T.

    Returns:
    --------
    tuple : (accept, delta_E, d_AA, d_BB, d_AB)
    """
    # Pick an A and B positions
    flat_ind_A = np.random.randint(nA)
    flat_ind_B = np.random.randint(nB)
    k_A = flat_A[flat_ind_A]
    k_B = flat_B[flat_ind_B]
    pos_A = (k_A // lattice_dim, k_A % lattice_dim)
    pos_B = (k_B // lattice_dim, k_B % lattice_dim)

    # Calculate energy change and pair count changes before the swap
    delta_E = _swap_delta_E(lattice, lattice_dim, pos_A, pos_B, E_AA, E_BB, E_AB)
    d_AA, d_BB, d_AB = _swap_delta_pairs(lattice, lattice_dim, pos_A, pos_B)

    # Metropolis criterion
    if delta_E < 0:
        accept = True
    else:
        accept = np.random.random() < np.exp(-delta_E)

    # Perform swap if accepted
    if accept:
        lattice[pos_A[0], pos_A[1]] = 0
        lattice[pos_B[0], pos_B[1]] = 1
        flat_A[flat_ind_A] = k_B
        flat_B[flat_ind_B] = k_A
    else:
        delta_E = 0
        d_AA = 0
        d_BB = 0
        d_AB = 0

    return accept, delta_E, d_AA, d_BB, d_AB


@numba.njit
def _run_mcmc_steps(lattice, flat_A, flat_B, lattice_dim, nA, nB, E, n_AA, n_BB, n_AB, E_AA, E_BB, E_AB, n_steps):
    """
    Run Monte Carlo steps without saving.

    Parameters:
    -----------
    lattice : 2D Numpy array
        The lattice_dim x lattice_dim lattice (modified in place)
    flat_A : 1D Numpy array
        Array of indices in flattened lattice that are occupied with A molecules
    flat_B : 1D Numpy array
        Array of indices in flattened lattice that are occupied with B molecules
    lattice_dim : int
        Lattice dimension
    nA : int
        Number of A molecules.
    nB : int
        Number of B molecules.
    E : float
        Total lattice energy
    n_AA, n_BB, n_AB : int
        Current neighbor pair counts.
    E_AA : float
        A-A interaction energy in units of kB T.
    E_BB : float
        B-B interaction energy in units of kB T.
    E_AB : float
        A-B interaction energy in units of kB T.
    n_steps : int
        Number of Monte Carlo steps to perform

    Returns:
    --------
    tuple : (acceptance_count, E, n_AA, n_BB, n_AB)
    """
    acceptance_count = 0

    for _ in range(n_steps):
        accepted, delta_E, d_AA, d_BB, d_AB = _mc_step(lattice, flat_A, flat_B, lattice_dim, nA, nB, E_AA, E_BB, E_AB)
        E += delta_E
        n_AA += d_AA
        n_BB += d_BB
        n_AB += d_AB

        if accepted:
            acceptance_count += 1

    return acceptance_count, E, n_AA, n_BB, n_AB


def mc_sample(
        lattice=None,
        E_AA=0.0,
        E_BB=0.0,
        E_AB=0.0,
        n_steps=1_000_000,
        save_interval=None,
        save_intermediate_lattices=False,
        quiet=False,
    ):
    """
    Run the Monte Carlo simulation.

    Parameters:
    -----------
    lattice : 2D Numpy array
        A square array with ones and zeros. 1 means a site is occupied
        by an A molecules and 0 means a site is occupied by a B
        molecule. If None, a 100x100 site lattice with 50% A and 50%
        B is initialized with A and B randomized on the sites.
    E_AA : float, default 0.0
        A-A interaction energy in units of kB T.
    E_BB : float, default 0.0
        B-B interaction energy in units of kB T.
    E_AB : float, default 0.0
        A-B interaction energy in units of kB T.
    n_steps : int
        Number of Monte Carlo steps to perform.
    save_interval : int, optional
        Save acceptance rate, A-B neighbor fraction, and lattice energy
        every save_interval steps.
    save_intermediate_lattices : bool, default True
        If True, save snapshots of the lattice configuration along with
        energy and acceptance rate. Ignored if save_interval is None.
    quiet : bool, default True
        If True, suppress printing intermediate results to screen.
        Ignored if save_interval is None.

    Returns:
    --------
    dict : Dictionary with the following keys:
        'final_energy' : float
            Total lattice energy after the final MC step.
        'acceptance_rate' : float
            Fraction of proposed swaps that were accepted.
        'energy' : list of float
            Lattice energy at each sampled step. When save_interval
            is not None, includes the initial energy and one value
            per interval. Otherwise contains only the final energy.
        'ab_fraction' : list of float
            Fraction of nearest-neighbor pairs that are A-B at each
            sampled step. Same structure as 'energy'.
        'final_ab_fraction' : float
            A-B neighbor pair fraction after the final MC step.
        'final_neighbor_counts' : tuple of int
            (n_AA, n_BB, n_AB) pair counts after the final MC step.
        'sample_step' : list of int
            MC step numbers at which energy and ab_fraction were
            recorded. When save_interval is not None, starts at 0
            and includes each interval. Otherwise contains only
            n_steps.
        'lattice' : list of 2D Numpy arrays
            When save_interval is not None and
            save_intermediate_lattices is True, contains lattice
            snapshots at each save interval (including initial).
            Otherwise contains only the final lattice.
        'final_lattice' : 2D Numpy array
            Lattice configuration after the final MC step.
    """
    # NOTE: We are risking floating point error by computing energy changes
    # incrementally. This is done for speed, but we may want to use 
    # `calculate_total_energy()` whenever we want to report energies to
    # avoid possible floating point error.

    # Instantiate lattice
    if lattice is None:
        lattice_dim = 100
        lattice = initialize_lattice(lattice_dim, lattice_dim ** 2 // 2)
    else:
        if lattice.shape[0] != lattice.shape[1]:
            raise RuntimeError('Lattice must be square.')

        lattice_dim = lattice.shape[0]

        if lattice_dim < 3:
            raise RuntimeError('Minimum lattice size is 3 x 3.')

        if np.sum(lattice == 0) + np.sum(lattice == 1) != lattice_dim ** 2:
            raise RuntimeError('Lattice may only contain ones and zeros.')

        lattice = lattice.astype(int)

    # Make sure we actually have a mixture
    if lattice.sum() in [0, lattice_dim * lattice_dim]:
        raise RuntimeError('Must have at least one molecule each of A and B.')

    # Flattened arrays of ones and zeros
    flat_A = np.flatnonzero(lattice)
    flat_B = np.flatnonzero(lattice == 0)

    # Compute total lattice energy
    E = calculate_total_energy(lattice, lattice_dim, E_AA, E_BB, E_AB)

    # Number of A and number of B molecules
    nA = int(np.sum(lattice))
    nB = lattice_dim * lattice_dim - nA

    # Compute A-B fraction
    n_AA, n_BB, n_AB = count_neighbor_pairs(lattice, lattice_dim)
    total_pairs = n_AA + n_BB + n_AB
    ab_frac = n_AB / total_pairs if total_pairs > 0 else 0.0

    # Keep track of energy, fraction of nearest-neighbor contacts
    # that are A-B and acceptance of MCMC moves
    sample_step = [0]
    energy = [E]
    ab_fraction = [ab_frac]
    stored_lattices = [lattice.copy()]
    total_acceptance = 0

    if save_interval:
        # Run in chunks with periodic saving
        steps_completed = 0
        while steps_completed < n_steps:
            steps_to_run = min(save_interval, n_steps - steps_completed)
            acceptance_count, E, n_AA, n_BB, n_AB = _run_mcmc_steps(
                lattice, flat_A, flat_B, lattice_dim, nA, nB, E, n_AA, n_BB, n_AB, E_AA, E_BB, E_AB, steps_to_run
            )
            total_acceptance += acceptance_count
            steps_completed += steps_to_run

            total_pairs = n_AA + n_BB + n_AB
            ab_frac = n_AB / total_pairs if total_pairs > 0 else 0.0

            sample_step.append(steps_completed)
            energy.append(E)
            ab_fraction.append(ab_frac)

            if save_intermediate_lattices:
                stored_lattices.append(lattice.copy())

            if not quiet:
                print(f"Step {steps_completed}: Energy = {E:.4f}, "
                    f"Acceptance rate = {total_acceptance / steps_completed:.4f}, "
                    f"A-B fraction = {ab_frac:.4f}")

        if not save_intermediate_lattices:
            stored_lattices = [lattice.copy()]
    else:
        # Run all steps at once without storing intermediate results
        total_acceptance, E, n_AA, n_BB, n_AB = _run_mcmc_steps(
            lattice, flat_A, flat_B, lattice_dim, nA, nB, E, n_AA, n_BB, n_AB, E_AA, E_BB, E_AB, n_steps
        )
        total_pairs = n_AA + n_BB + n_AB
        energy = [E]
        stored_lattices = [lattice.copy()]
        sample_step = [n_steps]
        ab_fraction = [n_AB / total_pairs if total_pairs > 0 else 0.0]

    return {
        'final_energy': E,
        'acceptance_rate': total_acceptance / n_steps,
        'energy': energy,
        'ab_fraction': ab_fraction,
        'final_ab_fraction': ab_fraction[-1],
        'final_neighbor_counts': (n_AA, n_BB, n_AB),
        'sample_step': sample_step,
        'lattice': stored_lattices,
        'final_lattice': lattice.copy(),
    }
