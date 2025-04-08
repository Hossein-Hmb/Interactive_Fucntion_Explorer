import numpy as np
from scipy.integrate import quad, dblquad, tplquad


def uncertainty(gamma):
    """
    Returns True if gamma (2n x 2n) satisfies
    gamma + (i/2) Omega >= 0,
    and False otherwise.
    """
    gamma = np.array(gamma, dtype=complex)
    dim = gamma.shape[0]

    # Must be 2n x 2n
    if gamma.shape[1] != dim or (dim % 2 != 0):
        return False

    n = dim // 2

    # Block-diagonal assembly
    Omega = np.array([
        [0, 1, 0, 0],
        [-1, 0, 0, 0],
        [0, 0, 0, -1],
        [0, 0, 1, 0]
    ], dtype=complex)

    # print(f"omega is :\n {Omega}")

    # Form M = gamma + (i/2)*Omega
    M = gamma + 1j/2 * Omega
    # print(f"M is :\n {M}")

    # Check Hermiticity (M should equal its own conjugate transpose)
    if not np.allclose(M, M.conj().T):
        return False

    # Check that M is positive semidefinite (all eigenvalues >= 0)
    eigenvals = np.linalg.eigvalsh(M)  # eigvalsh is for Hermitian matrices
    # Allow a tiny numerical tolerance
    return np.all(eigenvals > -1e-14)


def wigner_gaussian(x1, p1, x2, p2, gamma):
    # Verifgy Uncertainty Condition
    if uncertainty(gamma):
        # Create vector r
        vec_r = np.array([[x1, p1, x2, p2]]).reshape(-1, 1)

        # Transpose of vec_r
        vec_r_t = np.transpose(vec_r)

        # Determinant and inverse of gamma
        det = np.linalg.det(gamma)
        inv = np.linalg.inv(gamma)

        norm = 1 / ((2 * np.pi)**2 * np.sqrt(det))
        exponent = -0.5 * vec_r_t @ inv @ vec_r

        print(f"vec_r T: {vec_r_t.flatten()}")
        print("\n")
        print(f"vec_r: {vec_r}")
        print("\n")
        return norm * np.exp(exponent)
    else:
        return "Gamma doesn't respect uncertainty relation!"


# Example usage on your 4x4 gamma:
gamma = np.array([
    [1, 0, 1, 0],
    [0, 1, 0, 1],
    [1, 0, 2, 0],
    [0, 1, 0, 2]
], dtype=float)

# print(np.transpose(gamma))
# print(wigner_gaussian(1, 2, 0, 0, gamma))
# Define a wrapper function for integration
