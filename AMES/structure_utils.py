
import numpy as np

# some useful functions for structure calculations

def get_dihedral_angle(
        rki: np.ndarray, rij: np.ndarray, rjl: np.ndarray
) -> float:

    """

    A function to compute the dihedral angle 
    defined by three vectors;
    given vectors rki, rij, rjl, defined in the
    moiety k-i-j-l, it returns the cosine of the dihedral angle

    WARNING: for a dihedral angle to be properly defined, it is necessary
    that neither k-i-j nor i-j-l are colinear. If either of these conditions
    is not met (if the modulus of the cross product rki x rij or rij x rjl is
    below a tolerance) an undefined (None) value is returned.

    Args:

    :param np.ndarray rki: ri - rk
    :param np.ndarray rij: rj - ri
    :param np.ndarray rjl: rl - rj

    """

    # WARNING: Avoid using np.cross: this can cause a numerical error
    # by sometimes resulting in abs(cosphi)= 1 + delta, delta ~ 1.0e-16
    # which is sufficient to cause np.arccos to fail.
    # vkiij = np.cross( rki, rij )

    vkiij0 = rki[1] * rij[2] - rki[2] * rij[1]
    vkiij1 = rki[2] * rij[0] - rki[0] * rij[2]
    vkiij2 = rki[0] * rij[1] - rki[1] * rij[0]

    vkiij = np.array([vkiij0, vkiij1, vkiij2], dtype=float)

    nkiij = np.dot(vkiij, vkiij)

    # vijjl = np.cross( rij, rjl )
    vijjl0 = rij[1] * rjl[2] - rij[2] * rjl[1]
    vijjl1 = rij[2] * rjl[0] - rij[0] * rjl[2]
    vijjl2 = rij[0] * rjl[1] - rij[1] * rjl[0]

    vijjl = np.array([vijjl0, vijjl1, vijjl2], dtype=float)

    nijjl = np.dot(vijjl, vijjl)

    if (nkiij > 1.0e-5) and (nijjl > 1.0e-5):

       cosphi = np.dot(vkiij, vijjl) / np.sqrt(nkiij * nijjl)

    else:

       cosphi = None

    return cosphi

