# Copyright (c) 2025
# Licensed under the MIT License.
# See LICENSE file for details.

import numpy as np

class AttentionFilter:
    """
    Filtre hexagonal simple :
    applique une valeur sur une case centrale et ses 6 voisins directs.
    """
    def __init__(self, shape, radius=1, value=1):
        self.shape = shape
        self.radius = radius
        self.value = value

        # Offsets codés en dur (6 voisins hexagonaux)
        if radius == 1:
            offsets = [
                (0, 0),      # centre
                (-1, 0),     # haut
                (1, 0),      # bas
                (0, -1),     # gauche
                (0, 1),      # droite
                (-1, 1),     # haut-droite
                (1, -1)      # bas-gauche
            ]

        elif radius == 2:
            # distance hexagonale = 2 (total 19 cases : 1 centre + 6 + 12)
            offsets = [
                (0, 0),
                # distance = 1
                (-1, 0), (1, 0), (0, -1), (0, 1), (-1, 1), (1, -1),
                # distance = 2
                (-2, 0), (2, 0), (0, -2), (0, 2),
                (-1, -1), (1, 1),
                (-2, 1), (-1, 2), (1, -2), (2, -1),
                (-2, 2), (2, -2)
            ]
        else:
            raise ValueError("Seuls radius=1 et radius=2 sont supportés.")

        # Stockage sous forme numpy pour vectorisation
        self.dr = np.array([r for r, _ in offsets])
        self.dc = np.array([c for _, c in offsets])

    def apply(self, matrix, center):
        """
        Applique le filtre autour de `center` (r, c) sur la matrice donnée.
        """
        nrows, ncols = self.shape
        r0, c0 = center

        # Positions absolues des cellules affectées
        r = r0 + self.dr
        c = c0 + self.dc

        # Filtrage des positions valides dans les bornes
        valid = (r >= 0) & (r < nrows) & (c >= 0) & (c < ncols)
        r = r[valid]
        c = c[valid]

        # Application en place
        matrix[r, c] = self.value
