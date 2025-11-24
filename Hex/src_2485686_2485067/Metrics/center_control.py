# Copyright (c) 2025
# Licensed under the MIT License.
# See LICENSE file for details.


from game_state_hex import GameStateHex
import numpy as np
from typing import override
from typing import TYPE_CHECKING
from src_2485686_2485067.Metrics.metrics import Metrics
if TYPE_CHECKING:
    from my_player import MyPlayer  # import uniquement pour l'IDE




class Center_control(Metrics):
    """
    Génère une matrice de pondération pour le plateau Hex.
    Les cases centrales sont pondérées plus fortement, les bords moins.
    La matrice est calculée une seule fois pour optimiser les performances.
    Permet d'inciter le bot à controler le centre en priorité ( ne prend en compte que ses propres jetons)
    """

    def __init__(self,joueur:"MyPlayer"):
        super().__init__()
        self.joueur = joueur
        self.center_control_board = self.__generationPonderation()
        
   
    def __generationPonderation(self):

        hauteur, largeur = self.joueur._memory.BOARD_SIZE+2,self.joueur._memory.BOARD_SIZE+2
        pond = np.zeros((hauteur, largeur), dtype=float)

        # Coordonnées du centre du plateau
        center_i, center_j = (hauteur - 1) / 2, (largeur - 1) / 2

        # Calcul des poids
        for i in range(hauteur):
            for j in range(largeur):
                dist_to_center = np.sqrt((i - center_i)**2 + (j - center_j)**2)
                max_dist = np.sqrt(center_i**2 + center_j**2)
                pond[i, j] = 1 - (dist_to_center / max_dist)  # normalisé entre 0 et 1

        return pond

    @override
    def execute(self) -> float:
        """
        Retourne le poids total du plateau sous forme d'une valeur en prenant en compte
        les pions de chaque joueur par rapport au centre.

        Produit terme par terme

        """
        board = self.joueur._memory.get_board()
        
        # DÉBOGAGE
        player_board = (board == 1).astype(float)
        return np.sum(self.center_control_board * player_board)
    
    