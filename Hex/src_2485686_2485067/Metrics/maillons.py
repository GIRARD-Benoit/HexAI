# Copyright (c) 2025
# Licensed under the MIT License.
# See LICENSE file for details.

from game_state_hex import GameStateHex
from src_2485686_2485067.Metrics.metrics import Metrics
import time
from typing import override
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from my_player import MyPlayer  # import uniquement pour l'IDE

class Maillons(Metrics):
    """
    . . . . . . . . . . . .
    . . . . . . . . . . . .
    . . . . . R . . . . . .
    . . . . . .  . . . . .
    . . . . . R . . . . . .
    . . . . . . . . . . . .
    """
    def __init__(self,joueur:"MyPlayer"):
        super().__init__()
        self.joueur = joueur

    
    @override
    def execute(self):
        """ retourne le nombre de maillons à moi et à l'adversaire"""
        nbr_maillons_moi = len(self.joueur._memory.get_me_links().keys())
        nbr_maillons_adversaire = len(self.joueur._memory.get_adversary_links().keys())
        return nbr_maillons_moi,nbr_maillons_adversaire
