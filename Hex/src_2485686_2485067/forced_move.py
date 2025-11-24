# Copyright (c) 2025
# Licensed under the MIT License.
# See LICENSE file for details.

from typing import TYPE_CHECKING
import copy
from src_2485686_2485067.algorithme_minimax_alpha_beta_typeA import Algorithme_minimax_alpha_beta_typeA
from seahorse.game.light_action import LightAction
from src_2485686_2485067.heuristique_v1 import Heuristique_v1
if TYPE_CHECKING:
    from src_2485686_2485067.Memory.memory import Memory
    from src_2485686_2485067.algorithme_minimax_alpha_beta_typeA import Algorithme_minimax_alpha_beta_typeA
    from my_player import MyPlayer

# la classe travaille sur une memoire séparée
class ForcedMove:
    def __init__(self,joueur:"MyPlayer"):
        """
        _memory : instance de Memory
        """
        self.joueur = joueur
        self._memory = joueur._memory

    def find_me_forced_move(self,flag_local_analysis_area,current_state,depth):
        """
        1) Cherche un coup forcé à jouer :
        - Parcourt les maillons cassés
        - Identifie l'espace encore libre
        - Retourne ce coup sous forme de tuple (x, y)

        2) Cherche un coup forcé parmis les trapèzes
        """

        for broken in self._memory.get_me_broken_links():
            maillon = broken["maillon"]
            espaces = broken["espaces"]  # liste de 2 coordonnées [(x1, y1), (x2, y2)]
            casse_par = broken["cassé_par"]

            # On cherche dans les 2 espaces celui qui est encore vide
            for espace in espaces:
                if self._memory.is_empty(espace):
                    return LightAction({"piece": self._memory.get_my_color(), "position": (espace[0]-1,espace[1]-1)})

        
        # si un trapèze a été brisé 
        if flag_local_analysis_area == None and depth == 0: # On vérifie qu'on est pas dans une recherche locale sinon boucle infinie
            # si un trapèze a été brisé 
            for broken in self._memory.get_me_broken_trapezoid():
                area_analysis = broken["espaces"]
                area_analysis = [(x, y) for _, x, y in area_analysis]
                
                # lancement d'une recherche locale 

                (score, action) = self.joueur._ai_engine.execute(current_state, max_depth=1, local_analysis_area=area_analysis, branching_factor=200)

                if action is not None and not isinstance(action, LightAction):
                    piece_type = self._memory.get_my_color()
                    action = LightAction({"piece": piece_type, "position": action})
                return action

        # Aucun coup forcé
        return None

    def find_adversary_forced_move(self,flag_local_analysis_area,current_state,depth):
        for broken in self._memory.get_adversary_broken_links():
            maillon = broken["maillon"]
            espaces = broken["espaces"]  # liste de 2 coordonnées [(x1, y1), (x2, y2)]
            casse_par = broken["cassé_par"]

            # On cherche dans les 2 espaces celui qui est encore vide
            for espace in espaces:
                if self._memory.is_empty(espace):
                    return LightAction({"piece": self._memory.get_my_color(), "position": (espace[0]-1,espace[1]-1)})

        # Aucun coup forcé
        return None
    
    def first_move(self):
        if self._memory.get_move_history().isEmpty():
            return LightAction({"piece": self._memory.get_my_color(), "position": (5,7)}) # 5,7
        return None



    def to_json(self):
        """Sérialisation JSON compatible Seahorse."""
        return {
            "history": []
        }