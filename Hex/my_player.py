# Original work © 2024 Polytechnique Montréal.
# Modified by  in 2025.
# See LICENSE file for full licensing information.

from player_hex import PlayerHex
from seahorse.game.action import Action
from seahorse.game.game_state import GameState
from src_2485686_2485067.algorithme_minimax_alpha_beta_typeA import Algorithme_minimax_alpha_beta_typeA
from src_2485686_2485067.Memory.memory import Memory
from src_2485686_2485067.heuristique_v1 import Heuristique_v1
from src_2485686_2485067.forced_move import ForcedMove
from src_2485686_2485067.early_victory_detector import EarlyVictory
import random
from seahorse.game.light_action import LightAction
from src_2485686_2485067.game_debug import GameDebug
from src_2485686_2485067.Metrics.distance import Distance

# import cProfile
# import pstats
# import io


import numpy as np
# import matplotlib.pyplot as plt
# from matplotlib.colors import LinearSegmentedColormap


DEBUG = True

class MyPlayer(PlayerHex):
    """
    Player class for Hex game

    Attributes:
        piece_type (str): piece type of the player "R" for the first player and "B" for the second player
    """

    def __init__(self, piece_type: str, name: str = "MyPlayer"):
        """
        Initialize the PlayerHex instance.

        Args:
            piece_type (str): Type of the player's game piece
            name (str, optional): Name of the player (default is "bob")
        """
        super().__init__(piece_type, name)
        self._memory = Memory(self)

        self._early_victory_detector = EarlyVictory(self._memory)
        self._heuristique = Heuristique_v1(self)
        self._ai_engine = Algorithme_minimax_alpha_beta_typeA(self,self._heuristique)
        self._forced_move = ForcedMove(self)

        self.debug = GameDebug(self)


    def compute_action(self, current_state: GameState, remaining_time: int = 1e9, **kwargs) -> Action:
        """
        Use the minimax algorithm to choose the best action based on the heuristic evaluation of game states.

        Args:
            current_state (GameState): The current game state.

        Returns:
            Action: The best action as determined by minimax.
        """

        # profiler = cProfile.Profile()
        # profiler.enable()  # Démarre le profilage


        try:
            self._memory.update(current_state) # MAJ de la mémoire pour récupérer le coup adverse
            self._memory.heuristique_cache.clear() # vide le cache de l'heuristique
            print(self._memory.print__memory())
            #print(self._heuristique.print_debug())
            # if self._memory.last_move is not None and DEBUG == True:
            #     print(self._memory.print__memory())
            #     print(self._heuristique.print_debug())
                
            #     def mask(X,link_mask = True,trapezoid_mask=True):
            #         """
            #         Masque certaines positions du plateau pour lesquelles l'agent ne doit pas poser de pions
                
            #         Args:
            #             X: La matrice d'attention retourné par get_attention_board()
            #             link_mask: par défaut True, masque les maillons (applique -inf)
            #             trapezoid_mask: par defaut True, masque les trapèzes (applique -inf)
            #         Returns:
            #             X: masqué en fonction des choix 
            #         """
            #         adversary_links = self._memory.get_adversary_links()
            #         adversary_trapezoid = self._memory.get_adversary_trapezoid()
            #         if link_mask == True:
            #             for maillon, espaces in adversary_links.items():
            #                 for espace in espaces:
            #                     x, y = espace
            #                     X[x, y] = 0  # on bloque la case dans l'attention_board
            #         if trapezoid_mask == True:
            #             for pivot, espaces in adversary_trapezoid.items():
            #                 for espace in espaces:
            #                     _,x,y = espace
            #                     X[x, y] = 0  # on bloque la case dans l'attention_board
            #         return X
            #     probability_board = np.copy(self._memory.get_attention_board())
            #     probability_board = mask(X=probability_board) # masquage complet 
            #     self.display_hex_matrix(probability_board)
                
            #     print("-------------------------------------------------------------------------------------------")





            (score,move) = self._ai_engine.execute(current_state,max_depth=3)

            # mise à jour de la mémoire
            new_state = current_state.apply_action(move)
            self._memory.update(new_state)

            # on a trouvé un coup décisif, on a plus besoin du detecteur.
            if score == 40000: 
                self._early_victory_detector.deactivate()

        #   if DEBUG:
        #      self.display_hex_matrix(self._ai_engine.matrice_debug)


            # profiler.disable()  # Arrête le profilage
            # # Stocker et afficher les statistiques
            # # Crée un objet Stats pour afficher les résultats
            # s = io.StringIO()
            # ps = pstats.Stats(profiler, stream=s).sort_stats('cumtime')  # trie par temps cumulé
            # ps.print_stats()  # pas d'argument = affiche toutes les fonctions
            # print(s.getvalue())  # Affiche toutes les stats dans la console
            print(self._memory.print__memory())
            return move
        
        # Catch de toutes les erreurs
        except Exception as e:
            
            # Coup aléatoire en fallback
            possible = list(current_state.generate_possible_light_actions())
            if len(possible) == 0:
                fallback_move = random.choice([(i, j) for i in range(16) for j in range(16)])
            else:
                fallback_move = random.choice(possible)
            s_prime = current_state.apply_action(fallback_move)
            self._memory.update(s_prime)
            if not isinstance(fallback_move, LightAction):
                fallback_move = LightAction({"piece": self._memory.get_my_color(), "position": fallback_move})
            return 0, fallback_move

    # def display_hex_matrix(self, probability_board):
    #     """
    #     Affiche une matrice avec un affichage hexagonal,
    #     un dégradé bleu->rouge pour les valeurs >0,
    #     et une couleur spécifique (gris foncé) pour les valeurs à 0.
    #     La fonction s'adapte automatiquement à la plage des valeurs.
    #     """
    #     n = probability_board.shape[0]

    #     fig, ax = plt.subplots(figsize=(n, n))
    #     ax.set_aspect('equal')
    #     ax.axis('off')
    #     cell_size = 1

    #     # Colormap bleu->rouge
    #     cmap = LinearSegmentedColormap.from_list("blue_red_fine", ["blue", "red"], N=256)
    #     zero_color = (0.2, 0.2, 0.2, 1)  # gris foncé RGBA pour les 0

    #     # Normalisation automatique
    #     max_val = np.max(probability_board)
    #     if max_val == 0:
    #         max_val = 1  # éviter division par zéro

    #     for i in range(n):
    #         offset = i * cell_size * 0.5  # décalage pour effet hexagonal
    #         for j in range(n):
    #             value = probability_board[i, j]
    #             if value == 0:
    #                 color = zero_color
    #             else:
    #                 color = cmap(value / max_val)  # normalisation entre 0 et 1
    #             rect = plt.Rectangle((j + offset, n - i - 1), 1, 1, color=color)
    #             ax.add_patch(rect)
    #             ax.text(j + offset + 0.5, n - i - 1 + 0.5, f"{value:.4f}",
    #                     ha='center', va='center', color='white', fontsize=8)

    #     ax.set_xlim(0, n + n * 0.5)
    #     ax.set_ylim(0, n)
    #     ax.set_title("Attention Heatmap (0=Gray, >0=Blue->Red) - Hexagonal")
    #     plt.show()