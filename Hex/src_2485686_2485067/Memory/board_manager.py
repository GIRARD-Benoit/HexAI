# Copyright (c) 2025
# Licensed under the MIT License.
# See LICENSE file for details.

from typing import override
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from src_2485686_2485067.Memory.memory import Memory

from seahorse.game.game_state import GameState
import numpy as np
from src_2485686_2485067.Memory.manager import Manager
from src_2485686_2485067.Memory.UniqueStack import UniqueStack
import time

class BoardManager(Manager):
    """
    Gestion de la représentation du plateau + gestion de l'historique des coups.
    """
    def __init__(self,_memory:"Memory"):
        self._memory = _memory
        self.original_board = self.get_initial_board()
        self.history = UniqueStack()
        """ Historique des situations du plateau"""

    def get_initial_board(self):
        original_board = np.zeros((self._memory.BOARD_SIZE+2,self._memory.BOARD_SIZE+2), dtype=int) # represente le plateau
        # Remplissage des bords :
        # Bord supérieur (ligne 0) et bord inférieur (ligne -1)
        original_board[0, :] = 1 if self._memory.my_color == "R" else -1    # ou -1 selon le joueur que tu veux représenter
        original_board[-1, :] = 1 if self._memory.my_color == "R" else -1   # bord inférieur

        # Bord gauche (colonne 0) et bord droit (colonne -1)
        original_board[:, 0] = 1 if self._memory.my_color == "B" else -1    # bord gauche
        original_board[:, -1] = 1 if self._memory.my_color == "B" else -1   # bord droit
        return original_board

    @override
    def update(self, current_state: GameState):
        # Plateau actuel
        board = self._memory.get_board()
        prev_board = board.copy()
        self.history.push(prev_board)

        # Nouveau plateau
        max_piece = self._memory.joueur.get_piece_type()
        new_board = current_state.rep.get_env()
        # reinitialisation du plateau avant copie des nouveau coups
        board[:] = self.original_board
        # Reconstruction
        for (i, j), piece in new_board.items():
            # s'assurer qu'on a bien des int et non des float 
            i = int(i)
            j = int(j)
            board[i+1, j+1] = 1 if piece.piece_type == max_piece else -1 # décallage car le tableau est aggrandi

        # --- Détection du coup joué ---
        diff_mask = board != prev_board
        diff_positions = np.argwhere(diff_mask)

        if len(diff_positions) == 1:
            move = tuple(map(int, diff_positions[0]))  # convertit [i, j] en (i, j)
            self._memory.move_history.push(move)      # push le tuple complet
            self._memory.last_move = self._memory.move_history.peek()
            return
        elif len(diff_positions) == 0:
            return
        else:
            raise ValueError(f"Erreur levée dans board_manager : {len(diff_positions)} cases modifiées.")

    @override
    def undo(self):
        if not self.history:
            raise IndexError("Impossible d'annuler : l'historique est vide.")
        
        last_board = self.history.pop()
        board = self._memory.get_board()
        board[:] = last_board  # copie en place
    
        self._memory.last_move = self._memory.move_history.pop()  # on retire aussi le dernier coup