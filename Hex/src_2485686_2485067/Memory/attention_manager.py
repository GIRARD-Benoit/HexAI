# Copyright (c) 2025
# Licensed under the MIT License.
# See LICENSE file for details.

from seahorse.game.game_state import GameState
from src_2485686_2485067.Memory.attention_filter import AttentionFilter
from typing import override
import numpy as np
from typing import TYPE_CHECKING
from src_2485686_2485067.Memory.manager import Manager
if TYPE_CHECKING:
    from src_2485686_2485067.Memory.memory import Memory

class Attention_manager(Manager):
    
#    BUMP_THRESHOLD = 20 # reaugmentation des positions non vu depuis longtemps


    def __init__(self,_memory:"Memory"):
        self.MIN_ATTENTION = 0.3# attention minimal au cours de la decroissance
        self.value_max = 2 #attention maximale initiale
        self.decreasing = 0.6 # coefficient de decroissance

        self._memory = _memory
        self.local_attention_board = None # Uniquement dans cette classe (decorrelé de la mémoire)
        self._memory.attention_board_history.append(np.copy(_memory.get_attention_board()))
        self.board = _memory.get_board()
        self.filter = AttentionFilter(shape=(self._memory.BOARD_SIZE+2,self._memory.BOARD_SIZE+2),radius=2,value=self.value_max)

    @override
    def update(self, current_state: GameState):
        """
        Met à jour la matrice d'attention (avec somme de probabilités = 1)
        """


        # si c'est la première fois qu'on joue
        last_move = self._memory.get_move_history().peek()
        if last_move is None:
            return
        
        self.local_attention_board = np.copy(self._memory.attention_board_history[-1]) # on récupère la dernière matrice à jour
        

        # diminution de 1, arrêt à 0:
        # on ne touche que les cellules >= 0

        positive_mask = self.local_attention_board > 0
        self.local_attention_board[positive_mask] *= self.decreasing



        # ajout du chemin critique de chaque camp dans l'attention 
        if self._memory.my_critical_path is not None and self._memory.adversary_critical_path is not None:
            for i in self._memory.my_critical_path:
                self.local_attention_board[i] = self.value_max
            for j in self._memory.adversary_critical_path:
                self.local_attention_board[j] = self.value_max

        
        # application du filtre
        self.filter.apply(self.local_attention_board,center=last_move)

        # softmax 


        # nettoyage
        minimal_treshold = (self.local_attention_board < self.MIN_ATTENTION) & (self.local_attention_board > 0)
        self.local_attention_board[minimal_treshold] = float("-inf") #self.MIN_ATTENTION 

        # MAJ de la mémoire
        self._memory.attention_board = np.copy(self.softmax(self.local_attention_board))
        self._memory.get_attention_history().append(np.copy(self.local_attention_board))
        return

    @override
    def undo(self):
        self._memory.get_attention_history().pop()
        self._memory.attention_board = np.copy(self.softmax(self._memory.attention_board_history[-1]))
        self.local_attention_board = np.copy(self._memory.get_attention_history()[-1])

    def softmax(self,X):
        """
        Applique un softmax sur la matrice X
        """
        e_X = np.exp(X)
        return e_X / np.sum(e_X)