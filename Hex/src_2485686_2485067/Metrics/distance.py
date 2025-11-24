# Copyright (c) 2025
# Licensed under the MIT License.
# See LICENSE file for details.

from game_state_hex import GameStateHex
import numpy as np
import heapq
from typing import override, List, Tuple, Optional
from typing import TYPE_CHECKING
from src_2485686_2485067.Metrics.metrics import Metrics

if TYPE_CHECKING:
    from my_player import MyPlayer

class Distance(Metrics):
    """
    Calcule la distance (Dijkstra) à parcourir selon les règles suivantes :
    + 0 si un pion appartient à l'agent 
    + 1 si la case est vide 
    + 0.5 si la case est dans un maillon allié (favorise sans abuser)
    + 10 si la case est dans un maillon adverse (forte penalité) (passage impossible)
    
    Retourne (ma_distance, distance_adverse, mon_chemin, chemin_adverse)
    """
    
    # Coûts configurables
    COST_EMPTY = 1.0 # cout pour traverser une case vide
    COST_MY_LINK = 0.1 # cout pour traverser un espace dans un maillon 
    COST_ADV_LINK = 10  # cout pour traverser un maillon adverse
    COST_MY_PIECE = 0.0 # cout pour traverser une pièce à moi (de mon point de vue)
    COST_ADV_PIECE = 0.0 #cout pour traverser une pièce à l'adversaire (pdv de l'adversaire)
    COST_MY_TRAPEZOID = 0.75 # cout pour traverser un trapeze à moi 
    COST_ADV_TRAPEZOID = 3 # cout pour traverser un trapèze de l'adversaire 
    
    def __init__(self, joueur:"MyPlayer"):
        super().__init__()
        self.joueur = joueur
        self._my_path_cache = None
        self._adv_path_cache = None
        self._my_distance_cache = None
        self._adv_distance_cache = None

        # Récupération du dernier coup
        self._last_move = None

    @override
    def execute(self) -> Tuple[float, float]:
        """
        Retourne (ma_distance, distance_adversaire).
        """
        self._last_move = self.joueur._memory.move_history.peek()
        # moi, si le dernier coup n'impacte pas le chemin optimal, on ne recalcul pas 
        # if self._my_path_cache is None or (self._last_move and self._last_move in self._my_path_cache):
        my_distance, my_path = self._dijkstra_player(for_adversary=False)
        self._my_distance_cache = my_distance
        self._my_path_cache = my_path

        # Adversaire
        # if self._adv_path_cache is None or (self._last_move and self._last_move in self._adv_path_cache):

        adv_distance, adv_path = self._dijkstra_player(for_adversary=True)
        self._adv_distance_cache = adv_distance
        self._adv_path_cache = adv_path

        return my_distance, adv_distance
    
    def get_my_critical_path(self) -> List[Tuple[int, int]]:
        """Retourne le chemin critique du joueur (à appeler après execute())."""
        return self._my_path_cache or []
    
    def get_adversary_critical_path(self) -> List[Tuple[int, int]]:
        """Retourne le chemin critique de l'adversaire (à appeler après execute())."""
        return self._adv_path_cache or []

    def _dijkstra_player(self, for_adversary: bool) -> Tuple[float, List[Tuple[int, int]]]:
        """
        Implémentation générique de Dijkstra.
        
        Args:
            for_adversary: True pour calculer la distance de l'adversaire, False pour le joueur
            
        Returns:
            (distance, chemin_optimal)
        """
        board = self.joueur._memory.get_board()
        hauteur, largeur = board.shape
        
        # Déterminer le joueur concerné
        if for_adversary:
            player_color = self.joueur._memory.get_adversary_color()
            player_value = -1
            opponent_value = 1
        else:
            player_color = self.joueur.get_piece_type()
            player_value = 1
            opponent_value = -1
        
        # Initialisation
        dist = np.full((hauteur, largeur), np.inf)
        parent = {}  # Pour reconstruire le chemin
        pq = []
        objectives = set()
        
        # Déterminer points de départ et d'arrivée
        is_vertical = (player_color == "R")
        
        if is_vertical:  # Rouge : haut → bas
            start_row, end_row = 0, hauteur - 1
            for col in range(largeur):
                objectives.add((end_row, col))
                cell = board[start_row, col]
                
                # Accepter les cases vides OU occupées par le joueur
                if cell == player_value:
                    dist[start_row, col] = 0
                    parent[(start_row, col)] = None
                    heapq.heappush(pq, (0, (start_row, col)))
                elif cell == 0:
                    dist[start_row, col] = self.COST_EMPTY
                    parent[(start_row, col)] = None
                    heapq.heappush(pq, (self.COST_EMPTY, (start_row, col)))
        else:  # Bleu : gauche → droite
            start_col, end_col = 0, largeur - 1
            for row in range(hauteur):
                objectives.add((row, end_col))
                cell = board[row, start_col]
                
                if cell == player_value:
                    dist[row, start_col] = 0
                    parent[(row, start_col)] = None
                    heapq.heappush(pq, (0, (row, start_col)))
                elif cell == 0:
                    dist[row, start_col] = self.COST_EMPTY
                    parent[(row, start_col)] = None
                    heapq.heappush(pq, (self.COST_EMPTY, (row, start_col)))
        
        # Récupérer les maillons et trapezes
        if for_adversary:
            my_links = set(self.joueur._memory.get_adversary_space_links())
            opponent_links = set(self.joueur._memory.get_me_space_links())
            my_trapezoids = set(self.joueur._memory.get_adversary_space_trapezoid())
            opponent_trapezoids = set(self.joueur._memory.get_me_space_trapezoid())
        else:
            my_links = set(self.joueur._memory.get_me_space_links())
            opponent_links = set(self.joueur._memory.get_adversary_space_links())
            my_trapezoids = set(self.joueur._memory.get_me_space_trapezoid())
            opponent_trapezoids = set(self.joueur._memory.get_adversary_space_trapezoid())
        
        # Directions hexagonales
        directions = [(-1,0), (1,0), (0,-1), (0,1), (-1,1), (1,-1)]
        
        end_pos = None
        
        # Algorithme de Dijkstra
        while pq:
            cost, (i, j) = heapq.heappop(pq)
            
            # Optimisation : ignorer les doublons
            if cost > dist[i, j]:
                continue
            
            # Objectif atteint
            if (i, j) in objectives:
                end_pos = (i, j)
                break
            
            # Explorer les voisins
            for di, dj in directions:
                ni, nj = i + di, j + dj
                
                # Vérifier les limites
                if not (0 <= ni < hauteur and 0 <= nj < largeur):
                    continue
                
                cell = board[ni, nj]
                
                # Case bloquée par l'adversaire
                if cell == opponent_value:
                    continue
                
                # Calculer le coût de déplacement
                if cell == player_value:
                    move_cost = self.COST_MY_PIECE
                elif (ni, nj) in my_links:
                    move_cost = self.COST_MY_LINK
                elif (ni, nj) in opponent_links:
                    move_cost = self.COST_ADV_LINK
                elif (0,ni,nj) in my_trapezoids:
                    move_cost = self.COST_MY_TRAPEZOID
                elif (0,ni,nj) in opponent_trapezoids:
                    move_cost = self.COST_ADV_TRAPEZOID
                else:  # Case vide
                    move_cost = self.COST_EMPTY
                
                new_cost = cost + move_cost
                
                # Mise à jour si meilleur chemin trouvé
                if new_cost < dist[ni, nj]:
                    dist[ni, nj] = new_cost
                    parent[(ni, nj)] = (i, j)
                    heapq.heappush(pq, (new_cost, (ni, nj)))
        
        # Reconstruire le chemin optimal
        if end_pos is None:
            return np.inf, []
        
        path = []
        current = end_pos
        while current is not None:
            path.append(current)
            current = parent.get(current)
        
        path.reverse()
        
        return dist[end_pos], path
    
    def visualize_path(self, path: List[Tuple[int, int]], label: str = "Path"):
        """
        Affiche visuellement un chemin sur le plateau (pour debug).
        """
        board = self.joueur._memory.get_board()
        hauteur, largeur = board.shape
        visual = board.copy().astype(str)
        
        for (i, j) in path:
            if board[i, j] == 1:
                visual[i, j] = "M"  # My piece
            elif board[i, j] == -1:
                visual[i, j] = "A"  # Adversary piece
            else:
                visual[i, j] = "*"  # Path marker
        
        print(f"\n{label}:")
        print(visual)

    def to_json(self):
        """Sérialisation JSON compatible Seahorse."""
        return {
            "my_path_length": len(self._my_path_cache) if self._my_path_cache else 0,
            "adv_path_length": len(self._adv_path_cache) if self._adv_path_cache else 0
        }
    


    @override
    def update(self,s0):
        my_distance, adv_distance = self.execute()
        self.joueur._memory.my_distance = my_distance
        self.joueur._memory.adversary_distance = adv_distance
        self.joueur._memory.my_critical_path = self._my_path_cache
        self.joueur._memory.adversary_critical_path = self._adv_path_cache

    def undo(self):
        self.update(None)