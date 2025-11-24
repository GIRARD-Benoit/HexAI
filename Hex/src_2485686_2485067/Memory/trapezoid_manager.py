# Copyright (c) 2025
# Licensed under the MIT License.
# See LICENSE file for details.

from collections import defaultdict
from typing import override
import itertools
from game_state_hex import GameState
import time

import numpy as np
from src_2485686_2485067.Memory.manager import Manager
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from my_player import MyPlayer  # import uniquement pour l'IDE
if TYPE_CHECKING:
    from src_2485686_2485067.Memory.memory import Memory

class TrapezoidManager(Manager):
    """
    Gère les trapèzes du jeu de Hex.
    Un trapèze est défini par :
        - 1 jetons séparé de deux lignes/colonnes vides entre le bord de sa couleur
        - 8 cases vides autour 
    Permet :
        - d'ajouter un trapèze
        - de le supprimer
        - de savoir si un pion appartient à un trapèze
        - de supprimer un trapèze invalidé
    """

    def __init__(self, _memory :"Memory"):
        self._memory = _memory
        self.board = _memory.get_board()

        self.lines_down = [_memory.get_board()[10,2:15],_memory.get_board()[11,2:15],_memory.get_board()[12,2:15]]
        self.lines_up = [_memory.get_board()[6,2:15],_memory.get_board()[5,2:15],_memory.get_board()[4,2:15]]
        self.rows_right = [_memory.get_board()[2:15,10],_memory.get_board()[2:15,11],_memory.get_board()[2:15,12]]
        self.rows_left = [_memory.get_board()[2:15,5],_memory.get_board()[2:15,4],_memory.get_board()[2:15,3]]

        # (0,x,y) 0 : signifie case vide 1: pion m'appartenant -1: pion de l'adversaire
        self.shematic = {
    "3": [(0,1,-1),(0,1,0),(0,2,-2),(0,2,-1),(0,2,0),(0,2,-3),(0,1,-2),(0,0,-1)],
    "4": [(0,1,-1),(0,1,0),(0,2,-2),(0,2,-1),(0,2,0),(0,2,-3),(0,1,-2),(0,0,-1),
          (0,3,-2),(0,3,-1),(0,3,0),(0,3,-3),(0,3,-4),(0,3,-5),(0,3,1),(0,2,1),(0,1,1),(0,1,-3),(0,2,-4)],
    # "5": [(0,0,-1),(0,-1,-2),(0,-1,-1),(0,-1,0),(0,-2,-3),(0,-2,-2),(0,-2,-1),(0,-2,0),(0,-3,-4),
    #       (0,-3,-3),(1,-3,-2),(0,-3,-1),(0,-3,0),(0,-4,-5),(0,-4,-4),(0,-4,-3),(0,-4,-2),(0,-4,-1),(0,-4,0)]
    "5": [(0,0,0)]
}
        
        self.mirror_shematic = {
    "3": [(0,1,-1),(0,1,0),(0,2,-2),(0,2,-1),(0,2,0),(0,2,1),(0,1,1),(0,0,1)],
    "4": [(0,1,-1),(0,1,0),(0,2,-2),(0,2,-1),(0,2,0),(0,2,-3),(0,1,-2),(0,0,1),
          (0,3,-2),(0,3,-1),(0,3,0),(0,3,-3),(0,3,-4),(0,3,2),(0,3,1),(0,2,1),(0,1,1),(0,1,2),(0,2,2)],
    "5": [(0,0,0)] # case toujours pleine donc on ne fait pas de mirroir sur la ligne 5
}

        self.rotated_shematic60_left = {} # bas -> gauche
        for key, coords_list in self.shematic.items():
            self.rotated_shematic60_left[key] = [self.rotate60_clockwise_2D(t) for t in coords_list]

        self.rotated_mirror_shematic60_left = {} # bas -> gauche (mirror)
        for key, coords_list in self.mirror_shematic.items():
            self.rotated_mirror_shematic60_left[key] = [self.rotate60_clockwise_2D(t) for t in coords_list]

        self.rotated_shematic180_top = {} # bas -> haut
        for key, coords_list in self.shematic.items():
            self.rotated_shematic180_top[key] = [self.rotate180_clockwise_2D(t) for t in coords_list]

        self.rotated_mirror_shematic180_top = {} # bas -> haut (mirror)
        for key, coords_list in self.mirror_shematic.items():
            self.rotated_mirror_shematic180_top[key] = [self.rotate180_clockwise_2D(t) for t in coords_list]

        self.rotated_shematic60_right = {} #gauche -> droite
        for key, coords_list in self.rotated_shematic60_left.items():
            self.rotated_shematic60_right[key] = [self.rotate180_clockwise_2D(t) for t in coords_list]

        self.rotated_mirror_shematic60_right = {} #gauche -> droite (mirror)
        for key, coords_list in self.rotated_mirror_shematic60_left.items():
            self.rotated_mirror_shematic60_right[key] = [self.rotate180_clockwise_2D(t) for t in coords_list]

    def add_trapezoid(self,p1,spaces,color):
        """
        Ajoute un trapèze reliant un pion a son bord avec 2 lignes/colonnes d'espace.
        Args:
            p1 (tuple[int, int]): position du pion relié au bord par le trapèze
            spaces (list[tuple[int, int]]): deux positions vides entre les pions
        """
        # normalisation du pivot
        if isinstance(p1, frozenset):
            p1 = next(iter(p1))  # extraire le tuple à l’intérieur
        key = frozenset({p1})

        if key in self._memory.me_trapezoid or key in self._memory.adversary_trapezoid:
            # Le trapezoid existe déjà → on ne le recrée pas
            return

        # Mes trapezoids
        if self._memory.get_my_color() == color:
            self._memory.me_trapezoid[key] = spaces
            for space in spaces:
                self._memory.me_space_trapezoid.setdefault(space, set()).add(key)

        # trapezoids de l'adversaire
        else:
            self._memory.adversary_trapezoid[key] = spaces
            for space in spaces:
                self._memory.adversary_space_trapezoid.setdefault(space, set()).add(key)



    #  SUPPRESSION D'UN TRAPEZE
    def remove_trapezoid(self, pivot_or_key, color):
        # pivot_or_key peut être :
        #  - un tuple (row, col) -> on crée key = frozenset({pivot})
        #  - un frozenset déjà (la clé) -> on l'utilise telle quelle
        #  - éventuellement un set ou une liste -> on convertit si possible

        # Normaliser en clé frozenset
        if isinstance(pivot_or_key, frozenset):
            key = pivot_or_key
        else:
            # si c'est un set contenant un tuple (rare), extraire son élément
            if isinstance(pivot_or_key, (set, list)):
                # si c'est un set/list de pivots -> prendre le premier élément s'il est tuple
                try:
                    first = next(iter(pivot_or_key))
                except TypeError:
                    # non itérable -> le transformer en tuple
                    first = pivot_or_key
                # si le premier est un tuple, on crée frozenset({tuple})
                if isinstance(first, tuple):
                    key = frozenset({first})
                else:
                    # sinon tenter de créer frozenset([pivot_or_key]) (générique)
                    key = frozenset({pivot_or_key})
            else:
                # cas usuel : pivot_or_key est un tuple (row,col)
                key = frozenset({pivot_or_key})

        if self._memory.get_my_color() == color:
            if key in self._memory.me_trapezoid:
                for space in self._memory.me_trapezoid[key]:
                    trapezoids_for_space = self._memory.me_space_trapezoid.get(space)
                    if trapezoids_for_space:
                        trapezoids_for_space.discard(key)
                        if not trapezoids_for_space:
                            del self._memory.me_space_trapezoid[space]
                del self._memory.me_trapezoid[key]
        else:
            if key in self._memory.adversary_trapezoid:
                for space in self._memory.adversary_trapezoid[key]:
                    trapezoids_for_space = self._memory.adversary_space_trapezoid.get(space)
                    if trapezoids_for_space:
                        trapezoids_for_space.discard(key)
                        if not trapezoids_for_space:
                            del self._memory.adversary_space_trapezoid[space]
                del self._memory.adversary_trapezoid[key]



    # ---------------------------------
    # 🔹 NETTOYAGE : SUPPRIMER LES trapezoids INVALIDES
    # ---------------------------------

    def broken_me_trapezoid_detector(self):
        """
        Détecte si un trapèze du joueur a été détruit par le dernier coup de l'adversaire.
        Les maillons détectés sont ajoutés dans self._memory.me_broken_links. Le coup joué 
        est supprimé des maillons 
        """
        self._memory.me_broken_trapezoid.clear()

        last_move = self._memory.move_history.peek()
        if last_move is None:
            return
        
        # Vérifie si la case jouée appartient à un des espaces de MES trapèzes
        if (0,last_move[0],last_move[1]) in self._memory.get_me_space_trapezoid() and self.board[last_move[0],last_move[1]] == -1:
            for trapezoid in self._memory.get_me_space_trapezoid()[(0,last_move[0],last_move[1])]:
                # suppression de l'espace joué dans les espaces du trapèzes
                espaces = self._memory.me_trapezoid[trapezoid]
                espaces.remove((0,last_move[0],last_move[1]))
                # si il ne reste plus qu'une case vide, on le supprime definitivement 
                if len(espaces) > 1:
                    self._memory.me_broken_trapezoid.append({
                        "trapeze": trapezoid,
                        "espaces": espaces,
                        "cassé_par": last_move
                    })

    def broken_adversary_trapezoid_detector(self):
        """
        Détecte si un trapèze adverse a été affecté par le dernier coup.
        Les trapèzes détectés sont ajoutés dans self._memory.adversary_broken_trapezoid
        """

        self._memory.adversary_broken_trapezoid.clear()

        last_move = self._memory.move_history.peek()

        if last_move is None:
            return
    
        # Vérifie si la case jouée appartient à un des espaces de MES trapèzes
        if (0,last_move[0],last_move[1]) in self._memory.get_adversary_space_trapezoid():
            for trapezoid in self._memory.get_adversary_space_trapezoid()[(0,last_move[0],last_move[1])]:
                # suppression de l'espace joué dans les espaces du trapèzes
                espaces = self._memory.adversary_trapezoid[trapezoid]
                espaces.remove((0,last_move[0],last_move[1]))
                # si il ne reste plus qu'une case vide, on le supprime definitivement 
                if len(espaces) > 1:
                    self._memory.adversary_broken_trapezoid.append({
                        "trapeze": trapezoid,
                        "espaces": espaces,
                        "cassé_par": last_move
                    })

    # ---------------------------------
    # 🔹 AFFICHAGE / DEBUG
    # ---------------------------------
    def self_print(self):
        """
        Affiche les trapezoids connus pour debug.
        """
        print("=== trapezoids ===")
        for m, spaces in self._memory.me_trapezoid.items():
            print(f"trapezoid {tuple(m)} → Espaces {spaces}")

        print("\n=== ESPACEMENTS ===")
        for e, trapezoids in self._memory.me_space_trapezoid.items():
            print(f"Espace {e} → trapezoids {trapezoids}")



    @override
    def update(self, s: GameState = None):
        """
        Parcourt le plateau et met à jour la mémoire :
        détecte si le dernier coup créé un trapèze
        et l'ajoute dans self._memory via add_trapezoid().
        """

        # la detection des trapèzes brisés va permettre ensuite de les restaurer pour pouvoir 
        # continuer a jouer dedans (une fois qu'un trapèze a été setup,la victoire est assuré )

        self.broken_me_trapezoid_detector()
        self.broken_adversary_trapezoid_detector()

        self._memory.me_trapezoid.clear()
        self._memory.me_space_trapezoid.clear()
        self._memory.adversary_trapezoid.clear()
        self._memory.adversary_space_trapezoid.clear()

        my_color = self._memory.get_my_color()
        adversary_color = self._memory.get_adversary_color()  

        # on va parcourir les 3 lignes du bas
        idx_line = 10
        for line in self.lines_down:
            idx_colonne = 2 # indexe de l'element
            # pour chaque ligne, on parcourt les elements
            for pos in line:
                # si un pion est detecté sur la position et qu'il est de la même couleur que le bord
                if(pos != 0 and pos == self.board[15,1]):
                    position_initiale = (idx_line,idx_colonne)
                    shem = self.shematic[str(15-idx_line)]
                    shem_mirror = self.mirror_shematic[str(15-idx_line)]
                    if self.is_a_trapez(position_initiale,shem):
                        self.add_trapezoid(position_initiale,[(t[0], t[1]+position_initiale[0], t[2]+position_initiale[1]) for t in shem],my_color if self.board[position_initiale]==1 else adversary_color)
                    if self.is_a_trapez(position_initiale,shem_mirror):
                        self.add_trapezoid(position_initiale,[(t[0], t[1]+position_initiale[0], t[2]+position_initiale[1]) for t in shem_mirror],my_color if self.board[position_initiale]==1 else adversary_color)
                    
                idx_colonne+=1
            idx_line+=1

        # on va parcourir les 3 lignes du haut (attention on parcourt de gauche à droite)
        idx_line = 5
        for line in self.lines_up:
            idx_colonne = 1 # indexe de l'element
            # pour chaque ligne, on parcourt les elements
            for pos in line:
                # si un pion est detecté sur la position et qu'il est de la même couleur que le bord
                if(self.board[idx_line,idx_colonne] != 0 and self.board[idx_line,idx_colonne] == self.board[0,5]):
                    
                    position_initiale = (idx_line,idx_colonne)
                    shem = self.rotated_shematic180_top[str(idx_line)]
                    shem_mirror = self.rotated_mirror_shematic180_top[str(idx_line)]
                    if self.is_a_trapez(position_initiale,shem):
                        self.add_trapezoid(position_initiale,[(t[0], t[1]+position_initiale[0], t[2]+position_initiale[1]) for t in shem],my_color if self.board[position_initiale]==1 else adversary_color)
                    if self.is_a_trapez(position_initiale,shem_mirror):
                        self.add_trapezoid(position_initiale,[(t[0], t[1]+position_initiale[0], t[2]+position_initiale[1]) for t in shem_mirror],my_color if self.board[position_initiale]==1 else adversary_color)
                idx_colonne+=1
            idx_line-=1

        # on va parcourir les 3 colonnes de gauche
        idx_colonne = 5
        for col in self.rows_left:
            idx_line = 2 # indexe de l'element
            # pour chaque ligne, on parcourt les elements
            for pos in col:
                # si un pion est detecté sur la position et qu'il est de la même couleur que le bord
                if(self.board[idx_line,idx_colonne] != 0 and self.board[idx_line,idx_colonne] == self.board[5,0]):
                    position_initiale = (idx_line,idx_colonne)
                    shem = self.rotated_shematic60_left[str(idx_colonne)]
                    shem_mirror = self.rotated_mirror_shematic60_left[str(idx_colonne)]
                    if self.is_a_trapez(position_initiale,shem):
                        self.add_trapezoid(position_initiale,[(t[0], t[1]+position_initiale[0],t[2]+position_initiale[1]) for t in shem],my_color if pos==1 else adversary_color)
                    if self.is_a_trapez(position_initiale,shem_mirror):
                        self.add_trapezoid(position_initiale,[(t[0], t[1]+position_initiale[0],t[2]+position_initiale[1]) for t in shem_mirror],my_color if pos==1 else adversary_color)
                idx_line+=1
            idx_colonne-=1        

        # on va parcourir les 3 colonnes de droite
        idx_colonne = 10
        for col in self.rows_right:
            idx_line = 2 # indexe de l'element
            # pour chaque ligne, on parcourt les elements
            for pos in col:
                # si un pion est detecté sur la position et qu'il est de la même couleur que le bord
                if(self.board[idx_line,idx_colonne] != 0 and self.board[idx_line,idx_colonne] == self.board[5,15]):
                    position_initiale = (idx_line,idx_colonne)
                    shem = self.rotated_shematic60_right[str(15-idx_colonne)]
                    shem_mirror = self.rotated_mirror_shematic60_right[str(15-idx_colonne)]
                    if self.is_a_trapez(position_initiale,shem):
                        self.add_trapezoid(position_initiale,[(t[0], t[1]+position_initiale[0], t[2]+position_initiale[1]) for t in shem],my_color if pos==1 else adversary_color)
                    if self.is_a_trapez(position_initiale,shem_mirror):
                        self.add_trapezoid(position_initiale,[(t[0], t[1]+position_initiale[0], t[2]+position_initiale[1]) for t in shem_mirror],my_color if pos==1 else adversary_color)
                idx_line+=1
            idx_colonne+=1        
                    
        self._memory.adversary_broken_trapezoid.clear()



    def rotate60_clockwise_2D(self,coord):
        """
        coord : tuple (_,q, r)
        renvoie : tuple (_,q', r') après rotation de 60° clockwise
        """
        x,q, r = coord
        s = -q - r           # calcul du cube complet
        q_rot = -s            # rotation 60° clockwise
        r_rot = -q
        # s_rot = -r           # on n'a pas besoin de s pour la sortie 2D
        return (x,q_rot, r_rot)
    
    def rotate180_clockwise_2D(self,coord):
        """
        coord : tuple (_,q, r)
        renvoie : tuple (_,q', r') après rotation de 180° clockwise
        """
        return (coord[0],coord[1]*(-1), coord[2]*(-1))
    
            
    def is_a_trapez(self,pos_initiale:tuple,liste_espaces_a_verifier:list):
        for i in liste_espaces_a_verifier:
            # rappel liste_espace_a_verifier a des elements de type (0,x,y)
            position = (pos_initiale[0]+i[1],pos_initiale[1]+i[2])

            # verification des bornes 
            if not (0 <= position[0] < self.board.shape[0] and 0 <= position[1] < self.board.shape[1]):
                return False
            if self.board[position] != i[0]:
                return False 
        return True



    def undo(self):
        # Le plateau a déjà été restauré par BoardManager
        # Donc on recalcule simplement les trapezes à partir de l'état actuel
   
        self.update()