# Copyright (c) 2025
# Licensed under the MIT License.
# See LICENSE file for details.

from collections import defaultdict
from typing import override
import itertools
from game_state_hex import GameState
from src_2485686_2485067.Memory.manager import Manager
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from my_player import MyPlayer  # import uniquement pour l'IDE
if TYPE_CHECKING:
    from src_2485686_2485067.Memory.memory import Memory

class MaillonsManager(Manager):
    """
    Gère les maillons du jeu de Hex.
    Un maillon est défini par :
        - 2 jetons du même joueur
        - 2 cases vides entre eux
    Permet :
        - d'ajouter un maillon
        - de le supprimer
        - de savoir si un pion appartient à un maillon
        - de supprimer les maillons invalidés
    """

    def __init__(self, _memory :"Memory"):
        self._memory = _memory


    def add_links(self,p1, p2, spaces,color):
        """
        Ajoute un maillon reliant deux pions avec deux cases d'espacement vides.

        Args:
            p1 (tuple[int, int]): position du premier pion
            p2 (tuple[int, int]): position du second pion
            spaces (list[tuple[int, int]]): deux positions vides entre les pions
        """
        key = frozenset({p1, p2})

        if key in self._memory.me_links or key in self._memory.adversary_links:
            # Le maillon existe déjà → on ne le recrée pas
            return

        # Mes maillons
        if self._memory.get_my_color() == color:
            self._memory.me_links[key] = tuple(spaces)
            for space in spaces:
                self._memory.me_space_links.setdefault(space, set()).add(key)

        # Maillons de l'adversaire
        else:
            self._memory.adversary_links[key] = tuple(spaces)
            for space in spaces:
                self._memory.adversary_space_links.setdefault(space, set()).add(key)



    # -------------------------
    # 🔹 SUPPRESSION D'UN MAILLON
    # -------------------------
    def remove_maillon(self, p1, p2,color):
        """
        Supprime un maillon et met à jour les liens inverses.
        """
        key = frozenset({p1, p2})
        if key not in self._memory.me_links and key not in self._memory.adversary_links:
            return
        
        if self._memory.get_my_color() == color:
            for space in self._memory.me_links[key]:
                maillons_for_space = self._memory.me_space_links.get(space)
                if maillons_for_space:
                    maillons_for_space.discard(key)
                    if not maillons_for_space:  # nettoyage
                        del self._memory.me_space_links[space]
            del self._memory.me_links[key]

        else:
            for space in self._memory.adversary_links[key]:
                maillons_for_space = self._memory.adversary_space_links.get(space)
                if maillons_for_space:
                    maillons_for_space.discard(key)
                    if not maillons_for_space:  # nettoyage
                        del self._memory.adversary_space_links[space]
            del self._memory.adversary_links[key]

    # ---------------------------------
    # 🔹 NETTOYAGE : SUPPRIMER LES MAILLONS INVALIDES
    # ---------------------------------
    def broken_me_links_detector(self):
        """
        Détecte si un maillon du joueur a été détruit par le dernier coup de l'adversaire.
        Les maillons détectés sont ajoutés dans self._memory.me_broken_links.
        """
        self._memory.me_broken_links.clear()

        last_move = self._memory.move_history.peek()
        if last_move is None:
            return

        # Vérifie si la case jouée appartient à un des espaces de MES maillons
        if last_move in self._memory.get_me_space_links():
            for maillon in self._memory.get_me_space_links()[last_move]:
                self._memory.me_broken_links.append({
                    "maillon": maillon,
                    "espaces": self._memory.me_links[maillon],
                    "cassé_par": last_move
                })

    def broken_adversary_links_detector(self):
        self._memory.adversary_broken_links.clear()
        last_move = self._memory.move_history.peek()
        if last_move is None:
            return

        # Vérifie si la case jouée appartient à un des espaces des mailons adverses
        if last_move in self._memory.get_adversary_space_links():
            for maillon in self._memory.get_adversary_space_links()[last_move]:
                self._memory.adversary_broken_links.append({
                    "maillon": maillon,
                    "espaces": self._memory.adversary_links[maillon],
                    "cassé_par": last_move
                })


    # ---------------------------------
    # 🔹 AFFICHAGE / DEBUG
    # ---------------------------------
    def self_print(self):
        """
        Affiche les maillons connus pour debug.
        """
        print("=== MAILLONS ===")
        for m, spaces in self._memory.me_links.items():
            print(f"Maillon {tuple(m)} → Espaces {spaces}")

        print("\n=== ESPACEMENTS ===")
        for e, maillons in self._memory.me_space_links.items():
            print(f"Espace {e} → Maillons {maillons}")



    @override
    def update(self,s:GameState = None):
        """
        Parcourt le plateau et met à jour la mémoire :
        détecte tous les maillons valides (2 pions + 2 espaces vides)
        et les ajoute dans self._memory via add_links().
        """

        board = self._memory.get_board()
        hauteur, largeur = board.shape
        self.broken_me_links_detector()
        self.broken_adversary_links_detector()

        # Directions hexagonales à distance 2 (structure inchangée)
        directions = {
            "NORTH":       ((-2, 1), (-1, 0), (-1, 1)),
            "SOUTH":       ((2, -1), (1, 0), (1, -1)),
            "NORTH_EAST":  ((-1, 2), (-1, 1), (0, 1)),
            "NORTH_WEST":  ((-1, -1), (-1, 0), (0, -1)),
            "SOUTH_EAST":  ((1, 1), (1, 0), (0, 1)),
            "SOUTH_WEST":  ((1, -2), (1, -1), (0, -1))
        }

        # Nettoyage des anciens maillons avant mise à jour
        self._memory.me_links.clear()
        self._memory.me_space_links.clear()
        self._memory.adversary_links.clear()
        self._memory.adversary_space_links.clear()

        # Parcours du plateau
        for i in range(hauteur):
            for j in range(largeur):
                current = board[i, j]
                if current == 0:
                    continue  # case vide → on ignore

                # Pour chaque direction possible
                for _, ((di, dj), (di1, dj1), (di2, dj2)) in directions.items():
                    p1 = (i, j)
                    p2 = (i + di, j + dj)
                    space1 = (i + di1, j + dj1)
                    space2 = (i + di2, j + dj2)

                    # Vérification des bornes
                    if not (0 <= p2[0] < hauteur and 0 <= p2[1] < largeur):
                        continue
                    if not (0 <= space1[0] < hauteur and 0 <= space1[1] < largeur):
                        continue
                    if not (0 <= space2[0] < hauteur and 0 <= space2[1] < largeur):
                        continue

                    # Test de validité du maillon
                    if board[p2] == current and board[space1] == 0 and board[space2] == 0:
                        color = self._memory.get_my_color() if current == 1 else self._memory.get_adversary_color()
                        self.add_links(p1, p2, [space1, space2], color)


                    
    @override
    def undo(self):
        # Le plateau a déjà été restauré par BoardManager
        # Donc on recalcule simplement les maillons à partir de l'état actuel
        self.update()