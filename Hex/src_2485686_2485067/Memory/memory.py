# Copyright (c) 2025
# Licensed under the MIT License.
# See LICENSE file for details.

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from my_player import MyPlayer  # import uniquement pour l'IDE

import numpy as np
from seahorse.game.game_state import GameState
from src_2485686_2485067.Memory.maillons_manager import MaillonsManager
from src_2485686_2485067.Memory.board_manager import BoardManager
from src_2485686_2485067.Memory.attention_manager import Attention_manager
from src_2485686_2485067.Memory.trapezoid_manager import TrapezoidManager
from src_2485686_2485067.Memory.memoisation_manager import MemoisationManager
from src_2485686_2485067.Memory.UniqueStack import UniqueStack
from src_2485686_2485067.Metrics.distance import Distance




class Memory():
    BOARD_SIZE = 14

    def __init__(self :"MyPlayer",joueur):
        self.joueur = joueur

        # GENERAL
        self.my_color = joueur.get_piece_type()
        self.adversary_color = "R" if self.my_color == "B" else "B"

        # BOARD MANAGER 
        self.board = np.zeros((self.BOARD_SIZE+2, self.BOARD_SIZE+2), dtype=int) # represente le plateau
        # Remplissage des bords :
        # Bord supérieur (ligne 0) et bord inférieur (ligne -1)
        self.board[0, :] = 1 if self.my_color == "R" else -1    # ou -1 selon le joueur que tu veux représenter
        self.board[-1, :] = 1 if self.my_color == "R" else -1   # bord inférieur

        # Bord gauche (colonne 0) et bord droit (colonne -1)
        self.board[:, 0] = 1 if self.my_color == "B" else -1    # bord gauche
        self.board[:, -1] = 1 if self.my_color == "B" else -1   # bord droit

        self.move_history = UniqueStack() # tuple(x,y) historique des coups
        self.last_move = None # Permet de stocker le dernier coup retiré lors d'un pop 
        # MAILLON MANAGER

        # Mes maillons
        self.me_links = dict() # maillons -> espacement   clé : frozenset() || valeur : [(x1,y1),(x2,y2)]
        self.me_space_links= dict() # espacement -> maillons clé : (x_espacement,y_espacement) || valeur : frozenset()
        self.me_broken_links = list() # tampon pour mes propres maillons détruits par un pion adverse
        
        # Maillons de l'adversaire
        self.adversary_links = dict() # maillons -> espacement   clé : frozenset() || valeur : [(x1,y1),(x2,y2)]
        self.adversary_space_links= dict() # espacement -> maillons clé : (x_espacement,y_espacement) || valeur : frozenset()
        self.adversary_broken_links = list() # tampon pour les maillons adverses détruits par un pion 

        # Mes trapezes
        self.me_trapezoid = dict() # maillons -> espacement   clé : frozenset() || valeur : [(x1,y1),(x2,y2)]
        self.me_space_trapezoid= dict() # espacement -> maillons clé : (x_espacement,y_espacement) || valeur : frozenset()
        self.me_broken_trapezoid = list() # tampon pour mes propres maillons détruits par un pion adverse
        
        # trapezes de l'adversaire
        self.adversary_trapezoid = dict() # maillons -> espacement   clé : frozenset() || valeur : [(x1,y1),(x2,y2),...]
        self.adversary_space_trapezoid= dict() # espacement -> maillons clé : (x_espacement,y_espacement) || valeur : frozenset()
        self.adversary_broken_trapezoid = list() # tampon pour les maillons adverses détruits par un pion 

        # Matrice d'attention 
        self.attention_board = np.full((self.BOARD_SIZE + 2, self.BOARD_SIZE + 2), -np.inf, dtype=float) # softmax
        """
        Entre 0 et 1
        """
        self.attention_board_history = list() # Pas de softmax effectué sur les elements
        """
        Elements entre -inf et x
        """

        # cache pour l'heuristique (reset à chaque nouveau coup a jouer)
        self.heuristique_cache = MemoisationManager(self)

        # MAJ par le module distance de Metrics
        self.my_critical_path = None
        self.adversary_critical_path = None
        self.my_distance = None
        self.adversary_distance = None

        # Memory manager 
        self.manager = [BoardManager(self),MaillonsManager(self),Distance(joueur),TrapezoidManager(self),Attention_manager(self),self.heuristique_cache] # Attention l'ordre compte


    def to_json(self):
        """Sérialisation JSON compatible Seahorse."""
        return {
            "history": []
        }
    
    # -------------------------
    # 🔹 GETTERS
    # -------------------------
    def is_mine(self,position):
        if self.board[position] == 1: return True
        return False

    def is_empty(self,position):
        if self.board[position] == 0: return True
        return False

    def is_adversary(self,position):
        if self.board[position] == -1: return True
        return False

    def get_my_color(self) -> str:
        """Retourne la couleur du joueur ('R' ou 'B')."""
        return self.my_color

    def get_adversary_color(self) -> str:
        """Retourne ma couleur."""
        return self.adversary_color

    def get_board(self) -> np.ndarray:
        """Retourne le plateau actuel."""
        return self.board
    
    # Maillons

    def get_me_links(self) -> dict:
        """Retourne la correspondance maillon <-> espaces"""
        return self.me_links
    
    def get_adversary_links(self) -> dict:
        """
        Retourne le dictionnaire des maillons de l’adversaire.
        Clé : frozenset des positions des deux pions adverses.
        Valeur : liste des coordonnées de l’espacement [(x1, y1), (x2, y2)].
        """
        return self.adversary_links

    def get_me_space_links(self) -> dict:
        """
        Retourne le dictionnaire des espacements de mes maillons.
        Clé : coordonnée de l’espace vide (x, y).
        Valeur : frozenset représentant le maillon correspondant.
        """
        return self.me_space_links
    
    def get_me_broken_links(self) -> list:
        """ Retourne les maillons cassés par l'adversaire"""
        return self.me_broken_links
    
    def get_adversary_broken_links(self) -> list:
        """ Retourne les maillons brisés de l'adversaire par mes propres coups """
        return self.adversary_broken_links

    def get_adversary_space_links(self) -> dict:
        """
        Retourne le dictionnaire des espacements des maillons adverses.
        Clé : coordonnée de l’espace vide (x, y).
        Valeur : frozenset représentant le maillon correspondant.
        """
        return self.adversary_space_links

    # Trapèzes

    def get_me_trapezoid(self) -> dict:
        """Retourne la correspondance trapeze <-> espaces   pivots -> espaces vides""" 
        return self.me_trapezoid
    
    def get_adversary_trapezoid(self) -> dict:
        """
        Retourne le dictionnaire des maillons de l’adversaire.
        Clé : frozenset des positions des deux pions adverses.
        Valeur : liste des coordonnées de l’espacement [(x1, y1), (x2, y2)].
        """
        return self.adversary_trapezoid
    
    def get_me_space_trapezoid(self) -> dict:
        """
        Retourne le dictionnaire des espacements de mes maillons.
        Clé : coordonnée de l’espace vide (x, y).
        Valeur : frozenset représentant le trapeze(pivot) correspondant. espace_vide -> pivot
        """
        return self.me_space_trapezoid
    
    def get_adversary_space_trapezoid(self) -> dict:
        """
        Retourne le dictionnaire des espacements des trapezes adverses.
        Clé : coordonnée de l’espace vide (x, y).
        Valeur : frozenset représentant le trapeze(pivot) correspondant. espace_vide -> pivot
        """
        return self.adversary_space_trapezoid
    
    def get_me_broken_trapezoid(self) -> list:
        """ Retourne les trapezes cassés par l'adversaire"""
        return self.me_broken_trapezoid
    
    def get_adversary_broken_trapezoid(self) -> list:
        """ Retourne les trapezes brisés de l'adversaire par mes propres coups """
        return self.adversary_broken_trapezoid




    
    def get_move_history(self):
        """Retourne la liste de coups joués"""
        return self.move_history

    def get_attention_board(self):
        """
        Retourne la matrice d'attention
        """
        return self.attention_board
    
    def get_attention_history(self):
        """
        Retourne l'historique de la matrice d'attention
        """
        return self.attention_board_history


    
    def update(self,current_state:GameState):
        """
        Update de la mémoire p
        """
        for elem_manager in self.manager:
            elem_manager.update(current_state)
        return
    
    def undo(self):
        """ revient à l'état précédent """
        for elem_manager in self.manager:
            elem_manager.undo()
        return





    def print__memory(self):
        """
        Affiche l’état complet de la mémoire :
        - le plateau (board)
        - les maillons du joueur et de l’adversaire
        - les correspondances maillon ↔ espaces
        - l’historique des coups
        """

        print("\n" + "=" * 60)
        print(f"🧠  MÉMOIRE DU JOUEUR ({self.my_color})")
        print("=" * 60)

        # --- Plateau ---
        print("\n📦 Plateau actuel :")
        self.print_hex_board()
        print("\n📦 Plateau d'attention :")
        self.print_attention_board()

        # --- Historique ---
        print("\n🕓 Historique complet des coups :")
        if not self.move_history:
            print("  Aucun coup enregistré.")
        else:
            # move_history est une UniqueStack, on peut itérer dessus
            for idx, move in enumerate(self.move_history):
                print(f"  Coup {idx + 1} : {move}")

        # --- Maillons du joueur ---
        print("\n🔗 Mes maillons :", len(self.me_links))
        if not self.me_links:
            print("  Aucun maillon enregistré.")
        else:
            for m, spaces in self.me_links.items():
                print(f"  {tuple(m)} → Espaces : {spaces}")

        print("\n⬜ Espacements associés (moi) :", len(self.me_space_links))
        if not self.me_space_links:
            print("  Aucun espacement enregistré.")
        else:
            for space, linked in self.me_space_links.items():
                print(f"  {space} → Maillons : {linked}")

        # --- Maillons de l’adversaire ---
        print("\n🔴 Maillons adverses :", len(self.adversary_links))
        if not self.adversary_links:
            print("  Aucun maillon adverse enregistré.")
        else:
            for m, spaces in self.adversary_links.items():
                print(f"  {tuple(m)} → Espaces : {spaces}")

        print("\n⬛ Espacements associés (adversaire) :", len(self.adversary_space_links))
        if not self.adversary_space_links:
            print("  Aucun espacement enregistré.")
        else:
            for space, linked in self.adversary_space_links.items():
                print(f"  {space} → Maillons : {linked}")

        # --- Maillons détruits ---
        print("\n💥 mes maillons cassés :", len(self.me_broken_links))
        if not self.me_broken_links:
            print("  Aucun maillon détruit.")
        else:
            for m in self.me_broken_links:
                print(f"  {m}")
        
        print("\n💥 maillons adversaire cassés :", len(self.adversary_broken_links))
        if not self.adversary_broken_links:
            print("  Aucun maillon détruit.")
        else:
            for m in self.adversary_broken_links:
                print(f"  {m}")

        print("=" * 60 + "\n")

        # --- Trapèzes du joueur ---
        print("\n🔺 Mes trapèzes :", len(self.me_trapezoid))
        if not self.me_trapezoid:
            print("  Aucun trapèze enregistré.")
        else:
            for pivot, spaces in self.me_trapezoid.items():
                print(f"  Pivot {tuple(pivot)} → Espaces : {spaces}")

        print("\n⬜ Espacements associés aux trapèzes (moi) :", len(self.me_space_trapezoid))
        if not self.me_space_trapezoid:
            print("  Aucun espacement de trapèze enregistré.")
        else:
            for space, pivot_set in self.me_space_trapezoid.items():
                print(f"  Espace {space} → Pivots : {pivot_set}")

        # --- Trapèzes de l’adversaire ---
        print("\n🔻 Trapèzes adverses :", len(self.adversary_trapezoid))
        if not self.adversary_trapezoid:
            print("  Aucun trapèze adverse enregistré.")
        else:
            for pivot, spaces in self.adversary_trapezoid.items():
                print(f"  Pivot {tuple(pivot)} → Espaces : {spaces}")

        print("\n⬛ Espacements associés aux trapèzes adverses :", len(self.adversary_space_trapezoid))
        if not self.adversary_space_trapezoid:
            print("  Aucun espacement de trapèze adverse enregistré.")
        else:
            for space, pivot_set in self.adversary_space_trapezoid.items():
                print(f"  Espace {space} → Pivots : {pivot_set}")

        # --- Trapèzes détruits ---
        print("\n💥 mes trapèzes cassés :", len(self.me_broken_trapezoid))
        if not self.me_broken_trapezoid:
            print("  Aucun trapèze détruit.")
        else:
            for t in self.me_broken_trapezoid:
                print(f"  {t}")

        print("\n💥 trapèzes adverses cassés :", len(self.adversary_broken_trapezoid))
        if not self.adversary_broken_trapezoid:
            print("  Aucun trapèze détruit.")
        else:
            for t in self.adversary_broken_trapezoid:
                print(f"  {t}")

         # --- Chemins critiques & distances ---
        print("\n📏 DISTANCES & CHEMINS CRITIQUES")
        print("-" * 60)
        print(f"🧩 Distance (moi) : {self.my_distance}")
        print(f"🧩 Distance (adversaire) : {self.adversary_distance}")

        print(f"🛣️ Chemin critique (moi) : {self.my_critical_path}")
        print(f"🛣️ Chemin critique (adversaire) : {self.adversary_critical_path}")
        print("=" * 60 + "\n")

    def print_hex_board(self):
        n = self.board.shape[0]
        for i in range(n):
            offset = " " * i  # décalage pour forme hexagonale
            row = []
            for j in range(n):
                val = self.board[i, j]
                if val == 1:
                    row.append(self.get_my_color()) 
                elif val == -1:
                    row.append(self.get_adversary_color())  
                else:
                    row.append(".")
            print(offset + " ".join(row))

    def print_attention_board(self):
        n = self.attention_board.shape[0]
        for i in range(n):
            offset = " " * i  # décalage pour forme hexagonale
            row = []
            for j in range(n):
                val = self.attention_board[i, j]
                row.append(str(val))
            print(offset + " ".join(row))
