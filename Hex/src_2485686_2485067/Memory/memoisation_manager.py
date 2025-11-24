import numpy as np
import random
from typing import Optional
from src_2485686_2485067.Memory.manager import Manager
from typing import override, TYPE_CHECKING
if TYPE_CHECKING:
    from src_2485686_2485067.Memory.memory import Memory


class MemoisationManager(Manager):
    """
    Gère un cache de métriques pour les états du plateau.
    Utilise un hachage de Zobrist pour identifier chaque état.
    Se base directement sur self.board (fourni par Memory).
    """

    def __init__(self, memory: "Memory", board_shape=(16, 16), possible_values=(-1, 0, 1), seed: Optional[int] = None):
        self._cache = {}  # {board_hash: metrics_dict}
        self.memory = memory
        self.last_move = None
        self.board = memory.get_board()
        self.board_shape = board_shape
        self.possible_values = list(possible_values)
        self.hash_stack = []  # pour "undo"

        if seed is not None:
            random.seed(seed)

        # Table de hachage de Zobrist
        self.zobrist_table = {
            (x, y, v): random.getrandbits(64)
            for x in range(board_shape[0])
            for y in range(board_shape[1])
            for v in self.possible_values
        }

        # Hash courant du plateau
        self.current_hash = self._hash_board(self.board)

    # ------------------------------
    # Hachage complet du plateau
    # ------------------------------
    def _hash_board(self, board: np.ndarray) -> int:
        h = 0
        xs, ys = np.nonzero(board)  # indices des cases non vides
        for x, y in zip(xs, ys):
            val = board[x, y]
            h ^= self.zobrist_table[(x, y, val)]
        return h

    # ------------------------------
    # Interface publique du cache
    # ------------------------------
    def get(self, board: np.ndarray = None):
        """Récupère les métriques d’un plateau donné (ou du plateau courant)."""
        key = self.current_hash if board is None else self._hash_board(board)
        return self._cache.get(key)

    def set(self, board: np.ndarray = None, metrics=None):
        """Stocke les métriques d’un plateau donné (ou du plateau courant)."""
        key = self.current_hash if board is None else self._hash_board(board)
        self._cache[key] = metrics

    def contains(self, board: np.ndarray = None) -> bool:
        """Vérifie si un plateau est déjà dans le cache."""
        key = self.current_hash if board is None else self._hash_board(board)
        return key in self._cache

    def clear(self):
        """Vide le cache et réinitialise l’historique."""
        self._cache.clear()
        self.hash_stack.clear()
        self.current_hash = self._hash_board(self.board)

    # ------------------------------
    # Gestion incrémentale du hash
    # ------------------------------
    @override
    def update(self, s=None):
        """
        Met à jour le hash courant après un coup.
        Suppose que self.board a déjà été mis à jour.
        Nécessite que self.last_move et self.previous_value soient définis avant l'appel.
        """
        self.last_move = self.memory.move_history.peek()
        # Si aucun coup n’a été joué (début de partie)
        if not hasattr(self, "last_move") or self.last_move is None:
            self.current_hash = self._hash_board(self.board)
            self.hash_stack = [self.current_hash]
            return

        x, y = self.last_move
        old_val = getattr(self, "previous_value", 0)
        new_val = self.board[x, y]

        # Si c’est la première fois qu’on met à jour
        if not self.hash_stack:
            self.current_hash = self._hash_board(self.board)

        # Mise à jour incrémentale du hash (Zobrist)
        if old_val != 0:
            self.current_hash ^= self.zobrist_table[(x, y, old_val)]
        if new_val != 0:
            self.current_hash ^= self.zobrist_table[(x, y, new_val)]

        # Sauvegarde du hash courant
        self.hash_stack.append(self.current_hash)


    @override
    def undo(self):
        """Annule la dernière mise à jour du hash (ex : retour dans un arbre de recherche)."""
        if self.hash_stack:
            self.hash_stack.pop()
            self.current_hash = self.hash_stack[-1] if self.hash_stack else self._hash_board(self.board)
