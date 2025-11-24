# Copyright (c) 2025
# Licensed under the MIT License.
# See LICENSE file for details.

from collections import deque
import numpy as np

class UniqueStack:
    def __init__(self):
        self._stack = deque()
        self._set = set()

    def _make_key(self, item):
        """Crée une clé hashable selon le type de l'item"""
        if isinstance(item, np.ndarray):
            return item.tobytes()  # pour les tableaux numpy
        elif isinstance(item, tuple):
            return item  # les tuples sont déjà hashables
        else:
            raise TypeError(f"Type non supporté pour UniqueStack: {type(item)}")

    def push(self, item):
        key = self._make_key(item)
        if key not in self._set:
            self._stack.append(item.copy() if isinstance(item, np.ndarray) else item)
            self._set.add(key)

    def pop(self):
        if not self._stack:
            raise IndexError("Stack is empty")
        item = self._stack.pop()
        key = self._make_key(item)
        self._set.remove(key)
        return item
    
    def peek(self):
        """Retourne le dernier élément sans le retirer."""

        if not self._stack:
            return None
        return self._stack[-1]

    def __iter__(self):
        """Permet d'itérer directement sur les éléments du stack."""
        return iter(self._stack)
    
    def len(self):
        return len(self._stack)

    def isEmpty(self):
        """Retourne True si le stack est vide, sinon False."""
        return len(self._stack) == 0