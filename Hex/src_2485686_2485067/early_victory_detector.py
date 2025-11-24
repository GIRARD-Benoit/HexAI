# Copyright (c) 2025
# Licensed under the MIT License.
# See LICENSE file for details.

# from collections import deque
# from typing import TYPE_CHECKING
# if TYPE_CHECKING:
#     from src_2485686_2485067.Memory.memory import Memory  # import uniquement pour l'IDE

# class EarlyVictory:
#     #Todo a remplacer par une structure Union Find
#     def __init__(self, _memory:"Memory"):
#         self._memory = _memory
#         self.board = _memory.get_board()
#         self.size = _memory.BOARD_SIZE + 2  # inclut les bords
#         self.switch = True

#     def check_victory(self):
#         if self.switch:
#             score = self._bfs()
#             if score == True:  # MAX
#   #              print("EARLY VICTORY DETECTED MAX PLAYER")
#   #              print(self._memory.move_history.peek())
#                 return float("+inf")
#             else:
#                 return 0
#         else:
#             return None
        
#     def deactivate(self):
#         self.switch = False

#     def _bfs(self):
#         visited = set()
#         my_queue = deque()

#         # Départ selon l’orientation
#         links = self._memory.get_me_links()
#         if self._memory.get_my_color() == 'B':
#             # MAX : gauche → droite (horizontal)
#             my_queue.extend([(i, 0) for i in range(self.size) if self.board[i, 0] == 1])
#             goal = lambda x, y: y == self.size - 1
            
#         else:
#             # MAX : haut → bas (vertical)
#             my_queue.extend([(0, j) for j in range(self.size) if self.board[0, j] == 1])
#             goal = lambda x, y: x == self.size - 1 #goal est une fonction
#         my_queue.extend(next(iter(k)) for k in self._memory.get_me_trapezoid().keys() if len(k) == 1) #extrait les frozensets

#         while my_queue:
#             x, y = my_queue.popleft()
#             if (x, y) in visited:
#                 continue
#             visited.add((x, y))

#             if goal(x, y):
#                 return True

#             # voisins hex adjacents
#             for dx, dy in [(-1,0),(1,0),(0,-1),(0,1),(-1,1),(1,-1)]:
#                 nx, ny = x + dx, y + dy
#                 if 0 <= nx < self.size and 0 <= ny < self.size:
#                     if (nx, ny) not in visited and self.board[nx, ny] == 1:
#                         my_queue.append((nx, ny))

#             # voisins via maillons
#             for link, spaces in links.items():
#                 if (x, y) in link:
#                     for px, py in link:
#                         if (px, py) not in visited and self.board[px, py] == 1:
#                             my_queue.append((px, py))

#         return False

#     def to_json(self):
#         """Sérialisation JSON compatible Seahorse."""
#         return {
#             "history": []
#         }
    
# ANCIENNE VERSION 

################# STRUCTURE UNION FIND 

# NOUVELLE VERSION 
# Copyright (c) 2025
# Licensed under the MIT License.
# See LICENSE file for details.

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from src_2485686_2485067.Memory.memory import Memory


class UnionFind:
    """Structure Union-Find (disjoint-set) avec compression de chemin et union par rang."""
    def __init__(self, n: int):
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, x: int) -> int:
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, x: int, y: int):
        rx, ry = self.find(x), self.find(y)
        if rx == ry:
            return
        if self.rank[rx] < self.rank[ry]:
            self.parent[rx] = ry
        elif self.rank[rx] > self.rank[ry]:
            self.parent[ry] = rx
        else:
            self.parent[ry] = rx
            self.rank[rx] += 1


class EarlyVictory:
    """
    Détection de victoire précoce basée sur Union-Find.
    Fonctionne pour les deux joueurs (MAX et MIN).
    """
    def __init__(self, _memory: "Memory"):
        self._memory = _memory
        self.board = _memory.get_board()
        self.size = _memory.BOARD_SIZE + 2  # inclut les bords
        self.switch = True

    def deactivate(self):
        """Désactive la détection (utile pour certaines étapes du jeu)."""
        self.switch = False

    def check_victory(self):
        """Renvoie +inf si 'moi' gagne, -inf si l'adversaire gagne, sinon 0."""
        if not self.switch:
            return None

        my_color = self._memory.get_my_color()
        opp_color = 'R' if my_color == 'B' else 'B'

        # Victoire de mon camp ?
        if self._union_find_victory(my_color):
            return 40000

        # Victoire de l'adversaire ?
        if self._union_find_victory(opp_color):
            return -40000

        return 0

    def _union_find_victory(self, color: str) -> bool:
        """
        Teste si le joueur de la couleur donnée a relié ses deux bords.
        Utilise Union-Find pour explorer les connexions.
        """
        n = self.size * self.size
        uf = UnionFind(n + 4)  # +4 pour TOP, BOTTOM, LEFT, RIGHT
        board = self.board
        me = 1 if color == self._memory.get_my_color() else -1  

        # Indices virtuels
        TOP, BOTTOM, LEFT, RIGHT = n, n + 1, n + 2, n + 3
        index = lambda i, j: i * self.size + j

        # Union des voisins adjacents de même couleur
        for i in range(self.size):
            for j in range(self.size):
                if board[i, j] != me:
                    continue
                for di, dj in [(-1,0),(1,0),(0,-1),(0,1),(-1,1),(1,-1)]:
                    ni, nj = i + di, j + dj
                    if 0 <= ni < self.size and 0 <= nj < self.size and board[ni, nj] == me:
                        uf.union(index(i, j), index(ni, nj))

        # Union via les structures (liens + trapèzes)
        def union_cells(cells):
            cells_list = []
            for c in cells:
                if isinstance(c, (tuple, list)) and len(c) == 2:
                    cells_list.append(tuple(c))
            for a in cells_list:
                for b in cells_list:
                    uf.union(index(a[0], a[1]), index(b[0], b[1]))

        # Récupération des structures selon la couleur
        if color == self._memory.get_my_color():
            links = self._memory.get_me_links()
            traps = self._memory.get_me_trapezoid()
        else:
            links = self._memory.get_adversary_links()
            traps = self._memory.get_adversary_trapezoid()

        for link in links.keys():
            union_cells(link)
            
        for trap in traps.keys():
            if len(trap) > 1:
                # Trapèze classique → union interne
                union_cells(trap)
            else:
                # Trapèze unitaire → relier au bord le plus proche selon sa couleur
                (i, j) = next(iter(trap))
                if color == 'R':  # Rouge : haut ↔ bas
                    # On regarde le plus proche entre TOP et BOTTOM
                    dist_top = i
                    dist_bottom = self.size - 1 - i
                    if dist_top <= dist_bottom:
                        uf.union(index(i, j), TOP)
                    else:
                        uf.union(index(i, j), BOTTOM)

                elif color == 'B':  # Bleu : gauche ↔ droite
                    # On regarde le plus proche entre LEFT et RIGHT
                    dist_left = j
                    dist_right = self.size - 1 - j
                    if dist_left <= dist_right:
                        uf.union(index(i, j), LEFT)
                    else:
                        uf.union(index(i, j), RIGHT)

        # Connexion aux bords virtuels selon la couleur
        if color == 'R':
            # Rouge relie haut → bas
            for j in range(self.size):
                if board[0, j] == me:
                    uf.union(index(0, j), TOP)
                if board[self.size - 1, j] == me:
                    uf.union(index(self.size - 1, j), BOTTOM)
            return uf.find(TOP) == uf.find(BOTTOM)

        else:
            # Bleu relie gauche → droite
            for i in range(self.size):
                if board[i, 0] == me:
                    uf.union(index(i, 0), LEFT)
                if board[i, self.size - 1] == me:
                    uf.union(index(i, self.size - 1), RIGHT)
            return uf.find(LEFT) == uf.find(RIGHT)

    def to_json(self):
        """Sérialisation JSON compatible Seahorse."""
        return {"history": []}


    def debug_print(self):
        """
        Affiche un résumé compact de l'état interne pour le debug.
        Donne les métriques clés : plateau, couleurs, unions, tailles.
        """
        my_color = self._memory.get_my_color()
        opp_color = 'R' if my_color == 'B' else 'B'
        board = self.board

        total_cells = self.size * self.size
        my_cells = int((board == 1).sum())
        opp_cells = int((board == -1).sum())
        empty_cells = int((board == 0).sum())

        me_links = len(self._memory.get_me_links())
        adv_links = len(self._memory.get_adversary_links())
        me_traps = len(self._memory.get_me_trapezoid())
        adv_traps = len(self._memory.get_adversary_trapezoid())

        print("\n🧩 [EarlyVictory DEBUG SUMMARY]")
        print(f"  • Plateau : {self.size}x{self.size}  ({total_cells} cases)")
        print(f"  • Couleur moi : {my_color} | Adversaire : {opp_color}")
        print(f"  • Cases → Moi : {my_cells} | Adv : {opp_cells} | Vides : {empty_cells}")
        print(f"  • Liens → Moi : {me_links} | Adv : {adv_links}")
        print(f"  • Trapèzes → Moi : {me_traps} | Adv : {adv_traps}")
        print(f"  • Switch actif : {self.switch}")

        # Vérification rapide de connexion
        my_win = self._union_find_victory(my_color)
        opp_win = self._union_find_victory(opp_color)

        print(f"  • Connexion {my_color} : {'✅' if my_win else '❌'}")
        print(f"  • Connexion {opp_color} : {'✅' if opp_win else '❌'}")
        print("───────────────────────────────────────")