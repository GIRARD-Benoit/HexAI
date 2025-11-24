# Copyright (c) 2025
# Licensed under the MIT License.
# See LICENSE file for details.

from seahorse.game.game_state import GameState
from game_state_hex import GameStateHex
from src_2485686_2485067.algorithme import Algorithme
from typing import override
from src_2485686_2485067.heuristique import Heuristique
from seahorse.game.light_action import LightAction
import numpy as np
import random
from src_2485686_2485067.Memory.memory import Memory
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from my_player import MyPlayer  # import uniquement pour l'IDE


import time 

class Algorithme_minimax_alpha_beta_typeA(Algorithme):

    

    def __init__(self,joueur:"MyPlayer", heuristique:Heuristique):
        super().__init__()
        self.default_mode_branching_factor = 30
        self.maximum_depth = None
        self.local_analysis_area = None 
        self.branching_factor = self.default_mode_branching_factor
        self.heuristique = heuristique
        self.joueur = joueur
        self.matrice_debug = np.zeros((16, 16))

        
    @override
    def execute(self,s0: GameStateHex,max_depth,local_analysis_area = None,branching_factor = None):
        self.matrice_debug = np.zeros((16, 16))
        # MAJ avec le coup joué par l'adversaire 
        self.maximum_depth = max_depth
        self.local_analysis_area = local_analysis_area
        if branching_factor != None:
            self.branching_factor = branching_factor
        else:
            self.branching_factor = self.default_mode_branching_factor

        
        self.heuristique.coefficient_update()
        
        if local_analysis_area is not None: # uniquement dans le cadre d'une recherche locale 
            self.joueur._heuristique.beta_me = 0
            self.joueur._heuristique.alpha = 100
            self.joueur._heuristique.delta = 0
            self.joueur._heuristique.beta_adv = 0
            self.joueur._heuristique.epsilon = 0
            self.joueur._heuristique.zeta = 100

        # uniquement le premier coup
        first_move = self.joueur._forced_move.first_move()
        if first_move is not None:
            return 0,first_move
        
        (v, m) = self.__maxValue(s0, float("-inf"),float("+inf"),depth = 0)
        # Si jamais c'est vide, on génère un choix aléatoire
        if m is None:
            possible = list(s0.generate_possible_light_actions())
            m = random.choice(possible)
        # sécurité (forcer la conversion en cas de besoin)
        if not isinstance(m, LightAction):
            m = LightAction({"piece": self.joueur._memory.get_my_color(), "position": m})

        return (v,m)

    def __maxValue(self,s: GameStateHex,alpha,beta,depth):
        # si la profondeur de recherche est atteint
        if depth == self.maximum_depth:
            return self.heuristique.execute(), None
        
        #  Verification d'un coup forcé 
        _forced_move = self.joueur._forced_move.find_me_forced_move(self.local_analysis_area,s,depth)
        if _forced_move is not None:
            light_action = _forced_move  
            s_prime = s.apply_action(light_action)
            self.joueur._memory.update(s_prime)
            v, _ = self.__minValue(s_prime, alpha, beta, depth + 1)
            self.joueur._memory.undo()
            return v, light_action

        v_star = float("-inf")
        m_star = None

        # selection des meilleures actions
        actions = list(s.generate_possible_light_actions())
        actions = self.actions_selection(actions)

        if len(actions) == 0:
            return self.heuristique.execute(), None
        for move in actions:
            # si on sort de la zone d'intérêt, on de descend plus
            if move == "annex":
                if depth == 0:
                    self.maximum_depth = 2
                continue

            s_prime = s.apply_action(move)
             # MAJ Memory
            self.joueur._memory.update(s_prime)
           
            # Condition d'arrêt (finale)
            if s_prime.is_done():
                score = self.__evaluate(s_prime)
                self.joueur._memory.undo()
                return score, move  # score est +inf si MAX gagne, -inf si MIN gagne

            # on cherche la valeur minimum a la profondeur suivante 
            
            (v,_) = self.__minValue(s_prime,alpha,beta, depth + 1)

            # On cherche la plus grande valeur a la profondeur n+1
            if(v > v_star):
                v_star = v
                m_star = move
                alpha = max(alpha,v_star)
            self.joueur._memory.undo()
            if (v_star >= beta): return (v_star,m_star) #pruning
        # if(depth == 0):
        #     print("coup joué :",v_star,m_star)
        return (v_star,m_star)

    def __minValue(self,s: GameStateHex,alpha,beta,depth):
        # si la profondeur de recherche est atteint
        if depth == self.maximum_depth:
            return self.heuristique.execute(), None

        #  Verification d'un coup forcé (_forced_move est de type light_action)
        _forced_move = self.joueur._forced_move.find_adversary_forced_move(self.local_analysis_area,s,depth)
        if _forced_move is not None:
            light_action = _forced_move  
            s_prime = s.apply_action(light_action)
            self.joueur._memory.update(s_prime)
            v, _ = self.__maxValue(s_prime, alpha, beta, depth + 1)
            self.joueur._memory.undo()
            return v, light_action
        
        v_star = float("+inf")
        m_star = None

        # selection des meilleures actions
        actions = list(s.generate_possible_light_actions())
        actions = self.actions_selection(actions)

        # La liste des actions peut être vide sans être à un noeud terminal global (recherche locale)
        if len(actions) == 0:
            return self.heuristique.execute(), None

        for move in actions:     
            if move == "annex":
                continue
            s_prime = s.apply_action(move)
            self.joueur._memory.update(s_prime)

            # Condition d'arrêt (finale)
            if s_prime.is_done():
                score = self.__evaluate(s_prime)
                self.joueur._memory.undo()
                return score, move  # score est +inf si MAX gagne, -inf si MIN gagne


            # on cherche la valeur minimum a la profondeur suivante 
            (v,_) = self.__maxValue(s_prime,alpha,beta, depth + 1)

            # On cherche la plus petite valeur a la profondeur n+1
            if(v < v_star):
                v_star = v
                m_star = move
                beta = min(beta,v_star)
            self.joueur._memory.undo()
            if (v_star <= alpha): return (v_star,m_star) #pruning
        return (v_star,m_star)   

    def __evaluate(self, s: GameStateHex) -> float:
        my_id = self.joueur.get_id()

        return +10000 if s.scores[my_id] == 1 else -10000
    

    def to_json(self):
        return {
            "type": "Algorithme_minimax_alpha_beta_typeA",
        }
    

    def actions_selection(self,action_list):
        """
        Retourne une selection de noeuds parmis la liste de noeud possible
        """
        
        def mask(X,link_mask = True,trapezoid_mask=True):
            """
            Masque certaines positions du plateau pour lesquelles l'agent ne doit pas poser de pions
        
            Args:
                X: La matrice d'attention retourné par get_attention_board() (on travaille directement avec des proba)
                link_mask: par défaut True, masque les maillons (applique 0)
                trapezoid_mask: par defaut True, masque les trapèzes (applique 0)
            Returns:
                X: masqué en fonction des choix 
            """
            adversary_links = self.joueur._memory.get_adversary_links()
            adversary_trapezoid = self.joueur._memory.get_adversary_trapezoid()
            if link_mask == True:
                for maillon, espaces in adversary_links.items():
                    for espace in espaces:
                        x, y = espace
                        X[x, y] = 0  # on bloque la case dans l'attention_board
            if trapezoid_mask == True:
                for pivot, espaces in adversary_trapezoid.items():
                    for espace in espaces:
                        _,x,y = espace
                        X[x, y] = 0  # on bloque la case dans l'attention_board
            return X

        def positive_probability_selection(attention_probability_matrix):
            """
            Renvoie la liste des positions ayant une probabilité non nulle de selection

            Args:
                X: La matrice d'attention normalisé par softmax
            Returns:
                valid_action_list : la liste des Light_action qui peuvent être choisis (proba > 0)
                valid_probs : les probabilités associés à chaque Light_action de valid_action_list
            """
            valid_action_list = []
            valid_probs = []
            for i in action_list:
                if(attention_probability_matrix[i.data['position'][0]+1, i.data['position'][1]+1] > 0):
                    valid_probs.append(attention_probability_matrix[i.data['position'][0]+1, i.data['position'][1]+1])
                    valid_action_list.append(i)
            return valid_action_list,valid_probs
            
        
        # Cas d'une recherche locale
        if self.local_analysis_area is not None:
            valid_local_actions = []  # actions retournées
            possible_actions = [(i.data['position'][0]+1, i.data['position'][1]+1) for i in action_list]

            for action in self.local_analysis_area:  # parmi les actions valides
                if action in possible_actions:  # si l'action est dans la liste des actions possibles
                    idx = possible_actions.index(action)  # récupérer l'indice correspondant
                    valid_local_actions.append(action_list[idx])  # ajouter l'objet LightAction
            return valid_local_actions
        

        # Sinon cas classique avec la matrice d'attention
        topk=self.branching_factor
        probability_board = np.copy(self.joueur._memory.get_attention_board())
        probability_board = mask(X=probability_board) # masquage complet 
        
        valid_action_list,valid_probs = positive_probability_selection(probability_board)

        # si valid_action est vide, alors on recalcul en permettant le jeu dans les trapèzes
        if (len(valid_action_list) == 0):
            
            probability_board = np.copy(self.joueur._memory.get_attention_board())
            probability_board = mask(X=probability_board,link_mask=True,trapezoid_mask=False) # masquage complet 
            valid_action_list,valid_probs = positive_probability_selection(probability_board)

        # si valid_action est encore vide, alors on recalcul en permettant le jeu dans les trapèzes ET les maillons
        if (len(valid_action_list) == 0):
            
            probability_board = np.copy(self.joueur._memory.get_attention_board())
            probability_board = mask(X=probability_board,link_mask=False,trapezoid_mask=False) # masquage complet 
            valid_action_list,valid_probs = positive_probability_selection(probability_board)

        # si valid_action est encore vide, on renvoie la liste de base
        if (len(valid_action_list) == 0):
            probability_board = np.copy(self.joueur._memory.get_attention_board())
            probability_board[probability_board >= 0] += 20 # augmentation des valeurs 
            valid_action_list,valid_probs = positive_probability_selection(probability_board)
        
        # Passage à un vecteur numpy
        valid_probs = np.array(valid_probs)

        # Indices de toutes les actions
        all_indices = list(range(len(valid_action_list)))

        # sélectionner les actions max
        max_val = np.max(valid_probs)
        selected_indices = [idx for idx, p in enumerate(valid_probs) if p == max_val][:topk]

        remaining_slots = topk - len(selected_indices)
        if remaining_slots <= 0:
            return [valid_action_list[idx] for idx in selected_indices]

        # indices restants
        remaining_indices = [idx for idx in all_indices if idx not in selected_indices]
        if len(remaining_indices) == 0:
            return [valid_action_list[idx] for idx in selected_indices]

        # probabilités normalisées pour les restants
        remaining_probs = valid_probs[remaining_indices]
        if remaining_probs.sum() > 0:
            remaining_probs /= remaining_probs.sum()
        else:
            remaining_probs = np.ones(len(remaining_indices)) / len(remaining_indices)

        # tirage pondéré
        num_to_choose = min(remaining_slots, len(remaining_indices))  
        chosen_rest = np.random.choice(remaining_indices, size=num_to_choose, replace=False, p=remaining_probs)

        # assemblage final
        selected_actions = [valid_action_list[idx] for idx in selected_indices]+["annex"]+[valid_action_list[idx] for idx in chosen_rest]

        return selected_actions
