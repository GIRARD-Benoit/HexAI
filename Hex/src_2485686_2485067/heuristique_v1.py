# Copyright (c) 2025
# Licensed under the MIT License.
# See LICENSE file for details.

from game_state_hex import GameStateHex
from src_2485686_2485067.Metrics.center_control import Center_control
from src_2485686_2485067.Metrics.maillons import Maillons
from src_2485686_2485067.Metrics.distance import Distance
from src_2485686_2485067.heuristique import Heuristique
from typing import override
from typing import TYPE_CHECKING
import numpy as np

if TYPE_CHECKING:
    from my_player import MyPlayer

class Heuristique_v1(Heuristique):
    """
    Heuristique combinant plusieurs métriques :
    - Distance à parcourir (Dijkstra)
    - Nombre de maillons
    - Contrôle du centre
    - Blocage du chemin adverse
    """
    
    # Constantes de normalisation
    MAX_DISTANCE = 25.0
    MAX_MAILLONS = 50.0
    MAX_CENTER = 100.0

    def __init__(self, joueur:"MyPlayer"):
        super().__init__()
        self.joueur = joueur
        
        # Métriques
        self.metric_maillons = Maillons(self.joueur)
 #       self.metric_distance_jetons = Distance(self.joueur)
        self.metric_center_control = Center_control(self.joueur)
        
        # Coefficients (valeurs par défaut)
        self.alpha = 1.0
        """Poids de ma distance (négatif car on minimise)"""
        
        self.beta_me = 0.0
        """Poids de mes maillons"""
        
        self.beta_adv = 0.5
        """Poids des maillons adverses (négatif appliqué dans execute)"""

        self.gamma = 0.1
        """Contrôle du centre"""
        
        self.delta = 3.0
        """
        Poids de la distance adversaire
        (delta = 1) => équilibré
        (delta < 1) => offensif
        (delta > 1) => défensif
        """
        
        self.epsilon = 2.0
        """
        Poids du blocage actif
        Récompense les pions placés sur le chemin critique adverse
        """

        self.zeta = 0 
        """
        poid de mes trapèzes
        """
        self.mode = "COMPETITIF"

    @override
    def execute(self) -> float:
        """
        Calcule le score heuristique de la position actuelle.
        Plus le score est élevé, meilleure est la position pour MAX.
        """
        # si l'heuristique a déjà été calculé alors on la retourne directement 
        if(self.joueur._memory.heuristique_cache.contains(self.joueur._memory.get_board())):
            return self.joueur._memory.heuristique_cache.get(self.joueur._memory.get_board())

        # Early victory detector
        score = self.joueur._early_victory_detector.check_victory()
        if score is not None and score != 0:
            # Retourner directement le score de victoire/défaite
            return score

        # Calcul des métriques de base
 #       my_distance, adversary_distance = self.metric_distance_jetons.execute()
        my_distance, adversary_distance = self.joueur._memory.my_distance,self.joueur._memory.adversary_distance
        nbr_maillons_moi, nbr_maillons_adversaire = self.metric_maillons.execute()
        nbr_trapezes_moi = len(self.joueur._memory.me_trapezoid)
        center_score = self.metric_center_control.execute()
        
        # Gestion des distances infinies (pas de chemin possible)
        if my_distance == np.inf:
            return -10000  # Position perdante
        if adversary_distance == np.inf:
            return 10000   # Position gagnante
        
        # Calcul du score de blocage
        blocking_score = self.compute_blocking_score()
        
        # Normalisation des métriques
        my_dist_norm = my_distance / self.MAX_DISTANCE
        adv_dist_norm = adversary_distance / self.MAX_DISTANCE
        maillons_me_norm = nbr_maillons_moi / self.MAX_MAILLONS
        maillons_adv_norm = nbr_maillons_adversaire / self.MAX_MAILLONS
        center_norm = center_score / self.MAX_CENTER
        
        # Calcul du score heuristique
        score = (
            -self.alpha * my_dist_norm +              # Minimiser ma distance
            self.delta * adv_dist_norm +              # Maximiser distance adverse
            self.beta_me * maillons_me_norm -         # Récompenser mes maillons
            self.beta_adv * maillons_adv_norm +       # Pénaliser maillons adverses
            self.gamma * center_norm +                # Contrôle du centre
            self.epsilon * blocking_score+             # Récompenser le blocage
            self.zeta * nbr_trapezes_moi               # recompenser la creation de trapèzes (uniquement utilisé lors d'une recherche locale)
        )

        # sauvegarde de la valeur de l'heuristique pour cet état 
        self.joueur._memory.heuristique_cache.set(self.joueur._memory.get_board(),score)
        
        return score

    def compute_blocking_score(self) -> float:
        """
        Calcule un score de blocage basé sur le nombre de mes pions
        qui se trouvent sur le chemin critique de l'adversaire.
        
        Returns:
            Score entre 0 et 1 (pourcentage du chemin bloqué)
        """
        # Récupérer le chemin critique de l'adversaire (depuis le cache)
 #       adv_path = self.metric_distance_jetons.get_adversary_critical_path()
        adv_path = self.joueur._memory.adversary_critical_path
        
        if not adv_path or len(adv_path) == 0:
            return 0.0
        
        board = self.joueur._memory.get_board()
        
        # Compter combien de mes pions (valeur = 1) bloquent le chemin
        blocking_count = sum(1 for (i, j) in adv_path if board[i, j] == 1)
        
        # Normaliser par la longueur du chemin
        return blocking_count / len(adv_path)

    def coefficient_update(self):
        """
        Ajuste dynamiquement les coefficients selon l'état de la partie.
        Appelé après chaque coup.
        """
        num_tour = self.joueur._memory.get_move_history().len()
        
        # ===== PHASE DE JEU =====
        
        # Début de partie (tours 1-7) : Contrôle du centre important
        if num_tour < 7:
            self.gamma = 0.0  #0.6
            self.epsilon = 1.5  # Blocage modéré
            self.zeta = 0 
        
        # Milieu de partie (tours 8-15) : Équilibre
        elif num_tour < 30:
            self.gamma = 0.08
            self.epsilon = 2.0
            self.zeta = 0 
        
        # Fin de partie (tours 30+) : Centre moins important, blocage crucial
        else:
            self.gamma = 0.02
            self.epsilon = 2.5
            self.zeta = 0 
        
        # ===== AJUSTEMENT TACTIQUE =====
        
        my_distance, adversary_distance = self.joueur._memory.my_distance,self.joueur._memory.adversary_distance
        distance_diff = my_distance - adversary_distance
        
        # Gestion des distances infinies
        if my_distance == np.inf or adversary_distance == np.inf:
            return  # Garder les valeurs par défaut
        
        # Adversaire dangereux ET nous sommes loin
        # elif my_distance >= adversary_distance + 3:
        #     self.alpha = 0.0      # On stoppe notre progression
        #     self.delta = 5.0      # Très défensif
        #     self.epsilon = 2.0    
        #     self.gamma = 0.0
        #     self.beta_adv = 0.5
        #     self.mode = "DEFENSIF"
  #          print(f"MODE DÉFENSIF : {my_distance:.1f} vs {adversary_distance:.1f}")
        
        # Course serrée (milieu de partie)
        # elif abs(distance_diff) < 2:
        else:
            self.alpha = 0.7
            self.delta = 5.0
            self.epsilon = 2.0
            self.beta_adv = 0.5
            self.mode = "COMPETITIF"
 #           print(f"MODE COMPÉTITIF : {my_distance:.1f} vs {adversary_distance:.1f}")

    def to_json(self):
        """Sérialisation pour debug et logging."""
        return {
        }
    


    def print_debug(self):
        """
        Affiche toutes les informations utiles pour le debugging.
        À appeler manuellement : heuristique.print_debug()
        Recalcule automatiquement le score et toutes les métriques.
        """
        # Calculer toutes les métriques
        my_distance, adversary_distance = self.joueur._memory.my_distance,self.joueur._memory.adversary_distance
        nbr_maillons_moi, nbr_maillons_adversaire = self.metric_maillons.execute()
        center_score = self.metric_center_control.execute()
        blocking_score = self.compute_blocking_score()
        
        # Recalculer le score heuristique complet
        if my_distance == np.inf:
            score = -10000
        elif adversary_distance == np.inf:
            score = 10000
        else:
            my_dist_norm = my_distance / self.MAX_DISTANCE
            adv_dist_norm = adversary_distance / self.MAX_DISTANCE
            maillons_me_norm = nbr_maillons_moi / self.MAX_MAILLONS
            maillons_adv_norm = nbr_maillons_adversaire / self.MAX_MAILLONS
            center_norm = center_score / self.MAX_CENTER
            
            score = (
                -self.alpha * my_dist_norm +
                self.delta * adv_dist_norm +
                self.beta_me * maillons_me_norm -
                self.beta_adv * maillons_adv_norm +
                self.gamma * center_norm +
                self.epsilon * blocking_score
            )
        
        # Récupérer les chemins critiques
        # my_path = self.metric_distance_jetons.get_my_critical_path()
        # adv_path = self.metric_distance_jetons.get_adversary_critical_path()
        my_path = self.joueur._memory.my_critical_path
        adv_path = self.joueur._memory.adversary_critical_path
        
        # Calculs de normalisation (gérer les infinis)
        if my_distance != np.inf:
            my_dist_norm = my_distance / self.MAX_DISTANCE
            contrib_my_dist = -self.alpha * my_dist_norm
        else:
            my_dist_norm = np.inf
            contrib_my_dist = -np.inf
            
        if adversary_distance != np.inf:
            adv_dist_norm = adversary_distance / self.MAX_DISTANCE
            contrib_adv_dist = self.delta * adv_dist_norm
        else:
            adv_dist_norm = np.inf
            contrib_adv_dist = np.inf
        
        maillons_me_norm = nbr_maillons_moi / self.MAX_MAILLONS
        maillons_adv_norm = nbr_maillons_adversaire / self.MAX_MAILLONS
        center_norm = center_score / self.MAX_CENTER
        
        # Contributions individuelles
        contrib_my_links = self.beta_me * maillons_me_norm
        contrib_adv_links = -self.beta_adv * maillons_adv_norm
        contrib_center = self.gamma * center_norm
        contrib_blocking = self.epsilon * blocking_score
        
        num_tour = self.joueur._memory.get_move_history().len()
        player_id = self.joueur.get_id()
        player_color = self.joueur.get_piece_type()
        
        # Affichage formaté
        print("\n" + "="*70)
        print(f"🎮 HEURISTIQUE DEBUG - Tour {num_tour} | Joueur {player_id} ({player_color})")
        print("="*70)
        
        # Section 1 : Métriques brutes
        print("\n📊 MÉTRIQUES BRUTES:")
        print(f"  Distance moi        : {my_distance:.2f} / {self.MAX_DISTANCE:.0f}")
        print(f"  Distance adversaire : {adversary_distance:.2f} / {self.MAX_DISTANCE:.0f}")
        print(f"  Maillons moi        : {nbr_maillons_moi} / {self.MAX_MAILLONS:.0f}")
        print(f"  Maillons adversaire : {nbr_maillons_adversaire} / {self.MAX_MAILLONS:.0f}")
        print(f"  Contrôle centre     : {center_score:.2f} / {self.MAX_CENTER:.0f}")
        print(f"  Score blocage       : {blocking_score:.2%}")
        
        # Section 2 : Chemins critiques
        print("\n🛣️  CHEMINS CRITIQUES:")
        print(f"  Mon chemin          : {len(my_path)} cases")
        print(f"  Chemin adversaire   : {len(adv_path)} cases")
        if adv_path:
            board = self.joueur._memory.get_board()
            blocked_positions = [(i, j) for (i, j) in adv_path if board[i, j] == 1]
            print(f"  Cases bloquées      : {len(blocked_positions)}")
            if blocked_positions:
                print(f"    Positions: {blocked_positions[:5]}{'...' if len(blocked_positions) > 5 else ''}")
        
        # Section 3 : Coefficients actuels
        print("\n⚙️  COEFFICIENTS:")
        print(f"  α (ma distance)     : {self.alpha:.2f}")
        print(f"  δ (dist. adverse)   : {self.delta:.2f}")
        print(f"  β_me (mes maillons) : {self.beta_me:.2f}")
        print(f"  β_adv (leurs mail.) : {self.beta_adv:.2f}")
        print(f"  γ (centre)          : {self.gamma:.2f}")
        print(f"  ε (blocage)         : {self.epsilon:.2f}")
        print(f"  🧭 MODE ACTUEL       : {self.mode}")
        
        # Section 4 : Contributions au score
        print("\n💡 CONTRIBUTIONS AU SCORE:")
        
        if my_distance != np.inf and adversary_distance != np.inf:
            print(f"  Ma distance         : {contrib_my_dist:+.3f}")
            print(f"  Dist. adversaire    : {contrib_adv_dist:+.3f}")
            print(f"  Mes maillons        : {contrib_my_links:+.3f}")
            print(f"  Maillons adverses   : {contrib_adv_links:+.3f}")
            print(f"  Contrôle centre     : {contrib_center:+.3f}")
            print(f"  Blocage             : {contrib_blocking:+.3f}")
            print(f"  {'-'*40}")
            print(f"  SCORE FINAL         : {score:+.3f}")
        else:
            print(f"  SCORE FINAL         : {score:+.0f} (distance infinie)")
        
        # Section 5 : Analyse stratégique
        print("\n🎯 ANALYSE STRATÉGIQUE:")
        
        # Déterminer le mode de jeu
        if my_distance == np.inf:
            print("  ❌ AUCUN CHEMIN DISPONIBLE POUR NOUS")
        elif adversary_distance == np.inf:
            print("  ✅ ADVERSAIRE BLOQUÉ")
        elif adversary_distance < 5:
            print(f"  ⚠️  DANGER : Adversaire proche ({adversary_distance:.1f} cases)")
        elif my_distance < 5:
            print(f"  🚀 OPPORTUNITÉ : Nous sommes proches ({my_distance:.1f} cases)")
        elif abs(my_distance - adversary_distance) < 2:
            print(f"  ⚔️  COURSE SERRÉE : {my_distance:.1f} vs {adversary_distance:.1f}")
        else:
            ratio = my_distance / adversary_distance if adversary_distance > 0 else np.inf
            if ratio < 0.8:
                print(f"  ✅ NOUS AVONS L'AVANTAGE (ratio: {ratio:.2f})")
            elif ratio > 1.2:
                print(f"  ⚠️  ADVERSAIRE EN AVANCE (ratio: {ratio:.2f})")
            else:
                print(f"  ⚖️  SITUATION ÉQUILIBRÉE (ratio: {ratio:.2f})")
        
        # Efficacité du blocage
        if blocking_score > 0.3:
            print(f"  🛡️  BLOCAGE EFFICACE : {blocking_score:.0%} du chemin adverse")
        elif blocking_score > 0:
            print(f"  🛡️  Blocage partiel : {blocking_score:.0%}")
        else:
            print("  ⚠️  AUCUN BLOCAGE ACTIF")
        
        # Phase de jeu
        if num_tour < 7:
            print("  📍 PHASE : Début de partie (contrôle du centre)")
        elif num_tour < 30:
            print("  📍 PHASE : Milieu de partie (développement)")
        else:
            print("  📍 PHASE : Fin de partie (course finale)")
        
        print("="*70 + "\n")