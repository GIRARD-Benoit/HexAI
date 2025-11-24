# Copyright (c) 2025
# Licensed under the MIT License.
# See LICENSE file for details.

from game_state_hex import GameStateHex

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from my_player import MyPlayer  # import uniquement pour l'IDE




class GameDebug:
    """
    Classe d'analyse du plateau Hex pour un joueur donné.
    Fournit un aperçu structuré de la situation actuelle
    et des détails sur le calcul heuristique.
    """

    def __init__(self, joueur: "MyPlayer"):
        self.joueur = joueur
        self.state = None

    # ---------------------------
    # 🔹 Méthode principale
    # ---------------------------
    def debug_print(self):
        """Affiche une analyse complète de la situation du joueur."""
        print("------------ Évaluation de la situation -----------------")
        print("🎮 Joueur :", self.joueur.get_name())
        print("🟦 Couleur :", self.joueur.get_piece_type())
        print("🧩 Mon ID :", self.joueur.get_id())
        
        self.joueur._memory.print__memory()


    def to_json(self):
        """Sérialisation JSON compatible Seahorse."""
        return {
            "history": []
        }

    # # ---------------------------
    # # 🔹 Affichage détaillé
    # # ---------------------------
    # def _print_heuristique_details(
    #     self, alpha, beta, gamma,
    #     nbr_jetons, nbr_maillons_moi, nbr_maillons_adversaire,
    #     contrib_jetons, contrib_mail, contrib_plat, heur
    # ):
    #     """Affiche les détails du calcul heuristique."""
    #     print("=" * 70)
    #     print("🔍 DEBUG HEURISTIQUE")
    #     print("=" * 70)
    #     print("📊 Métriques normalisées (0-1):")
    #     print(f"   • Jetons à poser              : {nbr_jetons:8.4f}")
    #     print(f"   • Maillons (moi)              : {nbr_maillons_moi:8.4f}")
    #     print(f"   • Maillons (adversaire)       : {nbr_maillons_adversaire:8.4f}")
    #     print()
    #     print(f"⚖️  Contributions pondérées (α={alpha}, β={beta}, γ={gamma}):")
    #     print(f"   • -α × jetons     : {contrib_jetons:+8.4f}")
    #     print(f"   • +β × maillons   : {contrib_mail:+8.4f}")
    #     print(f"   • +γ × plateau    : {contrib_plat:+8.4f}")
    #     print()
    #     print(f"🎯 HEURISTIQUE TOTALE : {heur:+8.4f}")
    #     print("=" * 70)
    #     print("----------------------------------------------------------")
    