# Moteur de jeu HEX avec IA Minimax

Un moteur pour le jeu HEX basé sur un algorithme Minimax, permettant de jouer humain contre humain ou humain contre ordinateur.

Prérequis : Python 3.11 (ou supérieur) et pip.

Installation :  
1. Cloner le dépôt :  
   git clone <URL_DU_DEPOT>
   cd Hex  
2. Créer un environnement virtuel :  
   python3 -m venv venv  
3. Activer l’environnement :  
   - Linux / macOS : source venv/bin/activate  
   - Windows : venv\Scripts\activate  
4. Installer les dépendances :  
   pip install -r requirements.txt

Exécution :  
- Partie humain contre humain :  
  python main_hex.py -t human_vs_human  
- Partie humain contre IA :  
  python main_hex.py -t human_vs_computer my_player.py  
  (my_player.py : fichier contenant la logique de l’IA, dans le même dossier que main_hex.py ou avec chemin complet)

Règles du jeu :  
- Le premier joueur joue les pions rouges, le second joue les pions bleus.  
- Le rouge doit relier le haut et le bas du plateau, le bleu doit relier la gauche et la droite.  
- À chaque tour, un joueur pose un pion sur une case vide.  
- Le premier joueur à compléter son chemin gagne.

