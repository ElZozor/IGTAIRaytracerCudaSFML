# IGTAI_Raytracer_Cuda-SFML
Projet d'IGTAI implémenté sous CUDA et SFML.

Réalisé sous Visual Studio.
Les dépendances sont la librairie SMFL et CUDA.

**Installation des dépendances**

***CUDA***  
Télécharger et installer CUDA via le lien suivant: https://developer.nvidia.com/cuda-downloads

***SFML***  
Plusieurs possibilité, la plus simple étant d'installer VCPKG qui fonctionne très bien avec Visual Studio, disponible ici : https://github.com/microsoft/vcpkg.  
Il suffit ensuite de se rendre dans le répertoire d'installation et d'installer la SFML via la commande suivante :
`.\vcpkg.exe install sfml:x64-windows`


**Problèmes connus**  
N'étant pour l'instant pas très à l'aise avec CUDA le projet souffre de plusieurs problèmes comme par exemple la possibilité de stackoverflow dû à la récursivité lors du calcul des reflets. Cela devrait être corrigé en priorité.  
Il est possible que le programme ne tourne pas sur certaines carte graphique en raison de nombreux threads déclarés.  
Problèmes de caméra sur d'autre scèns comme la scène 2.


**Performances**  
La scène 0 tourne à ~120 FPS sur une GTX 1050 Max-Q Design.  
Les autres scènes, plus gourmandes tournent moins bien, cependant le KDTree n'a à l'heure actuelle, pas été implémenté.
