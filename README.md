# HarmoMesh

Prototype Rust de la note *« Reconstruire un mesh 3D avec des séries de Fourier et des harmoniques sphériques »* :
un mesh est remplacé par une somme finie de fonctions analytiques (coefficients + formule évaluable).

```bash
cargo build --release
```

## Utilisation

| Commande | Rôle |
|---|---|
| `harmomesh` ou `harmomesh gui [fichier]` | interface graphique (les 8 zones de la maquette) |
| `harmomesh diag suzanne.obj` | diagnostic topologique, méthodes disponibles, méthode conseillée |
| `harmomesh run suzanne.obj -m B -l 40 -o export` | chaîne complète + export JSON / CSV / GLSL / Python / PLY / OBJ |
| `harmomesh bench suzanne.obj` | tableau d'erreurs A / B / C de la section 3 |
| `harmomesh gen torus tore.obj` | formes de test (sphere, ellipsoid, torus) |

`harmomesh help` liste les options. Formats lus : OBJ, PLY, STL, glTF/GLB.

## Chaîne de traitement (section 4 de la note)

1. **Import** : soudure des sommets, triangulation, centre de gravité du volume à l'origine, sphère unité (transformation conservée pour l'export).
2. **Diagnostic** : composantes, variété, étanchéité, orientation, χ et genre, proportion de rayons à sortie unique.
3. **Réparation** : voxelisation, fermeture morphologique, remplissage, iso-surface lissée (marching tetrahedra, toujours une variété), simplification à topologie préservée. La pleine résolution sert de référence d'erreur.
4. **Choix de la méthode** : arbre de décision du tableau 4.
5. **Décomposition**
   - **A · radiale** : r(θ,φ) par lancer de rayons (BVH) sur grille de Gauss-Legendre, quadrature exacte.
   - **B · SPHARM** : paramétrage sphérique par flot de courbure moyenne conformalisé, correction de la distorsion d'aire, alignement du degré 1, moindres carrés pondérés et régularisés par gradient conjugué **sans matrice**.
   - **C · Fourier 3D** : distance signée exacte sur grille, FFT 3D, troncature |k| ≤ K, fenêtre de Lanczos, iso-surface.
6. **Reconstruction** par sommes partielles (contributions par degré précalculées : curseur en O(N)).
7. **Évaluation** : Chamfer, Hausdorff (% de la diagonale), normales, volume, aire, spectre d'énergie.
8. **Export** : coefficients + métadonnées de convention, formule complète en texte, shader GLSL, évaluateur Python, mesh reconstruit.

Dans l'interface, le panneau **Formule complète** (bouton *Formule* pour l'afficher ou le masquer) est ancré sous la vue 3D. Il compose la formule de l'objet avec de vrais symboles mathématiques (sommes avec bornes, indices, exposants, fractions, racines) : forme générale et définitions à gauche, développement terme à terme à droite, au degré du curseur. Les harmoniques peuvent rester symboliques ou être développées en cos θ, sin θ, cos mφ. La formule se copie en LaTeX ou en texte.

Convention : harmoniques réelles orthonormées, phase de Condon–Shortley, θ = colatitude, indice `l² + l + m`.

## Résultats sur Suzanne (`harmomesh bench suzanne.obj`)

| Méthode | Réglage | Coefficients | Erreur moy. | Erreur max. |
|---|---|---:|---:|---:|
| A · radiale | L = 16 / 32 / 63 | 289 / 1 089 / 4 096 | 0,49 / 0,21 / 0,13 % | 5,3 / 2,3 / 2,1 % |
| B · SPHARM | L = 20 / 40 / 60 | 1 323 / 5 043 / 11 163 | 0,18 / 0,07 / 0,05 % | 1,1 / 0,7 / 0,7 % |
| C · Fourier 3D | K = 16 / 24 / 32 | 35 937 / 117 649 / 274 625 | 0,35 / 0,19 / 0,13 % | 3,0 / 1,5 / 1,1 % |

La méthode B bénéficie de la correction d'aire (distorsion max ×236 → ×1,5 ; sans elle : 0,22 % / 3,4 % à L = 40, `--no-area-fix`).

## Code

`src/lib.rs` est le noyau, indépendant de l'interface : `mesh` (E/S), `bvh`, `diag`, `repair`, `sh` (base, synthèse/adjoint), `methods` (A, B, C), `metrics`, `export`, `pipeline` (API + tests à réponse connue : `cargo test --release`). `src/main.rs` est la ligne de commande, `src/gui.rs` l'interface (egui), `src/tex.rs` le petit moteur de composition mathématique et `src/formula.rs` la construction de la formule, avec threads de travail, progression et annulation.

`harmomesh gui fichier --shot capture.png` enchaîne réparation et calcul, enregistre une capture de la fenêtre, puis quitte (test de l'interface).
