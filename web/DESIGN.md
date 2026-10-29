# Système visuel de l'application

Application de travail (mode « Operate ») : calme par défaut, une action principale par écran,
les chiffres d'abord. Thème clair uniquement.

## Typographie

| Usage | Famille | Graisse | Taille |
| --- | --- | --- | --- |
| Titre de page | Archivo | 600 | 28 px |
| Titre de section | Archivo | 600 | 18 px |
| Texte courant, contrôles | IBM Plex Sans | 400 / 500 | 15 px |
| Libellés, légendes, axes | IBM Plex Sans | 500 | 13 px |
| Valeurs numériques | IBM Plex Sans | 400 / 500 | selon contexte, chiffres tabulaires |
| Identifiants, horodatages | IBM Plex Mono | 400 | 13 px |

Pas de texte sous 12 px, pas de capitales sur plus de trois mots, pas de petit libellé
au-dessus d'un titre. Longueur de ligne limitée à 70 caractères pour les paragraphes.

## Couleurs (OKLCH)

| Jeton | Valeur | Rôle |
| --- | --- | --- |
| `paper` | `oklch(1 0 0)` | Fond de page et surfaces |
| `mist` | `oklch(0.975 0.008 300)` | Fond discret (barre latérale, en-têtes de tableau) |
| `line` | `oklch(0.91 0.012 300)` | Filets et bordures |
| `ink` | `oklch(0.24 0.035 300)` | Texte principal |
| `ink-soft` | `oklch(0.46 0.03 300)` | Texte secondaire (contraste supérieur à 4,5:1) |
| `plum-50` à `plum-900` | teinte 300 | Marque, action principale, sélection, séries de données |
| `sun-100` à `sun-800` | teinte 92 | Attention : attente de pièce, seuils, mise en avant d'une série |
| `danger` | `oklch(0.56 0.19 27)` | Panne uniquement |

Le jaune ne porte jamais de texte fin sur fond blanc : il sert de remplissage, et le texte posé
dessus utilise `sun-800`. Aucun dégradé, aucun halo, aucun texte en dégradé.

## États des machines

| État | Couleur |
| --- | --- |
| Opérationnelle | `plum-100` |
| En panne | `danger` |
| Attente pièce | `sun-400` |
| Attente technicien | `sun-100` avec bordure `sun-600` |
| Réparation | `plum-700` |
| Maintenance préventive | `plum-400` |
| Test | `plum-200` |

## Forme et espace

- Rayons : 4 px (contrôles, cases), 8 px (panneaux). Rien au-delà.
- Une surface est définie par un filet **ou** par un fond, jamais par un filet et une ombre.
- Pas de cartes imbriquées ; les sections se séparent par l'espace et des filets.
- Espacements : 4, 8, 12, 16, 24, 32, 48 px. Plus d'espace entre groupes qu'à l'intérieur.

## Mouvement

Transitions de 150 ms en `ease-out` sur le survol et la sélection, sans rebond. Aucune animation
décorative ; `prefers-reduced-motion` désactive les transitions.

## Rédaction

Français, phrases courtes et précises. Pas de tirets cadratins, pas de formules publicitaires,
des libellés d'action explicites (« Simuler 365 jours » plutôt que « Continuer »).
