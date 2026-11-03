# Évaluation des politiques, scénario s1_nominal

## Scénario nominal

| Politique | Coût total (UR) | TRS | Disponibilité | Ruptures | p (coût, vs P1) |
| --- | --- | --- | --- | --- | --- |
| P0 Seuils fixes réglés à la main | 55355 [53702 ; 57008] | 96.08 % | 98.43 % | 12.4 % | 1.86e-09 |
| P1 Seuils fixes optimisés | 52136 [50882 ; 53391] | 96.69 % | 99.06 % | 1.1 % |  |
| P2 Maintenance conditionnelle | 47135 [45574 ; 48697] | 96.71 % | 98.85 % | 23.9 % | 1.86e-09 |
| P3 Prédictive sans apprentissage par renforcement | 49220 [48252 ; 50189] | 96.83 % | 99.25 % | 2.1 % | 1.3e-07 |
| DQN DQN, niveau d'action réduit | 54932 [53347 ; 56517] | 96.13 % | 98.45 % | 12.9 % | 1.86e-09 |
| P4 Agent DRL, niveau N1 | 52799 [51232 ; 54365] | 96.44 % | 98.89 % | 3.9 % | 0.0185 |
| P5 Agent DRL, niveau N2 | 67049 [65137 ; 68960] | 95.00 % | 97.97 % | 4.5 % | 1.86e-09 |
| BDQ Branching Dueling Q-Network, niveau N2 | 74825 [71271 ; 78379] | 95.58 % | 98.45 % | 3.7 % | 1.86e-09 |
| Oracle Connaissance des pannes futures | 42408 [41755 ; 43061] | 97.29 % | 99.28 % | 10.2 % | 1.86e-09 |

Test d'équivalence TOST (±2 % du coût de P1) : P0 moins bon ; P2 meilleur ; P3 meilleur ; DQN moins bon ; P4 moins bon ; P5 moins bon ; BDQ moins bon ; Oracle meilleur.

## Robustesse : délai fournisseur doublé

| Politique | Coût total (UR) | TRS | Disponibilité | Ruptures | p (coût, vs P1) |
| --- | --- | --- | --- | --- | --- |
| P0 Seuils fixes réglés à la main | 69472 [66862 ; 72083] | 93.65 % | 96.24 % | 40.4 % | 1.86e-09 |
| P1 Seuils fixes optimisés | 55314 [53314 ; 57314] | 96.11 % | 98.50 % | 13.4 % |  |
| P2 Maintenance conditionnelle | 58735 [56174 ; 61296] | 95.08 % | 97.60 % | 34.8 % | 3.45e-05 |
| P3 Prédictive sans apprentissage par renforcement | 51709 [50530 ; 52887] | 96.76 % | 99.20 % | 4.8 % | 7.99e-06 |
| DQN DQN, niveau d'action réduit | 69186 [66518 ; 71855] | 93.71 % | 96.28 % | 41.3 % | 1.86e-09 |
| P4 Agent DRL, niveau N1 | 59779 [57408 ; 62150] | 95.26 % | 97.73 % | 18.4 % | 5.72e-07 |
| P5 Agent DRL, niveau N2 | 73299 [70197 ; 76401] | 93.91 % | 96.86 % | 6.3 % | 1.86e-09 |
| BDQ Branching Dueling Q-Network, niveau N2 | 81301 [75523 ; 87079] | 95.26 % | 98.07 % | 4.3 % | 1.86e-09 |
| Oracle Connaissance des pannes futures | 45529 [44526 ; 46533] | 97.00 % | 99.03 % | 28.2 % | 1.86e-09 |

Test d'équivalence TOST (±2 % du coût de P1) : P0 moins bon ; P2 moins bon ; P3 meilleur ; DQN moins bon ; P4 moins bon ; P5 moins bon ; BDQ moins bon ; Oracle meilleur.

## Robustesse : taux de panne + 30 %

| Politique | Coût total (UR) | TRS | Disponibilité | Ruptures | p (coût, vs P1) |
| --- | --- | --- | --- | --- | --- |
| P0 Seuils fixes réglés à la main | 62663 [60908 ; 64418] | 95.46 % | 98.03 % | 14.8 % | 1.86e-09 |
| P1 Seuils fixes optimisés | 57616 [56307 ; 58925] | 96.40 % | 98.95 % | 1.6 % |  |
| P2 Maintenance conditionnelle | 55000 [53330 ; 56671] | 96.08 % | 98.49 % | 26.0 % | 2.76e-06 |
| P3 Prédictive sans apprentissage par renforcement | 55052 [53725 ; 56380] | 96.56 % | 99.21 % | 2.7 % | 4.42e-06 |
| DQN DQN, niveau d'action réduit | 62538 [60611 ; 64464] | 95.45 % | 97.96 % | 16.0 % | 1.86e-09 |
| P4 Agent DRL, niveau N1 | 59584 [57819 ; 61349] | 95.96 % | 98.61 % | 5.8 % | 3.24e-06 |
| P5 Agent DRL, niveau N2 | 72139 [69963 ; 74315] | 94.84 % | 97.95 % | 4.2 % | 1.86e-09 |
| BDQ Branching Dueling Q-Network, niveau N2 | 83671 [80548 ; 86794] | 95.15 % | 98.19 % | 3.4 % | 1.86e-09 |
| Oracle Connaissance des pannes futures | 46358 [45657 ; 47060] | 97.12 % | 99.19 % | 5.2 % | 1.86e-09 |

Test d'équivalence TOST (±2 % du coût de P1) : P0 moins bon ; P2 meilleur ; P3 meilleur ; DQN moins bon ; P4 moins bon ; P5 moins bon ; BDQ moins bon ; Oracle meilleur.

## Robustesse : un technicien de moins

| Politique | Coût total (UR) | TRS | Disponibilité | Ruptures | p (coût, vs P1) |
| --- | --- | --- | --- | --- | --- |
| P0 Seuils fixes réglés à la main | 59362 [57746 ; 60978] | 95.70 % | 98.63 % | 5.3 % | 1.02e-07 |
| P1 Seuils fixes optimisés | 57251 [55870 ; 58632] | 96.23 % | 99.12 % | 0.4 % |  |
| P2 Maintenance conditionnelle | 57662 [56015 ; 59309] | 95.91 % | 98.93 % | 5.3 % | 0.191 |
| P3 Prédictive sans apprentissage par renforcement | 58897 [57579 ; 60214] | 96.26 % | 99.31 % | 0.1 % | 5.97e-06 |
| DQN DQN, niveau d'action réduit | 59210 [57461 ; 60960] | 95.72 % | 98.64 % | 4.9 % | 1.02e-07 |
| P4 Agent DRL, niveau N1 | 57904 [56432 ; 59376] | 95.99 % | 98.89 % | 2.7 % | 0.00466 |
| P5 Agent DRL, niveau N2 | 68533 [66381 ; 70684] | 95.05 % | 98.18 % | 2.8 % | 1.86e-09 |
| BDQ Branching Dueling Q-Network, niveau N2 | 109858 [101904 ; 117812] | 95.62 % | 98.65 % | 1.3 % | 1.86e-09 |
| Oracle Connaissance des pannes futures | 50950 [49936 ; 51963] | 96.86 % | 99.29 % | 0.5 % | 1.86e-09 |

Test d'équivalence TOST (±2 % du coût de P1) : P0 moins bon ; P2 équivalent ; P3 moins bon ; DQN moins bon ; P4 équivalent ; P5 moins bon ; BDQ moins bon ; Oracle meilleur.

## Robustesse : flotte homogène S0

| Politique | Coût total (UR) | TRS | Disponibilité | Ruptures | p (coût, vs P1) |
| --- | --- | --- | --- | --- | --- |
| P0 Seuils fixes réglés à la main | 50143 [48554 ; 51732] | 96.22 % | 98.52 % | 13.2 % | 3.73e-09 |
| P1 Seuils fixes optimisés | 46996 [45876 ; 48116] | 96.82 % | 99.14 % | 0.7 % |  |
| P2 Maintenance conditionnelle | 42452 [40888 ; 44015] | 96.83 % | 98.89 % | 29.1 % | 1.86e-09 |
| Oracle Connaissance des pannes futures | 38587 [37854 ; 39320] | 97.34 % | 99.34 % | 9.7 % | 1.86e-09 |

Test d'équivalence TOST (±2 % du coût de P1) : P0 moins bon ; P2 meilleur ; Oracle meilleur.
