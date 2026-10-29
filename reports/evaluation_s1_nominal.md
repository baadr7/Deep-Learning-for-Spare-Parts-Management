# Évaluation des politiques, scénario s1_nominal

## Scénario nominal

| Politique | Coût total (UR) | TRS | Disponibilité | Ruptures | p (coût, vs P1) |
| --- | --- | --- | --- | --- | --- |
| P0 Seuils fixes réglés à la main | 55355 [53702 ; 57008] | 96.08 % | 98.43 % | 12.4 % | 1.86e-09 |
| P1 Seuils fixes optimisés | 52136 [50882 ; 53391] | 96.69 % | 99.06 % | 1.1 % |  |
| P2 Maintenance conditionnelle | 47135 [45574 ; 48697] | 96.71 % | 98.85 % | 23.9 % | 1.86e-09 |
| P3 Prédictive sans apprentissage par renforcement | 51038 [49936 ; 52139] | 96.70 % | 99.25 % | 2.6 % | 0.00374 |
| DQN DQN, niveau d'action réduit | 58483 [56807 ; 60159] | 95.65 % | 98.36 % | 21.8 % | 1.86e-09 |
| P4 Agent DRL, niveau N1 | 52725 [51333 ; 54117] | 96.61 % | 98.94 % | 2.4 % | 0.0293 |
| P5 Agent DRL, niveau N2 | 62078 [60009 ; 64147] | 95.77 % | 98.31 % | 5.4 % | 1.86e-09 |
| BDQ Branching Dueling Q-Network, niveau N2 | 92844 [89493 ; 96195] | 94.00 % | 97.24 % | 3.8 % | 1.86e-09 |
| Oracle Connaissance des pannes futures | 42408 [41755 ; 43061] | 97.29 % | 99.28 % | 10.2 % | 1.86e-09 |

## Robustesse : délai fournisseur doublé

| Politique | Coût total (UR) | TRS | Disponibilité | Ruptures | p (coût, vs P1) |
| --- | --- | --- | --- | --- | --- |
| P0 Seuils fixes réglés à la main | 69472 [66862 ; 72083] | 93.65 % | 96.24 % | 40.4 % | 1.86e-09 |
| P1 Seuils fixes optimisés | 55314 [53314 ; 57314] | 96.11 % | 98.50 % | 13.4 % |  |
| P2 Maintenance conditionnelle | 58735 [56174 ; 61296] | 95.08 % | 97.60 % | 34.8 % | 3.45e-05 |
| P3 Prédictive sans apprentissage par renforcement | 54354 [53015 ; 55694] | 96.54 % | 99.11 % | 7.8 % | 0.339 |
| DQN DQN, niveau d'action réduit | 79633 [76814 ; 82452] | 92.20 % | 95.13 % | 54.8 % | 1.86e-09 |
| P4 Agent DRL, niveau N1 | 57038 [54968 ; 59108] | 95.88 % | 98.27 % | 7.2 % | 6.29e-05 |
| P5 Agent DRL, niveau N2 | 66892 [64208 ; 69577] | 94.89 % | 97.43 % | 8.7 % | 1.86e-09 |
| BDQ Branching Dueling Q-Network, niveau N2 | 104883 [100493 ; 109273] | 94.26 % | 97.47 % | 3.8 % | 1.86e-09 |
| Oracle Connaissance des pannes futures | 45529 [44526 ; 46533] | 97.00 % | 99.03 % | 28.2 % | 1.86e-09 |

## Robustesse : taux de panne + 30 %

| Politique | Coût total (UR) | TRS | Disponibilité | Ruptures | p (coût, vs P1) |
| --- | --- | --- | --- | --- | --- |
| P0 Seuils fixes réglés à la main | 62663 [60908 ; 64418] | 95.46 % | 98.03 % | 14.8 % | 1.86e-09 |
| P1 Seuils fixes optimisés | 57616 [56307 ; 58925] | 96.40 % | 98.95 % | 1.6 % |  |
| P2 Maintenance conditionnelle | 55000 [53330 ; 56671] | 96.08 % | 98.49 % | 26.0 % | 2.76e-06 |
| P3 Prédictive sans apprentissage par renforcement | 57075 [55728 ; 58422] | 96.42 % | 99.20 % | 3.7 % | 0.124 |
| DQN DQN, niveau d'action réduit | 66876 [65039 ; 68714] | 94.89 % | 97.81 % | 25.1 % | 1.86e-09 |
| P4 Agent DRL, niveau N1 | 58942 [57464 ; 60420] | 96.19 % | 98.72 % | 3.4 % | 2.08e-05 |
| P5 Agent DRL, niveau N2 | 70230 [67247 ; 73213] | 94.94 % | 97.63 % | 7.4 % | 1.86e-09 |
| BDQ Branching Dueling Q-Network, niveau N2 | 96515 [93430 ; 99600] | 94.25 % | 97.64 % | 3.6 % | 1.86e-09 |
| Oracle Connaissance des pannes futures | 46358 [45657 ; 47060] | 97.12 % | 99.19 % | 5.2 % | 1.86e-09 |

## Robustesse : un technicien de moins

| Politique | Coût total (UR) | TRS | Disponibilité | Ruptures | p (coût, vs P1) |
| --- | --- | --- | --- | --- | --- |
| P0 Seuils fixes réglés à la main | 59362 [57746 ; 60978] | 95.70 % | 98.63 % | 5.3 % | 1.02e-07 |
| P1 Seuils fixes optimisés | 57251 [55870 ; 58632] | 96.23 % | 99.12 % | 0.4 % |  |
| P2 Maintenance conditionnelle | 57662 [56015 ; 59309] | 95.91 % | 98.93 % | 5.3 % | 0.191 |
| P3 Prédictive sans apprentissage par renforcement | 59044 [57893 ; 60196] | 96.21 % | 99.29 % | 0.2 % | 6.91e-07 |
| DQN DQN, niveau d'action réduit | 63260 [61347 ; 65173] | 95.16 % | 98.14 % | 20.3 % | 1.86e-09 |
| P4 Agent DRL, niveau N1 | 58420 [56972 ; 59868] | 96.03 % | 98.94 % | 1.5 % | 4.41e-05 |
| P5 Agent DRL, niveau N2 | 63855 [62104 ; 65606] | 95.78 % | 98.63 % | 1.8 % | 1.86e-09 |
| BDQ Branching Dueling Q-Network, niveau N2 | 97840 [93712 ; 101968] | 93.35 % | 96.58 % | 3.9 % | 1.86e-09 |
| Oracle Connaissance des pannes futures | 50950 [49936 ; 51963] | 96.86 % | 99.29 % | 0.5 % | 1.86e-09 |
