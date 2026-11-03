# Généralisation du classement — compositions de flotte

Référence : s1_nominal · 30 réplications (graines 9000-9029) · 365 jours.

P1 et P3 sont réglés composition par composition (graines 101-103) ; P0, P2, Oracle et les agents sont transférés tels quels, faute de réentraînement par composition.

## Synthèse

| Composition | Techniciens | Part U | τ Kendall vs nominal | Ordre CDC | Coût P1 | Coût P2 | Meilleur agent |
| --- | --- | --- | --- | --- | --- | --- | --- |
| nominal | 2 | — | 1.000 | oui | 52136 | 47135 | P4 (52799) |
| composition_00 | 1 | 0.42 | 0.667 | non | 52642 | 53061 | P4 (52954) |
| composition_01 | 2 | 0.10 | 0.944 | oui | 33921 | 31368 | P4 (34226) |
| composition_02 | 2 | 0.48 | 1.000 | oui | 50033 | 45997 | P4 (50672) |
| composition_03 | 3 | 0.62 | 0.944 | non | 56676 | 50470 | P4 (56419) |
| composition_04 | 3 | 0.88 | 1.000 | oui | 65887 | 59504 | P4 (68330) |

## Détail des classements

### nominal

| Rang | Politique | Coût total moyen (UR) |
| --- | --- | --- |
| 1 | Oracle Connaissance des pannes futures | 42408 |
| 2 | P2 Maintenance conditionnelle | 47135 |
| 3 | P3 Prédictive sans apprentissage par renforcement | 49220 |
| 4 | P1 Seuils fixes optimisés | 52136 |
| 5 | P4 Agent DRL, niveau N1 | 52799 |
| 6 | DQN DQN, niveau d'action réduit | 54932 |
| 7 | P0 Seuils fixes réglés à la main | 55355 |
| 8 | P5 Agent DRL, niveau N2 | 67049 |
| 9 | BDQ Branching Dueling Q-Network, niveau N2 | 74825 |

### composition_00

| Rang | Politique | Coût total moyen (UR) |
| --- | --- | --- |
| 1 | Oracle Connaissance des pannes futures | 46395 |
| 2 | P1 Seuils fixes optimisés | 52642 |
| 3 | P4 Agent DRL, niveau N1 | 52954 |
| 4 | P3 Prédictive sans apprentissage par renforcement | 53005 |
| 5 | P2 Maintenance conditionnelle | 53061 |
| 6 | P0 Seuils fixes réglés à la main | 54047 |
| 7 | DQN DQN, niveau d'action réduit | 54446 |
| 8 | P5 Agent DRL, niveau N2 | 63525 |
| 9 | BDQ Branching Dueling Q-Network, niveau N2 | 92820 |

### composition_01

| Rang | Politique | Coût total moyen (UR) |
| --- | --- | --- |
| 1 | Oracle Connaissance des pannes futures | 27678 |
| 2 | P2 Maintenance conditionnelle | 31368 |
| 3 | P3 Prédictive sans apprentissage par renforcement | 31545 |
| 4 | P1 Seuils fixes optimisés | 33921 |
| 5 | P4 Agent DRL, niveau N1 | 34226 |
| 6 | P0 Seuils fixes réglés à la main | 37467 |
| 7 | DQN DQN, niveau d'action réduit | 37938 |
| 8 | P5 Agent DRL, niveau N2 | 45456 |
| 9 | BDQ Branching Dueling Q-Network, niveau N2 | 48792 |

### composition_02

| Rang | Politique | Coût total moyen (UR) |
| --- | --- | --- |
| 1 | Oracle Connaissance des pannes futures | 40982 |
| 2 | P2 Maintenance conditionnelle | 45997 |
| 3 | P3 Prédictive sans apprentissage par renforcement | 47531 |
| 4 | P1 Seuils fixes optimisés | 50033 |
| 5 | P4 Agent DRL, niveau N1 | 50672 |
| 6 | DQN DQN, niveau d'action réduit | 53042 |
| 7 | P0 Seuils fixes réglés à la main | 53540 |
| 8 | P5 Agent DRL, niveau N2 | 65072 |
| 9 | BDQ Branching Dueling Q-Network, niveau N2 | 71734 |

### composition_03

| Rang | Politique | Coût total moyen (UR) |
| --- | --- | --- |
| 1 | Oracle Connaissance des pannes futures | 46356 |
| 2 | P2 Maintenance conditionnelle | 50470 |
| 3 | P3 Prédictive sans apprentissage par renforcement | 53526 |
| 4 | P4 Agent DRL, niveau N1 | 56419 |
| 5 | P1 Seuils fixes optimisés | 56676 |
| 6 | DQN DQN, niveau d'action réduit | 59266 |
| 7 | P0 Seuils fixes réglés à la main | 59554 |
| 8 | P5 Agent DRL, niveau N2 | 73777 |
| 9 | BDQ Branching Dueling Q-Network, niveau N2 | 78635 |

### composition_04

| Rang | Politique | Coût total moyen (UR) |
| --- | --- | --- |
| 1 | Oracle Connaissance des pannes futures | 55426 |
| 2 | P2 Maintenance conditionnelle | 59504 |
| 3 | P3 Prédictive sans apprentissage par renforcement | 63956 |
| 4 | P1 Seuils fixes optimisés | 65887 |
| 5 | P4 Agent DRL, niveau N1 | 68330 |
| 6 | DQN DQN, niveau d'action réduit | 70277 |
| 7 | P0 Seuils fixes réglés à la main | 71211 |
| 8 | P5 Agent DRL, niveau N2 | 91413 |
| 9 | BDQ Branching Dueling Q-Network, niveau N2 | 96325 |
