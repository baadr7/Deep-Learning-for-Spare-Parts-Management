"""États d'une machine (cahier des charges, figure 1)."""

from enum import IntEnum


class MachineState(IntEnum):
    OPERATIONAL = 0
    FAILED = 1
    WAITING_PART = 2
    WAITING_TECH = 3
    REPAIR = 4
    PREVENTIVE = 5
    TEST = 6


STATE_LABELS = {
    MachineState.OPERATIONAL: "Opérationnelle",
    MachineState.FAILED: "En panne",
    MachineState.WAITING_PART: "Attente pièce",
    MachineState.WAITING_TECH: "Attente technicien",
    MachineState.REPAIR: "Réparation",
    MachineState.PREVENTIVE: "Maintenance préventive",
    MachineState.TEST: "Test",
}

N_STATES = len(MachineState)
