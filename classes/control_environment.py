from typing import Dict
from classes.control_system import ControlSystem


class ControlEnvironment:
    def __init__(self, name):
        self.name = name
        self.global_variables = []
        self.global_constants = {}

        self.tfs: Dict[str, ControlSystem] = {}