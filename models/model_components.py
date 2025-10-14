from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Any
from classes.transfer_function import TransferFunction

@dataclass
class ComponentsModel:
    tfs: Dict[str, TransferFunction] = field(default_factory=dict)