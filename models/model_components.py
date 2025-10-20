from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Any
from classes.transfer_function import TransferFunctionObject

@dataclass
class ComponentsModel:
    tfs: Dict[str, TransferFunctionObject] = field(default_factory=dict)