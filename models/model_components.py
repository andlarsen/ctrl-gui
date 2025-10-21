from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Any
from classes.class_transfer_function import TransferFunctionClass

@dataclass
class ComponentsModel:
    tfs: Dict[str, TransferFunctionClass] = field(default_factory=dict)