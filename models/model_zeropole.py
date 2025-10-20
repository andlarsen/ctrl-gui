from dataclasses import dataclass, field
import sympy as sp
import scipy.signal as sig
from typing import Optional, List, Tuple

@dataclass
class ZeroPoleModel:
    symbolic: List[sp.Expr] = field(default_factory=list)
    numeric: List[sp.Expr] = field(default_factory=list)

    def __str__(self, indent: int = 4, name_width: int = 12, type_width: int = 30):
        pad_title = " " * indent
        pad = " " * (indent+2)
        lines = [
            f"{pad_title}MetricModel(",
            f"{pad}{'symbolic:':<{name_width}} {str(type(self.symbolic)):>{type_width}} = {self.symbolic}",
            f"{pad}{'numeric:':<{name_width}} {str(type(self.numeric)):>{type_width}} = {self.numeric}",
            f"{' ' * (indent)})"
        ]
        return "\n".join(lines)