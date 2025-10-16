from dataclasses import dataclass
from typing import List, Optional
import sympy as sp

@dataclass
class Coefficients:
    symbolic: Optional[List[sp.Expr]] = None
    numeric: Optional[List[float]] = None

    def __str__(self, indent: int = 4, name_width: int = 12, type_width: int = 30):
        pad = " " * indent
        lines = [
            f"Coefficients(",
            f"{pad}{'symbolic:':<{name_width}} {str(type(self.symbolic)):>{type_width}} = {self.symbolic}",
            f"{pad}{'numeric:':<{name_width}} {str(type(self.numeric)):>{type_width}} = {self.numeric}",
            f"{' ' * (indent - 2)})"
        ]
        return "\n".join(lines)
