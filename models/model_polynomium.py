from dataclasses import dataclass, field
from typing import Optional
import sympy as sp
from models.model_coefficients import Coefficients

@dataclass
class Polynomium:
    symbolic: Optional[sp.Expr] = None
    numeric: Optional[sp.Expr] = None
    coefficients: Coefficients = field(default_factory=Coefficients)

    def __str__(self, indent: int = 2, name_width: int = 12, type_width: int = 30):
        pad = " " * indent
        lines = [
            f"Polynomium(",
            f"{pad}{'symbolic:':<{name_width}} {str(type(self.symbolic)):>{type_width}} = {self.symbolic}",
            f"{pad}{'numeric:':<{name_width}} {str(type(self.numeric)):>{type_width}} = {self.numeric}",
            f"{pad}{'coefficients:':<{name_width}} {str(type(self.coefficients)):>{type_width}} =",
            f"{self.coefficients.__str__(indent + 4, name_width, type_width)}",
            f"{' ' * (indent - 2)})"
        ]
        return "\n".join(lines)
