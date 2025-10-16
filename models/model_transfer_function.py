from dataclasses import dataclass, field
from typing import Optional
import sympy as sp
import scipy.signal as sig
from models.model_polynomium import Polynomium

@dataclass
class TransferFunctionModel:
    symbolic: Optional[sp.Expr] = None
    numeric: Optional[sp.Expr] = None
    signal: Optional[sig.TransferFunction] = None
    numerator: Polynomium = field(default_factory=Polynomium)
    denominator: Polynomium = field(default_factory=Polynomium)

    def __str__(self, name_width: int = 12, type_width: int = 30):
        lines = [
            f"TransferFunctionModel(",
            f"  {'symbolic:':<{name_width}} {str(type(self.symbolic)):>{type_width}} = {self.symbolic}",
            f"  {'numeric:':<{name_width}} {str(type(self.numeric)):>{type_width}} = {self.numeric}",
            f"  {'signal:':<{name_width}} {str(type(self.signal)):>{type_width}} = {self.signal}",
            f"  {'numerator:':<{name_width}} {str(type(self.numerator)):>{type_width}} =",
            f"{self.numerator.__str__(6, name_width, type_width)}",
            f"  {'denominator:':<{name_width}} {str(type(self.denominator)):>{type_width}} =",
            f"{self.denominator.__str__(6, name_width, type_width)}",
            f")"
        ]
        return "\n".join(lines)
