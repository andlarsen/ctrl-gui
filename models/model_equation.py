from dataclasses import dataclass
import sympy as sp
import scipy.signal as sig

@dataclass
class EquationModel:
    symbolic: sp.Eq | None = None
    numeric: sp.Eq | None = None

    def __str__(self, indent: int = 4, name_width: int = 12, type_width: int = 30):
        pad_title = " " * indent
        pad = " " * (indent+2)
        lines = [
            f"{pad_title}EquationModel(",
            f"{pad}{'symbolic:':<{name_width}} {str(type(self.symbolic)):>{type_width}} {self.symbolic.lhs} = {self.symbolic.rhs}",
            f"{pad}{'numeric:':<{name_width}} {str(type(self.numeric)):>{type_width}} {self.numeric.lhs} = {self.numeric.rhs}",
            f"{' ' * (indent)})"
        ]
        return "\n".join(lines)