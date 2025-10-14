from dataclasses import dataclass
import sympy as sp
import scipy.signal as sig

@dataclass
class EquationModel:
    string: str = ""
    symbolic: sp.Eq | None = None
    numeric: sp.Eq | None = None