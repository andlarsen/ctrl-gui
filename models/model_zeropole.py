from dataclasses import dataclass, field
import sympy as sp
import scipy.signal as sig

@dataclass
class ZeroPoleModel:
    symbolic: sp.Eq = field(default_factory=dict)
    numeric: sp.Eq = field(default_factory=dict)