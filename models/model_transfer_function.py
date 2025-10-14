from dataclasses import dataclass
import sympy as sp
import scipy.signal as sig

@dataclass
class TransferFunctionModel:
    symbolic: sp.Expr | None = None
    numeric: sig.TransferFunction | None = None