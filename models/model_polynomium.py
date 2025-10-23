
import utilities.utils_transfer_function as utils_tf
import sympy as sp

from dataclasses import dataclass, field
from typing import Optional
from models.model_coefficients import CoefficientsModel

# Import logger
import logging
from utilities.logger import get_logger, header, subheader, subsubheader
log = get_logger(__name__, level=logging.DEBUG, logfile='logs/main.log')

@dataclass
class PolynomiumModel:
    _symbolic: sp.Expr = field(default=sp.S.One, init=False, repr=False)
    initial_symbolic: sp.Expr = None

    _numeric: sp.Expr = field(default=sp.S.One, init=False, repr=False)
    initial_numeric: sp.Expr = None

    coefficients: CoefficientsModel = field(default_factory=CoefficientsModel)

    def __str__(self, indent: int = 2, name_width: int = 12, type_width: int = 30):
        pad_title = " " * indent
        pad = " " * (indent+2)
        lines = [
            f"{pad_title}PolynomiumModel(",
            f"{pad}{'symbolic:':<{name_width}} {str(type(self.symbolic)):>{type_width}} = {self.symbolic}",
            f"{pad}{'numeric:':<{name_width}} {str(type(self.numeric)):>{type_width}} = {self.numeric}",
            f"{pad}{'coefficients:':<{name_width}} {str(type(self.coefficients)):>{type_width}} =",
            f"{self.coefficients.__str__(indent + 6, name_width, type_width)}",
            f"{' ' * (indent)})"
        ]
        return "\n".join(lines)
    
    def __post_init__(self):
        self.symbolic = self.initial_symbolic
        self.numeric = self.initial_numeric

    @property
    def symbolic(self) -> sp.Expr:
        log.debug(f"Getting @property.symbolic")
        return self._symbolic

    @symbolic.setter
    def symbolic(self, new_expr: sp.Expr):
        self._symbolic = new_expr
        if new_expr is not None:
            subsubheader(log, "Updating symbolic coefficients", level=logging.DEBUG)
            self.coefficients.symbolic = utils_tf.get_coefficients(new_expr)
            subsubheader(log, "End of updating symbolic coefficients", level=logging.DEBUG)

    @property
    def numeric(self) -> sp.Expr:
        log.debug(f"Getting @property.numeric")
        return self._numeric

    @numeric.setter
    def numeric(self, new_expr: sp.Expr):
        self._numeric = new_expr
        if new_expr is not None:
            subsubheader(log, "Updating numeric coefficients", level=logging.DEBUG)
            self.coefficients.numeric = utils_tf.get_coefficients(new_expr)
            subsubheader(log, "End of updating numeric coefficients", level=logging.DEBUG)