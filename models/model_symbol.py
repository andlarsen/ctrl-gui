
import sympy as sp
from typing import NamedTuple, Optional

class SymbolModel(NamedTuple):
    symbol: sp.Symbol = None
    description: str = "N/A"
    is_real: bool = None
    is_positive: bool = None
    is_constant: bool = None
    is_global: bool = None

    def __str__(self, indent: int = 4, name_width: int = 12, type_width: int = 30):
        pad_title = " " * indent
        pad = " " * (indent+2)
        lines = [
            f"{pad_title}SymbolModel(",
            f"{pad}{'symbol:':<{name_width}} {str(type(self.symbol)):>{type_width}} = {self.symbol}",
            f"{pad}{'description:':<{name_width}} {str(type(self.description)):>{type_width}} = {self.description}",
            f"{pad}{'is_real:':<{name_width}} {str(type(self.is_real)):>{type_width}} = {self.is_real}",
            f"{pad}{'is_positive:':<{name_width}} {str(type(self.is_positive)):>{type_width}} = {self.is_positive}",
            f"{pad}{'is_constant:':<{name_width}} {str(type(self.is_constant)):>{type_width}} = {self.is_constant}",
            f"{pad}{'is_global:':<{name_width}} {str(type(self.is_global)):>{type_width}} = {self.is_global}",
            f"{' ' * (indent)})"
        ]
        return "\n".join(lines)
