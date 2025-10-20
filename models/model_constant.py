
import sympy as sp
from typing import NamedTuple, Optional

class ConstantModel(NamedTuple):
    value: Optional[float] = None
    description: str = "N/A"
    unit: str = "N/A"
    symbol: sp.Symbol = None
    is_global: bool = False

    def __str__(self, indent: int = 4, name_width: int = 12, type_width: int = 30):
        pad_title = " " * indent
        pad = " " * (indent+2)
        lines = [
            f"{pad_title}ConstantModel(",
            f"{pad}{'value:':<{name_width}} {str(type(self.value)):>{type_width}} = {self.value}",
            f"{pad}{'description:':<{name_width}} {str(type(self.description)):>{type_width}} = {self.description}",
            f"{pad}{'unit:':<{name_width}} {str(type(self.unit)):>{type_width}} = {self.unit}",
            f"{pad}{'symbol:':<{name_width}} {str(type(self.symbol)):>{type_width}} = {self.symbol}",
            f"{pad}{'is_global:':<{name_width}} {str(type(self.is_global)):>{type_width}} = {self.is_global}",
            f"{' ' * (indent)})"
        ]
        return "\n".join(lines)
