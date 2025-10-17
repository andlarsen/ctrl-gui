
from typing import NamedTuple, List, Dict, Any, Optional

class Metric(NamedTuple):
    value: Optional[float] = None
    label: str = "N/A"
    unit: str = ""

    def __str__(self, indent: int = 4, name_width: int = 12, type_width: int = 30):
        pad = " " * indent
        lines = [
            f"Polynomium(",
            f"{pad}{'value:':<{name_width}} {str(type(self.value)):>{type_width}} = {self.value}",
            f"{pad}{'label:':<{name_width}} {str(type(self.label)):>{type_width}} = {self.label}",
            f"{pad}{'unit:':<{name_width}} {str(type(self.unit)):>{type_width}} = {self.unit}",
            f"{' ' * (indent - 2)})"
        ]
        return "\n".join(lines)
