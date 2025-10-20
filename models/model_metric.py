
from typing import NamedTuple, List, Dict, Any, Optional

class MetricModel(NamedTuple):
    value: Optional[float] = None
    label: str = "N/A"
    unit: str = "N/A"

    def __str__(self, indent: int = 4, name_width: int = 12, type_width: int = 30):
        pad_title = " " * indent
        pad = " " * (indent+2)
        lines = [
            f"{pad_title}MetricModel(",
            f"{pad}{'value:':<{name_width}} {str(type(self.value)):>{type_width}} = {self.value}",
            f"{pad}{'label:':<{name_width}} {str(type(self.label)):>{type_width}} = {self.label}",
            f"{pad}{'unit:':<{name_width}} {str(type(self.unit)):>{type_width}} = {self.unit}",
            f"{' ' * (indent)})"
        ]
        return "\n".join(lines)
