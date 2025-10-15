
from typing import NamedTuple, List, Dict, Any, Optional

class Metric(NamedTuple):
    value: Optional[float] = None
    label: str = "N/A"
    unit: str = ""