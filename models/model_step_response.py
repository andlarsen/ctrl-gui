from dataclasses import dataclass, field
from typing import NamedTuple, List, Tuple, Dict, Any, Optional
import sympy as sp
import scipy
import scipy.signal as signal
from models.model_step_response_info import StepResponseInfo

@dataclass
class StepResponse:
    t_vals: List[float] = field(default_factory=list)
    y_vals: List[float] = field(default_factory=list)
    t_range: Tuple[float,float] = field(default_factory=tuple)
    delay_time: float = 0
    info: Dict[str, Any] = field(default_factory=StepResponseInfo)

