from dataclasses import dataclass, field
from typing import NamedTuple, List, Dict, Any, Optional
import sympy as sp
import scipy
import scipy.signal as signal
from models.model_ramp_response_info import RampResponseInfo

@dataclass
class RampResponse:
    t_vals: List[float] = field(default_factory=list)
    r_vals: List[float] = field(default_factory=list)
    y_vals: List[float] = field(default_factory=list)
    delay_time: float = 1
    info: Dict[str, Any] = field(default_factory=RampResponseInfo)

