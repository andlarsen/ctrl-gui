from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Any, Optional
import sympy as sp
import scipy
import scipy.signal as signal
from models.model_impulse_response_info import ImpulseResponseInfo
    
@dataclass
class ImpulseResponse:
    t_vals: List[float] = field(default_factory=list)
    y_vals: List[float] = field(default_factory=list)
    t_range: Tuple[float,float] = field(default_factory=tuple)
    delay_time: float = 0
    info: Dict[str, Any] = field(default_factory=ImpulseResponseInfo)

