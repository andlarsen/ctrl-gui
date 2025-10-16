from dataclasses import dataclass, field
from typing import List, Tuple, Union, Optional
import numpy as np
from models.model_response_impulse_info import ImpulseResponseInfo
from models.model_response_step_info import StepResponseInfo
from models.model_response_ramp_info import RampResponseInfo

@dataclass
class Response:
    t_vals: List[float] = field(default_factory=list)
    y_vals: List[float] = field(default_factory=list)
    r_vals: List[float] = field(default_factory=list)
    t_range: Tuple[float, float] = field(default_factory=tuple)
    delay_time: float = 0
    response_type: str = 'step'
    info: Union[ImpulseResponseInfo, StepResponseInfo, RampResponseInfo] = field(default_factory=lambda: StepResponseInfo())