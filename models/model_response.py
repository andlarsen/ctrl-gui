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

    def __str__(self, name_width: int = 12, type_width: int = 30):
        lines = [
            f"Response(",
            f"  {'t_vals:':<{name_width}} {str(type(self.t_vals)):>{type_width}} = {self.t_vals[:4]}...{self.t_vals[-1]}",
            f"  {'y_vals:':<{name_width}} {str(type(self.y_vals)):>{type_width}} = {self.y_vals[:4]}...{self.y_vals[-1]}",
            f"  {'r_vals:':<{name_width}} {str(type(self.r_vals)):>{type_width}} = {self.r_vals[:4]}...{self.r_vals[-1]}",
            f"  {'t_range:':<{name_width}} {str(type(self.t_range)):>{type_width}} = ({self.t_range[0]},{self.t_range[1]})",
            f"  {'delay_time:':<{name_width}} {str(type(self.delay_time)):>{type_width}} = {self.delay_time}",
            f"  {'response_type:':<{name_width}} {str(type(self.response_type)):>{type_width}} = {self.response_type}",
            f"  {'info:':<{name_width}} {str(type(self.info)):>{type_width}} =",
            f"{self.info.__str__(6, name_width, type_width)}",
            f")"
        ]
        return "\n".join(lines)