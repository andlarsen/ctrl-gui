from dataclasses import dataclass, field
from typing import List, Tuple, Union, Optional
import numpy as np
from models.model_response_impulse_info import ImpulseResponseInfoModel
from models.model_response_step_info import StepResponseInfoModel
from models.model_response_ramp_info import RampResponseInfoModel

@dataclass
class ResponseModel:
    t_vals: Optional[List[float]] = None
    y_vals: Optional[List[float]] = None
    r_vals: Optional[List[float]] = None
    w_vals: Optional[List[float]] = None
    F_vals: Optional[List[float]] = None
    mag_vals: Optional[List[float]] = None
    phase_vals: Optional[List[float]] = None
    delay_time: Optional[float] = None
    response_type: Optional[str] = None
    info: Optional[Union[ImpulseResponseInfoModel, StepResponseInfoModel, RampResponseInfoModel]] = None

    def __str__(self, indent: int = 2, name_width: int = 14, type_width: int = 35):
        pad_title = " " * indent
        pad = " " * (indent + 2)
        next_indent = 6

        def format_list_preview(values: List[float], n: int = 3):
            if not values:
                return "None"
            if len(values) <= n + 1:
                return str(values)
            return f"[{', '.join(f'{v:.3g}' for v in values[:n])}, ..., {values[-1]:.3g}]"

        # Build output lines
        if self.response_type == 'frequency':
            lines = [
                f"{pad_title}ResponseModel(",
                f"{pad}{'response_type:':<{name_width}} {str(type(self.response_type)).replace('class ', ''):>{type_width}} = {self.response_type}",
                f"{pad}{'w_vals:':<{name_width}} {str(type(self.w_vals)).replace('class ', ''):>{type_width}} = {format_list_preview(self.w_vals)}",
                f"{pad}{'F_vals:':<{name_width}} {str(type(self.F_vals)).replace('class ', ''):>{type_width}} = {format_list_preview(self.F_vals)}",
                f"{pad}{'mag_vals:':<{name_width}} {str(type(self.mag_vals)).replace('class ', ''):>{type_width}} = {format_list_preview(self.mag_vals)}",
                f"{pad}{'phase_vals:':<{name_width}} {str(type(self.phase_vals)).replace('class ', ''):>{type_width}} = {format_list_preview(self.phase_vals)}",
                f"{pad}{'info:':<{name_width}} {str(type(self.info)):>{type_width}} =",
                f"{self.info.__str__(indent + next_indent, name_width, type_width)}",
                f"{pad_title})"
                ]
        elif self.response_type == 'impulse' or self.response_type == 'step' or self.response_type == 'ramp':
            lines = [
                f"{pad_title}ResponseModel(",
                f"{pad}{'response_type:':<{name_width}} {str(type(self.response_type)).replace('class ', ''):>{type_width}} = {self.response_type}",
                f"{pad}{'delay_time:':<{name_width}} {str(type(self.delay_time)).replace('class ', ''):>{type_width}} = {self.delay_time}",
                f"{pad}{'t_vals:':<{name_width}} {str(type(self.t_vals)).replace('class ', ''):>{type_width}} = {format_list_preview(self.t_vals)}",
                f"{pad}{'y_vals:':<{name_width}} {str(type(self.y_vals)).replace('class ', ''):>{type_width}} = {format_list_preview(self.y_vals)}",
                f"{pad}{'r_vals:':<{name_width}} {str(type(self.r_vals)).replace('class ', ''):>{type_width}} = {format_list_preview(self.r_vals)}",
                f"{pad}{'info:':<{name_width}} {str(type(self.info)):>{type_width}} =",
                f"{self.info.__str__(indent + next_indent, name_width, type_width)}",
                f"{pad_title})"
                ]
        else:
            raise ValueError(f"Unknown response_type: {self.response_type}")

        return "\n".join(lines)
    
    def __post_init__(self):
        if self.info is None:
            if self.response_type == "impulse":
                self.info = ImpulseResponseInfoModel()
            elif self.response_type == "step":
                self.info = StepResponseInfoModel()
            elif self.response_type == "ramp":
                self.info = RampResponseInfoModel()
            elif self.response_type == "frequency":
                self.info = RampResponseInfoModel()
            else:
                raise ValueError(f"Unknown response_type: {self.response_type}")
