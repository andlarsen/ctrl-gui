import numpy as np
from typing import NamedTuple, List, Dict, Any, Optional
from models.model_metric import MetricModel

class RampResponseInfoModel(NamedTuple):          
    y_final: MetricModel = MetricModel(value=np.nan,label="Final value", unit="-")                     
    t_peak: MetricModel = MetricModel(value=np.nan,label="Peak time", unit="s")             
    y_peak: MetricModel = MetricModel(value=np.nan,label="Peak value", unit="-")
    e_final: MetricModel = MetricModel(value=np.nan,label="Final error (e_ss)", unit="-")  
    Kv: MetricModel = MetricModel(value=np.nan,label="Kv = 1 / e_ss", unit="-")     
    t_lag: MetricModel = MetricModel(value=np.nan,label="Time lag between input and output", unit="s")    
    e_max_tracking: MetricModel = MetricModel(value=np.nan,label="Maximum tracking error", unit="-")    

    def __str__(self, indent: int = 2, name_width: int = 12, type_width: int = 30):
        pad_title = " " * indent
        pad = " " * (indent+2)
        next_indent = 6
        lines = [
            f"{pad_title}RampResponseInfoModel(",
            f"{pad}{'y_final:':<{name_width}} {str(type(self.y_final)):>{type_width}} = ",
            f"{self.y_final.__str__(indent + next_indent, name_width, type_width)}",
            f"{pad}{'t_peak:':<{name_width}} {str(type(self.t_peak)):>{type_width}} = ",
            f"{self.t_peak.__str__(indent + next_indent, name_width, type_width)}",
            f"{pad}{'y_peak:':<{name_width}} {str(type(self.y_peak)):>{type_width}} = ",
            f"{self.y_peak.__str__(indent + next_indent, name_width, type_width)}",
            f"{pad}{'e_final:':<{name_width}} {str(type(self.e_final)):>{type_width}} = ",
            f"{self.e_final.__str__(indent + next_indent, name_width, type_width)}",
            f"{pad}{'Kv:':<{name_width}} {str(type(self.Kv)):>{type_width}} = ",
            f"{self.Kv.__str__(indent + next_indent, name_width, type_width)}",
            f"{pad}{'t_lag:':<{name_width}} {str(type(self.t_lag)):>{type_width}} = ",
            f"{self.t_lag.__str__(indent + next_indent, name_width, type_width)}",
            f"{pad}{'e_max_tracking:':<{name_width}} {str(type(self.e_max_tracking)):>{type_width}} = ",
            f"{self.e_max_tracking.__str__(indent + next_indent, name_width, type_width)}",
            f"{' ' * (indent)})"
        ]
        return "\n".join(lines)  