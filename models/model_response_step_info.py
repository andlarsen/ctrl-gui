import numpy as np
from typing import NamedTuple, List, Dict, Any, Optional
from models.model_metric import MetricModel

class StepResponseInfoModel(NamedTuple):
    y_initial: MetricModel = MetricModel(value=np.nan,label="Initial value", unit="-")
    y_final: MetricModel = MetricModel(value=np.nan,label="Final value", unit="-")            
    t_rise: MetricModel = MetricModel(value=np.nan,label="Rise time (10% --> 90%)", unit="s")              
    t_peak: MetricModel = MetricModel(value=np.nan,label="Peak time", unit="s")             
    y_peak: MetricModel = MetricModel(value=np.nan,label="Peak value", unit="-")             
    overshoot_percent: MetricModel = MetricModel(value=np.nan,label="Overshoot", unit="%")
    t_settling: MetricModel = MetricModel(value=np.nan,label="Settling time (within 2%)", unit="s")        
    num_oscillations: MetricModel = MetricModel(value=np.nan,label="Number of oscillations", unit="-")    
    damping_ratio: MetricModel = MetricModel(value=np.nan,label="Estimated damping ratio", unit="-")       
    natural_freq: MetricModel = MetricModel(value=np.nan,label="Estimated undamped natural frequency", unit="rad/s")       

    def __str__(self, indent: int = 2, name_width: int = 12, type_width: int = 30):
        pad_title = " " * indent
        pad = " " * (indent+2)
        next_indent = 6
        lines = [
            f"{pad_title}StepResponseInfoModel(",
            f"{pad}{'y_initial:':<{name_width}} {str(type(self.y_initial)):>{type_width}} = ",
            f"{self.y_initial.__str__(indent + next_indent, name_width, type_width)}",
            f"{pad}{'y_final:':<{name_width}} {str(type(self.y_final)):>{type_width}} = ",
            f"{self.y_final.__str__(indent + next_indent, name_width, type_width)}",
            f"{pad}{'t_rise:':<{name_width}} {str(type(self.t_rise)):>{type_width}} = ",
            f"{self.t_rise.__str__(indent + next_indent, name_width, type_width)}",
            f"{pad}{'t_peak:':<{name_width}} {str(type(self.t_peak)):>{type_width}} = ",
            f"{self.t_peak.__str__(indent + next_indent, name_width, type_width)}",
            f"{pad}{'y_peak:':<{name_width}} {str(type(self.y_peak)):>{type_width}} = ",
            f"{self.y_peak.__str__(indent + next_indent, name_width, type_width)}",
            f"{pad}{'overshoot_percent:':<{name_width}} {str(type(self.overshoot_percent)):>{type_width}} = ",
            f"{self.overshoot_percent.__str__(indent + next_indent, name_width, type_width)}",
            f"{pad}{'t_settling:':<{name_width}} {str(type(self.t_settling)):>{type_width}} = ",
            f"{self.t_settling.__str__(indent + next_indent, name_width, type_width)}",
            f"{pad}{'num_oscillations:':<{name_width}} {str(type(self.num_oscillations)):>{type_width}} = ",
            f"{self.num_oscillations.__str__(indent + next_indent, name_width, type_width)}",
            f"{pad}{'damping_ratio:':<{name_width}} {str(type(self.damping_ratio)):>{type_width}} = ",
            f"{self.damping_ratio.__str__(indent + next_indent, name_width, type_width)}",
            f"{pad}{'natural_freq:':<{name_width}} {str(type(self.natural_freq)):>{type_width}} = ",
            f"{self.natural_freq.__str__(indent + next_indent, name_width, type_width)}",
            f"{' ' * (indent)})"
        ]
        return "\n".join(lines)  