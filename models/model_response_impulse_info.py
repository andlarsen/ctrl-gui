import numpy as np
from typing import NamedTuple, List, Dict, Any, Optional
from models.model_metric import MetricModel

class ImpulseResponseInfoModel(NamedTuple):       
    t_peak: MetricModel = MetricModel(value=np.nan,label="Peak time", unit="s")             
    y_peak: MetricModel = MetricModel(value=np.nan,label="Peak value", unit="-")
    t_settling: MetricModel = MetricModel(value=np.nan,label="Settling time (within 2%)", unit="s")  
    t_half: MetricModel = MetricModel(value=np.nan,label="Half-peak time", unit="s")     
    decay_rate: MetricModel = MetricModel(value=np.nan,label="Exponential decay constant", unit="1/s")    
    integral: MetricModel = MetricModel(value=np.nan,label="DC gain (total area)", unit="-")   
    energy: MetricModel = MetricModel(value=np.nan,label="Signal energy (∫h²(t)dt)", unit="-")  
    num_oscillations: MetricModel = MetricModel(value=np.nan,label="Number of oscillations", unit="-")    
    damping_ratio: MetricModel = MetricModel(value=np.nan,label="Estimated damping ratio", unit="-")       
    natural_freq: MetricModel = MetricModel(value=np.nan,label="Estimated undamped natural frequency", unit="rad/s")     

    def __str__(self, indent: int = 2, name_width: int = 12, type_width: int = 30):
        pad_title = " " * indent
        pad = " " * (indent+1)
        next_indent = 6
        lines = [
            f"{pad_title}ImpulseResponseInfoModel(",
            f"{pad}{'t_peak:':<{name_width}} {str(type(self.t_peak)):>{type_width}} = ",
            f"{self.t_peak.__str__(indent + next_indent, name_width, type_width)}",
            f"{pad}{'y_peak:':<{name_width}} {str(type(self.y_peak)):>{type_width}} = ",
            f"{self.y_peak.__str__(indent + next_indent, name_width, type_width)}",
            f"{pad}{'t_settling:':<{name_width}} {str(type(self.t_settling)):>{type_width}} = ",
            f"{self.t_settling.__str__(indent + next_indent, name_width, type_width)}",
            f"{pad}{'t_half:':<{name_width}} {str(type(self.t_half)):>{type_width}} = ",
            f"{self.t_half.__str__(indent + next_indent, name_width, type_width)}",
            f"{pad}{'decay_rate:':<{name_width}} {str(type(self.decay_rate)):>{type_width}} = ",
            f"{self.decay_rate.__str__(indent + next_indent, name_width, type_width)}",
            f"{pad}{'integral:':<{name_width}} {str(type(self.integral)):>{type_width}} = ",
            f"{self.integral.__str__(indent + next_indent, name_width, type_width)}",
            f"{pad}{'energy:':<{name_width}} {str(type(self.energy)):>{type_width}} = ",
            f"{self.energy.__str__(indent + next_indent, name_width, type_width)}",
            f"{pad}{'num_oscillations:':<{name_width}} {str(type(self.num_oscillations)):>{type_width}} = ",
            f"{self.num_oscillations.__str__(indent + next_indent, name_width, type_width)}",
            f"{pad}{'damping_ratio:':<{name_width}} {str(type(self.damping_ratio)):>{type_width}} = ",
            f"{self.damping_ratio.__str__(indent + next_indent, name_width, type_width)}",
            f"{pad}{'natural_freq:':<{name_width}} {str(type(self.natural_freq)):>{type_width}} = ",
            f"{self.natural_freq.__str__(indent + next_indent, name_width, type_width)}",
            f"{' ' * (indent)})"
        ]
        return "\n".join(lines)  