import numpy as np
from typing import NamedTuple, List, Dict, Any, Optional
from models.model_metric import Metric

class ImpulseResponseInfo(NamedTuple):       
    t_peak: Metric = Metric(value=np.nan,label="Peak time", unit="s")             
    y_peak: Metric = Metric(value=np.nan,label="Peak value", unit="-")
    t_settling: Metric = Metric(value=np.nan,label="Settling time (within 2%)", unit="s")  
    t_half: Metric = Metric(value=np.nan,label="Half-peak time", unit="s")     
    decay_rate: Metric = Metric(value=np.nan,label="Exponential decay constant", unit="1/s")    
    integral: Metric = Metric(value=np.nan,label="DC gain (total area)", unit="-")   
    energy: Metric = Metric(value=np.nan,label="Signal energy (∫h²(t)dt)", unit="-")  
    num_oscillations: Metric = Metric(value=np.nan,label="Number of oscillations", unit="-")    
    damping_ratio: Metric = Metric(value=np.nan,label="Estimated damping ratio", unit="-")       
    natural_freq: Metric = Metric(value=np.nan,label="Estimated undamped natural frequency", unit="rad/s")     

    def __str__(self, indent: int = 2, name_width: int = 12, type_width: int = 30):
        pad = " " * indent
        lines = [
            f"ImpulseResponseInfo(",
            f"  {'t_peak:':<{name_width}} {str(type(self.t_peak)):>{type_width}} = {self.t_peak}",
            f"  {'y_peak:':<{name_width}} {str(type(self.y_peak)):>{type_width}} = {self.y_peak}",
            f"  {'t_settling:':<{name_width}} {str(type(self.t_settling)):>{type_width}} = {self.t_settling}",
            f"  {'t_half:':<{name_width}} {str(type(self.t_half)):>{type_width}} = {self.t_half}",
            f"  {'decay_rate:':<{name_width}} {str(type(self.decay_rate)):>{type_width}} = {self.decay_rate}",
            f"  {'integral:':<{name_width}} {str(type(self.integral)):>{type_width}} = {self.integral}",
            f"  {'energy:':<{name_width}} {str(type(self.energy)):>{type_width}} = {self.energy}",
            f"  {'num_oscillations:':<{name_width}} {str(type(self.num_oscillations)):>{type_width}} = {self.num_oscillations}",
            f"  {'damping_ratio:':<{name_width}} {str(type(self.damping_ratio)):>{type_width}} = {self.damping_ratio}",
            f"  {'natural_freq:':<{name_width}} {str(type(self.natural_freq)):>{type_width}} = {self.natural_freq}",
            f")"
        ]
        return "\n".join(lines)  