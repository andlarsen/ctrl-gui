import numpy as np
from typing import NamedTuple, List, Dict, Any, Optional
from models.model_metric import Metric

class StepResponseInfo(NamedTuple):
    y_initial: Metric = Metric(value=np.nan,label="Initial value", unit="-")
    y_final: Metric = Metric(value=np.nan,label="Final value", unit="-")            
    t_rise: Metric = Metric(value=np.nan,label="Rise time (10% --> 90%)", unit="s")              
    t_peak: Metric = Metric(value=np.nan,label="Peak time", unit="s")             
    y_peak: Metric = Metric(value=np.nan,label="Peak value", unit="-")             
    overshoot_percent: Metric = Metric(value=np.nan,label="Overshoot", unit="%")
    t_settling: Metric = Metric(value=np.nan,label="Settling time (within 2%)", unit="s")        
    num_oscillations: Metric = Metric(value=np.nan,label="Number of oscillations", unit="-")    
    damping_ratio: Metric = Metric(value=np.nan,label="Estimated damping ratio", unit="-")       
    natural_freq: Metric = Metric(value=np.nan,label="Estimated undamped natural frequency", unit="rad/s")       

    def __str__(self, indent: int = 2, name_width: int = 12, type_width: int = 30):
        pad = " " * indent
        lines = [
            f"StepResponseInfo(",
            f"  {'y_initial:':<{name_width}} {str(type(self.y_initial)):>{type_width}} = {self.y_initial}",
            f"  {'y_final:':<{name_width}} {str(type(self.y_final)):>{type_width}} = {self.y_final}",
            f"  {'t_rise:':<{name_width}} {str(type(self.t_rise)):>{type_width}} = {self.t_rise}",
            f"  {'t_peak:':<{name_width}} {str(type(self.t_peak)):>{type_width}} = {self.t_peak}",
            f"  {'y_peak:':<{name_width}} {str(type(self.y_peak)):>{type_width}} = {self.y_peak}",
            f"  {'overshoot_percent:':<{name_width}} {str(type(self.overshoot_percent)):>{type_width}} = {self.overshoot_percent}",
            f"  {'t_settling:':<{name_width}} {str(type(self.t_settling)):>{type_width}} = {self.t_settling}",
            f"  {'num_oscillations:':<{name_width}} {str(type(self.num_oscillations)):>{type_width}} = {self.num_oscillations}",
            f"  {'damping_ratio:':<{name_width}} {str(type(self.damping_ratio)):>{type_width}} = {self.damping_ratio}",
            f"  {'natural_freq:':<{name_width}} {str(type(self.natural_freq)):>{type_width}} = {self.natural_freq}",
            f")"
        ]
        return "\n".join(lines)  