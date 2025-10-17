import numpy as np
from typing import NamedTuple, List, Dict, Any, Optional
from models.model_metric import Metric

class RampResponseInfo(NamedTuple):          
    y_final: Metric = Metric(value=np.nan,label="Final value", unit="-")                     
    t_peak: Metric = Metric(value=np.nan,label="Peak time", unit="s")             
    y_peak: Metric = Metric(value=np.nan,label="Peak value", unit="-")
    e_final: Metric = Metric(value=np.nan,label="Final error (e_ss)", unit="-")  
    Kv: Metric = Metric(value=np.nan,label="Kv = 1 / e_ss", unit="-")     
    t_lag: Metric = Metric(value=np.nan,label="Time lag between input and output", unit="s")    
    e_max_tracking: Metric = Metric(value=np.nan,label="Maximum tracking error", unit="-")    

    def __str__(self, indent: int = 2, name_width: int = 12, type_width: int = 30):
        pad = " " * indent
        lines = [
            f"RampResponseInfo(",
            f"  {'y_final:':<{name_width}} {str(type(self.y_final)):>{type_width}} = {self.y_final}",
            f"  {'t_peak:':<{name_width}} {str(type(self.t_peak)):>{type_width}} = {self.t_peak}",
            f"  {'y_peak:':<{name_width}} {str(type(self.y_peak)):>{type_width}} = {self.y_peak}",
            f"  {'e_final:':<{name_width}} {str(type(self.e_final)):>{type_width}} = {self.e_final}",
            f"  {'Kv:':<{name_width}} {str(type(self.Kv)):>{type_width}} = {self.Kv}",
            f"  {'t_lag:':<{name_width}} {str(type(self.t_lag)):>{type_width}} = {self.t_lag}",
            f"  {'e_max_tracking:':<{name_width}} {str(type(self.e_max_tracking)):>{type_width}} = {self.e_max_tracking}",
            f")"
        ]
        return "\n".join(lines)  