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