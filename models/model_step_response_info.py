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