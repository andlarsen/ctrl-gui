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