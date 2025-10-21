from typing import NamedTuple, List, Dict, Any, Optional

class MarginModel(NamedTuple):
    gain_margin: float
    w_gain_crossover: float
    w_gain_crossover_found: bool 
    
    phase_margin: float 
    w_phase_crossover: float
    w_phase_crossover_found: bool 

    def __str__(self, indent: int = 2, name_width: int = 12, type_width: int = 30):
        pad_title = " " * indent
        pad = " " * (indent+2)
        lines = [
            f"{pad_title}MarginModel(",
            f"{pad}{'gain_margin:':<{name_width}} {str(type(self.gain_margin)):>{type_width}} = {self.gain_margin}",
            f"{pad}{'w_gain_crossover:':<{name_width}} {str(type(self.w_gain_crossover)):>{type_width}} = {self.w_gain_crossover}",
            f"{pad}{'w_gain_crossover_found:':<{name_width}} {str(type(self.w_gain_crossover_found)):>{type_width}} = {self.w_gain_crossover_found}",
            f"{pad}{'phase_margin:':<{name_width}} {str(type(self.phase_margin)):>{type_width}} = {self.phase_margin}",
            f"{pad}{'w_phase_crossover:':<{name_width}} {str(type(self.w_phase_crossover)):>{type_width}} = {self.w_phase_crossover}",
            f"{pad}{'w_phase_crossover_found:':<{name_width}} {str(type(self.w_phase_crossover_found)):>{type_width}} = {self.w_phase_crossover_found}",
            f"{' ' * (indent)})"
        ]
        return "\n".join(lines)
    