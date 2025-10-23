"""
Transfer Function Model Module

This module defines the TransferFunctionModel dataclass which represents a complete
transfer function system with symbolic/numeric representations, time/frequency responses,
and various analytical properties.

Author: Andreas
Date: 2025
"""

# Import of modules
import sympy as sp
import numpy as np
import scipy.signal as sig

from dataclasses import dataclass, field
from typing import Optional, List, Tuple, Dict, Any

# Import all utilities
import utilities.utils_transfer_function as utils_tf

# Import all model dependencies
from models.model_polynomium import PolynomiumModel
from models.model_equation import EquationModel
from models.model_response import (
    ResponseModel, 
    ImpulseResponseInfoModel, 
    StepResponseInfoModel, 
    RampResponseInfoModel)
from models.model_constant import ConstantModel
from models.model_symbol import SymbolModel
from models.model_zeropole import ZeroPoleModel
from models.model_metric import MetricModel
from models.model_margin import MarginModel

# Import logger
import logging
from utilities.logger import get_logger, header, subheader, subsubheader
log = get_logger(__name__, level=logging.DEBUG, logfile='logs/main.log')

# ============================================================================
# MAIN TRANSFER FUNCTION MODEL
# ============================================================================

@dataclass
class TransferFunctionModel:
    # ========================================
    # Basic Identification
    # ========================================
    name: str = ""
    description: str = ""

    # ========================================
    # Signal Definitions
    # ========================================
    tf_name: Tuple[str,str] = field(default_factory=tuple)
    input: Tuple[str,str] = field(default_factory=tuple)
    output: Tuple[str,str] = field(default_factory=tuple)

    # ========================================
    # Symbolic Environment
    # ========================================
    symbols: Dict[str,SymbolModel] = field(default_factory=dict)
    global_symbols: Optional[Dict[str,sp.Symbol]] = field(default_factory=dict)
    constants: Dict[str, ConstantModel] = field(default_factory=dict)
    global_constants: Optional[Dict[str,ConstantModel]] = field(default_factory=dict)
    functions: Dict[str,sp.Expr] = field(default_factory=dict)

    # ========================================
    # Transfer Function Representations
    # ========================================
    _symbolic: Optional[sp.Expr] = field(default=sp.S.One, init=False, repr=False)
    initial_symbolic: Optional[sp.Expr] = field(default=sp.S.One)

    _numeric: Optional[sp.Expr] = field(default=sp.S.One, init=False, repr=False)
    initial_numeric: Optional[sp.Expr] = field(default=sp.S.One)

    signal: Optional[sig.TransferFunction] = None

    # ========================================
    # Polynomial Representations
    # ========================================
    numerator: PolynomiumModel = field(default_factory=PolynomiumModel)
    denominator: PolynomiumModel = field(default_factory=PolynomiumModel)
    poles: ZeroPoleModel = field(default_factory=ZeroPoleModel)
    zeros: ZeroPoleModel = field(default_factory=ZeroPoleModel)

    # ========================================
    # Differential Equation
    # ========================================
    differential_equation: Optional[EquationModel] = field(default_factory=EquationModel)
    
    # ========================================
    # Time-Domain Analysis
    # ========================================
    t_range: Optional[Tuple[float,float]] = None
    impulse_response: Optional[ResponseModel] = field(default_factory=lambda: ResponseModel(response_type='impulse'))
    step_response: Optional[ResponseModel] = field(default_factory=lambda: ResponseModel(response_type='step'))
    ramp_response: Optional[ResponseModel] = field(default_factory=lambda: ResponseModel(response_type='ramp'))

    # ========================================
    # Frequency-Domain Analysis
    # ========================================
    w_range: Optional[Tuple[float,float]] = None
    frequency_response: Optional[ResponseModel] = field(default_factory=lambda: ResponseModel(response_type='frequency'))
    margin: Optional[MarginModel] = None

# ========================================================================
# LOGGER HELPER FUNCTIONS
# ========================================================================
    def _safe_update(self, function: callable, component_name: str, 
                    fallback_value: Any = None, critical: bool = False) -> Any:
        """
        Safely execute an update function with error handling and logging.
        """
        try:
            result = function()
            log.debug(f"TransferFunctionModel({self.name}): Successfully updated {component_name}",
                      stacklevel=2)
            return result
        except Exception as e:
            log.error(f"TransferFunctionModel({self.name}): Failed to update {component_name}: {e}", exc_info=True)
            if critical:
                raise
            return fallback_value

# ========================================================================
# PROPERTY ACCESSORS (WITH CASCADE UPDATES)
# ========================================================================
    @property
    def symbolic(self) -> Optional[sp.Expr]:
        """Get the symbolic transfer function expression."""
        log.debug(f"TransferFunctionModel({self.name}): Getting @property.symbolic")
        return self._symbolic

    @symbolic.setter
    def symbolic(self, new_tf_symbolic: sp.Expr):
        """
        Set symbolic transfer function and cascade updates to child models.
        """
        subsubheader(log, f"TransferFunctionModel({self.name}): symbolic.setter triggered", level=logging.DEBUG)
        log.debug(f"TransferFunctionModel({self.name}): Setting new_tf_symbolic = {new_tf_symbolic}")

        # Type check
        if new_tf_symbolic is not None and not isinstance(new_tf_symbolic, sp.Basic):
            log.error(f"TransferFunctionModel({self.name}): Invalid type passed to symbolic setter. Must be a SymPy expression or None")
            raise TypeError(f"TransferFunctionModel({self.name}): Invalid type passed to symbolic setter. Must be a SymPy expression or None")
        
        # Skip update if new symbolic transfer function is identical to the current one.
        if new_tf_symbolic == self._symbolic:
            log.debug(f"TransferFunctionModel({self.name}): new_tf_symbolic is identical to self._num_symboliceric = {self._symbolic}")
            log.debug(f"TransferFunctionModel({self.name}): Skipping update of symbolic representations")
            return
        
        # Store the new symbolic transfer function
        self._symbolic = new_tf_symbolic
        log.info(f"TransferFunctionModel({self.name}): self._symbolic = {new_tf_symbolic}")

        if new_tf_symbolic is not None:
            subsubheader(log, f"TransferFunctionModel({self.name}): Updating symbolic representations", level=logging.DEBUG)
            self.numerator.symbolic = self._safe_update(
                function=lambda: utils_tf.get_numerator(new_tf_symbolic),
                component_name="self.numerator.symbolic",
                critical=True,
                fallback_value=None)
            
            self.denominator.symbolic = self._safe_update(
                function=lambda: utils_tf.get_denominator(new_tf_symbolic),
                component_name="self.denominator.symbolic",
                critical=True,
                fallback_value=None)
            
            self.zeros.symbolic = self._safe_update(
                function=lambda: utils_tf.get_zeros(new_tf_symbolic),
                component_name="self.zeros.symbolic",
                critical=True,
                fallback_value=None)
            
            self.poles.symbolic = self._safe_update(
                function=lambda: utils_tf.get_poles(new_tf_symbolic),
                component_name="self.poles.symbolic",
                critical=True,
                fallback_value=None)
            
            self.differential_equation.symbolic = self._safe_update(
                function=lambda: utils_tf.get_differential_equation(new_tf_symbolic, self.input, self.output, self.symbols),
                component_name="self.differential_equation.symbolic",
                critical=True,
                fallback_value=None)
                
            subsubheader(log, f"TransferFunctionModel({self.name}): End of updating symbolic representation", level=logging.DEBUG)

        else:
            log.warning(f"TransferFunctionModel({self.name}): new_tf_symbolic == None. Skipping symbolic representations")
        
    @property
    def numeric(self) -> Optional[sp.Expr]:
        """Get the numeric transfer function expression."""
        log.debug(f"TransferFunctionModel({self.name}): Getting @property.numeric")
        return self._numeric

    @numeric.setter
    def numeric(self, new_tf_numeric: sp.Expr):
        """
        Set numeric transfer function and cascade updates to all dependent models.
        """

        subsubheader(log, f"TransferFunctionModel({self.name}): numeric.setter triggered", level=logging.DEBUG)
        log.debug(f"TransferFunctionModel({self.name}): Setting new_tf_numeric = {new_tf_numeric}")

        # Type safety
        if new_tf_numeric is not None and not isinstance(new_tf_numeric, sp.Basic):
            log.error(f"TransferFunctionModel({self.name}): Invalid type passed to numeric setter. Must be a SymPy expression or None")
            raise TypeError(f"TransferFunctionModel({self.name}): Invalid type passed to numeric setter. Must be a SymPy expression or None")
        
        # Skip update if new numeric transfer function is identical to the current one.
        if new_tf_numeric == self._numeric:
            log.debug(f"TransferFunctionModel({self.name}): new_tf_numeric is identical to self._numeric = {self._numeric}")
            log.debug(f"TransferFunctionModel({self.name}): Skipping update of numeric representations")
            return

        # Store the new numeric transfer function
        self._numeric = new_tf_numeric
        log.info(f"TransferFunctionModel({self.name}): self._numeric = {new_tf_numeric}")

        if new_tf_numeric is not None:
            subsubheader(log, f"TransferFunctionModel({self.name}): Updating numeric representations", level=logging.DEBUG)
            self.numerator.numeric = self._safe_update(
                function=lambda: utils_tf.get_numerator(new_tf_numeric),
                component_name="self.numerator.numeric",
                critical=True,
                fallback_value=None)
            
            self.denominator.numeric = self._safe_update(
                function=lambda: utils_tf.get_denominator(new_tf_numeric),
                component_name="self.denominator.numeric",
                critical=True,
                fallback_value=None)
            
            self.zeros.numeric = self._safe_update(
                function=lambda: utils_tf.get_zeros(new_tf_numeric),
                component_name="self.zeros.numeric",
                critical=True,
                fallback_value=None)
            
            self.poles.numeric = self._safe_update(
                function=lambda: utils_tf.get_poles(new_tf_numeric),
                component_name="self.poles.numeric",
                critical=True,
                fallback_value=None)
            
            self.differential_equation.numeric = self._safe_update(
                function=lambda: utils_tf.get_differential_equation(new_tf_numeric, self.input, self.output, self.symbols),
                component_name="self.differential_equation.numeric",
                critical=True,
                fallback_value=None)
            subsubheader(log, f"TransferFunctionModel({self.name}): End of updating numeric representations", level=logging.DEBUG)
        else: 
            log.warning(f"TransferFunctionModel({self.name}): new_tf_numeric == None. Skipping numeric representations")
            
        if new_tf_numeric is not None:
            subsubheader(log, f"TransferFunctionModel({self.name}): Updating signal representation", level=logging.DEBUG)
            self.signal = self._safe_update(
                function=lambda: utils_tf.define_signal(self.numerator.coefficients.numeric,self.denominator.coefficients.numeric),
                component_name="self.signal",
                critical=True,
                fallback_value=None)
            subsubheader(log, f"TransferFunctionModel({self.name}): End of signal representation", level=logging.DEBUG)
        else: 
            log.warning(f"TransferFunctionModel({self.name}): new_tf_numeric == None. Skipping signal representation")
            
        if new_tf_numeric is not None:
            subsubheader(log, f"TransferFunctionModel({self.name}): Updating time-domain analysis", level=logging.DEBUG)
            self.t_range = self._safe_update(
                function=lambda: self.get_t_range(t_range=(),delay_time=0,n_tau=8,tol=1e-6),
                component_name="self.t_range",
                critical=True,
                fallback_value=None)
            
            self.impulse_response = self._safe_update(
                function=lambda: self.get_time_response(response_type='impulse', t_range=None, delay_time=0, n_points=500, tol=1e-6),
                component_name="self.impulse_response",
                critical=True,
                fallback_value=None)
            
            self.step_response = self._safe_update(
                function=lambda: self.get_time_response(response_type='step', t_range=None, delay_time=0, n_points=500, tol=1e-6),
                component_name="self.step_response",
                critical=True,
                fallback_value=None)
            
            self.ramp_response = self._safe_update(
                function=lambda: self.get_time_response(response_type='ramp', t_range=None, delay_time=0, n_points=500, tol=1e-6),
                component_name="self.ramp_response",
                critical=True,
                fallback_value=None)
            
            subsubheader(log, f"TransferFunctionModel({self.name}): End of updating time-domain analysis", level=logging.DEBUG)
        else: 
            log.warning(f"TransferFunctionModel({self.name}): new_tf_numeric == None. Skipping time-domain analysis")

        if new_tf_numeric is not None:
            subsubheader(log, f"TransferFunctionModel({self.name}): Updating frequency-domain analysis", level=logging.DEBUG)
            self.w_range = self._safe_update(
                function=lambda: self.get_w_range(n_decades_above_w_max=2,n_decades_below_w_min=0),
                component_name="self.w_range",
                critical=True,
                fallback_value=None)
            
            self.frequency_response = self._safe_update(
                function=lambda: self.get_frequency_response(response_type='frequency', w_range=None, n_points=500),
                component_name="self.frequency_response",
                critical=True,
                fallback_value=None)
            
            self.margin = self._safe_update(
                function=lambda: self.get_margin(),
                component_name="self.margin",
                critical=True,
                fallback_value=None)
            
            subsubheader(log, f"TransferFunctionModel({self.name}): End of updating frequency-domain analysis", level=logging.DEBUG)
        else: 
            log.debug(f"TransferFunctionModel({self.name}): new_tf_numeric == None. Skipping frequency-domain analysis")

# ========================================================================
# INITIALIZATION
# ========================================================================

    def __post_init__(self):
        header(log,f"Initializing TransferFunctionModel({self.name})")

        # if self.global_symbols:
        #     self.symbols = self.global_symbols.copy()
        # if self.global_constants:
        #     self.constants = self.global_constants.copy()

        self.name = self.name.capitalize()
        self.add_st()
        self.define_tf_name(self.name)
        self.define_input('u')
        self.define_output('y')
        subheader(log, f'TransferFunctionModel({self.name}): Initializing symbolic and numeric representations')
        self.symbolic = self.initial_symbolic
        self.numeric = self.initial_numeric
       

# ========================================================================
# STRING REPRESENTATION
# ========================================================================
    
    def __str__(self, indent: int = 2, name_width: int = 12, type_width: int = 30) -> str:
        pad_title = " " * indent
        pad = " " * (indent + 2)

        # --- Format simple fields ---
        t_range_str = f"({self.t_range[0]:.3g}, {self.t_range[1]:.3g})" if self.t_range else "None"
        w_range_str = f"({self.w_range[0]:.3g}, {self.w_range[1]:.3g})" if self.w_range else "None"

        # --- Indent Multiline Helper (Modified for better alignment) ---
        def indent_multiline(text: str, pad_start: str) -> str:
            lines = str(text).splitlines()
            if len(lines) <= 1:
                return text
            
            # All lines after the first one are prepended with the calculated padding
            return "\n".join([lines[0]] + [pad_start + line for line in lines[1:]])
        
        def format_nested_object(attr_name: str, obj: Any, indent: int, name_width: int, type_width: int) -> List[str]:
            """Helper function to format and print a nested object that has a custom __str__."""
            pad = " " * (indent + 2)
            header = f"{pad}{attr_name + ':':<{name_width}} {str(type(obj)):>{type_width}} ="
            
            # Recursively call the custom __str__ of the nested object with increased indentation
            body_lines = obj.__str__(indent=indent + 6, name_width=name_width, type_width=type_width)
            
            return [header, body_lines]
        
        def format_dict_entry(key: str, model: Any, indent: int, name_width: int, type_width: int) -> List[str]:
            sub_indent = indent + 6 
            key_line = f"{' ' * sub_indent}Key: '{key}'"
            try:
                model_str = model.__str__(
                    indent=sub_indent, 
                    name_width=name_width, 
                    type_width=type_width
                )
            except TypeError:
                model_str = f"{' ' * sub_indent}Value: {str(model)}"
            except AttributeError:
                model_str = f"{' ' * sub_indent}Value: {model}"
            return [key_line, model_str]
        
        # Calculate the exact padding needed to align subsequent lines under the signal value
        signal_alignment_pad = " " * (indent + name_width + type_width + 6)

        # Prepare aligned signal string
        signal_str = indent_multiline(self.signal, signal_alignment_pad)
        
        # --- Build Lines List ---
        lines = [
            f"{pad_title}TransferFunctionModel("
        ]

        # BASIC FIELDS
        lines.extend([
            f"{pad}{'name:':<{name_width}} {str(type(self.name)):>{type_width}} = {self.name}",
            f"{pad}{'description:':<{name_width}} {str(type(self.description)):>{type_width}} = {self.description}",
            f"----------------",
            f"{pad}{'tf_name:':<{name_width}} {str(type(self.tf_name)):>{type_width}} = {self.tf_name}",
            f"{pad}{'input:':<{name_width}} {str(type(self.input)):>{type_width}} = {self.input}",
            f"{pad}{'output:':<{name_width}} {str(type(self.output)):>{type_width}} = {self.output}"])
        lines.append(
            f"{pad}{'symbols:':<{name_width}} {str(type(self.symbols)):>{type_width}} = ")
        for key, symbol_model_instance in self.symbols.items():
            lines.extend(format_dict_entry(
                key, 
                symbol_model_instance, 
                indent=indent + 4, 
                name_width=name_width, 
                type_width=type_width
            ))
        lines.append(
            f"{pad}{'constants:':<{name_width}} {str(type(self.constants)):>{type_width}} = "        )
        for key, constant_model_instance in self.constants.items():
            lines.extend(format_dict_entry(
                key, 
                constant_model_instance, 
                indent=indent + 4, 
                name_width=name_width, 
                type_width=type_width
            ))
        lines.extend([
            f"{pad}{'functions:':<{name_width}} {str(type(self.functions)):>{type_width}} = {self.functions}",
            f"----------------",
            f"{pad}{'symbolic:':<{name_width}} {str(type(self.symbolic)):>{type_width}} = {self.symbolic}",
            f"{pad}{'numeric:':<{name_width}} {str(type(self.numeric)):>{type_width}} = {self.numeric}",
            f"{pad}{'signal:':<{name_width}} {str(type(self.signal)):>{type_width}} = TransferFunctionContinuous(num={self.signal.num.tolist()}, den={self.signal.den.tolist()}, dt={self.signal.dt})",
            f"----------------",
        ])
        
        lines.extend(format_nested_object('numerator', self.numerator, indent, name_width, type_width))
        lines.extend(format_nested_object('denominator', self.denominator, indent, name_width, type_width))
        lines.extend(format_nested_object('zeros', self.zeros, indent, name_width, type_width))
        lines.extend(format_nested_object('poles', self.poles, indent, name_width, type_width))

        lines.extend([
            f"----------------"
        ])
        lines.extend(format_nested_object('differential_equation', self.differential_equation, indent, name_width, type_width))

        lines.extend([
            f"----------------",
            f"{pad}{'t_range:':<{name_width}} {str(type(self.t_range)):>{type_width}} = {t_range_str}",
        ])
        lines.extend(format_nested_object('impulse_response', self.impulse_response, indent, name_width, type_width))
        lines.extend(format_nested_object('step_response', self.step_response, indent, name_width, type_width))
        lines.extend(format_nested_object('ramp_response', self.ramp_response, indent, name_width, type_width))

        lines.extend([
            f"----------------",
            f"{pad}{'w_range:':<{name_width}} {str(type(self.w_range)):>{type_width}} = {w_range_str}",
        ])
        lines.extend(format_nested_object('frequency_response', self.frequency_response, indent, name_width, type_width))
        lines.extend(format_nested_object('margin', self.margin, indent, name_width, type_width))

        lines.append(f"{' ' * indent})")
        
        return "\n".join(lines)

# ========================================================================
# DEFINITIONS
# ========================================================================

## I/O functions
    def define_input(
            self, 
            name: str) -> None:
        subheader(log, f'TransferFunctionModel({self.name}): Defining input as {name}')
        try:
            s = self.symbols['s'].symbol
            t = self.symbols['t'].symbol
        except KeyError:
            print(f"TransferFunctionModel({self.name}): Error! SymPy symbols 's' and 't' must be defined in self.symbols before calling define_input")
            return 
        
        ft_name = name.lower()
        Fs_name = name.capitalize() 

        if self.input and self.input[0].func.name.lower() == ft_name:
            print(f"TransferFunctionModel({self.name}): Warning! Input name '{name}' is the same as the existing input name")
            return

        if self.input:
            old_ft_name = self.input[0].func.name
            old_Fs_name = self.input[1].func.name
            
            if old_ft_name in self.functions:
                del self.functions[old_ft_name]
            if old_Fs_name in self.functions:
                del self.functions[old_Fs_name]

        self.input = (sp.Function(ft_name)(t), sp.Function(Fs_name)(s))
        
        self.functions[ft_name] = self.input[0]
        self.functions[Fs_name] = self.input[1]

    def define_output(
            self, 
            name: str) -> None:
        subheader(log, f'TransferFunctionModel({self.name}): Defining output as {name}')
        try:
            s = self.symbols['s'].symbol
            t = self.symbols['t'].symbol
        except KeyError:
            print(f"TransferFunctionModel({self.name}): Error! SymPy symbols 's' and 't' must be defined in self.symbols before calling define_output")
            return 
        
        ft_name = name.lower()
        Fs_name = name.capitalize() 

        if self.output and self.output[0].func.name.lower() == ft_name:
            print(f"TransferFunctionModel({self.name}): Warning! Output name '{name}' is the same as the existing output name")
            return

        if self.output:
            old_ft_name = self.output[0].func.name
            old_Fs_name = self.output[1].func.name
            
            if old_ft_name in self.functions:
                del self.functions[old_ft_name]
            if old_Fs_name in self.functions:
                del self.functions[old_Fs_name]

        self.output = (sp.Function(ft_name)(t), sp.Function(Fs_name)(s))
        
        self.functions[ft_name] = self.output[0]
        self.functions[Fs_name] = self.output[1]

    def define_tf_name(
            self, 
            name: str) -> None:
        subheader(log, f'TransferFunctionModel({self.name}): Defining tf_name as {name}')

        try:
            s = self.symbols['s'].symbol
            t = self.symbols['t'].symbol
        except KeyError:
            print(f"TransferFunctionModel({self.name}): Error! SymPy symbols 's' and 't' must be defined in self.symbols before calling define_tf_name")
            return 
        
        ft_name = name.lower()
        Fs_name = name.capitalize() 

        if self.tf_name and self.tf_name[0].func.name.lower() == ft_name:
            print(f"TransferFunctionModel({self.name}): Warning! Name '{name}' is the same as the existing name")
            return

        if self.tf_name:
            old_ft_name = self.tf_name[0].func.name
            old_Fs_name = self.tf_name[1].func.name
            
            if old_ft_name in self.functions:
                del self.functions[old_ft_name]
            if old_Fs_name in self.functions:
                del self.functions[old_Fs_name]

        self.tf_name = (sp.Function(ft_name)(t), sp.Function(Fs_name)(s))
        
        self.functions[ft_name] = self.tf_name[0]
        self.functions[Fs_name] = self.tf_name[1]

## Symbol functions
    def add_st(
            self) -> None:
        if 's' not in self.symbols:
            self.symbols['s'] = self.create_symbol('s', is_constant=False)
        if 't' not in self.symbols:
            self.symbols['t'] = self.create_symbol('t', is_real=True, is_constant=False)
    
    @staticmethod
    def create_symbol(
            name: str, 
            description: str='None',
            is_real: bool=None, 
            is_positive: bool=None,
            is_constant: bool=False,
            is_global: bool=False) -> SymbolModel:
        symbol = sp.symbols(name, real = is_real, positive = is_positive)
        return SymbolModel(
            symbol=symbol,
            description=description,
            is_real=is_real,
            is_positive=is_positive,
            is_constant=is_constant,
            is_global=is_global)
    
    def remove_symbol(
            self, 
            name: str) -> None:
        
        if name not in self.symbols:
            print(f"TransferFunctionModel({self.name}): Warning! Symbol '{name}' not found")
            return
        symbol_to_remove = self.symbols[name]

        if self.symbolic is not None and self.symbolic.has(symbol_to_remove):
            print(f"TransferFunctionModel({self.name}): Cannot remove symbol '{name}'. "
                f"It is currently used in the symbolic transfer function: {self.symbolic}")
            return
        del self.symbols[name]
        
## Constant functions
    def add_constant(
            self, 
            name: str, 
            value: float, 
            description='None', 
            unit='-') -> None:
        subheader(log, f'TransferFunctionModel({self.name}): Adding constant {name} = {value} [{unit}]')
        
        if name in self.constants:
            print(f"TransferFunctionModel({self.name}): Warning! Constant '{name}' already exists in self.constants")
            return
        
        if name in self.symbols:
            print(f"TransferFunctionModel({self.name}): Warning! Name used for constant '{name}' already exists in self.symbol")
            return
        else:
            self.symbols[name] = self.create_symbol(name=name, description=description, is_real=True, is_constant=True, is_global=False)

        symbol = self.symbols[name].symbol

        self.constants[name] = self.create_constant(name=name,value=value,symbol=symbol,description=description,unit=unit,is_global=False)

    @staticmethod
    def create_constant(
            name: str, 
            value: float, 
            symbol: sp.Symbol=None,
            description: str='None', 
            unit: str='-', 
            is_global=False) -> None:
        
        # Fallback
        if symbol is None:
            symbol = sp.Symbol(name, is_real=True)
        
        return ConstantModel(
            value=float(value),
            description=f"{description}",
            unit=f"{unit}",
            symbol=symbol,
            is_global=is_global)     
        
    def remove_constant(
            self,
            name: str) -> None:
        subheader(log, f'TransferFunctionModel({self.name}): Removing constant {name} from model {self.name}')
        
        if name not in self.constants:
            print(f"TransferFunctionModel({self.name}): Warning! Constant '{name}' not found")
            return

        constant_data = self.constants[name]
        symbol_to_remove = constant_data["symbol"]

        if self.symbolic is not None and self.symbolic.has(symbol_to_remove):
            print(f"TransferFunctionModel({self.name}): Cannot remove constant '{name}'. "
                f"It is currently used in the symbolic transfer function: {self.symbolic}")
            return
        
        # Remove from the constants dictionary
        del self.constants[name]
        
        # Remove from the symbols list
        if symbol_to_remove in self.symbols:
            self.symbols.remove(symbol_to_remove)

        # 4. Optional: Remove from global tracking (if applicable)
        # If the constant was marked as 'is_global', you would call a
        # global control system method here to unregister it.
        if constant_data.get('is_global'):
            print(f"TransferFunctionModel({self.name}): Cannot remove constant '{name}'. "
                f"It is a global constant: {self.symbolic}"
            )
            # self.control_system.remove_global_constant(name)
            pass 

        # 5. Model Update: Trigger a full numeric recalculation
        # Though the constant wasn't used, recalculating self.numeric 
        # is a safe way to refresh the model state if needed.
        if self.symbolic is not None:
             # Assigning to the public property triggers the numeric setter cascade.
            self.numeric = self.symbolic.subs(self.get_constant_values())
            
        print(f"TransferFunctionModel({self.name}): Constant '{name}' successfully removed")

    def edit_constant(
            self, 
            name: str, 
            value: float, 
            description='None', 
            unit='-', 
            is_global=False) -> None:
        subheader(log, f'TransferFunctionModel({self.name}): Editing constant {name} = {value} [{unit}]')
        
        if name in self.constants:
            # Reuse the existing symbol
            symbol = self.constants[name].symbol
        else:
            # If it doesn't exist, create a new symbol
            symbol = sp.symbols(name, real = True)

        # Replace the old constant with a new ConstantModel object
        self.constants[name] = self.create_constant(
            name=name,
            value=value,
            description=description,
            unit=unit,
            is_global=is_global,
            symbol=symbol)
        
        # Trigger an update of transfer function
        if self.symbolic is not None:
            self.numeric = self.symbolic.subs(self.get_constant_values())

    def get_constant_values(self) -> Dict[sp.Symbol, float]:
        constants_dict = {}
        for name, data in self.constants.items():
            constants_dict[data.symbol] = data.value
        return constants_dict

## Transfer function functions
    def define_tf(
            self, 
            *args, 
            **kwargs) -> None:
        """
        Define transfer function from various input formats:
            define_tf("1/(s+1)")                            # from string
            define_tf(num_coefs=[1], den_coefs=[1, 1])      # from coefficients
            define_tf(lhs=..., rhs=...)                     # from differential equation
        """
        header(log, f"TransferFunctionModel({self.name}): Defining transfer function")
        
        # Determine input type and get symbolic transfer function
        if len(args) == 1 and isinstance(args[0], str):
            try:
                # From string
                subheader(log, f"TransferFunctionModel({self.name}): From string {args[0]}")
                tf_symbolic = utils_tf.from_string(args[0], self.symbols)
                self.symbolic = tf_symbolic
            except (ValueError, KeyError, TypeError) as e:
                log.error(f"TransferFunctionModel({self.name}): Failed to define symbolic transfer function from string '{args[0]}': {e}")
                raise
            
        elif 'num_coefs' in kwargs and 'den_coefs' in kwargs:
            # From coefficients
            num_coefs = kwargs.get('num_coefs', [])
            den_coefs = kwargs.get('den_coefs', [])
            subheader(log, f"TransferFunctionModel({self.name}): From coefficients num=[{num_coefs}], den=[{den_coefs}]")
            self.symbolic = utils_tf.from_coefs(num_coefs, den_coefs, self.constants)
            
        elif 'lhs' in kwargs and 'rhs' in kwargs:
            # From differential equation
            lhs = kwargs['lhs']
            rhs = kwargs['rhs']
            subheader(log, f"TransferFunctionModel({self.name}): From differential equation {lhs} = {rhs}")
            self.symbolic = utils_tf.from_equation(lhs, rhs, self.input, self.output, self.constants)
            
        else:
            raise ValueError(
                f"TransferFunctionModel({self.name}): Invalid arguments. Use one of:\n"
                "  define_tf('string')  # e.g., '1/(s+1)'\n"
                "  define_tf(num_coefs=[...], den_coefs=[...])\n"
                "  define_tf(lhs=..., rhs=...)"
            )
        self.numeric = self.symbolic.subs(self.get_constant_values())
        
## Zeros and poles functions


## Time and frequency responses functions

    def get_t_range(
            self, 
            t_range:    Optional[tuple] = (), 
            delay_time: float = 0, 
            n_tau:      float = 8, 
            tol:        float = 1e-6) -> tuple[float,float]:
        
        poles = self.signal.poles
        
        t_start = t_range[0] if t_range else 0.0
        t_end = t_range[1] if t_range else None

        if t_end is None or t_end <= t_start:
                
            # Initialize T_sim
            T_sim = 10 

            # Find the dominant pole for stable systems
            if len(poles) > 0:
                # Check stability: any pole with Re(s) >= 0 (excluding numerical noise)
                unstable_or_marginal = np.any(np.real(poles) >= -tol)

                if not unstable_or_marginal:
                    # System is stable. Estimate settling time.
                    
                    # Characteristic Time Constant (tau) is 1 / |Re(p_dominant)|
                    # The dominant pole is the one closest to the jw-axis (smallest |Re(p)|)
                    stable_poles = poles[np.real(poles) < -tol]
                    if len(stable_poles) > 0:
                        
                        # Find the stable pole with the smallest magnitude of its real part
                        abs_real_parts = np.abs(np.real(stable_poles))
                        tau_max = 1.0 / np.min(abs_real_parts)
                        
                        # Simulation time T_sim is based on n_tau * tau_max
                        T_settle_factor = n_tau # Default set to 6 for impulse
                        T_sim = T_settle_factor * tau_max
                
                # Add delay time to the required simulation time
                t_end = T_sim + delay_time
                
            else:
                # If system has no poles (e.g., pure gain), use a default
                t_end = T_sim + delay_time

        self.t_range = (float(t_start),float(t_end))

        return self.t_range

    def get_w_range(
            self,
            n_decades_above_w_max: int = 1,
            n_decades_below_w_min: int = 1) -> tuple[float,float]:
        
        zeros = self.zeros.numeric
        poles = self.poles.numeric
        
        all_freqs = []
        
        # Extract the absolute values of the real part of poles/zeros (corner frequencies)
        if poles is not None:
            all_freqs += [
                abs(p.evalf().as_real_imag()[0]) 
                for p in poles 
                if abs(p.evalf().as_real_imag()[0]) > 0
            ]
        if zeros is not None:
            all_freqs += [
                abs(z.evalf().as_real_imag()[0]) 
                for z in zeros 
                if abs(z.evalf().as_real_imag()[0]) > 0
            ]
            
        if all_freqs:
            w_min_raw = float(min(all_freqs))
            w_max_raw = float(max(all_freqs))

            min_exponent = np.floor(np.log10(w_min_raw)) - n_decades_below_w_min
            max_exponent = np.ceil(np.log10(w_max_raw)) + n_decades_above_w_max

            min_freq = 10**min_exponent
            max_freq = 10**max_exponent
            
            self.w_range = (min_freq, max_freq)
            return self.w_range
        else:
            # Fallback case if no finite poles or zeros exist (e.g., a simple gain)
            # The decade parameters don't apply here, so use the sensible default
            self.w_range = (0.01, 100)
            return self.w_range

    def get_time_response(
            self, 
            response_type: str = 'impulse',  # 'impulse', 'step', or 'ramp'
            t_range: Optional[tuple] = (), 
            delay_time: float = 0, 
            n_points: int = 500, 
            tol: float = 1e-6) -> ResponseModel: 
        
        if not t_range:
            t_range = self.t_range
        
        t_start = t_range[0]
        t_end = t_range[1]
        t_end = max(t_end, delay_time + 1e-3)
        
        t_vals = np.linspace(t_start, t_end, n_points)
        
        mask_after = t_vals >= delay_time
        y_vals = np.zeros_like(t_vals)
        r_vals = np.zeros_like(t_vals)
        
        if np.any(mask_after):
            t_shifted = t_vals[mask_after] - delay_time
            
            # Choose the appropriate response function and define input signal
            if response_type == 'impulse':
                _, y_after = sig.impulse(self.signal, T=t_shifted)
                # Impulse approximation
                if len(t_shifted) > 0:
                    r_vals[mask_after][0] = 1.0 / (t_shifted[1] - t_shifted[0]) if len(t_shifted) > 1 else 1.0
                    
            elif response_type == 'step':
                _, y_after = sig.step(self.signal, T=t_shifted)
                r_vals[mask_after] = 1.0
                
            elif response_type == 'ramp':
                num = self.signal.num
                den = np.concatenate([self.signal.den, [0]])
                ramp_tf = sig.TransferFunction(num, den)
                _, y_after = sig.step(ramp_tf, T=t_shifted)
                r_vals[mask_after] = t_shifted
                
            else:
                raise ValueError(f"Unknown response_type: {response_type}. Choose 'impulse', 'step', or 'ramp'")
            
            y_vals[mask_after] = y_after
        
        t_range = (t_vals[0], t_vals[-1])
        
        # Get appropriate info based on response type
        if response_type == 'impulse':
            info = self.get_impulse_response_info(t_vals, y_vals, r_vals, delay_time, tol)
        elif response_type == 'step':
            info = self.get_step_response_info(t_vals, y_vals, r_vals, delay_time, tol)
        elif response_type == 'ramp':
            info = self.get_ramp_response_info(t_vals, y_vals, r_vals, delay_time, tol)
        
        return ResponseModel( 
            t_vals=t_vals.tolist(), 
            y_vals=y_vals.tolist(),
            r_vals=r_vals.tolist(),
            delay_time=delay_time,
            response_type=response_type,
            info=info
        )
    
    def get_frequency_response(
            self, 
            response_type: str = 'frequency',  # 'impulse', 'step', or 'ramp'
            w_range: Optional[tuple] = (), 
            n_points: int = 500) -> ResponseModel: 
        
        if not w_range:
            w_range = self.w_range

        w_vals = np.logspace(np.log10(w_range[0]), np.log10(w_range[1]), n_points)
        if self.numeric == 1:
            F_vals = np.ones_like(w_vals, dtype=complex)
        else:
            s = self.symbols['s'].symbol
            s_vals = 1j * w_vals
            F_func = sp.lambdify(s, self.numeric, modules=['numpy'])
            F_vals = F_func(s_vals)
        
        mag_vals = 20 * np.log10(np.abs(F_vals))
        phase_vals = np.unwrap(np.angle(F_vals)) * (180/np.pi)
        
        return ResponseModel( 
            w_vals=w_vals.tolist(), 
            F_vals=F_vals.tolist(),
            mag_vals=mag_vals.tolist(),
            phase_vals=phase_vals.tolist(),
            response_type=response_type,
            # info=info
        )

    def get_impulse_response_info(
            self,
            t_vals: list[float] = (), 
            y_vals: list[float] = (), 
            r_vals: list[float] = (), 
            delay_time: float = 0, 
            tol: float = 0.02) -> ImpulseResponseInfoModel:
        
        y_vals = np.real_if_close(y_vals)
        
        # Remove values before delay
        mask = t_vals >= delay_time
        t = t_vals[mask]
        y = y_vals[mask]
        
        if len(y) == 0:
            return ImpulseResponseInfoModel()
        
        # --- Peak characteristics ---
        peak_idx = np.argmax(np.abs(y))
        t_peak = t[peak_idx] - delay_time
        y_peak = y[peak_idx]
        
        # --- Integral (DC gain) ---
        integral = np.trapz(y, t)
        
        # --- Energy ---
        energy = np.trapz(y**2, t)
        
        # --- Settling time ---
        y_final = y[-1]  # Should approach 0 for stable system
        threshold = tol * np.abs(y_peak)
        settled = np.abs(y) < threshold
        
        if np.any(settled):
            # Find last time it exceeds threshold
            last_outside = np.where(~settled)[0]
            if len(last_outside) > 0 and last_outside[-1] < len(t) - 1:
                t_settling = t[last_outside[-1] + 1] - delay_time
            else:
                t_settling = np.nan
        else:
            t_settling = np.nan
        
        # --- Time to half peak ---
        half_peak = y_peak / 2

        try:
            after_peak = y[peak_idx:]
            t_after_peak = t[peak_idx:]
            half_idx = np.where(np.abs(after_peak) <= np.abs(half_peak))[0][0]
            t_half = t_after_peak[half_idx] - delay_time
        except (IndexError, ValueError):
            t_half = np.nan
        
        # --- Decay rate (exponential fit after peak) ---
        try:
            if peak_idx < len(y) - 10:  # Need some points after peak
                y_decay = np.abs(y[peak_idx:])
                t_decay = t[peak_idx:] - t[peak_idx]
                
                # Fit exponential: y = y_peak * exp(-decay_rate * t)
                with np.errstate(divide='ignore', invalid='ignore'):
                    log_y = np.log(y_decay / y_decay[0])
                    valid = np.isfinite(log_y)
                    if np.sum(valid) > 5:
                        # Linear fit in log space
                        coeffs = np.polyfit(t_decay[valid], log_y[valid], 1)
                        decay_rate = -coeffs[0]  # Negative of slope
                    else:
                        decay_rate = np.nan
            else:
                decay_rate = np.nan
        except:
            decay_rate = np.nan
        
        # --- Count oscillations (zero crossings) ---
        zero_crossings = np.where(np.diff(np.sign(y)))[0]
        num_oscillations = len(zero_crossings)
        
        # --- Estimate damping ratio and natural frequency (if oscillatory) ---
        if num_oscillations >= 2:
            # Use logarithmic decrement method
            peaks_idx = []
            for i in range(1, len(y)-1):
                if y[i] > y[i-1] and y[i] > y[i+1]:
                    peaks_idx.append(i)
            
            if len(peaks_idx) >= 2:
                # Period from two consecutive peaks
                T_d = (t[peaks_idx[1]] - t[peaks_idx[0]])
                omega_d = 2 * np.pi / T_d  # Damped natural frequency
                
                # Logarithmic decrement
                delta = np.log(np.abs(y[peaks_idx[0]]) / np.abs(y[peaks_idx[1]]))
                damping_ratio = delta / np.sqrt((2*np.pi)**2 + delta**2)
                
                # Undamped natural frequency
                natural_freq = omega_d / np.sqrt(1 - damping_ratio**2) if damping_ratio < 1 else omega_d
            else:
                damping_ratio = None
                natural_freq = None
        else:
            damping_ratio = None
            natural_freq = None

        defaults = ImpulseResponseInfoModel._field_defaults
        
        calculated_data = {
            "t_peak": t_peak,
            "y_peak": y_peak,
            "t_settling": t_settling,
            "t_half": t_half,
            "decay_rate": decay_rate,
            "integral": integral,
            "energy": energy,
            "num_oscillations": num_oscillations,
            "damping_ratio": damping_ratio,
            "natural_freq": natural_freq,
        }
        
        info_args = {}
        for field_name, value in calculated_data.items():
            default_metric = defaults.get(field_name)
            
            if default_metric:
                info_args[field_name] = default_metric._replace(value=value)
            else:
                info_args[field_name] = MetricModel(value=value, label=field_name)
        
        return ImpulseResponseInfoModel(**info_args)

    def get_step_response_info(
            self,
            t_vals: list[float] = (), 
            y_vals: list[float] = (), 
            r_vals: list[float] = (),
            delay_time: float = 0, 
            tol: float = 0.02) -> StepResponseInfoModel:
            
        y_vals = np.real_if_close(y_vals)

        # Remove values before delay
        mask = t_vals >= delay_time
        t = t_vals[mask]
        y = y_vals[mask]
        
        if len(y) == 0:
            return StepResponseInfoModel
        
        t_relative = t - delay_time
        
        # --- Steady-State Value ---
        y_final = y[-1]
        y_initial = y[0] if len(y) > 0 else 0.0
        
        # --- Peak Characteristics ---
        peak_idx = np.argmax(y)
        y_peak = y[peak_idx]
        t_peak = t_relative[peak_idx]
        
        # --- Overshoot ---
        overshoot_percent = None
        if y_final != 0 and y_peak > y_final:
            overshoot_percent = ((y_peak - y_final) / np.abs(y_final)) * 100.0
        
        # --- Rise Time (10% to 90%) ---
        target_10 = y_initial + 0.1 * (y_final - y_initial)
        target_90 = y_initial + 0.9 * (y_final - y_initial)
        
        t_10, t_90 = np.nan, np.nan
        try:
            idx_10 = np.where(y >= target_10)[0][0]
            t_10 = t_relative[idx_10]
        except IndexError:
            pass 
            
        try:
            idx_90 = np.where(y >= target_90)[0][0]
            t_90 = t_relative[idx_90]
        except IndexError:
            pass 
        
        t_rise = t_90 - t_10 if not np.isnan(t_10) and not np.isnan(t_90) and t_90 > t_10 else np.nan
        
        # --- Settling Time ---
        t_settling = np.nan
        threshold_low = y_final * (1 - tol)
        threshold_high = y_final * (1 + tol)
        
        outside_band = (y < threshold_low) | (y > threshold_high)
        
        if np.any(outside_band):
            last_outside_idx = np.where(outside_band)[0][-1]
            
            if last_outside_idx < len(y) - 1:
                t_settling = t_relative[last_outside_idx + 1]
            else:
                t_settling = np.nan
        elif len(y) > 0:
            t_settling = t_relative[0] 

        # --- Number of Oscillations (Crossings of Final Value) ---
        # Look at the difference between y and y_final
        error = y - y_final
        # Zero crossings of the error signal
        zero_crossings = np.where(np.diff(np.sign(error)))[0]
        num_oscillations = len(zero_crossings)
        
        # --- Damping Ratio ($\zeta$) and Natural Frequency ($\omega_n$) Estimation ---
        damping_ratio, natural_freq = None, None
        
        if overshoot_percent is not None and overshoot_percent > 0 and y_final != 0:
            try:
                M_p = overshoot_percent / 100.0
                ln_M_p = np.log(M_p)
                damping_ratio = -ln_M_p / np.sqrt(np.pi**2 + ln_M_p**2)
                
                if t_peak > 0 and damping_ratio < 1:
                    natural_freq = np.pi / (t_peak * np.sqrt(1 - damping_ratio**2))
            except (ValueError, ZeroDivisionError, TypeError):
                pass 

        defaults = StepResponseInfoModel._field_defaults
        
        calculated_data = {
            "y_final": y_final,
            "y_initial": y_initial,
            "t_rise": t_rise,
            "t_peak": t_peak,
            "y_peak": y_peak,
            "overshoot_percent": overshoot_percent,
            "t_settling": t_settling,
            "num_oscillations": num_oscillations,
            "damping_ratio": damping_ratio,
            "natural_freq": natural_freq,
        }
        
        info_args = {}
        for field_name, value in calculated_data.items():
            default_metric = defaults.get(field_name)
            
            if default_metric:
                info_args[field_name] = default_metric._replace(value=value)
            else:
                info_args[field_name] = MetricModel(value=value, label=field_name)
        return StepResponseInfoModel(**info_args)

    def get_ramp_response_info(
            self,
            t_vals: list[float] = (), 
            y_vals: list[float] = (), 
            r_vals: list[float] = (), 
            delay_time: float = 0, 
            tol: float = 0.02) -> RampResponseInfoModel:
        
        y_vals = np.real_if_close(y_vals)
        
        # --- Data Filtering ---
        mask = t_vals >= delay_time
        t = t_vals[mask]
        r = r_vals[mask]
        y = y_vals[mask]
        
        if len(y) < 20: # Need enough points to determine steady-state behavior
            return RampResponseInfoModel(
                steady_state_error=MetricModel(), velocity_error_const=MetricModel(), t_lag=MetricModel(), 
                max_tracking_error=MetricModel(), delay_time=MetricModel(value=delay_time, label="Time Delay", unit="s"), 
                y_final=MetricModel(), t_peak=MetricModel(), y_peak=MetricModel()
            )
        
        # --- 1. Steady-State Error (ess) ---
        # The error is e(t) = r(t) - y(t)
        error = r - y
        
        # Calculate steady-state error by averaging the error over the last 10% of the simulation
        n_avg = max(10, len(error) // 10)
        e_final = np.mean(error[-n_avg:])
        
        # --- 2. Velocity Error Constant (Kv) ---
        # For a Type 1 stable system: ess = 1 / Kv
        Kv = 1.0 / e_final if e_final != 0 else np.inf
        
        # --- 3. Maximum Tracking Error ---
        e_max_tracking = np.max(np.abs(error))
        
        # --- 4. Time Lag (t_lag) ---
        # If the system tracks the ramp, the error is constant (ess). 
        # This constant error relates to the time lag: ess ≈ t_lag * slope (slope=1 for standard ramp)
        t_lag = e_final 
        
        # --- 5. Peak Characteristics (less common for ramp, but good to check) ---
        peak_idx = np.argmax(y)
        t_peak = t[peak_idx] - delay_time
        y_peak = y[peak_idx]
        
        # --- 6. Final Values ---
        y_final = y[-1]

        defaults = RampResponseInfoModel._field_defaults
        
        calculated_data = {
            "y_final": y_final,
            "t_peak": t_peak,
            "y_peak": y_peak,
            "e_final": e_final,
            "Kv": Kv,
            "t_lag": t_lag,
            "e_max_tracking": e_max_tracking,
        }
        
        info_args = {}
        for field_name, value in calculated_data.items():
            default_metric = defaults.get(field_name)
            
            if default_metric:
                info_args[field_name] = default_metric._replace(value=value)
            else:
                info_args[field_name] = MetricModel(value=value, label=field_name)

        return RampResponseInfoModel(**info_args)
    
    def get_margin(self):
        magnitude = np.array(self.frequency_response.mag_vals)
        phase = np.array(self.frequency_response.phase_vals)

        gain_crossings = np.where(np.diff(np.sign(magnitude)))[0]
        phase_crossings = np.where(np.diff(np.sign(phase + 180)))[0]

        if gain_crossings.size > 0:
            w_gain_crossover_found = True
            w_gain_crossover = self.frequency_response.w_vals[gain_crossings[0]]
            phase_margin = 180 + phase[gain_crossings[0]]
        else:
            w_gain_crossover_found = False
            w_gain_crossover = None
            phase_margin = None

        if phase_crossings.size > 0:
            w_phase_crossover_found = True
            w_phase_crossover = self.frequency_response.w_vals[phase_crossings[0]]
            gain_margin = -magnitude[phase_crossings[0]]
        else:
            w_phase_crossover_found = False
            w_phase_crossover = None
            gain_margin = None

        return MarginModel(
            gain_margin=gain_margin,
            w_phase_crossover=w_phase_crossover,
            w_phase_crossover_found=w_phase_crossover_found,
            phase_margin=phase_margin,
            w_gain_crossover=w_gain_crossover,
            w_gain_crossover_found=w_gain_crossover_found,
        )