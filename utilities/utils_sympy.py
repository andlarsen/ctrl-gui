
import sympy as sp
import numpy as np
import re

from typing import Dict, List, Any, Tuple

def add_symbol(name: str, is_real=None, is_positive=None, print_output=False):
    symbol = sp.symbols(name, real = is_real, positive = is_positive)
    if print_output:
            print(f"  {symbol}, is_real: {symbol.is_real}, is_positive: {symbol.is_positive}")
    return symbol

def remove_symbol(name: str, symbol_list: list, print_output=False):
    for symbol in symbol_list:
        if symbol == name:
            symbol_list.remove(symbol)
            if print_output:
                print(f"Removed symbol: {name}")
            return
    if print_output:
        print(f"Symbol '{name}' not found.")   
    return  

def translate_string(string_input: str):
    # Insert functions as sympy
    if 'exp' in string_input and 'sp.exp' not in string_input:
        string_input = string_input.replace('exp', 'sp.exp')
    if 'sin' in string_input and 'sp.sin' not in string_input:
        string_input = string_input.replace('sin', 'sp.sin')
    if 'cos' in string_input and 'sp.cos' not in string_input:
        string_input = string_input.replace('cos', 'sp.cos')
    if 'sqrt' in string_input and 'sp.sqrt' not in string_input:
        string_input = string_input.replace('sqrt', 'sp.sqrt')
    # Replace dot() with diff(var, t, x) where x is the number of derivatives. Number of derivatives is counted from number of d's in string_input
    if 'dot(' in string_input:
        matches = re.findall(r'(d+)ot\(([^)]+)\)', string_input)
        for d_string, var in matches:
            derivative_order = len(d_string) 
            pattern = f'{d_string}ot\\({re.escape(var)}\\)'
            replacement = f'sp.diff({var}, t, {derivative_order})'
            string_input = re.sub(pattern, replacement, string_input)
    return string_input

