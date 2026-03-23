import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import pandas as pd
import io
import warnings
import matplotlib.cm as cm
from matplotlib.colors import Normalize
from typing import Dict, Any, Optional, Tuple
import time
from scipy.stats import norm, probplot
from statsmodels.graphics.tsaplots import plot_acf
import warnings
warnings.filterwarnings('ignore')

# ============================================
# КОНФИГУРАЦИЯ И ИНИЦИАЛИЗАЦИЯ
# ============================================

# Set plot style for scientific publications
plt.rcParams.update({
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.labelweight': 'bold',
    'axes.titlesize': 12,
    'axes.titleweight': 'bold',
    'axes.edgecolor': 'black',
    'axes.linewidth': 1.0,
    'xtick.color': 'black',
    'ytick.color': 'black',
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'legend.frameon': True,
    'legend.framealpha': 0.9,
    'legend.edgecolor': 'black',
    'figure.dpi': 600,
    'savefig.dpi': 600,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.1
})

# Page configuration
st.set_page_config(
    page_title="Thermo-Mechanical Expansion Modeling",
    page_icon="📈",
    layout="wide"
)

# ============================================
# ИНИЦИАЛИЗАЦИЯ СЕССИОННОГО СОСТОЯНИЯ
# ============================================

def initialize_session_state():
    """Initialize all session state variables"""
    defaults = {
        'experimental_data': None,
        'fit_results': None,
        'geometric_fit_results': None,
        'last_fit_params': None,
        'geometric_fit_complete': False,
        'plot_style': {
            'point_color': '#1f77b4',
            'point_alpha': 0.8,
            'model_line_color': '#000000',
            'thermal_line_color': '#1f77b4',
            'chemical_line_color': '#d62728',
            'tec_exp_color': '#1f77b4',
            'tec_model_color': '#d62728',
            'oh_color': '#2ca02c',
            'bar_thermal_color': '#1f77b4',
            'bar_chemical_color': '#d62728',
            'cmap_style': 'viridis',
            'point_size': 50,
            'line_width': 2
        },
        'model_params': {
            'Acc': {'value': 0.6, 'fixed': False},
            'alpha_1e6': {'value': 14.4, 'fixed': False},
            'beta': {'value': 0.017, 'fixed': False},
            'dH': {'value': -81.0, 'fixed': False},
            'dS': {'value': -131.0, 'fixed': False},
            'pH2O': {'value': 0.083, 'fixed': False},
            'residue': {'value': 0.0, 'fixed': False}
        },
        'cation_selection': {
            'A_cation': 'Ba',
            'B_cation': 'Zr',
            'Acc_cation': 'Y'
        },
        'ionic_radii_db': None,
        'data_loaded': False,
        'plots_generated': False,
        'fitting_complete': False
    }
    
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

# Initialize session state
initialize_session_state()

# ============================================
# МОДУЛЬ ИОННЫХ РАДИУСОВ (ТАБЛИЦА ШЕННОНА)
# ============================================

def parse_shannon_table() -> Dict[str, Dict[int, float]]:
    """Parse Shannon ionic radii table and return database"""
    # Создаем базу данных радиусов для различных ионов и координационных чисел
    # Данные из предоставленной таблицы (ионные радиусы, Å)
    radii_db = {
        'Ba': {2: {6: 1.35, 7: 1.38, 8: 1.42, 9: 1.47, 10: 1.52, 11: 1.57, 12: 1.61}},
        'Sr': {2: {6: 1.18, 7: 1.21, 8: 1.26, 9: 1.31, 10: 1.36, 12: 1.44}},
        'Ca': {2: {6: 1.00, 7: 1.06, 8: 1.12, 9: 1.18, 10: 1.23, 12: 1.34}},
        'La': {3: {6: 1.032, 7: 1.10, 8: 1.16, 9: 1.216, 10: 1.27, 12: 1.36}},
        'Zr': {4: {4: 0.59, 5: 0.66, 6: 0.72, 7: 0.78, 8: 0.84, 9: 0.89}},
        'Ce': {4: {6: 0.87, 8: 0.97, 10: 1.07, 12: 1.14}},
        'Ti': {4: {4: 0.42, 5: 0.51, 6: 0.605, 8: 0.74}},
        'Hf': {4: {4: 0.58, 6: 0.71, 7: 0.76, 8: 0.83}},
        'Sn': {4: {4: 0.55, 5: 0.62, 6: 0.69, 7: 0.75, 8: 0.81}},
        'In': {3: {4: 0.62, 6: 0.80, 8: 0.92}},
        'Sc': {3: {6: 0.745, 8: 0.87}},
        'Y': {3: {6: 0.90, 7: 0.96, 8: 1.019, 9: 1.075}},
        'Yb': {3: {6: 0.868, 7: 0.925, 8: 0.985, 9: 1.042}},
        'Mg': {2: {4: 0.57, 5: 0.66, 6: 0.72, 8: 0.89}},
        'Zn': {2: {4: 0.60, 5: 0.68, 6: 0.74, 8: 0.90}},
        'Al': {3: {4: 0.39, 5: 0.48, 6: 0.535}},
        'Ga': {3: {4: 0.47, 5: 0.55, 6: 0.62}},
        'Ho': {3: {6: 0.901, 8: 1.015, 9: 1.072, 10: 1.12}},
        'Dy': {3: {6: 0.912, 7: 0.97, 8: 1.027, 9: 1.083}},
        'Gd': {3: {6: 0.938, 7: 1.00, 8: 1.053, 9: 1.107}},
        'Sm': {3: {6: 0.958, 7: 1.02, 8: 1.079, 9: 1.132, 12: 1.24}},
        'Nd': {3: {6: 0.983, 8: 1.109, 9: 1.163, 12: 1.27}},
        'Pr': {3: {6: 0.99, 8: 1.126, 9: 1.179}},
        'Er': {3: {6: 0.89, 8: 1.004, 9: 1.062}},
        'Tm': {3: {6: 0.88, 8: 0.994, 9: 1.052}},
        'Lu': {3: {6: 0.861, 8: 0.977, 9: 1.032}},
    }
    return radii_db

def get_ionic_radius(ion: str, charge: int, cn: float, radii_db: Dict) -> float:
    """Get ionic radius for given ion, charge, and coordination number with linear interpolation/extrapolation"""
    if ion not in radii_db:
        raise ValueError(f"Ion {ion} not found in database")
    
    if charge not in radii_db[ion]:
        raise ValueError(f"Charge {charge} for ion {ion} not found in database")
    
    cn_data = radii_db[ion][charge]
    cn_values = sorted(cn_data.keys())
    
    if cn in cn_values:
        return cn_data[cn]
    
    # Linear interpolation between known CN values
    if cn < min(cn_values):
        # Extrapolation to lower CN
        cn1 = min(cn_values)
        cn2 = sorted(cn_values)[1] if len(cn_values) > 1 else cn1
        r1 = cn_data[cn1]
        r2 = cn_data[cn2]
        slope = (r2 - r1) / (cn2 - cn1)
        return r1 + slope * (cn - cn1)
    elif cn > max(cn_values):
        # Extrapolation to higher CN
        cn1 = max(cn_values)
        cn2 = sorted(cn_values)[-2] if len(cn_values) > 1 else cn1
        r1 = cn_data[cn1]
        r2 = cn_data[cn2]
        slope = (r1 - r2) / (cn1 - cn2)
        return r1 + slope * (cn - cn1)
    else:
        # Interpolation between known CN values
        lower_cn = max([c for c in cn_values if c < cn])
        upper_cn = min([c for c in cn_values if c > cn])
        r_lower = cn_data[lower_cn]
        r_upper = cn_data[upper_cn]
        return r_lower + (r_upper - r_lower) * (cn - lower_cn) / (upper_cn - lower_cn)

# ============================================
# ГЕОМЕТРИЧЕСКАЯ МОДЕЛЬ ПЕРОВСКИТА
# ============================================

def calculate_coordination_numbers(x: float, y: float) -> Tuple[float, float]:
    """Calculate coordination numbers for A and B cations according to the model"""
    CN_B = 6 - x + y
    CN_A = 12 - 2 * x + 2 * y
    return CN_A, CN_B

def calculate_average_radii(x: float, y: float, A_cation: str, B_cation: str, 
                           Acc_cation: str, radii_db: Dict, 
                           r_V: float, r_OH: float) -> Tuple[float, float, float]:
    """Calculate average radii for A, B, and anion sublattices"""
    
    CN_A, CN_B = calculate_coordination_numbers(x, y)
    
    # A-cation radius (undoped)
    r_A = get_ionic_radius(A_cation, 2 if A_cation in ['Ba', 'Sr', 'Ca'] else 3, CN_A, radii_db)
    
    # B-cation radius (mixed with acceptor)
    r_B_host = get_ionic_radius(B_cation, 4, CN_B, radii_db)
    r_Acc = get_ionic_radius(Acc_cation, 3, CN_B, radii_db)
    r_B_avg = (1 - x) * r_B_host + x * r_Acc
    
    # Anion sublattice
    r_O = 1.38  # O2- radius for CN=4 (ionic radius)
    
    # Concentrations
    V_O_conc = (x - y) / 2
    O_O_conc = 3 - (x + y) / 2
    
    # Average anion radius
    r_anion_avg = (O_O_conc * r_O + V_O_conc * r_V + y * r_OH) / 3
    
    return r_A, r_B_avg, r_anion_avg

def calculate_lattice_parameter(x: float, y: float, A_cation: str, B_cation: str,
                                Acc_cation: str, radii_db: Dict,
                                r_V: float, r_OH: float) -> float:
    """Calculate lattice parameter for cubic perovskite"""
    r_A, r_B, r_O_avg = calculate_average_radii(x, y, A_cation, B_cation, 
                                                Acc_cation, radii_db, r_V, r_OH)
    # For cubic perovskite: a = √2 * (r_A + r_B + r_O)
    a = np.sqrt(2) * (r_A + r_B + r_O_avg)
    return a

def calculate_geometric_expansion(T_data: np.ndarray, y_data: np.ndarray, 
                                  x: float, A_cation: str, B_cation: str,
                                  Acc_cation: str, radii_db: Dict,
                                  r_V: float, r_OH: float,
                                  alpha: float, gamma: float,
                                  T_ref: float, a_ref: float) -> np.ndarray:
    """Calculate ΔL/L0 based on geometric model"""
    expansion = np.zeros_like(T_data)
    
    for i in range(len(T_data)):
        y = y_data[i]
        a = calculate_lattice_parameter(x, y, A_cation, B_cation, Acc_cation, 
                                       radii_db, r_V, r_OH)
        expansion[i] = (a - a_ref) / a_ref + alpha * (T_data[i] - T_ref) + gamma
    
    return expansion

# ============================================
# ОПТИМИЗИРОВАННЫЕ ФУНКЦИИ С КЭШИРОВАНИЕМ
# ============================================

@st.cache_data(ttl=3600, show_spinner=False)
def parse_data_cached(data_string: str) -> np.ndarray:
    """Parse data with various separators (cached)"""
    lines = data_string.strip().split('\n')
    data = []
    
    for line in lines:
        if not line.strip():
            continue
            
        for sep in ['\t', ' ', ',', ';']:
            if sep in line:
                parts = line.split(sep)
                parts = [p.strip() for p in parts if p.strip()]
                if len(parts) >= 2:
                    try:
                        t = float(parts[0])
                        dl = float(parts[1])
                        data.append([t, dl])
                        break
                    except ValueError:
                        continue
    
    if not data:
        raise ValueError("Unable to parse data. Check the format.")
    
    return np.array(data)

@st.cache_data(ttl=3600, show_spinner=False)
def calculate_oh_cached(T: np.ndarray, Acc: float, dH: float, dS: float, pH2O: float) -> Tuple[np.ndarray, np.ndarray]:
    """Calculate [OH] for given temperatures (cached)"""
    T_K = T + 273.15
    R = 8.314
    Khydr = np.exp(-dH * 1000 / (R * T_K) + dS / R)
    
    A = Khydr * pH2O
    oh = (3*A - np.sqrt(A * (9*A - 6*A*Acc + A*Acc**2 + 24*Acc - 4*Acc**2))) / (A - 4)
    return oh, Khydr

@st.cache_data(ttl=3600, show_spinner=False)
def calculate_tec_cached(T: np.ndarray, dl: np.ndarray) -> np.ndarray:
    """Calculate Thermal Expansion Coefficient (dΔL/Lo/dT) (cached)"""
    tec = np.zeros_like(T)
    if len(T) < 2:
        return tec
    
    for i in range(1, len(T)-1):
        tec[i] = (dl[i+1] - dl[i-1]) / (T[i+1] - T[i-1])
    
    # Handle boundaries
    tec[0] = (dl[1] - dl[0]) / (T[1] - T[0])
    tec[-1] = (dl[-1] - dl[-2]) / (T[-1] - T[-2])
    
    return tec

def model_func_cached(T: np.ndarray, Acc: float, alpha_1e6: float, beta: float, 
                     dH: float, dS: float, pH2O: float, residue: float, 
                     T_start: float, oh_start: float) -> np.ndarray:
    """Model function for fitting (cached calculation)"""
    oh, _ = calculate_oh_cached(T, Acc, dH, dS, pH2O)
    dl_dl0 = (alpha_1e6/1e6) * (T - T_start) + beta * (oh - oh_start) + residue
    return dl_dl0

@st.cache_data(ttl=3600, show_spinner=False)
def fit_model_cached(data: np.ndarray, fixed_params: Dict[str, Any], 
                    initial_guess: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Perform model fitting with caching"""
    T_data = data[:, 0]
    dl_data = data[:, 1]
    
    T_start = T_data[0]
    
    # Get current parameter values
    current_params = {}
    for name in ['Acc', 'dH', 'dS', 'pH2O']:
        if fixed_params[name] is not None:
            current_params[name] = fixed_params[name]
        else:
            current_params[name] = initial_guess[name]
    
    oh_start, _ = calculate_oh_cached(
        np.array([T_start]), 
        current_params['Acc'],
        current_params['dH'],
        current_params['dS'],
        current_params['pH2O']
    )
    oh_start = oh_start[0]
    
    vary_params = []
    bounds_lower = []
    bounds_upper = []
    initial_params = []
    
    param_names = ['Acc', 'alpha_1e6', 'beta', 'dH', 'dS', 'pH2O', 'residue']
    
    for name in param_names:
        if fixed_params[name] is None:
            vary_params.append(name)
            bounds_lower.append(initial_guess[f'{name}_bounds'][0])
            bounds_upper.append(initial_guess[f'{name}_bounds'][1])
            initial_params.append(initial_guess[name])
    
    def fit_func(T, *params):
        all_params = fixed_params.copy()
        param_idx = 0
        for name in param_names:
            if fixed_params[name] is not None:
                all_params[name] = fixed_params[name]
            else:
                all_params[name] = params[param_idx]
                param_idx += 1
        
        return model_func_cached(T, all_params['Acc'], all_params['alpha_1e6'], 
                               all_params['beta'], all_params['dH'], all_params['dS'], 
                               all_params['pH2O'], all_params['residue'], T_start, oh_start)
    
    try:
        popt, pcov = curve_fit(fit_func, T_data, dl_data, 
                             p0=initial_params, 
                             bounds=(bounds_lower, bounds_upper),
                             maxfev=5000)
        
        result_params = fixed_params.copy()
        param_idx = 0
        for name in param_names:
            if fixed_params[name] is None:
                result_params[name] = popt[param_idx]
                param_idx += 1
            else:
                result_params[name] = fixed_params[name]
        
        dl_model = fit_func(T_data, *popt)
        
        residuals = dl_data - dl_model
        mse = np.mean(residuals**2)
        rmse = np.sqrt(mse)
        r2 = 1 - np.sum(residuals**2) / np.sum((dl_data - np.mean(dl_data))**2)
        
        N = len(dl_data)
        p = len(vary_params)
        
        if N > p:
            chi2 = np.sum(residuals**2) / (N - p)
        else:
            chi2 = np.nan
        
        tec_exp = calculate_tec_cached(T_data, dl_data)
        tec_model = calculate_tec_cached(T_data, dl_model)
        
        oh, _ = calculate_oh_cached(T_data, result_params['Acc'], result_params['dH'], 
                                  result_params['dS'], result_params['pH2O'])
        
        thermal_contrib = (result_params['alpha_1e6']/1e6) * (T_data - T_start)
        chem_contrib = result_params['beta'] * (oh - oh_start)
        
        return {
            'params': result_params,
            'popt': popt,
            'pcov': pcov,
            'dl_model': dl_model,
            'residuals': residuals,
            'mse': mse,
            'rmse': rmse,
            'r2': r2,
            'chi2': chi2,
            'reduced_chi2': chi2,
            'N_points': N,
            'n_free_params': p,
            'T_start': T_start,
            'oh_start': oh_start,
            'vary_params': vary_params,
            'tec_exp': tec_exp,
            'tec_model': tec_model,
            'oh_concentration': oh,
            'thermal_contrib': thermal_contrib,
            'chem_contrib': chem_contrib,
            'T_data': T_data,
            'dl_data': dl_data,
            'fixed_params': fixed_params,
            'initial_guess': initial_guess
        }
    
    except Exception as e:
        st.error(f"Fitting error: {str(e)}")
        return None

@st.cache_data(ttl=3600, show_spinner=False)
def fit_geometric_model_cached(data: np.ndarray, oh_concentration: np.ndarray,
                               x: float, A_cation: str, B_cation: str,
                               Acc_cation: str, radii_db: Dict,
                               alpha: float, gamma: float, T_ref: float,
                               initial_r_V: float, initial_r_OH: float,
                               fit_r_V: bool, fit_r_OH: bool,
                               fit_alpha: bool, fit_gamma: bool) -> Optional[Dict[str, Any]]:
    """Fit geometric model to estimate r_V and r_OH"""
    
    T_data = data[:, 0]
    dl_data = data[:, 1]
    y_data = oh_concentration
    
    # Calculate reference lattice parameter at T_ref with y=0
    a_ref = calculate_lattice_parameter(x, 0.0, A_cation, B_cation, Acc_cation,
                                        radii_db, initial_r_V, initial_r_OH)
    
    # Prepare fitting parameters
    vary_params = []
    bounds_lower = []
    bounds_upper = []
    initial_params = []
    param_names = []
    
    if fit_r_V:
        param_names.append('r_V')
        vary_params.append('r_V')
        bounds_lower.append(0.5)
        bounds_upper.append(1.5)
        initial_params.append(initial_r_V)
    
    if fit_r_OH:
        param_names.append('r_OH')
        vary_params.append('r_OH')
        bounds_lower.append(0.5)
        bounds_upper.append(1.5)
        initial_params.append(initial_r_OH)
    
    if fit_alpha:
        param_names.append('alpha')
        vary_params.append('alpha')
        bounds_lower.append(0.0)
        bounds_upper.append(1e-4)
        initial_params.append(alpha)
    
    if fit_gamma:
        param_names.append('gamma')
        vary_params.append('gamma')
        bounds_lower.append(-0.01)
        bounds_upper.append(0.01)
        initial_params.append(gamma)
    
    def geometric_model(T, y, *params):
        r_V_val = params[param_names.index('r_V')] if fit_r_V else initial_r_V
        r_OH_val = params[param_names.index('r_OH')] if fit_r_OH else initial_r_OH
        alpha_val = params[param_names.index('alpha')] if fit_alpha else alpha
        gamma_val = params[param_names.index('gamma')] if fit_gamma else gamma
        
        expansion = np.zeros_like(T)
        for i in range(len(T)):
            a = calculate_lattice_parameter(x, y[i], A_cation, B_cation, Acc_cation,
                                           radii_db, r_V_val, r_OH_val)
            expansion[i] = (a - a_ref) / a_ref + alpha_val * (T[i] - T_ref) + gamma_val
        return expansion
    
    def fit_func_wrapper(T, *params):
        return geometric_model(T, y_data, *params)
    
    try:
        if len(vary_params) > 0:
            popt, pcov = curve_fit(fit_func_wrapper, T_data, dl_data,
                                  p0=initial_params,
                                  bounds=(bounds_lower, bounds_upper),
                                  maxfev=5000)
            
            result_params = {
                'r_V': popt[param_names.index('r_V')] if fit_r_V else initial_r_V,
                'r_OH': popt[param_names.index('r_OH')] if fit_r_OH else initial_r_OH,
                'alpha': popt[param_names.index('alpha')] if fit_alpha else alpha,
                'gamma': popt[param_names.index('gamma')] if fit_gamma else gamma
            }
        else:
            popt = []
            pcov = None
            result_params = {
                'r_V': initial_r_V,
                'r_OH': initial_r_OH,
                'alpha': alpha,
                'gamma': gamma
            }
        
        # Calculate model expansion
        dl_model = geometric_model(T_data, y_data,
                                  *([result_params[p] for p in param_names] if param_names else []))
        
        residuals = dl_data - dl_model
        mse = np.mean(residuals**2)
        rmse = np.sqrt(mse)
        r2 = 1 - np.sum(residuals**2) / np.sum((dl_data - np.mean(dl_data))**2)
        
        N = len(dl_data)
        p = len(vary_params)
        if N > p:
            chi2 = np.sum(residuals**2) / (N - p)
        else:
            chi2 = np.nan
        
        # Calculate individual sublattice contributions
        a_ref_val = a_ref
        sublattice_contributions = []
        for i in range(len(T_data)):
            r_A, r_B, r_O = calculate_average_radii(x, y_data[i], A_cation, B_cation,
                                                    Acc_cation, radii_db,
                                                    result_params['r_V'], result_params['r_OH'])
            a_total = np.sqrt(2) * (r_A + r_B + r_O)
            
            # Calculate contributions from each sublattice
            a_A_only = np.sqrt(2) * (r_A + 0 + 0)
            a_B_only = np.sqrt(2) * (0 + r_B + 0)
            a_O_only = np.sqrt(2) * (0 + 0 + r_O)
            
            sublattice_contributions.append({
                'T': T_data[i],
                'y': y_data[i],
                'a_total': a_total,
                'a_A': a_A_only,
                'a_B': a_B_only,
                'a_O': a_O_only
            })
        
        return {
            'params': result_params,
            'popt': popt,
            'pcov': pcov,
            'dl_model': dl_model,
            'residuals': residuals,
            'mse': mse,
            'rmse': rmse,
            'r2': r2,
            'chi2': chi2,
            'N_points': N,
            'n_free_params': p,
            'vary_params': vary_params,
            'param_names': param_names,
            'T_data': T_data,
            'dl_data': dl_data,
            'y_data': y_data,
            'a_ref': a_ref_val,
            'sublattice_contributions': sublattice_contributions
        }
    
    except Exception as e:
        st.error(f"Geometric fitting error: {str(e)}")
        return None

# ============================================
# ФУНКЦИИ ДЛЯ СОЗДАНИЯ ГРАФИКОВ (С КЭШИРОВАНИЕМ)
# ============================================

@st.cache_data(ttl=3600, show_spinner=False)
def create_plot1_cached(fit_results: Dict[str, Any], style: Dict[str, Any]) -> plt.Figure:
    """Create plot 1: Experimental data and model with residual plot (cached)"""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(5, 6), 
                                   height_ratios=[3, 1],
                                   sharex=True)
    fig.subplots_adjust(hspace=0.05)
    
    T = fit_results['T_data']
    dl_exp = fit_results['dl_data']
    dl_model = fit_results['dl_model']
    residuals = fit_results['residuals']
    
    ax1.scatter(T, dl_exp, s=style['point_size'], color=style['point_color'], 
               edgecolor='none', 
               label='Experimental', zorder=3, alpha=style['point_alpha'])
    
    ax1.plot(T, dl_model, '-', color=style['model_line_color'], 
            linewidth=style['line_width'], label='Model', zorder=4)
    
    ax1.set_ylabel('ΔL/L₀', fontweight='bold', fontsize=11)
    ax1.legend(loc='best')
    ax1.grid(True, alpha=0.3, linestyle='--')
    
    # Residual plot
    ax2.fill_between(T, 0, residuals, alpha=0.3, color='gray')
    ax2.plot(T, residuals, 'k-', linewidth=1)
    ax2.axhline(y=0, color='red', linestyle='--', linewidth=1, alpha=0.7)
    ax2.set_xlabel('Temperature (°C)', fontweight='bold', fontsize=11)
    ax2.set_ylabel('Residual', fontweight='bold', fontsize=11)
    ax2.grid(True, alpha=0.3, linestyle='--')
    
    fig.set_dpi(600)
    return fig

@st.cache_data(ttl=3600, show_spinner=False)
def create_plot2_cached(fit_results: Dict[str, Any], style: Dict[str, Any]) -> plt.Figure:
    """Create plot 2: Model contributions (cached)"""
    fig, ax = plt.subplots(figsize=(5, 3.5))
    
    T = fit_results['T_data']
    dl_model = fit_results['dl_model']
    thermal_contrib = fit_results['thermal_contrib']
    chem_contrib = fit_results['chem_contrib']
    residue = fit_results['params']['residue']
    
    ax.plot(T, dl_model, '-', color=style['model_line_color'], 
           linewidth=style['line_width'], label='Total model')
    ax.plot(T, thermal_contrib + residue, '--', color=style['thermal_line_color'], 
           linewidth=style['line_width'], label='Thermal contribution')
    ax.plot(T, chem_contrib - chem_contrib[-1], '--', color=style['chemical_line_color'], 
           linewidth=style['line_width'], label='Chemical contribution')
    
    ax.set_xlabel('Temperature (°C)', fontweight='bold', fontsize=11)
    ax.set_ylabel('ΔL/L₀', fontweight='bold', fontsize=11)
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3, linestyle='--')
    
    fig.set_dpi(600)
    return fig

@st.cache_data(ttl=3600, show_spinner=False)
def create_plot3_cached(fit_results: Dict[str, Any], style: Dict[str, Any]) -> plt.Figure:
    """Create plot 3: Histograms of changes (cached)"""
    fig, ax = plt.subplots(figsize=(5, 3.5))
    
    T = fit_results['T_data']
    thermal_contrib = fit_results['thermal_contrib']
    chem_contrib = fit_results['chem_contrib']
    residue = fit_results['params']['residue']
    
    T_diff = T[-1] - T[0]
    
    if T_diff > 0:
        thermal_start = thermal_contrib[0] + residue
        thermal_changes = (thermal_contrib + residue) - thermal_start
        chem_end = chem_contrib[-1] + residue
        chem_changes = chem_end - (chem_contrib + residue)
    else:
        thermal_end = thermal_contrib[-1] + residue
        thermal_changes = (thermal_contrib + residue) - thermal_end
        chem_start = chem_contrib[0] + residue
        chem_changes = (chem_contrib + residue) - chem_start
    
    bar_width = (T[1] - T[0]) * 0.7 if len(T) > 1 else 10
    ax.bar(T - bar_width/2, thermal_changes, width=bar_width, 
           color=style['bar_thermal_color'], alpha=0.7, label='Δ Thermal')
    ax.bar(T + bar_width/2, chem_changes, width=bar_width, 
           color=style['bar_chemical_color'], alpha=0.7, label='Δ Chemical')
    
    ax.set_xlabel('Temperature (°C)', fontweight='bold', fontsize=11)
    ax.set_ylabel('Change in ΔL/L₀', fontweight='bold', fontsize=11)
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3, linestyle='--', axis='y')
    
    direction = "Heating" if T_diff > 0 else "Cooling"
    ax.text(0.98, 0.98, direction, transform=ax.transAxes,
           ha='right', va='top', fontweight='bold',
           bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
    
    fig.set_dpi(600)
    return fig

@st.cache_data(ttl=3600, show_spinner=False)
def create_plot4_cached(fit_results: Dict[str, Any], style: Dict[str, Any]) -> plt.Figure:
    """Create plot 4: TEC and Proton concentration (cached)"""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(5, 6), 
                                   height_ratios=[3, 1],
                                   sharex=True)
    fig.subplots_adjust(hspace=0.05)
    
    T = fit_results['T_data']
    
    ax1.plot(T, fit_results['tec_exp']*1e6, 'o-', color=style['tec_exp_color'], 
            linewidth=style['line_width']-0.5, markersize=4, alpha=0.7)
    ax1.plot(T, fit_results['tec_model']*1e6, '-', color=style['tec_model_color'], 
            linewidth=style['line_width'])
    ax1_right = ax1.twinx()
    ax1_right.plot(T, fit_results['oh_concentration'], '--', color=style['oh_color'], 
                  linewidth=style['line_width'])
    
    ax1.set_ylabel('TEC (10⁻⁶ K⁻¹)', fontweight='bold', fontsize=11, color='black')
    ax1.tick_params(axis='y', labelcolor='black')
    ax1.grid(True, alpha=0.3, linestyle='--')
    
    ax1_right.set_ylabel('[OH] (arb. units)', fontweight='bold', 
                       fontsize=11, color=style['oh_color'])
    ax1_right.tick_params(axis='y', labelcolor=style['oh_color'])
    
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color=style['tec_exp_color'], 
               label='Experimental TEC', markersize=6, linewidth=style['line_width']-0.5),
        Line2D([0], [0], color=style['tec_model_color'], 
               label='Model TEC', linewidth=style['line_width']),
        Line2D([0], [0], color=style['oh_color'], linestyle='--',
               label='[OH] concentration', linewidth=style['line_width'])
    ]
    
    ax1.legend(handles=legend_elements, loc='center left', 
              bbox_to_anchor=(0.45, 0.5), frameon=True, framealpha=0.9)
    
    tec_residuals = fit_results['tec_exp'] - fit_results['tec_model']
    ax2.fill_between(T, 0, tec_residuals*1e6, alpha=0.3, color='purple')
    ax2.plot(T, tec_residuals*1e6, 'k-', linewidth=1)
    ax2.axhline(y=0, color='red', linestyle='--', linewidth=1, alpha=0.7)
    ax2.set_xlabel('Temperature (°C)', fontweight='bold', fontsize=11)
    ax2.set_ylabel('TEC Residual\n(10⁻⁶ K⁻¹)', fontweight='bold', fontsize=11)
    ax2.grid(True, alpha=0.3, linestyle='--')
    
    fig.set_dpi(600)
    return fig

@st.cache_data(ttl=3600, show_spinner=False)
def create_plot5_cached(fit_results: Dict[str, Any], style: Dict[str, Any]) -> plt.Figure:
    """Create plot 5: (ΔL/Lo)exp vs (ΔL/Lo)model with temperature color scale (cached)"""
    fig, ax = plt.subplots(figsize=(5, 3.5))
    
    dl_exp = fit_results['dl_data']
    dl_model = fit_results['dl_model']
    T = fit_results['T_data']
    
    cmap = getattr(cm, style['cmap_style'])
    norm = Normalize(vmin=T.min(), vmax=T.max())
    
    sc = ax.scatter(dl_model, dl_exp, c=T, cmap=cmap, norm=norm,
                   s=style['point_size']*0.7, edgecolor='none')
    
    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label('Temperature (°C)', fontweight='bold', fontsize=10)
    
    min_val = min(dl_exp.min(), dl_model.min())
    max_val = max(dl_exp.max(), dl_model.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', 
           linewidth=1, alpha=0.7, label='Perfect fit')
    
    ax.set_xlabel('Model ΔL/L₀', fontweight='bold', fontsize=11)
    ax.set_ylabel('Experimental ΔL/L₀', fontweight='bold', fontsize=11)
    ax.set_title('Correlation between Model and Experiment', fontweight='bold', fontsize=12)
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3, linestyle='--')
    
    fig.set_dpi(600)
    return fig

@st.cache_data(ttl=3600, show_spinner=False)
def create_plot6_cached(fit_results: Dict[str, Any], style: Dict[str, Any]) -> plt.Figure:
    """Create plot 6: α_exp vs α_model with temperature color scale (cached)"""
    fig, ax = plt.subplots(figsize=(5, 3.5))
    
    tec_exp = fit_results['tec_exp'] * 1e6
    tec_model = fit_results['tec_model'] * 1e6
    T = fit_results['T_data']
    
    cmap = getattr(cm, style['cmap_style'])
    norm = Normalize(vmin=T.min(), vmax=T.max())
    
    sc = ax.scatter(tec_model, tec_exp, c=T, cmap=cmap, norm=norm,
                   s=style['point_size']*0.7, edgecolor='none')
    
    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label('Temperature (°C)', fontweight='bold', fontsize=10)
    
    min_val = min(tec_exp.min(), tec_model.min())
    max_val = max(tec_exp.max(), tec_model.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', 
           linewidth=1, alpha=0.7, label='Perfect fit')
    
    ax.set_xlabel('Model TEC (10⁻⁶ K⁻¹)', fontweight='bold', fontsize=11)
    ax.set_ylabel('Experimental TEC (10⁻⁶ K⁻¹)', fontweight='bold', fontsize=11)
    ax.set_title('Correlation between Model and Experimental TEC', 
                fontweight='bold', fontsize=12)
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3, linestyle='--')
    
    fig.set_dpi(600)
    return fig

@st.cache_data(ttl=3600, show_spinner=False)
def create_plot7_cached(fit_results: Dict[str, Any], style: Dict[str, Any]) -> plt.Figure:
    """Create plot 7: Долевой вклад теплового и химического компонентов (cached)"""
    T = fit_results['T_data']
    thermal = fit_results['thermal_contrib']
    chemical = fit_results['chem_contrib']
    residue = fit_results['params']['residue']
    
    thermal_abs = thermal + residue
    chemical_abs = chemical - chemical[-1]
    
    thermal_percent = np.abs(thermal_abs) / (np.abs(thermal_abs) + np.abs(chemical_abs)) * 100
    chemical_percent = np.abs(chemical_abs) / (np.abs(thermal_abs) + np.abs(chemical_abs)) * 100
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(5, 6), sharex=True)
    fig.subplots_adjust(hspace=0.1)
    
    ax1.plot(T, thermal_abs, '-', color=style['thermal_line_color'], 
            linewidth=style['line_width'], label='Thermal contribution')
    ax1.plot(T, chemical_abs, '-', color=style['chemical_line_color'], 
            linewidth=style['line_width'], label='Chemical contribution')
    ax1.set_ylabel('Absolute Contribution', fontweight='bold', fontsize=11)
    ax1.legend(loc='best')
    ax1.grid(True, alpha=0.3, linestyle='--')
    
    ax2.stackplot(T, [thermal_percent, chemical_percent], 
                  colors=[style['thermal_line_color'], style['chemical_line_color']],
                  labels=['Thermal %', 'Chemical %'], alpha=0.7)
    ax2.set_xlabel('Temperature (°C)', fontweight='bold', fontsize=11)
    ax2.set_ylabel('Relative Contribution (%)', fontweight='bold', fontsize=11)
    ax2.legend(loc='upper left')
    ax2.grid(True, alpha=0.3, linestyle='--')
    ax2.set_ylim(0, 100)
    
    max_thermal_temp = T[np.argmax(thermal_percent)]
    max_chemical_temp = T[np.argmax(chemical_percent)]
    
    ax2.text(0.02, 0.98, f'Max thermal: {max_thermal_temp:.1f}°C', 
             transform=ax2.transAxes, ha='left', va='top',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    ax2.text(0.02, 0.90, f'Max chemical: {max_chemical_temp:.1f}°C', 
             transform=ax2.transAxes, ha='left', va='top',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    fig.set_dpi(600)
    return fig

@st.cache_data(ttl=3600, show_spinner=False)
def create_plot8_cached(fit_results: Dict[str, Any], style: Dict[str, Any]) -> plt.Figure:
    """Create plot 8: Чувствительность модели к вариациям ключевых параметров (cached)"""
    T = fit_results['T_data']
    base_dl = fit_results['dl_model']
    params = fit_results['params']
    T_start = fit_results['T_start']
    oh_start = fit_results['oh_start']
    
    fig, axes = plt.subplots(2, 2, figsize=(8, 6))
    axes = axes.flatten()
    
    variations = [-0.10, -0.05, 0, 0.05, 0.10]
    for var in variations:
        alpha_var = params['alpha_1e6'] * (1 + var)
        dl_var = model_func_cached(T, params['Acc'], alpha_var, params['beta'],
                                  params['dH'], params['dS'], params['pH2O'], 
                                  params['residue'], T_start, oh_start)
        axes[0].plot(T, (dl_var - base_dl)*1e6, label=f'{var*100:+}%', linewidth=1.5)
    axes[0].set_title('Sensitivity to α (CTE)', fontweight='bold', fontsize=11)
    axes[0].set_ylabel('Δ(ΔL/L₀) (10⁻⁶)', fontweight='bold', fontsize=10)
    axes[0].grid(True, alpha=0.3, linestyle='--')
    axes[0].legend(title='α variation', fontsize=8, title_fontsize=9)
    
    for var in variations:
        beta_var = params['beta'] * (1 + var)
        dl_var = model_func_cached(T, params['Acc'], params['alpha_1e6'], beta_var,
                                  params['dH'], params['dS'], params['pH2O'], 
                                  params['residue'], T_start, oh_start)
        axes[1].plot(T, (dl_var - base_dl)*1e6, label=f'{var*100:+}%', linewidth=1.5)
    axes[1].set_title('Sensitivity to β (chemical coeff.)', fontweight='bold', fontsize=11)
    axes[1].set_ylabel('Δ(ΔL/L₀) (10⁻⁶)', fontweight='bold', fontsize=10)
    axes[1].grid(True, alpha=0.3, linestyle='--')
    axes[1].legend(title='β variation', fontsize=8, title_fontsize=9)
    
    dH_variations = [-5, -2, 0, 2, 5]
    for var in dH_variations:
        dH_var = params['dH'] + var
        dl_var = model_func_cached(T, params['Acc'], params['alpha_1e6'], params['beta'],
                                  dH_var, params['dS'], params['pH2O'], 
                                  params['residue'], T_start, oh_start)
        axes[2].plot(T, (dl_var - base_dl)*1e6, label=f'{var:+} kJ/mol', linewidth=1.5)
    axes[2].set_title('Sensitivity to ΔH', fontweight='bold', fontsize=11)
    axes[2].set_xlabel('Temperature (°C)', fontweight='bold', fontsize=10)
    axes[2].set_ylabel('Δ(ΔL/L₀) (10⁻⁶)', fontweight='bold', fontsize=10)
    axes[2].grid(True, alpha=0.3, linestyle='--')
    axes[2].legend(title='ΔH shift', fontsize=8, title_fontsize=9)
    
    pH2O_variations = [-0.5, -0.2, 0, 0.2, 0.5]
    for var in pH2O_variations:
        pH2O_var = params['pH2O'] * 10**var
        dl_var = model_func_cached(T, params['Acc'], params['alpha_1e6'], params['beta'],
                                  params['dH'], params['dS'], pH2O_var, 
                                  params['residue'], T_start, oh_start)
        axes[3].plot(T, (dl_var - base_dl)*1e6, label=f'×{10**var:.1f}', linewidth=1.5)
    axes[3].set_title('Sensitivity to pH₂O', fontweight='bold', fontsize=11)
    axes[3].set_xlabel('Temperature (°C)', fontweight='bold', fontsize=10)
    axes[3].set_ylabel('Δ(ΔL/L₀) (10⁻⁶)', fontweight='bold', fontsize=10)
    axes[3].grid(True, alpha=0.3, linestyle='--')
    axes[3].legend(title='pH₂O factor', fontsize=8, title_fontsize=9)
    
    plt.tight_layout()
    fig.set_dpi(600)
    return fig

@st.cache_data(ttl=3600, show_spinner=False)
def create_plot9_cached(fit_results: Dict[str, Any], style: Dict[str, Any]) -> plt.Figure:
    """Create plot 9: Фазовый портрет системы (cached)"""
    T = fit_results['T_data']
    dl = fit_results['dl_data']
    dl_model = fit_results['dl_model']
    
    dldT_exp = np.gradient(dl, T)
    dldT_model = np.gradient(dl_model, T)
    
    d2ldT2_exp = np.gradient(dldT_exp, T)
    d2ldT2_model = np.gradient(dldT_model, T)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    
    sc1 = ax1.scatter(dl[1:-1], dldT_exp[1:-1]*1e6, c=T[1:-1], 
                     cmap=style['cmap_style'], s=50, edgecolor='none', alpha=0.8)
    ax1.plot(dl_model, dldT_model*1e6, 'k-', linewidth=2, alpha=0.7, label='Model trajectory')
    ax1.set_xlabel('ΔL/L₀', fontweight='bold', fontsize=11)
    ax1.set_ylabel('d(ΔL/L₀)/dT (10⁻⁶ K⁻¹)', fontweight='bold', fontsize=11)
    ax1.set_title('Phase Portrait: Position vs Velocity', fontweight='bold', fontsize=12)
    ax1.legend(loc='best')
    ax1.grid(True, alpha=0.3, linestyle='--')
    cbar1 = plt.colorbar(sc1, ax=ax1)
    cbar1.set_label('Temperature (°C)', fontweight='bold', fontsize=10)
    
    sc2 = ax2.scatter(dldT_exp[2:-2]*1e6, d2ldT2_exp[2:-2]*1e9, c=T[2:-2], 
                     cmap=style['cmap_style'], s=50, edgecolor='none', alpha=0.8)
    ax2.plot(dldT_model*1e6, d2ldT2_model*1e9, 'k-', linewidth=2, alpha=0.7, label='Model trajectory')
    ax2.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
    ax2.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax2.set_xlabel('d(ΔL/L₀)/dT (10⁻⁶ K⁻¹)', fontweight='bold', fontsize=11)
    ax2.set_ylabel('d²(ΔL/L₀)/dT² (10⁻⁹ K⁻²)', fontweight='bold', fontsize=11)
    ax2.set_title('Phase Portrait: Velocity vs Acceleration', fontweight='bold', fontsize=12)
    ax2.legend(loc='best')
    ax2.grid(True, alpha=0.3, linestyle='--')
    cbar2 = plt.colorbar(sc2, ax=ax2)
    cbar2.set_label('Temperature (°C)', fontweight='bold', fontsize=10)
    
    plt.tight_layout()
    fig.set_dpi(600)
    return fig

@st.cache_data(ttl=3600, show_spinner=False)
def create_plot10_cached(fit_results: Dict[str, Any], style: Dict[str, Any]) -> plt.Figure:
    """Create plot 10: Статистический анализ остатков (cached)"""
    residuals = fit_results['residuals']
    T = fit_results['T_data']
    dl_model = fit_results['dl_model']
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(10, 8))
    
    n_bins = min(15, len(residuals)//5)
    if n_bins < 5:
        n_bins = 5
    
    ax1.hist(residuals, bins=n_bins, density=True, alpha=0.7, 
            color=style['point_color'], edgecolor='black')
    
    mu, std = norm.fit(residuals)
    x = np.linspace(residuals.min(), residuals.max(), 100)
    ax1.plot(x, norm.pdf(x, mu, std), 'r-', linewidth=2, 
            label=f'Normal fit\nμ={mu:.2e}\nσ={std:.2e}')
    ax1.set_xlabel('Residuals', fontweight='bold', fontsize=11)
    ax1.set_ylabel('Density', fontweight='bold', fontsize=11)
    ax1.set_title('Distribution of Residuals', fontweight='bold', fontsize=12)
    ax1.legend(loc='best')
    ax1.grid(True, alpha=0.3, linestyle='--')
    
    probplot(residuals, dist="norm", plot=ax2)
    ax2.get_lines()[0].set_marker('o')
    ax2.get_lines()[0].set_markersize(4)
    ax2.get_lines()[0].set_markerfacecolor(style['point_color'])
    ax2.get_lines()[0].set_markeredgecolor(style['point_color'])
    ax2.get_lines()[0].set_alpha(0.7)
    ax2.get_lines()[1].set_color('red')
    ax2.get_lines()[1].set_linewidth(2)
    ax2.set_title('Q-Q Plot (Normality Test)', fontweight='bold', fontsize=12)
    ax2.grid(True, alpha=0.3, linestyle='--')
    ax2.set_xlabel('Theoretical Quantiles', fontweight='bold', fontsize=11)
    ax2.set_ylabel('Sample Quantiles', fontweight='bold', fontsize=11)
    
    max_lags = min(20, len(residuals)//2)
    if max_lags >= 5:
        from statsmodels.graphics.tsaplots import plot_acf
        plot_acf(residuals, lags=max_lags, ax=ax3, 
                alpha=0.05, title='Autocorrelation of Residuals',
                marker='o', markersize=4,
                markerfacecolor=style['point_color'],
                markeredgecolor=style['point_color'])
    else:
        ax3.text(0.5, 0.5, 'Insufficient data\nfor autocorrelation',
                ha='center', va='center', transform=ax3.transAxes,
                fontweight='bold')
        ax3.set_title('Autocorrelation of Residuals', fontweight='bold', fontsize=12)
    ax3.set_xlabel('Lag', fontweight='bold', fontsize=11)
    ax3.set_ylabel('Autocorrelation', fontweight='bold', fontsize=11)
    ax3.grid(True, alpha=0.3, linestyle='--')
    
    ax4.scatter(dl_model, residuals, s=30, alpha=0.7, 
               color=style['point_color'], edgecolor='none')
    ax4.axhline(y=0, color='red', linestyle='--', alpha=0.7, linewidth=1.5)
    ax4.set_xlabel('Predicted ΔL/L₀', fontweight='bold', fontsize=11)
    ax4.set_ylabel('Residuals', fontweight='bold', fontsize=11)
    ax4.set_title('Residuals vs Fitted Values', fontweight='bold', fontsize=12)
    ax4.grid(True, alpha=0.3, linestyle='--')
    
    ax4.axhline(y=2*std, color='gray', linestyle=':', alpha=0.5, linewidth=1)
    ax4.axhline(y=-2*std, color='gray', linestyle=':', alpha=0.5, linewidth=1)
    ax4.axhline(y=3*std, color='gray', linestyle=':', alpha=0.3, linewidth=0.8)
    ax4.axhline(y=-3*std, color='gray', linestyle=':', alpha=0.3, linewidth=0.8)
    
    ax4.text(0.05, 0.95, f'±2σ = ±{2*abs(std):.2e}\n±3σ = ±{3*abs(std):.2e}', 
             transform=ax4.transAxes, va='top', ha='left', fontsize=9, 
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    fig.set_dpi(600)
    return fig

@st.cache_data(ttl=3600, show_spinner=False)
def create_plot11_cached(fit_results: Dict[str, Any], style: Dict[str, Any]) -> plt.Figure:
    """Create plot 11: Динамика концентрации OH и её температурных производных (cached)"""
    T = fit_results['T_data']
    oh = fit_results['oh_concentration']
    
    dohdT = np.gradient(oh, T) * 1e3
    d2ohdT2 = np.gradient(dohdT, T)
    
    oh_safe = oh.copy()
    oh_safe[oh_safe < 1e-10] = 1e-10
    dlnohdT = dohdT / oh_safe * 1e-3
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6, 8), sharex=True)
    fig.subplots_adjust(hspace=0.1)
    
    ax1.plot(T, oh, '-', color=style['oh_color'], linewidth=2, label='[OH] concentration')
    ax1.set_ylabel('[OH] (arb. units)', color=style['oh_color'], fontweight='bold', fontsize=11)
    ax1.tick_params(axis='y', labelcolor=style['oh_color'])
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3, linestyle='--')
    
    ax1_r = ax1.twinx()
    ax1_r.plot(T, dohdT, '--', color='purple', linewidth=1.5, 
              label="d[OH]/dT (10⁻³ K⁻¹)")
    ax1_r.set_ylabel('d[OH]/dT (10⁻³ K⁻¹)', color='purple', fontweight='bold', fontsize=11)
    ax1_r.tick_params(axis='y', labelcolor='purple')
    ax1_r.legend(loc='upper right')
    ax1_r.grid(True, alpha=0.3, linestyle='--')
    
    ax1.set_title('OH Concentration Dynamics', fontweight='bold', fontsize=12)
    
    ax2.plot(T, d2ohdT2, '-', color='orange', linewidth=2, 
            label='d²[OH]/dT²', alpha=0.8)
    ax2.set_xlabel('Temperature (°C)', fontweight='bold', fontsize=11)
    ax2.set_ylabel('d²[OH]/dT²', fontweight='bold', fontsize=11, color='orange')
    ax2.tick_params(axis='y', labelcolor='orange')
    ax2.legend(loc='upper left')
    ax2.grid(True, alpha=0.3, linestyle='--')
    
    ax2_r = ax2.twinx()
    ax2_r.plot(T, dlnohdT, '--', color='green', linewidth=1.5, 
              label='d(ln[OH])/dT', alpha=0.8)
    ax2_r.set_ylabel('d(ln[OH])/dT (K⁻¹)', color='green', fontweight='bold', fontsize=11)
    ax2_r.tick_params(axis='y', labelcolor='green')
    ax2_r.legend(loc='upper right')
    
    max_dohdT_idx = np.argmax(np.abs(dohdT))
    ax1.annotate(f'Max rate\n{T[max_dohdT_idx]:.1f}°C',
                xy=(T[max_dohdT_idx], oh[max_dohdT_idx]),
                xytext=(T[max_dohdT_idx]+5, oh[max_dohdT_idx]*0.8),
                arrowprops=dict(arrowstyle='->', color='gray', alpha=0.7),
                fontsize=9, ha='center')
    
    sign_changes = np.where(np.diff(np.sign(d2ohdT2)))[0]
    if len(sign_changes) > 0:
        inflection_idx = sign_changes[0]
        ax2.annotate(f'Inflection\n{T[inflection_idx]:.1f}°C',
                    xy=(T[inflection_idx], d2ohdT2[inflection_idx]),
                    xytext=(T[inflection_idx]+5, d2ohdT2[inflection_idx]*0.8),
                    arrowprops=dict(arrowstyle='->', color='gray', alpha=0.7),
                    fontsize=9, ha='center')
    
    plt.tight_layout()
    fig.set_dpi(600)
    return fig

@st.cache_data(ttl=3600, show_spinner=False)
def create_plot12_cached(fit_results: Dict[str, Any], style: Dict[str, Any]) -> Optional[plt.Figure]:
    """Create plot 12: Корреляционная матрица параметров модели (cached)"""
    params = fit_results['params']
    pcov = fit_results.get('pcov', None)
    vary_params = fit_results.get('vary_params', [])
    
    if pcov is None or np.all(pcov == 0) or len(vary_params) < 2:
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.text(0.5, 0.5, 'Covariance matrix not available\nfor correlation analysis\n(too few fitted parameters or matrix is zero)',
                ha='center', va='center', transform=ax.transAxes,
                fontweight='bold', fontsize=12)
        ax.set_title('Parameter Correlation Matrix', fontweight='bold', fontsize=14)
        ax.axis('off')
        fig.set_dpi(600)
        return fig
    
    try:
        param_names_display = []
        param_indices = []
        
        all_param_names = ['Acc', 'alpha_1e6', 'beta', 'dH', 'dS', 'pH2O', 'residue']
        display_names = {
            'Acc': '[Acc]',
            'alpha_1e6': 'α·10⁶',
            'beta': 'β',
            'dH': 'ΔH',
            'dS': 'ΔS',
            'pH2O': 'pH₂O',
            'residue': 'Residue'
        }
        
        for i, name in enumerate(all_param_names):
            if name in vary_params:
                param_names_display.append(display_names.get(name, name))
                param_indices.append(i)
        
        pcov_subset = pcov[np.ix_(param_indices, param_indices)]
        
        std_dev = np.sqrt(np.diag(pcov_subset))
        std_dev[std_dev == 0] = 1e-10
        corr_matrix = pcov_subset / np.outer(std_dev, std_dev)
        
        corr_matrix = np.clip(corr_matrix, -1, 1)
        
        fig, ax = plt.subplots(figsize=(6, 5))
        
        im = ax.imshow(corr_matrix, cmap='coolwarm', vmin=-1, vmax=1, aspect='auto')
        
        for i in range(len(param_names_display)):
            for j in range(len(param_names_display)):
                text_color = 'white' if abs(corr_matrix[i, j]) > 0.7 else 'black'
                ax.text(j, i, f'{corr_matrix[i, j]:.2f}',
                       ha="center", va="center", color=text_color, fontsize=9,
                       fontweight='bold' if abs(corr_matrix[i, j]) > 0.8 else 'normal')
        
        ax.set_xticks(range(len(param_names_display)))
        ax.set_yticks(range(len(param_names_display)))
        ax.set_xticklabels(param_names_display, rotation=45, ha='right', fontsize=10)
        ax.set_yticklabels(param_names_display, fontsize=10)
        ax.set_title('Parameter Correlation Matrix', fontweight='bold', fontsize=14)
        
        ax.text(0.02, 1.02, f'Fitted parameters: {len(vary_params)}', 
                transform=ax.transAxes, ha='left', va='bottom', fontsize=9,
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Correlation Coefficient', fontweight='bold', fontsize=10)
        
        plt.tight_layout()
        fig.set_dpi(600)
        return fig
        
    except Exception as e:
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.text(0.5, 0.5, f'Error creating correlation matrix:\n{str(e)[:100]}...',
                ha='center', va='center', transform=ax.transAxes,
                fontweight='bold', fontsize=12)
        ax.set_title('Parameter Correlation Matrix', fontweight='bold', fontsize=14)
        ax.axis('off')
        fig.set_dpi(600)
        return fig

# ============================================
# НОВЫЕ ГРАФИКИ ДЛЯ ГЕОМЕТРИЧЕСКОЙ МОДЕЛИ
# ============================================

@st.cache_data(ttl=3600, show_spinner=False)
def create_plot13_cached(geometric_fit: Dict[str, Any], style: Dict[str, Any]) -> plt.Figure:
    """Plot 13: Effective ionic radii vs Coordination Number"""
    fig, ax = plt.subplots(figsize=(6, 5))
    
    T = geometric_fit['T_data']
    y = geometric_fit['y_data']
    x = st.session_state.model_params['Acc']['value'] if st.session_state.fit_results else 0.6
    
    CN_A, CN_B = calculate_coordination_numbers(x, y)
    
    ax.plot(CN_A, label='CN_A', marker='o', linewidth=2)
    ax.plot(CN_B, label='CN_B', marker='s', linewidth=2)
    ax.set_xlabel('Data point index', fontweight='bold', fontsize=11)
    ax.set_ylabel('Coordination Number', fontweight='bold', fontsize=11)
    ax.set_title('Coordination Numbers Evolution', fontweight='bold', fontsize=12)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    fig.set_dpi(600)
    return fig

@st.cache_data(ttl=3600, show_spinner=False)
def create_plot14_cached(geometric_fit: Dict[str, Any], style: Dict[str, Any]) -> plt.Figure:
    """Plot 14: Lattice parameter evolution"""
    fig, ax = plt.subplots(figsize=(6, 5))
    
    T = geometric_fit['T_data']
    dl_exp = geometric_fit['dl_data']
    dl_model = geometric_fit['dl_model']
    
    ax.scatter(T, dl_exp, s=style['point_size'], color=style['point_color'], 
               alpha=style['point_alpha'], label='Experimental')
    ax.plot(T, dl_model, '-', color=style['model_line_color'], 
            linewidth=style['line_width'], label='Geometric model')
    ax.set_xlabel('Temperature (°C)', fontweight='bold', fontsize=11)
    ax.set_ylabel('ΔL/L₀', fontweight='bold', fontsize=11)
    ax.set_title('Geometric Model Fit', fontweight='bold', fontsize=12)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    fig.set_dpi(600)
    return fig

@st.cache_data(ttl=3600, show_spinner=False)
def create_plot15_cached(geometric_fit: Dict[str, Any], style: Dict[str, Any]) -> plt.Figure:
    """Plot 15: Sublattice contributions to expansion"""
    fig, ax = plt.subplots(figsize=(8, 5))
    
    contributions = geometric_fit['sublattice_contributions']
    T = [c['T'] for c in contributions]
    
    a_A = [c['a_A'] for c in contributions]
    a_B = [c['a_B'] for c in contributions]
    a_O = [c['a_O'] for c in contributions]
    a_total = [c['a_total'] for c in contributions]
    
    a_ref = geometric_fit['a_ref']
    
    dl_A = [(a - a_ref)/a_ref for a in a_A]
    dl_B = [(a - a_ref)/a_ref for a in a_B]
    dl_O = [(a - a_ref)/a_ref for a in a_O]
    dl_total = [(a - a_ref)/a_ref for a in a_total]
    
    ax.fill_between(T, 0, dl_A, alpha=0.5, label='A-sublattice', color='red')
    ax.fill_between(T, dl_A, np.array(dl_A) + np.array(dl_B), alpha=0.5, 
                    label='B-sublattice', color='green')
    ax.fill_between(T, np.array(dl_A) + np.array(dl_B), 
                    np.array(dl_A) + np.array(dl_B) + np.array(dl_O), 
                    alpha=0.5, label='Anion sublattice', color='blue')
    ax.plot(T, dl_total, 'k-', linewidth=2, label='Total')
    
    ax.set_xlabel('Temperature (°C)', fontweight='bold', fontsize=11)
    ax.set_ylabel('ΔL/L₀', fontweight='bold', fontsize=11)
    ax.set_title('Sublattice Contributions to Expansion', fontweight='bold', fontsize=12)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    fig.set_dpi(600)
    return fig

@st.cache_data(ttl=3600, show_spinner=False)
def create_plot16_cached(geometric_fit: Dict[str, Any], style: Dict[str, Any]) -> plt.Figure:
    """Plot 16: Fitted r_V and r_OH comparison"""
    fig, ax = plt.subplots(figsize=(6, 5))
    
    params = geometric_fit['params']
    
    literature_data = {
        'r_V': {'BaZrO3': 1.18, 'SrZrO3': 1.15, 'CaZrO3': 1.12},
        'r_OH': {'BaZrO3': 1.35, 'SrZrO3': 1.34, 'CaZrO3': 1.33}
    }
    
    categories = ['r_V (Å)', 'r_OH (Å)']
    fitted_values = [params['r_V'], params['r_OH']]
    
    x_pos = np.arange(len(categories))
    ax.bar(x_pos, fitted_values, width=0.4, label='Fitted', color='blue', alpha=0.7)
    
    # Add literature comparison if available
    ax.axhline(y=1.18, color='red', linestyle='--', alpha=0.7, label='Literature r_V (BaZrO₃)')
    ax.axhline(y=1.35, color='orange', linestyle='--', alpha=0.7, label='Literature r_OH (BaZrO₃)')
    
    ax.set_ylabel('Effective Ionic Radius (Å)', fontweight='bold', fontsize=11)
    ax.set_title('Fitted Effective Radii', fontweight='bold', fontsize=12)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(categories)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    for i, v in enumerate(fitted_values):
        ax.text(i, v + 0.02, f'{v:.3f}', ha='center', fontweight='bold')
    
    fig.set_dpi(600)
    return fig

@st.cache_data(ttl=3600, show_spinner=False)
def create_plot17_cached(geometric_fit: Dict[str, Any], style: Dict[str, Any]) -> plt.Figure:
    """Plot 17: Chemical expansion coefficient vs [OH]"""
    fig, ax = plt.subplots(figsize=(6, 5))
    
    T = geometric_fit['T_data']
    y = geometric_fit['y_data']
    dl_model = geometric_fit['dl_model']
    dl_exp = geometric_fit['dl_data']
    
    # Calculate local chemical expansion coefficient
    chem_exp_coeff = np.gradient(dl_model, y) * y
    
    ax.plot(y, chem_exp_coeff * 1e3, 'o-', color=style['chemical_line_color'], 
            linewidth=2, markersize=4)
    ax.set_xlabel('[OH] concentration', fontweight='bold', fontsize=11)
    ax.set_ylabel('Chemical Expansion Coefficient (10⁻³)', fontweight='bold', fontsize=11)
    ax.set_title('Nonlinearity of Chemical Expansion', fontweight='bold', fontsize=12)
    ax.grid(True, alpha=0.3)
    
    fig.set_dpi(600)
    return fig

@st.cache_data(ttl=3600, show_spinner=False)
def create_plot18_cached(geometric_fit: Dict[str, Any], style: Dict[str, Any]) -> plt.Figure:
    """Plot 18: Arrhenius-type plot for Kw(T)"""
    fig, ax = plt.subplots(figsize=(6, 5))
    
    T = geometric_fit['T_data']
    y = geometric_fit['y_data']
    
    # Calculate Kw from the model
    x = st.session_state.model_params['Acc']['value'] if st.session_state.fit_results else 0.6
    T_K = T + 273.15
    R = 8.314
    dH = st.session_state.fit_results['params']['dH'] if st.session_state.fit_results else -81.0
    dS = st.session_state.fit_results['params']['dS'] if st.session_state.fit_results else -131.0
    pH2O = st.session_state.fit_results['params']['pH2O'] if st.session_state.fit_results else 0.083
    
    Kw = np.exp(-dH * 1000 / (R * T_K) + dS / R)
    
    ax.plot(1000/T_K, np.log(Kw), 'o-', color='purple', linewidth=2, markersize=4)
    
    # Linear fit
    coeffs = np.polyfit(1000/T_K, np.log(Kw), 1)
    fit_line = coeffs[0] * 1000/T_K + coeffs[1]
    ax.plot(1000/T_K, fit_line, 'r--', linewidth=1.5, 
            label=f'Fit: slope = {coeffs[0]:.1f} K')
    
    ax.set_xlabel('1000/T (K⁻¹)', fontweight='bold', fontsize=11)
    ax.set_ylabel('ln(K_w)', fontweight='bold', fontsize=11)
    ax.set_title('Arrhenius Plot for Hydration Equilibrium', fontweight='bold', fontsize=12)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    fig.set_dpi(600)
    return fig

@st.cache_data(ttl=3600, show_spinner=False)
def create_plot19_cached(geometric_fit: Dict[str, Any], style: Dict[str, Any]) -> plt.Figure:
    """Plot 19: Residual analysis for geometric model"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    
    residuals = geometric_fit['residuals']
    T = geometric_fit['T_data']
    dl_model = geometric_fit['dl_model']
    
    # Residuals vs temperature
    ax1.scatter(T, residuals, s=style['point_size'], color=style['point_color'], 
                alpha=style['point_alpha'])
    ax1.axhline(y=0, color='red', linestyle='--', linewidth=1.5)
    ax1.set_xlabel('Temperature (°C)', fontweight='bold', fontsize=11)
    ax1.set_ylabel('Residuals', fontweight='bold', fontsize=11)
    ax1.set_title('Residuals vs Temperature', fontweight='bold', fontsize=12)
    ax1.grid(True, alpha=0.3)
    
    # Residuals vs predicted
    ax2.scatter(dl_model, residuals, s=style['point_size'], color=style['point_color'], 
                alpha=style['point_alpha'])
    ax2.axhline(y=0, color='red', linestyle='--', linewidth=1.5)
    ax2.set_xlabel('Predicted ΔL/L₀', fontweight='bold', fontsize=11)
    ax2.set_ylabel('Residuals', fontweight='bold', fontsize=11)
    ax2.set_title('Residuals vs Fitted Values', fontweight='bold', fontsize=12)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    fig.set_dpi(600)
    return fig

# ============================================
# ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
# ============================================

def get_metric_explanation() -> Dict[str, Dict[str, str]]:
    """Return explanations for the fitting metrics"""
    return {
        'MSE': {
            'title': 'Mean Squared Error (MSE)',
            'formula': r'$\text{MSE} = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2$',
            'explanation': 'Measures the average squared difference between experimental and model values. Lower values indicate better fit.',
            'units': '(ΔL/L₀)²'
        },
        'RMSE': {
            'title': 'Root Mean Squared Error (RMSE)',
            'formula': r'$\text{RMSE} = \sqrt{\frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2}$',
            'explanation': 'Square root of MSE, provides error in the same units as the measured quantity. More interpretable than MSE.',
            'units': 'ΔL/L₀'
        },
        'R²': {
            'title': 'Coefficient of Determination (R²)',
            'formula': r'$R^2 = 1 - \frac{\sum (y_i - \hat{y}_i)^2}{\sum (y_i - \bar{y})^2}$',
            'explanation': 'Represents the proportion of variance in the experimental data explained by the model. Ranges from 0 to 1, with 1 indicating perfect fit.',
            'units': 'dimensionless'
        },
        'χ²': {
            'title': 'Reduced Chi-Squared Statistic (χ²_red)',
            'formula': r'$\chi^2_{\text{red}} = \frac{1}{n-p} \sum_{i=1}^{n} \frac{(y_i - \hat{y}_i)^2}{\sigma_i^2}$',
            'explanation': 'For regression without known measurement errors, we assume σ = 1. This normalized metric accounts for degrees of freedom. Values close to 1 indicate good fit.',
            'units': 'dimensionless'
        }
    }

def update_model_param(param_name: str, value: float, is_fixed: bool):
    """Update model parameter in session state"""
    st.session_state.model_params[param_name]['value'] = value
    st.session_state.model_params[param_name]['fixed'] = is_fixed

def update_plot_style(param_name: str, value):
    """Update plot style parameter in session state"""
    st.session_state.plot_style[param_name] = value
    if st.session_state.fitting_complete:
        st.rerun()

def get_available_b_cations(a_cation: str) -> list:
    """Get available B-cations based on A-cation selection"""
    if a_cation == 'La':
        return ['In', 'Sc', 'Y', 'Yb']
    else:
        return ['Ce', 'Zr', 'Ti', 'Hf', 'Sn']

def get_available_acc_cations(a_cation: str) -> list:
    """Get available acceptor cations based on A-cation selection"""
    if a_cation == 'La':
        return ['Mg', 'Zn']
    else:
        return ['Al', 'Ga', 'In', 'Sc', 'Y', 'Yb', 'Ho', 'Dy', 'Gd', 'Sm', 'Nd', 'La']

# ============================================
# ОСНОВНОЙ ИНТЕРФЕЙС
# ============================================

def main():
    st.title("📈 Thermo-Mechanical Expansion Modeling")
    st.markdown("Modeling of proton-conducting oxides thermal expansion")
    
    # Initialize ionic radii database
    if st.session_state.ionic_radii_db is None:
        st.session_state.ionic_radii_db = parse_shannon_table()
    
    # Sidebar для ввода данных и параметров модели
    with st.sidebar:
        st.header("Input Data")
        
        # Data input options
        data_option = st.radio("Data input method:", 
                              ["Manual entry", "File upload", "Example data"])
        
        if data_option == "Manual entry":
            data_text = st.text_area(
                "Enter data (Temperature ΔL/L₀):",
                value="20\t0.0045\n40\t0.004787988\n60\t0.005075916\n80\t0.005363555\n100\t0.005650042\n120\t0.005932612\n140\t0.006203565",
                height=200,
                help="Separators: tab, space, comma, or semicolon"
            )
            if st.button("Load Data", type="primary"):
                try:
                    st.session_state.experimental_data = parse_data_cached(data_text)
                    st.session_state.data_loaded = True
                    st.success(f"Loaded {len(st.session_state.experimental_data)} data points")
                    st.session_state.fitting_complete = False
                    st.session_state.geometric_fit_complete = False
                    st.session_state.fit_results = None
                    st.session_state.geometric_fit_results = None
                except Exception as e:
                    st.error(f"Error: {str(e)}")
        
        elif data_option == "File upload":
            uploaded_file = st.file_uploader("Choose a file", type=['txt', 'csv', 'dat'])
            if uploaded_file is not None:
                try:
                    data_text = uploaded_file.getvalue().decode()
                    st.session_state.experimental_data = parse_data_cached(data_text)
                    st.session_state.data_loaded = True
                    st.success(f"Loaded {len(st.session_state.experimental_data)} data points")
                    st.session_state.fitting_complete = False
                    st.session_state.geometric_fit_complete = False
                    st.session_state.fit_results = None
                    st.session_state.geometric_fit_results = None
                except Exception as e:
                    st.error(f"Error: {str(e)}")
        
        else:
            if st.button("Load Example Data"):
                example_data = """20\t0.0045
40\t0.004787988
60\t0.005075916
80\t0.005363555
100\t0.005650042
120\t0.005932612
140\t0.006203565"""
                st.session_state.experimental_data = parse_data_cached(example_data)
                st.session_state.data_loaded = True
                st.success(f"Loaded {len(st.session_state.experimental_data)} example data points")
                st.session_state.fitting_complete = False
                st.session_state.geometric_fit_complete = False
                st.session_state.fit_results = None
                st.session_state.geometric_fit_results = None
        
        st.divider()
        
        # Cation Selection
        st.header("Cation Selection")
        
        A_cation = st.selectbox(
            "A-site cation",
            ['Ba', 'Sr', 'Ca', 'La'],
            index=['Ba', 'Sr', 'Ca', 'La'].index(st.session_state.cation_selection['A_cation'])
        )
        
        available_B = get_available_b_cations(A_cation)
        B_cation = st.selectbox(
            "B-site cation",
            available_B,
            index=available_B.index(st.session_state.cation_selection['B_cation']) 
            if st.session_state.cation_selection['B_cation'] in available_B else 0
        )
        
        available_Acc = get_available_acc_cations(A_cation)
        Acc_cation = st.selectbox(
            "Acceptor cation (Acc)",
            available_Acc,
            index=available_Acc.index(st.session_state.cation_selection['Acc_cation'])
            if st.session_state.cation_selection['Acc_cation'] in available_Acc else 0
        )
        
        st.session_state.cation_selection = {
            'A_cation': A_cation,
            'B_cation': B_cation,
            'Acc_cation': Acc_cation
        }
        
        st.divider()
        
        # Model Parameters Form
        st.header("Model Parameters")
        st.markdown("Check 'Fix' to keep parameter constant during fitting")
        
        with st.form("model_params_form"):
            col1, col2 = st.columns(2)
            
            with col1:
                acc_value = st.number_input(
                    "[Acc]", 
                    value=st.session_state.model_params['Acc']['value'],
                    step=0.01, 
                    format="%.4f",
                    key=f"acc_input_{st.session_state.model_params['Acc']['value']}"
                )
                acc_fixed = st.checkbox("Fix", value=st.session_state.model_params['Acc']['fixed'], 
                                       key=f"acc_fix_{st.session_state.model_params['Acc']['fixed']}")
                
                alpha_value = st.number_input(
                    "α·10⁶", 
                    value=st.session_state.model_params['alpha_1e6']['value'],
                    step=0.1, 
                    format="%.4f",
                    key=f"alpha_input_{st.session_state.model_params['alpha_1e6']['value']}"
                )
                alpha_fixed = st.checkbox("Fix", value=st.session_state.model_params['alpha_1e6']['fixed'],
                                         key=f"alpha_fix_{st.session_state.model_params['alpha_1e6']['fixed']}")
                
                beta_value = st.number_input(
                    "β", 
                    value=st.session_state.model_params['beta']['value'],
                    step=0.001, 
                    format="%.4f",
                    key=f"beta_input_{st.session_state.model_params['beta']['value']}"
                )
                beta_fixed = st.checkbox("Fix", value=st.session_state.model_params['beta']['fixed'],
                                        key=f"beta_fix_{st.session_state.model_params['beta']['fixed']}")
                
                dH_value = st.number_input(
                    "ΔH (kJ/mol)", 
                    value=st.session_state.model_params['dH']['value'],
                    step=1.0, 
                    format="%.2f",
                    key=f"dH_input_{st.session_state.model_params['dH']['value']}"
                )
                dH_fixed = st.checkbox("Fix", value=st.session_state.model_params['dH']['fixed'],
                                      key=f"dH_fix_{st.session_state.model_params['dH']['fixed']}")
            
            with col2:
                dS_value = st.number_input(
                    "ΔS (J/mol·K)", 
                    value=st.session_state.model_params['dS']['value'],
                    step=1.0, 
                    format="%.2f",
                    key=f"dS_input_{st.session_state.model_params['dS']['value']}"
                )
                dS_fixed = st.checkbox("Fix", value=st.session_state.model_params['dS']['fixed'],
                                      key=f"dS_fix_{st.session_state.model_params['dS']['fixed']}")
                
                pH2O_value = st.number_input(
                    "pH₂O", 
                    value=st.session_state.model_params['pH2O']['value'],
                    step=0.001, 
                    format="%.4f",
                    key=f"pH2O_input_{st.session_state.model_params['pH2O']['value']}"
                )
                pH2O_fixed = st.checkbox("Fix", value=st.session_state.model_params['pH2O']['fixed'],
                                        key=f"pH2O_fix_{st.session_state.model_params['pH2O']['fixed']}")
                
                residue_value = st.number_input(
                    "Residue", 
                    value=st.session_state.model_params['residue']['value'],
                    step=0.0001, 
                    format="%.6f",
                    key=f"residue_input_{st.session_state.model_params['residue']['value']}"
                )
                residue_fixed = st.checkbox("Fix", value=st.session_state.model_params['residue']['fixed'],
                                           key=f"residue_fix_{st.session_state.model_params['residue']['fixed']}")
            
            fit_button = st.form_submit_button("🚀 Fit Model and Create Plots", type="primary", use_container_width=True)
            
            if fit_button:
                if st.session_state.experimental_data is None:
                    st.error("Please load data first!")
                else:
                    param_updates = [
                        ('Acc', acc_value, acc_fixed),
                        ('alpha_1e6', alpha_value, alpha_fixed),
                        ('beta', beta_value, beta_fixed),
                        ('dH', dH_value, dH_fixed),
                        ('dS', dS_value, dS_fixed),
                        ('pH2O', pH2O_value, pH2O_fixed),
                        ('residue', residue_value, residue_fixed)
                    ]
                    
                    for param_name, value, is_fixed in param_updates:
                        update_model_param(param_name, value, is_fixed)
                    
                    fixed_params = {}
                    initial_guess = {}
                    
                    param_config = {
                        'Acc': (st.session_state.model_params['Acc']['fixed'], st.session_state.model_params['Acc']['value']),
                        'alpha_1e6': (st.session_state.model_params['alpha_1e6']['fixed'], st.session_state.model_params['alpha_1e6']['value']),
                        'beta': (st.session_state.model_params['beta']['fixed'], st.session_state.model_params['beta']['value']),
                        'dH': (st.session_state.model_params['dH']['fixed'], st.session_state.model_params['dH']['value']),
                        'dS': (st.session_state.model_params['dS']['fixed'], st.session_state.model_params['dS']['value']),
                        'pH2O': (st.session_state.model_params['pH2O']['fixed'], st.session_state.model_params['pH2O']['value']),
                        'residue': (st.session_state.model_params['residue']['fixed'], st.session_state.model_params['residue']['value'])
                    }
                    
                    for name, (is_fixed, value) in param_config.items():
                        fixed_params[name] = value if is_fixed else None
                        initial_guess[name] = value
                        
                        if name == 'Acc':
                            initial_guess[f'{name}_bounds'] = (0.001, 0.9)
                        elif name == 'alpha_1e6':
                            initial_guess[f'{name}_bounds'] = (1.0, 100.0)
                        elif name == 'beta':
                            initial_guess[f'{name}_bounds'] = (0.0001, 0.1)
                        elif name == 'dH':
                            initial_guess[f'{name}_bounds'] = (-200.0, -10.0)
                        elif name == 'dS':
                            initial_guess[f'{name}_bounds'] = (-300.0, -10.0)
                        elif name == 'pH2O':
                            initial_guess[f'{name}_bounds'] = (0.00001, 1.0)
                        elif name == 'residue':
                            initial_guess[f'{name}_bounds'] = (-0.05, 0.05)
                    
                    with st.spinner("Fitting model..."):
                        start_time = time.time()
                        st.session_state.fit_results = fit_model_cached(
                            st.session_state.experimental_data, 
                            fixed_params, 
                            initial_guess
                        )
                        end_time = time.time()
                        
                        if st.session_state.fit_results is not None:
                            st.session_state.fitting_complete = True
                            st.session_state.last_fit_params = {
                                'fixed_params': fixed_params,
                                'initial_guess': initial_guess
                            }
                            
                            for param_name in ['Acc', 'alpha_1e6', 'beta', 'dH', 'dS', 'pH2O', 'residue']:
                                if not st.session_state.model_params[param_name]['fixed']:
                                    fitted_value = st.session_state.fit_results['params'][param_name]
                                    st.session_state.model_params[param_name]['value'] = fitted_value
                            
                            st.success(f"Fitting completed in {end_time - start_time:.2f} seconds")
                            st.rerun()
                        else:
                            st.error("Fitting failed. Please check your parameters.")
        
        st.divider()
        
        # Geometric Model Fitting Section
        if st.session_state.fitting_complete and st.session_state.fit_results is not None:
            st.header("Geometric Model (Stage 2)")
            st.markdown("Fit effective ionic radii r_V and r_OH")
            
            with st.form("geometric_fit_form"):
                col1, col2 = st.columns(2)
                
                with col1:
                    fit_r_V = st.checkbox("Fit r_V", value=True)
                    initial_r_V = st.number_input("Initial r_V (Å)", value=1.18, step=0.01, format="%.3f")
                
                with col2:
                    fit_r_OH = st.checkbox("Fit r_OH", value=True)
                    initial_r_OH = st.number_input("Initial r_OH (Å)", value=1.35, step=0.01, format="%.3f")
                
                col3, col4 = st.columns(2)
                with col3:
                    fit_alpha_geo = st.checkbox("Refit α", value=False)
                    initial_alpha_geo = st.number_input("Initial α (K⁻¹)", 
                                                         value=st.session_state.fit_results['params']['alpha_1e6']/1e6,
                                                         step=1e-6, format="%.2e")
                
                with col4:
                    fit_gamma_geo = st.checkbox("Refit γ", value=False)
                    initial_gamma_geo = st.number_input("Initial γ", 
                                                         value=st.session_state.fit_results['params']['residue'],
                                                         step=0.0001, format="%.5f")
                
                geometric_fit_button = st.form_submit_button("🔬 Fit Geometric Model", type="primary", use_container_width=True)
                
                if geometric_fit_button:
                    x = st.session_state.model_params['Acc']['value']
                    alpha = st.session_state.fit_results['params']['alpha_1e6'] / 1e6
                    gamma = st.session_state.fit_results['params']['residue']
                    T_ref = st.session_state.fit_results['T_start']
                    oh = st.session_state.fit_results['oh_concentration']
                    
                    with st.spinner("Fitting geometric model..."):
                        geometric_results = fit_geometric_model_cached(
                            st.session_state.experimental_data,
                            oh,
                            x,
                            st.session_state.cation_selection['A_cation'],
                            st.session_state.cation_selection['B_cation'],
                            st.session_state.cation_selection['Acc_cation'],
                            st.session_state.ionic_radii_db,
                            alpha, gamma, T_ref,
                            initial_r_V, initial_r_OH,
                            fit_r_V, fit_r_OH,
                            fit_alpha_geo, fit_gamma_geo
                        )
                        
                        if geometric_results is not None:
                            st.session_state.geometric_fit_results = geometric_results
                            st.session_state.geometric_fit_complete = True
                            st.success("Geometric model fitting completed!")
                            st.rerun()
                        else:
                            st.error("Geometric fitting failed. Check parameters.")
        
        st.divider()
        
        # Plot Settings
        st.header("Plot Settings")
        
        with st.expander("Customize Plot Appearance", expanded=False):
            col1, col2 = st.columns(2)
            
            with col1:
                new_point_color = st.color_picker(
                    "Point color", 
                    st.session_state.plot_style['point_color'],
                    key="point_color_picker"
                )
                if new_point_color != st.session_state.plot_style['point_color']:
                    update_plot_style('point_color', new_point_color)
                
                new_point_alpha = st.slider(
                    "Point transparency", 
                    0.1, 1.0, 
                    st.session_state.plot_style['point_alpha'],
                    key="point_alpha_slider"
                )
                if new_point_alpha != st.session_state.plot_style['point_alpha']:
                    update_plot_style('point_alpha', new_point_alpha)
                
                new_model_line_color = st.color_picker(
                    "Model line color", 
                    st.session_state.plot_style['model_line_color'],
                    key="model_line_color_picker"
                )
                if new_model_line_color != st.session_state.plot_style['model_line_color']:
                    update_plot_style('model_line_color', new_model_line_color)
                
                new_thermal_line_color = st.color_picker(
                    "Thermal line color", 
                    st.session_state.plot_style['thermal_line_color'],
                    key="thermal_line_color_picker"
                )
                if new_thermal_line_color != st.session_state.plot_style['thermal_line_color']:
                    update_plot_style('thermal_line_color', new_thermal_line_color)
                
                new_chemical_line_color = st.color_picker(
                    "Chemical line color", 
                    st.session_state.plot_style['chemical_line_color'],
                    key="chemical_line_color_picker"
                )
                if new_chemical_line_color != st.session_state.plot_style['chemical_line_color']:
                    update_plot_style('chemical_line_color', new_chemical_line_color)
            
            with col2:
                new_tec_exp_color = st.color_picker(
                    "Exp. TEC color", 
                    st.session_state.plot_style['tec_exp_color'],
                    key="tec_exp_color_picker"
                )
                if new_tec_exp_color != st.session_state.plot_style['tec_exp_color']:
                    update_plot_style('tec_exp_color', new_tec_exp_color)
                
                new_tec_model_color = st.color_picker(
                    "Model TEC color", 
                    st.session_state.plot_style['tec_model_color'],
                    key="tec_model_color_picker"
                )
                if new_tec_model_color != st.session_state.plot_style['tec_model_color']:
                    update_plot_style('tec_model_color', new_tec_model_color)
                
                new_oh_color = st.color_picker(
                    "[OH] line color", 
                    st.session_state.plot_style['oh_color'],
                    key="oh_color_picker"
                )
                if new_oh_color != st.session_state.plot_style['oh_color']:
                    update_plot_style('oh_color', new_oh_color)
                
                new_bar_thermal_color = st.color_picker(
                    "Thermal bar color", 
                    st.session_state.plot_style['bar_thermal_color'],
                    key="bar_thermal_color_picker"
                )
                if new_bar_thermal_color != st.session_state.plot_style['bar_thermal_color']:
                    update_plot_style('bar_thermal_color', new_bar_thermal_color)
                
                new_bar_chemical_color = st.color_picker(
                    "Chemical bar color", 
                    st.session_state.plot_style['bar_chemical_color'],
                    key="bar_chemical_color_picker"
                )
                if new_bar_chemical_color != st.session_state.plot_style['bar_chemical_color']:
                    update_plot_style('bar_chemical_color', new_bar_chemical_color)
            
            col1, col2 = st.columns(2)
            with col1:
                new_point_size = st.slider(
                    "Point size", 
                    10, 100, 
                    st.session_state.plot_style['point_size'],
                    step=1,
                    key="point_size_slider"
                )
                if new_point_size != st.session_state.plot_style['point_size']:
                    update_plot_style('point_size', new_point_size)
                
                new_line_width = st.slider(
                    "Line width", 
                    1.0, 5.0, 
                    float(st.session_state.plot_style['line_width']),
                    step=0.1,
                    key="line_width_slider"
                )
                if new_line_width != st.session_state.plot_style['line_width']:
                    update_plot_style('line_width', new_line_width)
            
            with col2:
                cmap_options = ['viridis', 'plasma', 'inferno', 'magma', 'cividis', 
                               'coolwarm', 'RdYlBu', 'Spectral', 'rainbow', 'jet']
                new_cmap_style = st.selectbox(
                    "Temperature colormap", 
                    cmap_options, 
                    index=cmap_options.index(st.session_state.plot_style['cmap_style']),
                    key="cmap_select"
                )
                if new_cmap_style != st.session_state.plot_style['cmap_style']:
                    update_plot_style('cmap_style', new_cmap_style)
    
    # ============================================
    # ОСНОВНОЙ КОНТЕНТ
    # ============================================
    
    if st.session_state.experimental_data is not None:
        st.header("Loaded Data Preview")
        
        df = pd.DataFrame(st.session_state.experimental_data, columns=['Temperature (°C)', 'ΔL/L₀'])
        st.dataframe(df, use_container_width=True)
        
        fig_raw, ax_raw = plt.subplots(figsize=(6, 3))
        ax_raw.scatter(df['Temperature (°C)'], df['ΔL/L₀'], s=40, 
                      color=st.session_state.plot_style['point_color'], 
                      edgecolor='none',
                      alpha=st.session_state.plot_style['point_alpha'])
        ax_raw.set_xlabel('Temperature (°C)', fontweight='bold')
        ax_raw.set_ylabel('ΔL/L₀', fontweight='bold')
        ax_raw.set_title('Raw Experimental Data', fontweight='bold')
        ax_raw.grid(True, alpha=0.3)
        fig_raw.set_dpi(600)
        st.pyplot(fig_raw)
    
    if st.session_state.fit_results is not None and st.session_state.fitting_complete:
        st.header("Fitting Results (Stage 1)")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("MSE", f"{st.session_state.fit_results['mse']:.3e}")
        with col2:
            st.metric("RMSE", f"{st.session_state.fit_results['rmse']:.3e}")
        with col3:
            st.metric("R²", f"{st.session_state.fit_results['r2']:.6f}")
        with col4:
            chi2_value = st.session_state.fit_results['chi2']
            chi2_label = "χ²_red" if not np.isnan(chi2_value) else "χ²"
            st.metric(chi2_label, f"{chi2_value:.6f}")
        
        with st.expander("📊 Metric Explanations (for scientific paper)"):
            explanations = get_metric_explanation()
            for metric, info in explanations.items():
                st.markdown(f"**{info['title']}**")
                st.latex(info['formula'])
                st.markdown(f"*{info['explanation']}*")
                st.markdown(f"**Units:** {info['units']}")
                st.markdown("---")
        
        st.subheader("Model Parameters")
        
        params_data = []
        for param_name in ['Acc', 'alpha_1e6', 'beta', 'dH', 'dS', 'pH2O', 'residue']:
            if st.session_state.model_params[param_name]['fixed']:
                value = st.session_state.model_params[param_name]['value']
                status = "Fixed"
            else:
                value = st.session_state.fit_results['params'][param_name]
                status = "Fitted"
            
            display_name = param_name
            if param_name == 'alpha_1e6':
                display_name = 'α·10⁶'
            elif param_name == 'dH':
                display_name = 'ΔH (kJ/mol)'
            elif param_name == 'dS':
                display_name = 'ΔS (J/mol·K)'
            elif param_name == 'pH2O':
                display_name = 'pH₂O'
            
            params_data.append({
                "Parameter": display_name,
                "Value": value,
                "Status": status
            })
        
        params_df = pd.DataFrame(params_data)
        st.dataframe(params_df.style.format({"Value": "{:.6f}"}), use_container_width=True)
        
        st.divider()
        st.header("Plots (Stage 1)")
        
        tab1, tab2, tab3, tab4 = st.tabs([
            "📈 Basic Plots", 
            "🔍 Advanced Analysis", 
            "📊 Statistical Analysis", 
            "🧪 Model Insights"
        ])
        
        with tab1:
            st.subheader("Basic Analysis Plots")
            
            plot1 = create_plot1_cached(st.session_state.fit_results, st.session_state.plot_style)
            plot2 = create_plot2_cached(st.session_state.fit_results, st.session_state.plot_style)
            plot3 = create_plot3_cached(st.session_state.fit_results, st.session_state.plot_style)
            plot4 = create_plot4_cached(st.session_state.fit_results, st.session_state.plot_style)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Experimental Data and Model**")
                st.pyplot(plot1)
                
                st.markdown("**Histograms of Changes**")
                st.pyplot(plot3)
            
            with col2:
                st.markdown("**Model Contributions**")
                st.pyplot(plot2)
                
                st.markdown("**TEC and Proton Concentration**")
                st.pyplot(plot4)
        
        with tab2:
            st.subheader("Advanced Analysis Plots")
            
            plot5 = create_plot5_cached(st.session_state.fit_results, st.session_state.plot_style)
            plot6 = create_plot6_cached(st.session_state.fit_results, st.session_state.plot_style)
            plot9 = create_plot9_cached(st.session_state.fit_results, st.session_state.plot_style)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**ΔL/L₀: Model vs Experiment**")
                st.pyplot(plot5)
                
                st.markdown("**TEC: Model vs Experiment**")
                st.pyplot(plot6)
            
            with col2:
                st.markdown("**Phase Portrait Analysis**")
                st.pyplot(plot9)
                
                st.info("""
                **Phase Portrait Insights:**
                - **Left plot**: Shows how the rate of expansion (velocity) changes with expansion itself (position)
                - **Right plot**: Shows acceleration (rate of change of velocity) vs velocity
                - The model trajectory (black line) shows the predicted path through phase space
                - Deviations from the trajectory indicate regions where the model doesn't fully capture the dynamics
                """)
        
        with tab3:
            st.subheader("Statistical Analysis Plots")
            
            plot10 = create_plot10_cached(st.session_state.fit_results, st.session_state.plot_style)
            plot12 = create_plot12_cached(st.session_state.fit_results, st.session_state.plot_style)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Residual Analysis**")
                st.pyplot(plot10)
                
                st.info("""
                **Statistical Diagnostics:**
                - **Top-left**: Histogram of residuals with normal distribution fit
                - **Top-right**: Q-Q plot for normality testing (points should follow red line)
                - **Bottom-left**: Autocorrelation of residuals (should be within blue bands for independence)
                - **Bottom-right**: Residuals vs fitted values (should be randomly scattered)
                """)
            
            with col2:
                st.markdown("**Parameter Correlations**")
                st.pyplot(plot12)
                
                st.info("""
                **Correlation Matrix Insights:**
                - Shows correlations between fitted parameters
                - Values close to ±1 indicate strong (anti-)correlation
                - High correlations may indicate identifiability issues
                - Ideally, parameters should be relatively independent (values near 0)
                """)
        
        with tab4:
            st.subheader("Model Insights Plots")
            
            plot7 = create_plot7_cached(st.session_state.fit_results, st.session_state.plot_style)
            plot8 = create_plot8_cached(st.session_state.fit_results, st.session_state.plot_style)
            plot11 = create_plot11_cached(st.session_state.fit_results, st.session_state.plot_style)
            
            st.markdown("**Contribution Analysis**")
            st.pyplot(plot7)
            
            st.markdown("**Parameter Sensitivity Analysis**")
            st.pyplot(plot8)
            
            st.markdown("**OH Concentration Dynamics**")
            st.pyplot(plot11)
            
            st.info("""
            **Insights from these plots:**
            
            **1. Contribution Analysis (top):**
            - Shows the absolute and relative contributions of thermal vs chemical effects
            - Reveals at which temperatures each mechanism dominates
            
            **2. Parameter Sensitivity (middle):**
            - Shows how small changes in parameters affect the model predictions
            - Helps identify which parameters are most critical for accurate predictions
            
            **3. OH Concentration Dynamics (bottom):**
            - Shows how OH concentration and its derivatives change with temperature
            - Inflection points indicate changes in reaction kinetics
            """)
        
        # Geometric Model Results
        if st.session_state.geometric_fit_complete and st.session_state.geometric_fit_results is not None:
            st.divider()
            st.header("Geometric Model Results (Stage 2)")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("r_V (Å)", f"{st.session_state.geometric_fit_results['params']['r_V']:.4f}")
            with col2:
                st.metric("r_OH (Å)", f"{st.session_state.geometric_fit_results['params']['r_OH']:.4f}")
            with col3:
                st.metric("R² (Geometric)", f"{st.session_state.geometric_fit_results['r2']:.6f}")
            
            st.subheader("Geometric Model Plots")
            
            tab5, tab6 = st.tabs(["📐 Geometric Analysis", "📊 Geometric Statistics"])
            
            with tab5:
                col1, col2 = st.columns(2)
                
                with col1:
                    plot13 = create_plot13_cached(st.session_state.geometric_fit_results, st.session_state.plot_style)
                    st.markdown("**Coordination Numbers Evolution**")
                    st.pyplot(plot13)
                    
                    plot15 = create_plot15_cached(st.session_state.geometric_fit_results, st.session_state.plot_style)
                    st.markdown("**Sublattice Contributions**")
                    st.pyplot(plot15)
                
                with col2:
                    plot14 = create_plot14_cached(st.session_state.geometric_fit_results, st.session_state.plot_style)
                    st.markdown("**Geometric Model Fit**")
                    st.pyplot(plot14)
                    
                    plot16 = create_plot16_cached(st.session_state.geometric_fit_results, st.session_state.plot_style)
                    st.markdown("**Fitted Effective Radii**")
                    st.pyplot(plot16)
            
            with tab6:
                col1, col2 = st.columns(2)
                
                with col1:
                    plot17 = create_plot17_cached(st.session_state.geometric_fit_results, st.session_state.plot_style)
                    st.markdown("**Chemical Expansion Nonlinearity**")
                    st.pyplot(plot17)
                    
                    plot18 = create_plot18_cached(st.session_state.geometric_fit_results, st.session_state.plot_style)
                    st.markdown("**Arrhenius Plot for Hydration**")
                    st.pyplot(plot18)
                
                with col2:
                    plot19 = create_plot19_cached(st.session_state.geometric_fit_results, st.session_state.plot_style)
                    st.markdown("**Geometric Model Residuals**")
                    st.pyplot(plot19)
        
        # Download section
        st.divider()
        st.subheader("Export Results")
        
        download_tab1, download_tab2, download_tab3, download_tab4 = st.tabs([
            "📥 Basic Plots", 
            "📥 Advanced Plots", 
            "📥 Statistical Plots", 
            "📥 Insight Plots"
        ])
        
        with download_tab1:
            col1, col2 = st.columns(2)
            with col1:
                if st.button("📥 Download Plot 1 (Data & Model)", key="dl_plot1"):
                    buf = io.BytesIO()
                    plot1.savefig(buf, format='png', dpi=600)
                    st.download_button(
                        label="Download PNG (600 DPI)",
                        data=buf.getvalue(),
                        file_name="plot1_data_and_model.png",
                        mime="image/png",
                        key="dl_btn_plot1"
                    )
                
                if st.button("📥 Download Plot 2 (Contributions)", key="dl_plot2"):
                    buf = io.BytesIO()
                    plot2.savefig(buf, format='png', dpi=600)
                    st.download_button(
                        label="Download PNG (600 DPI)",
                        data=buf.getvalue(),
                        file_name="plot2_contributions.png",
                        mime="image/png",
                        key="dl_btn_plot2"
                    )
            
            with col2:
                if st.button("📥 Download Plot 3 (Histograms)", key="dl_plot3"):
                    buf = io.BytesIO()
                    plot3.savefig(buf, format='png', dpi=600)
                    st.download_button(
                        label="Download PNG (600 DPI)",
                        data=buf.getvalue(),
                        file_name="plot3_histograms.png",
                        mime="image/png",
                        key="dl_btn_plot3"
                    )
                
                if st.button("📥 Download Plot 4 (TEC & [OH])", key="dl_plot4"):
                    buf = io.BytesIO()
                    plot4.savefig(buf, format='png', dpi=600)
                    st.download_button(
                        label="Download PNG (600 DPI)",
                        data=buf.getvalue(),
                        file_name="plot4_tec_and_oh.png",
                        mime="image/png",
                        key="dl_btn_plot4"
                    )
        
        with download_tab2:
            col1, col2 = st.columns(2)
            with col1:
                if st.button("📥 Download Plot 5 (ΔL/L₀ Correlation)", key="dl_plot5"):
                    buf = io.BytesIO()
                    plot5.savefig(buf, format='png', dpi=600)
                    st.download_button(
                        label="Download PNG (600 DPI)",
                        data=buf.getvalue(),
                        file_name="plot5_dl_correlation.png",
                        mime="image/png",
                        key="dl_btn_plot5"
                    )
                
                if st.button("📥 Download Plot 6 (TEC Correlation)", key="dl_plot6"):
                    buf = io.BytesIO()
                    plot6.savefig(buf, format='png', dpi=600)
                    st.download_button(
                        label="Download PNG (600 DPI)",
                        data=buf.getvalue(),
                        file_name="plot6_tec_correlation.png",
                        mime="image/png",
                        key="dl_btn_plot6"
                    )
            
            with col2:
                if st.button("📥 Download Plot 9 (Phase Portrait)", key="dl_plot9"):
                    buf = io.BytesIO()
                    plot9.savefig(buf, format='png', dpi=600)
                    st.download_button(
                        label="Download PNG (600 DPI)",
                        data=buf.getvalue(),
                        file_name="plot9_phase_portrait.png",
                        mime="image/png",
                        key="dl_btn_plot9"
                    )
        
        with download_tab3:
            col1, col2 = st.columns(2)
            with col1:
                if st.button("📥 Download Plot 10 (Residual Analysis)", key="dl_plot10"):
                    buf = io.BytesIO()
                    plot10.savefig(buf, format='png', dpi=600)
                    st.download_button(
                        label="Download PNG (600 DPI)",
                        data=buf.getvalue(),
                        file_name="plot10_residual_analysis.png",
                        mime="image/png",
                        key="dl_btn_plot10"
                    )
            
            with col2:
                if st.button("📥 Download Plot 12 (Correlation Matrix)", key="dl_plot12"):
                    buf = io.BytesIO()
                    plot12.savefig(buf, format='png', dpi=600)
                    st.download_button(
                        label="Download PNG (600 DPI)",
                        data=buf.getvalue(),
                        file_name="plot12_correlation_matrix.png",
                        mime="image/png",
                        key="dl_btn_plot12"
                    )
        
        with download_tab4:
            col1, col2 = st.columns(2)
            with col1:
                if st.button("📥 Download Plot 7 (Contribution Analysis)", key="dl_plot7"):
                    buf = io.BytesIO()
                    plot7.savefig(buf, format='png', dpi=600)
                    st.download_button(
                        label="Download PNG (600 DPI)",
                        data=buf.getvalue(),
                        file_name="plot7_contribution_analysis.png",
                        mime="image/png",
                        key="dl_btn_plot7"
                    )
                
                if st.button("📥 Download Plot 8 (Sensitivity Analysis)", key="dl_plot8"):
                    buf = io.BytesIO()
                    plot8.savefig(buf, format='png', dpi=600)
                    st.download_button(
                        label="Download PNG (600 DPI)",
                        data=buf.getvalue(),
                        file_name="plot8_sensitivity_analysis.png",
                        mime="image/png",
                        key="dl_btn_plot8"
                    )
            
            with col2:
                if st.button("📥 Download Plot 11 (OH Dynamics)", key="dl_plot11"):
                    buf = io.BytesIO()
                    plot11.savefig(buf, format='png', dpi=600)
                    st.download_button(
                        label="Download PNG (600 DPI)",
                        data=buf.getvalue(),
                        file_name="plot11_oh_dynamics.png",
                        mime="image/png",
                        key="dl_btn_plot11"
                    )
        
        # Download geometric plots if available
        if st.session_state.geometric_fit_complete:
            st.divider()
            st.subheader("Export Geometric Model Results")
            
            geo_tab1, geo_tab2 = st.tabs(["📐 Geometric Plots", "📊 Geometric Statistics"])
            
            with geo_tab1:
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("📥 Download Plot 13 (CN Evolution)", key="dl_plot13"):
                        buf = io.BytesIO()
                        plot13.savefig(buf, format='png', dpi=600)
                        st.download_button(
                            label="Download PNG (600 DPI)",
                            data=buf.getvalue(),
                            file_name="plot13_cn_evolution.png",
                            mime="image/png",
                            key="dl_btn_plot13"
                        )
                    
                    if st.button("📥 Download Plot 15 (Sublattice Contributions)", key="dl_plot15"):
                        buf = io.BytesIO()
                        plot15.savefig(buf, format='png', dpi=600)
                        st.download_button(
                            label="Download PNG (600 DPI)",
                            data=buf.getvalue(),
                            file_name="plot15_sublattice_contributions.png",
                            mime="image/png",
                            key="dl_btn_plot15"
                        )
                
                with col2:
                    if st.button("📥 Download Plot 14 (Geometric Fit)", key="dl_plot14"):
                        buf = io.BytesIO()
                        plot14.savefig(buf, format='png', dpi=600)
                        st.download_button(
                            label="Download PNG (600 DPI)",
                            data=buf.getvalue(),
                            file_name="plot14_geometric_fit.png",
                            mime="image/png",
                            key="dl_btn_plot14"
                        )
                    
                    if st.button("📥 Download Plot 16 (Fitted Radii)", key="dl_plot16"):
                        buf = io.BytesIO()
                        plot16.savefig(buf, format='png', dpi=600)
                        st.download_button(
                            label="Download PNG (600 DPI)",
                            data=buf.getvalue(),
                            file_name="plot16_fitted_radii.png",
                            mime="image/png",
                            key="dl_btn_plot16"
                        )
            
            with geo_tab2:
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("📥 Download Plot 17 (Chem Expansion)", key="dl_plot17"):
                        buf = io.BytesIO()
                        plot17.savefig(buf, format='png', dpi=600)
                        st.download_button(
                            label="Download PNG (600 DPI)",
                            data=buf.getvalue(),
                            file_name="plot17_chem_expansion.png",
                            mime="image/png",
                            key="dl_btn_plot17"
                        )
                    
                    if st.button("📥 Download Plot 18 (Arrhenius)", key="dl_plot18"):
                        buf = io.BytesIO()
                        plot18.savefig(buf, format='png', dpi=600)
                        st.download_button(
                            label="Download PNG (600 DPI)",
                            data=buf.getvalue(),
                            file_name="plot18_arrhenius.png",
                            mime="image/png",
                            key="dl_btn_plot18"
                        )
                
                with col2:
                    if st.button("📥 Download Plot 19 (Geo Residuals)", key="dl_plot19"):
                        buf = io.BytesIO()
                        plot19.savefig(buf, format='png', dpi=600)
                        st.download_button(
                            label="Download PNG (600 DPI)",
                            data=buf.getvalue(),
                            file_name="plot19_geometric_residuals.png",
                            mime="image/png",
                            key="dl_btn_plot19"
                        )
        
        # Download data
        st.divider()
        st.subheader("Download Data")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("📊 Download Fitted Data (CSV)", key="dl_csv"):
                fitted_df = pd.DataFrame({
                    'Temperature_C': st.session_state.fit_results['T_data'],
                    'DeltaL_L0_exp': st.session_state.fit_results['dl_data'],
                    'DeltaL_L0_model': st.session_state.fit_results['dl_model'],
                    'Residual': st.session_state.fit_results['residuals'],
                    'TEC_exp_1e6K-1': st.session_state.fit_results['tec_exp'] * 1e6,
                    'TEC_model_1e6K-1': st.session_state.fit_results['tec_model'] * 1e6,
                    'OH_concentration': st.session_state.fit_results['oh_concentration'],
                    'Thermal_contribution': st.session_state.fit_results['thermal_contrib'],
                    'Chemical_contribution': st.session_state.fit_results['chem_contrib']
                })
                csv = fitted_df.to_csv(index=False)
                st.download_button(
                    label="Download CSV",
                    data=csv,
                    file_name="fitted_data.csv",
                    mime="text/csv",
                    key="dl_btn_csv"
                )
        
        with col2:
            if st.button("⚙️ Download Parameters (TXT)", key="dl_params"):
                params_text = f"""FITTING RESULTS (Stage 1)
================
MSE: {st.session_state.fit_results['mse']:.6e}
RMSE: {st.session_state.fit_results['rmse']:.6e}
R²: {st.session_state.fit_results['r2']:.6f}
χ²: {st.session_state.fit_results['chi2']:.6f}
N points: {st.session_state.fit_results['N_points']}
Fitted parameters: {st.session_state.fit_results['n_free_params']}

MODEL PARAMETERS (Stage 1)
================
[Acc] = {st.session_state.fit_results['params']['Acc']:.6f}
α·10⁶ = {st.session_state.fit_results['params']['alpha_1e6']:.6f}
β = {st.session_state.fit_results['params']['beta']:.6f}
ΔH = {st.session_state.fit_results['params']['dH']:.6f} kJ/mol
ΔS = {st.session_state.fit_results['params']['dS']:.6f} J/mol·K
pH₂O = {st.session_state.fit_results['params']['pH2O']:.6f}
Residue = {st.session_state.fit_results['params']['residue']:.6f}

CATION COMPOSITION
================
A-cation: {st.session_state.cation_selection['A_cation']}
B-cation: {st.session_state.cation_selection['B_cation']}
Acceptor: {st.session_state.cation_selection['Acc_cation']}
"""
                
                if st.session_state.geometric_fit_complete:
                    params_text += f"""
GEOMETRIC MODEL RESULTS (Stage 2)
================
r_V = {st.session_state.geometric_fit_results['params']['r_V']:.6f} Å
r_OH = {st.session_state.geometric_fit_results['params']['r_OH']:.6f} Å
R² (Geometric) = {st.session_state.geometric_fit_results['r2']:.6f}
MSE (Geometric) = {st.session_state.geometric_fit_results['mse']:.6e}
Fitted parameters: {', '.join(st.session_state.geometric_fit_results['vary_params'])}
"""
                
                st.download_button(
                    label="Download Parameters",
                    data=params_text,
                    file_name="model_parameters.txt",
                    mime="text/plain",
                    key="dl_btn_params"
                )
    
    else:
        st.info("👈 Please load data and configure parameters in the sidebar, then click 'Fit Model and Create Plots'")
    
    st.divider()
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center'>
            <p>Thermo-Mechanical Expansion Modeling Tool | For scientific publications | 600 DPI export</p>
            <p>Includes 19 comprehensive plots for detailed analysis (12 standard + 7 geometric)</p>
        </div>
        """,
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()
