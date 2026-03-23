import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import pandas as pd
import io
import warnings
import matplotlib.cm as cm
from matplotlib.colors import Normalize
from typing import Dict, Any, Optional, Tuple, List
import time
from scipy.stats import norm, probplot
from statsmodels.graphics.tsaplots import plot_acf
import warnings
warnings.filterwarnings('ignore')

# ============================================
# МОДУЛЬ ИОННЫХ РАДИУСОВ (НОВЫЙ)
# ============================================

class IonRadiusDatabase:
    """Database of Shannon ionic radii with interpolation capabilities"""
    
    def __init__(self):
        """Initialize database with Shannon radii data"""
        self.radii_data = {}
        self._load_radii_data()
    
    def _load_radii_data(self):
        """Load Shannon radii data from embedded table"""
        
        # Data structure: {ion: {charge: {CN: (radius_crystal, radius_ionic)}}}
        
        # A-site cations
        self.radii_data['Ca'] = {
            2: {6: (1.14, 1.00), 7: (1.20, 1.06), 8: (1.26, 1.12), 
                9: (1.32, 1.18), 10: (1.37, 1.23), 12: (1.48, 1.34)}
        }
        self.radii_data['Sr'] = {
            2: {6: (1.32, 1.18), 7: (1.35, 1.21), 8: (1.40, 1.26), 
                9: (1.45, 1.31), 10: (1.50, 1.36), 12: (1.58, 1.44)}
        }
        self.radii_data['Ba'] = {
            2: {6: (1.49, 1.35), 7: (1.52, 1.38), 8: (1.56, 1.42), 
                9: (1.61, 1.47), 10: (1.66, 1.52), 11: (1.71, 1.57), 
                12: (1.75, 1.61)}
        }
        self.radii_data['La'] = {
            3: {6: (1.172, 1.032), 7: (1.24, 1.10), 8: (1.30, 1.16), 
                9: (1.356, 1.216), 10: (1.41, 1.27), 12: (1.50, 1.36)}
        }
        
        # B-site cations
        self.radii_data['Ce'] = {
            3: {6: (1.15, 1.01), 7: (1.21, 1.07), 8: (1.283, 1.143), 
                9: (1.336, 1.196), 10: (1.39, 1.25), 12: (1.48, 1.34)},
            4: {6: (1.01, 0.87), 8: (1.11, 0.97), 10: (1.21, 1.07), 12: (1.28, 1.14)}
        }
        self.radii_data['Zr'] = {
            4: {4: (0.73, 0.59), 5: (0.80, 0.66), 6: (0.86, 0.72), 
                7: (0.92, 0.78), 8: (0.98, 0.84), 9: (1.03, 0.89)}
        }
        self.radii_data['Ti'] = {
            4: {4: (0.56, 0.42), 5: (0.65, 0.51), 6: (0.745, 0.605), 8: (0.88, 0.74)}
        }
        self.radii_data['Hf'] = {
            4: {4: (0.72, 0.58), 6: (0.85, 0.71), 7: (0.90, 0.76), 8: (0.97, 0.83)}
        }
        self.radii_data['Sn'] = {
            4: {4: (0.69, 0.55), 5: (0.76, 0.62), 6: (0.83, 0.69), 
                7: (0.89, 0.75), 8: (0.95, 0.81)}
        }
        self.radii_data['In'] = {
            3: {4: (0.76, 0.62), 6: (0.94, 0.80), 8: (1.06, 0.92)}
        }
        self.radii_data['Sc'] = {
            3: {6: (0.885, 0.745), 8: (1.01, 0.87)}
        }
        self.radii_data['Y'] = {
            3: {6: (1.04, 0.90), 7: (1.10, 0.96), 8: (1.159, 1.019), 9: (1.215, 1.075)}
        }
        self.radii_data['Yb'] = {
            3: {6: (1.008, 0.868), 7: (1.065, 0.925), 8: (1.125, 0.985), 9: (1.182, 1.042)}
        }
        
        # Acceptor cations
        self.radii_data['Al'] = {3: {4: (0.53, 0.39), 5: (0.62, 0.48), 6: (0.675, 0.535)}}
        self.radii_data['Ga'] = {3: {4: (0.61, 0.47), 5: (0.69, 0.55), 6: (0.76, 0.62)}}
        self.radii_data['Ho'] = {3: {6: (1.041, 0.901), 8: (1.155, 1.015), 9: (1.212, 1.072), 10: (1.26, 1.12)}}
        self.radii_data['Dy'] = {3: {6: (1.052, 0.912), 7: (1.11, 0.97), 8: (1.167, 1.027), 9: (1.223, 1.083)}}
        self.radii_data['Gd'] = {3: {6: (1.078, 0.938), 7: (1.14, 1.00), 8: (1.193, 1.053), 9: (1.247, 1.107)}}
        self.radii_data['Sm'] = {3: {6: (1.098, 0.958), 7: (1.16, 1.02), 8: (1.219, 1.079), 9: (1.272, 1.132), 12: (1.38, 1.24)}}
        self.radii_data['Nd'] = {3: {6: (1.123, 0.983), 8: (1.249, 1.109), 9: (1.303, 1.163), 12: (1.41, 1.27)}}
        self.radii_data['Mg'] = {2: {4: (0.71, 0.57), 5: (0.80, 0.66), 6: (0.86, 0.72), 8: (1.03, 0.89)}}
        self.radii_data['Zn'] = {2: {4: (0.74, 0.60), 5: (0.82, 0.68), 6: (0.88, 0.74), 8: (1.04, 0.90)}}
        
        # Anions
        self.radii_data['O'] = {-2: {2: (1.21, 1.35), 3: (1.22, 1.36), 4: (1.24, 1.38), 6: (1.26, 1.40), 8: (1.28, 1.42)}}
        self.radii_data['OH'] = {-1: {2: (1.18, 1.32), 3: (1.20, 1.34), 4: (1.21, 1.35), 6: (1.23, 1.37)}}
    
    def get_radius(self, ion: str, charge: int, cn: float, use_ionic: bool = True) -> float:
        """
        Get radius for ion at given coordination number with interpolation
        
        Args:
            ion: Element symbol
            charge: Ionic charge (positive for cations, negative for anions)
            cn: Coordination number (can be non-integer)
            use_ionic: True for ionic radius, False for crystal radius
        
        Returns:
            Radius in Angstroms
        """
        if ion not in self.radii_data:
            raise ValueError(f"Ion {ion} not found in database")
        
        if charge not in self.radii_data[ion]:
            raise ValueError(f"Charge {charge} for {ion} not found in database")
        
        cn_data = self.radii_data[ion][charge]
        cn_values = sorted(cn_data.keys())
        
        # Exact match
        if cn in cn_values:
            radius = cn_data[cn][1 if use_ionic else 0]
            return radius
        
        # Interpolation or extrapolation
        if cn < cn_values[0]:
            # Extrapolate using lowest CN
            r1 = cn_data[cn_values[0]][1 if use_ionic else 0]
            r2 = cn_data[cn_values[1]][1 if use_ionic else 0]
            slope = (r2 - r1) / (cn_values[1] - cn_values[0])
            radius = r1 + slope * (cn - cn_values[0])
        elif cn > cn_values[-1]:
            # Extrapolate using highest CN
            r1 = cn_data[cn_values[-2]][1 if use_ionic else 0]
            r2 = cn_data[cn_values[-1]][1 if use_ionic else 0]
            slope = (r2 - r1) / (cn_values[-1] - cn_values[-2])
            radius = r2 + slope * (cn - cn_values[-1])
        else:
            # Linear interpolation between nearest CNs
            lower_cn = max([c for c in cn_values if c <= cn])
            upper_cn = min([c for c in cn_values if c >= cn])
            
            r_lower = cn_data[lower_cn][1 if use_ionic else 0]
            r_upper = cn_data[upper_cn][1 if use_ionic else 0]
            
            if upper_cn == lower_cn:
                radius = r_lower
            else:
                radius = r_lower + (r_upper - r_lower) * (cn - lower_cn) / (upper_cn - lower_cn)
        
        return radius
    
    def get_available_ions(self) -> Dict[str, List[int]]:
        """Get list of available ions and their charges"""
        ions = {}
        for ion, charges in self.radii_data.items():
            ions[ion] = list(charges.keys())
        return ions


# Initialize global radius database
@st.cache_resource
def get_radius_database():
    return IonRadiusDatabase()


# ============================================
# ГЕОМЕТРИЧЕСКАЯ МОДЕЛЬ (НОВАЯ)
# ============================================

def calculate_coordination_numbers(Acc: float, oh: float) -> Tuple[float, float]:
    """
    Calculate coordination numbers for A and B cations
    
    According to the task:
    CN_B = 6 - [Acc] + [OH]
    CN_A = 12 - 2[Acc] + 2[OH]
    
    Args:
        Acc: Acceptor concentration (x)
        oh: Proton concentration [OH]
    
    Returns:
        (CN_B, CN_A)
    """
    cn_b = 6 - Acc + oh
    cn_a = 12 - 2 * Acc + 2 * oh
    return cn_b, cn_a


def calculate_average_b_radius(ion_B: str, ion_Acc: str, charge_B: int, charge_Acc: int,
                               Acc: float, cn_b: float, radius_db: IonRadiusDatabase,
                               use_ionic: bool = True) -> float:
    """
    Calculate average radius of B-site sublattice
    
    Args:
        ion_B: Host B-site cation (e.g., 'Zr')
        ion_Acc: Acceptor cation (e.g., 'Y')
        charge_B: Charge of host B-site cation
        charge_Acc: Charge of acceptor cation
        Acc: Acceptor concentration
        cn_b: Coordination number of B-site
        radius_db: IonRadiusDatabase instance
        use_ionic: Use ionic radius (True) or crystal radius (False)
    
    Returns:
        Average B-site radius in Angstroms
    """
    r_B_host = radius_db.get_radius(ion_B, charge_B, cn_b, use_ionic)
    r_Acc = radius_db.get_radius(ion_Acc, charge_Acc, cn_b, use_ionic)
    
    # Weighted average
    r_avg = (1 - Acc) * r_B_host + Acc * r_Acc
    return r_avg


def calculate_average_a_radius(ion_A: str, charge_A: int, cn_a: float,
                               radius_db: IonRadiusDatabase,
                               use_ionic: bool = True) -> float:
    """
    Calculate radius of A-site cation
    
    Args:
        ion_A: A-site cation (e.g., 'Ba')
        charge_A: Charge of A-site cation
        cn_a: Coordination number of A-site
        radius_db: IonRadiusDatabase instance
        use_ionic: Use ionic radius (True) or crystal radius (False)
    
    Returns:
        A-site radius in Angstroms
    """
    return radius_db.get_radius(ion_A, charge_A, cn_a, use_ionic)


def calculate_anion_radius(oh: float, Acc: float, r_V: float, r_OH: float,
                           r_O: float = 1.4, use_ionic: bool = True) -> float:
    """
    Calculate average anion sublattice radius
    
    Using: [O] + [V] + [OH] = 3
    [V] = (Acc - oh) / 2
    [O] = 3 - [V] - oh
    
    Args:
        oh: Proton concentration [OH]
        Acc: Acceptor concentration
        r_V: Effective radius of oxygen vacancy (fitted)
        r_OH: Effective radius of OH group (fitted or fixed)
        r_O: Radius of O2- ion (fixed, typically 1.4 Å)
        use_ionic: Not used here, kept for consistency
    
    Returns:
        Average anion radius in Angstroms
    """
    v = (Acc - oh) / 2
    o = 3 - v - oh
    
    # Weighted average
    r_avg = (o * r_O + v * r_V + oh * r_OH) / 3
    return r_avg


def calculate_lattice_parameter_geometric(r_B_avg: float, r_anion_avg: float,
                                          r_A_avg: Optional[float] = None,
                                          model_type: str = 'B_only') -> float:
    """
    Calculate lattice parameter from ionic radii
    
    For cubic perovskite:
    - model_type = 'B_only': a = 2*(r_B + r_anion)
    - model_type = 'A_only': a = 2*(r_A + r_anion)
    - model_type = 'full': a = 2/√2*(r_A + r_B + r_anion)  # more accurate
    
    Args:
        r_B_avg: Average B-site radius
        r_anion_avg: Average anion radius
        r_A_avg: Average A-site radius (optional)
        model_type: Type of geometric model
    
    Returns:
        Lattice parameter in Angstroms
    """
    if model_type == 'B_only':
        return 2 * (r_B_avg + r_anion_avg)
    elif model_type == 'A_only':
        return 2 * (r_A_avg + r_anion_avg)
    elif model_type == 'full':
        if r_A_avg is None:
            raise ValueError("r_A_avg required for 'full' model")
        return 2 / np.sqrt(2) * (r_A_avg + r_B_avg + r_anion_avg)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")


def calculate_geometric_expansion(T: np.ndarray, Acc: float, oh: np.ndarray,
                                  ion_A: str, ion_B: str, ion_Acc: str,
                                  charge_A: int, charge_B: int, charge_Acc: int,
                                  r_V: float, r_OH: float, r_O: float = 1.4,
                                  a0: Optional[float] = None,
                                  model_type: str = 'B_only',
                                  radius_db: Optional[IonRadiusDatabase] = None) -> Tuple[np.ndarray, float, np.ndarray, np.ndarray]:
    """
    Calculate relative expansion using geometric model
    
    Args:
        T: Temperature array
        Acc: Acceptor concentration
        oh: Proton concentration array
        ion_A, ion_B, ion_Acc: Ion symbols
        charge_A, charge_B, charge_Acc: Ionic charges
        r_V, r_OH: Effective radii for vacancies and OH groups
        r_O: Oxygen ion radius
        a0: Reference lattice parameter (if None, use first point)
        model_type: Geometric model type
        radius_db: IonRadiusDatabase instance
    
    Returns:
        (deltaL_L0, a0_reference, r_B_avg_array, r_anion_avg_array)
    """
    if radius_db is None:
        radius_db = get_radius_database()
    
    n_points = len(T)
    a_values = np.zeros(n_points)
    r_B_avg_array = np.zeros(n_points)
    r_anion_avg_array = np.zeros(n_points)
    r_A_avg_array = np.zeros(n_points)
    
    for i in range(n_points):
        # Calculate coordination numbers
        cn_b, cn_a = calculate_coordination_numbers(Acc, oh[i])
        
        # Calculate radii
        r_B_avg = calculate_average_b_radius(ion_B, ion_Acc, charge_B, charge_Acc,
                                              Acc, cn_b, radius_db)
        r_A_avg = calculate_average_a_radius(ion_A, charge_A, cn_a, radius_db)
        r_anion_avg = calculate_anion_radius(oh[i], Acc, r_V, r_OH, r_O)
        
        r_B_avg_array[i] = r_B_avg
        r_anion_avg_array[i] = r_anion_avg
        r_A_avg_array[i] = r_A_avg
        
        # Calculate lattice parameter
        if model_type == 'full':
            a_values[i] = calculate_lattice_parameter_geometric(r_B_avg, r_anion_avg, r_A_avg, 'full')
        else:
            a_values[i] = calculate_lattice_parameter_geometric(r_B_avg, r_anion_avg, model_type=model_type)
    
    # Reference value
    if a0 is None:
        a0 = a_values[0]
    
    deltaL_L0 = (a_values - a0) / a0
    
    return deltaL_L0, a0, r_B_avg_array, r_anion_avg_array


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
        'last_fit_params': None,
        'geometric_fit_results': None,
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
        # New: Geometric model parameters
        'geometric_params': {
            'ion_A': 'Ba',
            'ion_B': 'Zr',
            'ion_Acc': 'Y',
            'charge_A': 2,
            'charge_B': 4,
            'charge_Acc': 3,
            'r_V': 1.2,  # Initial guess for vacancy radius
            'r_OH': 1.35,  # Fixed or fitted
            'r_O': 1.4,  # Fixed
            'r_OH_fixed': True,  # Whether r_OH is fixed during fitting
            'geometric_model_type': 'B_only',  # 'B_only', 'A_only', 'full'
            'use_geometric_model': False  # Switch between phenomenological and geometric
        },
        'data_loaded': False,
        'plots_generated': False,
        'fitting_complete': False,
        'geometric_fitting_complete': False
    }
    
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

# Initialize session state
initialize_session_state()

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
def fit_geometric_model_cached(data: np.ndarray, 
                               fixed_params: Dict[str, Any],
                               initial_guess: Dict[str, Any],
                               ion_A: str, ion_B: str, ion_Acc: str,
                               charge_A: int, charge_B: int, charge_Acc: int,
                               r_O: float, r_OH_fixed: float,
                               geometric_model_type: str,
                               fit_r_V: bool = True,
                               fit_r_OH: bool = False) -> Optional[Dict[str, Any]]:
    """
    Fit geometric model to experimental data
    
    This performs the second-stage fitting to determine r_V and optionally r_OH
    """
    T_data = data[:, 0]
    dl_data = data[:, 1]
    
    # Get phenomenological parameters from first stage
    Acc = fixed_params.get('Acc')
    if Acc is None:
        Acc = initial_guess.get('Acc', 0.6)
    
    # Get thermodynamic parameters
    dH = fixed_params.get('dH')
    if dH is None:
        dH = initial_guess.get('dH', -81.0)
    dS = fixed_params.get('dS')
    if dS is None:
        dS = initial_guess.get('dS', -131.0)
    pH2O = fixed_params.get('pH2O')
    if pH2O is None:
        pH2O = initial_guess.get('pH2O', 0.083)
    
    # Calculate OH concentrations
    oh, _ = calculate_oh_cached(T_data, Acc, dH, dS, pH2O)
    
    # Reference temperature
    T_start = T_data[0]
    oh_start = oh[0]
    
    # Prepare parameters for fitting
    radius_db = get_radius_database()
    
    vary_params = []
    bounds_lower = []
    bounds_upper = []
    initial_params = []
    
    # Always fit alpha and residue if not fixed? Actually we keep them from first stage
    # For geometric model, we fit r_V and optionally r_OH, with fixed alpha and residue
    # But we need to allow alpha to adjust? Actually we can keep α from first stage
    
    # For now, we fit only r_V and maybe r_OH
    if fit_r_V:
        vary_params.append('r_V')
        bounds_lower.append(0.5)  # Minimum possible vacancy radius
        bounds_upper.append(2.0)  # Maximum possible vacancy radius
        initial_params.append(initial_guess.get('r_V', 1.2))
    
    if fit_r_OH:
        vary_params.append('r_OH')
        bounds_lower.append(1.0)
        bounds_upper.append(1.8)
        initial_params.append(initial_guess.get('r_OH', 1.35))
    
    # Fixed parameters from first stage
    alpha = fixed_params.get('alpha_1e6', initial_guess.get('alpha_1e6', 14.4)) / 1e6
    residue = fixed_params.get('residue', initial_guess.get('residue', 0.0))
    
    # Calculate reference lattice parameter from first data point
    # For geometric model, we need to calculate a0 from the first data point
    # We'll use the geometric model to compute a0 at T_start
    
    def geometric_model_func(T, oh_vals, *params):
        """Geometric model function for fitting"""
        param_dict = {}
        param_idx = 0
        if fit_r_V:
            param_dict['r_V'] = params[param_idx]
            param_idx += 1
        if fit_r_OH:
            param_dict['r_OH'] = params[param_idx]
            param_idx += 1
        
        r_V_val = param_dict.get('r_V', initial_guess.get('r_V', 1.2))
        r_OH_val = param_dict.get('r_OH', r_OH_fixed)
        
        # Calculate geometric expansion
        deltaL_geo, a0, _, _ = calculate_geometric_expansion(
            T, Acc, oh_vals,
            ion_A, ion_B, ion_Acc,
            charge_A, charge_B, charge_Acc,
            r_V_val, r_OH_val, r_O,
            a0=None,
            model_type=geometric_model_type,
            radius_db=radius_db
        )
        
        # Add thermal expansion and residue
        total = deltaL_geo + alpha * (T - T_start) + residue
        return total
    
    try:
        # Perform fitting
        popt, pcov = curve_fit(
            lambda T, *params: geometric_model_func(T, oh, *params),
            T_data, dl_data,
            p0=initial_params,
            bounds=(bounds_lower, bounds_upper),
            maxfev=5000
        )
        
        # Assemble results
        result_params = {}
        param_idx = 0
        if fit_r_V:
            result_params['r_V'] = popt[param_idx]
            param_idx += 1
        if fit_r_OH:
            result_params['r_OH'] = popt[param_idx]
            param_idx += 1
        
        # Calculate model values with fitted parameters
        r_V_final = result_params.get('r_V', initial_guess.get('r_V', 1.2))
        r_OH_final = result_params.get('r_OH', r_OH_fixed)
        
        deltaL_geo, a0, r_B_avg, r_anion_avg = calculate_geometric_expansion(
            T_data, Acc, oh,
            ion_A, ion_B, ion_Acc,
            charge_A, charge_B, charge_Acc,
            r_V_final, r_OH_final, r_O,
            a0=None,
            model_type=geometric_model_type,
            radius_db=radius_db
        )
        
        dl_model = deltaL_geo + alpha * (T_data - T_start) + residue
        
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
        
        # Calculate CNs for visualization
        cn_b_array = np.zeros_like(T_data)
        cn_a_array = np.zeros_like(T_data)
        for i in range(len(T_data)):
            cn_b_array[i], cn_a_array[i] = calculate_coordination_numbers(Acc, oh[i])
        
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
            'T_data': T_data,
            'dl_data': dl_data,
            'oh_concentration': oh,
            'a0_reference': a0,
            'r_B_avg': r_B_avg,
            'r_anion_avg': r_anion_avg,
            'cn_B': cn_b_array,
            'cn_A': cn_a_array,
            'geometric_model_type': geometric_model_type,
            'alpha': alpha,
            'residue': residue,
            'Acc': Acc,
            'dH': dH,
            'dS': dS,
            'pH2O': pH2O
        }
        
    except Exception as e:
        st.error(f"Geometric model fitting error: {str(e)}")
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
    
    # Draw experimental points
    ax1.scatter(T, dl_exp, s=style['point_size'], color=style['point_color'], 
               edgecolor='none', 
               label='Experimental', zorder=3, alpha=style['point_alpha'])
    
    # Draw model line
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
    
    # Determine direction of temperature change (heating or cooling)
    T_diff = T[-1] - T[0]
    
    if T_diff > 0:  # Heating
        thermal_start = thermal_contrib[0] + residue
        thermal_changes = (thermal_contrib + residue) - thermal_start
        chem_end = chem_contrib[-1] + residue
        chem_changes = chem_end - (chem_contrib + residue)
    else:  # Cooling
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
    
    # TEC plot
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
    
    # Combine legends
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
    
    # TEC residual plot
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
def create_plot13_cached(geo_results: Dict[str, Any], pheno_results: Dict[str, Any], 
                         style: Dict[str, Any]) -> plt.Figure:
    """Plot 13: Comparison of phenomenological and geometric models"""
    fig, ax = plt.subplots(figsize=(6, 4))
    
    T = geo_results['T_data']
    
    # Phenomenological model (from first stage)
    dl_pheno = pheno_results['dl_model'] if pheno_results else None
    
    # Geometric model
    dl_geo = geo_results['dl_model']
    
    ax.plot(T, geo_results['dl_data'], 'o', color=style['point_color'], 
           markersize=style['point_size']/10, alpha=style['point_alpha'], label='Experimental')
    
    if dl_pheno is not None:
        ax.plot(T, dl_pheno, '-', color=style['thermal_line_color'], 
               linewidth=style['line_width'], label='Phenomenological model')
    
    ax.plot(T, dl_geo, '--', color=style['chemical_line_color'], 
           linewidth=style['line_width'], label='Geometric model')
    
    ax.set_xlabel('Temperature (°C)', fontweight='bold', fontsize=11)
    ax.set_ylabel('ΔL/L₀', fontweight='bold', fontsize=11)
    ax.set_title('Model Comparison: Phenomenological vs Geometric', fontweight='bold', fontsize=12)
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3, linestyle='--')
    
    fig.set_dpi(600)
    return fig


@st.cache_data(ttl=3600, show_spinner=False)
def create_plot14_cached(geo_results: Dict[str, Any], style: Dict[str, Any]) -> plt.Figure:
    """Plot 14: Evolution of effective radii with temperature"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    
    T = geo_results['T_data']
    r_B = geo_results['r_B_avg']
    r_anion = geo_results['r_anion_avg']
    
    # Plot radii
    ax1.plot(T, r_B, '-', color=style['thermal_line_color'], 
            linewidth=style['line_width'], label='⟨r_B⟩')
    ax1.plot(T, r_anion, '-', color=style['chemical_line_color'], 
            linewidth=style['line_width'], label='⟨r_anion⟩')
    ax1.plot(T, r_B + r_anion, '--', color=style['model_line_color'], 
            linewidth=style['line_width'], label='r_B + r_anion')
    ax1.set_xlabel('Temperature (°C)', fontweight='bold', fontsize=11)
    ax1.set_ylabel('Radius (Å)', fontweight='bold', fontsize=11)
    ax1.set_title('Effective Radii Evolution', fontweight='bold', fontsize=12)
    ax1.legend(loc='best')
    ax1.grid(True, alpha=0.3, linestyle='--')
    
    # Plot relative contributions
    r_B_contrib = np.gradient(r_B, T) / r_B * 1e6
    r_anion_contrib = np.gradient(r_anion, T) / r_anion * 1e6
    
    ax2.plot(T, r_B_contrib, '-', color=style['thermal_line_color'], 
            linewidth=style['line_width'], label='d(ln r_B)/dT')
    ax2.plot(T, r_anion_contrib, '-', color=style['chemical_line_color'], 
            linewidth=style['line_width'], label='d(ln r_anion)/dT')
    ax2.axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.5)
    ax2.set_xlabel('Temperature (°C)', fontweight='bold', fontsize=11)
    ax2.set_ylabel('Relative Change Rate (10⁻⁶ K⁻¹)', fontweight='bold', fontsize=11)
    ax2.set_title('Radii Change Rates', fontweight='bold', fontsize=12)
    ax2.legend(loc='best')
    ax2.grid(True, alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    fig.set_dpi(600)
    return fig


@st.cache_data(ttl=3600, show_spinner=False)
def create_plot15_cached(geo_results: Dict[str, Any], style: Dict[str, Any]) -> plt.Figure:
    """Plot 15: Lattice parameter vs proton concentration"""
    fig, ax = plt.subplots(figsize=(6, 5))
    
    oh = geo_results['oh_concentration']
    T = geo_results['T_data']
    
    # Calculate lattice parameter from geometric model
    radius_db = get_radius_database()
    Acc = geo_results['Acc']
    r_V = geo_results['params']['r_V']
    r_OH = geo_results['params'].get('r_OH', 1.35)
    r_O = 1.4
    model_type = geo_results['geometric_model_type']
    
    a_values = np.zeros_like(T)
    for i in range(len(T)):
        cn_b, cn_a = calculate_coordination_numbers(Acc, oh[i])
        r_B = calculate_average_b_radius(
            st.session_state.geometric_params['ion_B'],
            st.session_state.geometric_params['ion_Acc'],
            st.session_state.geometric_params['charge_B'],
            st.session_state.geometric_params['charge_Acc'],
            Acc, cn_b, radius_db
        )
        r_anion = calculate_anion_radius(oh[i], Acc, r_V, r_OH, r_O)
        
        if model_type == 'full':
            r_A = calculate_average_a_radius(
                st.session_state.geometric_params['ion_A'],
                st.session_state.geometric_params['charge_A'],
                cn_a, radius_db
            )
            a_values[i] = calculate_lattice_parameter_geometric(r_B, r_anion, r_A, 'full')
        else:
            a_values[i] = calculate_lattice_parameter_geometric(r_B, r_anion, model_type=model_type)
    
    # Color by temperature
    sc = ax.scatter(oh, a_values, c=T, cmap=style['cmap_style'], 
                   s=style['point_size'], edgecolor='none', alpha=style['point_alpha'])
    
    cbar = plt.colorbar(sc, ax=ax)
    cbar.set_label('Temperature (°C)', fontweight='bold', fontsize=10)
    
    ax.set_xlabel('[OH] concentration (arb. units)', fontweight='bold', fontsize=11)
    ax.set_ylabel('Lattice parameter a (Å)', fontweight='bold', fontsize=11)
    ax.set_title('Lattice Parameter vs Proton Concentration', fontweight='bold', fontsize=12)
    ax.grid(True, alpha=0.3, linestyle='--')
    
    fig.set_dpi(600)
    return fig


@st.cache_data(ttl=3600, show_spinner=False)
def create_plot16_cached(geo_results: Dict[str, Any], style: Dict[str, Any]) -> plt.Figure:
    """Plot 16: Coordination numbers evolution"""
    fig, ax = plt.subplots(figsize=(6, 4))
    
    T = geo_results['T_data']
    cn_B = geo_results['cn_B']
    cn_A = geo_results['cn_A']
    
    ax.plot(T, cn_B, '-', color=style['thermal_line_color'], 
           linewidth=style['line_width'], label='CN_B')
    ax.plot(T, cn_A, '-', color=style['chemical_line_color'], 
           linewidth=style['line_width'], label='CN_A')
    
    ax.fill_between(T, 0, cn_B, alpha=0.1, color=style['thermal_line_color'])
    ax.fill_between(T, 0, cn_A, alpha=0.1, color=style['chemical_line_color'])
    
    ax.set_xlabel('Temperature (°C)', fontweight='bold', fontsize=11)
    ax.set_ylabel('Coordination Number', fontweight='bold', fontsize=11)
    ax.set_title('Coordination Numbers Evolution with Temperature', fontweight='bold', fontsize=12)
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3, linestyle='--')
    
    # Add horizontal lines for reference
    ax.axhline(y=6, color='gray', linestyle=':', alpha=0.5, label='CN_B ideal (6)')
    ax.axhline(y=12, color='gray', linestyle=':', alpha=0.5, label='CN_A ideal (12)')
    
    fig.set_dpi(600)
    return fig


@st.cache_data(ttl=3600, show_spinner=False)
def create_plot17_cached(geo_results: Dict[str, Any], style: Dict[str, Any]) -> plt.Figure:
    """Plot 17: Goldschmidt tolerance factor evolution"""
    fig, ax = plt.subplots(figsize=(6, 4))
    
    T = geo_results['T_data']
    oh = geo_results['oh_concentration']
    Acc = geo_results['Acc']
    
    radius_db = get_radius_database()
    r_O = 1.4
    
    t_factor = np.zeros_like(T)
    
    for i in range(len(T)):
        cn_b, cn_a = calculate_coordination_numbers(Acc, oh[i])
        
        r_B = calculate_average_b_radius(
            st.session_state.geometric_params['ion_B'],
            st.session_state.geometric_params['ion_Acc'],
            st.session_state.geometric_params['charge_B'],
            st.session_state.geometric_params['charge_Acc'],
            Acc, cn_b, radius_db
        )
        
        r_A = calculate_average_a_radius(
            st.session_state.geometric_params['ion_A'],
            st.session_state.geometric_params['charge_A'],
            cn_a, radius_db
        )
        
        # Goldschmidt tolerance factor: t = (r_A + r_O) / (√2 * (r_B + r_O))
        t_factor[i] = (r_A + r_O) / (np.sqrt(2) * (r_B + r_O))
    
    ax.plot(T, t_factor, '-', color=style['model_line_color'], 
           linewidth=style['line_width'])
    
    # Ideal cubic perovskite range
    ax.axhspan(0.89, 1.02, alpha=0.2, color='green', label='Cubic stability range')
    ax.axhline(y=1.0, color='green', linestyle='--', linewidth=1, alpha=0.7, label='Ideal cubic (t=1.0)')
    
    ax.set_xlabel('Temperature (°C)', fontweight='bold', fontsize=11)
    ax.set_ylabel('Goldschmidt tolerance factor t', fontweight='bold', fontsize=11)
    ax.set_title('Structural Stability (Goldschmidt Factor)', fontweight='bold', fontsize=12)
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3, linestyle='--')
    
    fig.set_dpi(600)
    return fig


@st.cache_data(ttl=3600, show_spinner=False)
def create_plot18_cached(geo_results: Dict[str, Any], style: Dict[str, Any]) -> plt.Figure:
    """Plot 18: r_V and r_OH sensitivity analysis"""
    fig, ax = plt.subplots(figsize=(6, 5))
    
    T = geo_results['T_data']
    oh = geo_results['oh_concentration']
    dl_exp = geo_results['dl_data']
    r_V_fitted = geo_results['params']['r_V']
    
    # Calculate model with different r_V values
    r_V_variations = [-0.2, -0.1, 0, 0.1, 0.2]
    r_OH = geo_results['params'].get('r_OH', 1.35)
    Acc = geo_results['Acc']
    r_O = 1.4
    model_type = geo_results['geometric_model_type']
    alpha = geo_results['alpha']
    residue = geo_results['residue']
    T_start = T[0]
    oh_start = oh[0]
    
    radius_db = get_radius_database()
    
    colors = plt.cm.viridis(np.linspace(0, 1, len(r_V_variations)))
    
    for i, var in enumerate(r_V_variations):
        r_V_test = r_V_fitted * (1 + var)
        
        deltaL_geo, _, _, _ = calculate_geometric_expansion(
            T, Acc, oh,
            st.session_state.geometric_params['ion_A'],
            st.session_state.geometric_params['ion_B'],
            st.session_state.geometric_params['ion_Acc'],
            st.session_state.geometric_params['charge_A'],
            st.session_state.geometric_params['charge_B'],
            st.session_state.geometric_params['charge_Acc'],
            r_V_test, r_OH, r_O,
            a0=None,
            model_type=model_type,
            radius_db=radius_db
        )
        
        dl_model = deltaL_geo + alpha * (T - T_start) + residue
        
        ax.plot(T, (dl_model - geo_results['dl_model'])*1e6, 
               color=colors[i], linewidth=style['line_width'],
               label=f'r_V = {r_V_test:.3f} Å ({var*100:+.0f}%)')
    
    ax.set_xlabel('Temperature (°C)', fontweight='bold', fontsize=11)
    ax.set_ylabel('Δ(ΔL/L₀) (10⁻⁶)', fontweight='bold', fontsize=11)
    ax.set_title('Sensitivity to r_V (Vacancy Radius)', fontweight='bold', fontsize=12)
    ax.legend(loc='best', fontsize=8)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.5)
    
    fig.set_dpi(600)
    return fig


@st.cache_data(ttl=3600, show_spinner=False)
def create_plot19_cached(geo_results: Dict[str, Any], style: Dict[str, Any]) -> plt.Figure:
    """Plot 19: CN_B vs OH concentration correlation"""
    fig, ax = plt.subplots(figsize=(6, 5))
    
    oh = geo_results['oh_concentration']
    cn_B = geo_results['cn_B']
    T = geo_results['T_data']
    
    sc = ax.scatter(oh, cn_B, c=T, cmap=style['cmap_style'], 
                   s=style['point_size'], edgecolor='none', alpha=style['point_alpha'])
    
    cbar = plt.colorbar(sc, ax=ax)
    cbar.set_label('Temperature (°C)', fontweight='bold', fontsize=10)
    
    ax.set_xlabel('[OH] concentration', fontweight='bold', fontsize=11)
    ax.set_ylabel('CN_B (Coordination number of B-site)', fontweight='bold', fontsize=11)
    ax.set_title('CN_B vs OH Concentration (from CN_B = 6 - [Acc] + [OH])', 
                fontweight='bold', fontsize=12)
    ax.grid(True, alpha=0.3, linestyle='--')
    
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

def update_geometric_param(param_name: str, value):
    """Update geometric model parameter in session state"""
    st.session_state.geometric_params[param_name] = value

def update_plot_style(param_name: str, value):
    """Update plot style parameter in session state"""
    st.session_state.plot_style[param_name] = value
    if st.session_state.fitting_complete:
        st.rerun()


# ============================================
# ОСНОВНОЙ ИНТЕРФЕЙС
# ============================================

def main():
    st.title("📈 Thermo-Mechanical Expansion Modeling")
    st.markdown("Modeling of proton-conducting oxides thermal expansion with geometric defect analysis")
    
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
                    st.session_state.geometric_fitting_complete = False
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
                    st.session_state.geometric_fitting_complete = False
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
                st.session_state.geometric_fitting_complete = False
                st.session_state.fit_results = None
                st.session_state.geometric_fit_results = None
        
        st.divider()
        
        # Model Type Selection (NEW)
        st.header("Model Selection")
        
        model_type_toggle = st.radio(
            "Model type:",
            ["Phenomenological (α·ΔT + β·Δ[OH] + γ)", "Geometric (ionic radii based)"],
            index=0 if not st.session_state.geometric_params.get('use_geometric_model', False) else 1,
            key="model_type_select"
        )
        
        use_geometric = (model_type_toggle == "Geometric (ionic radii based)")
        st.session_state.geometric_params['use_geometric_model'] = use_geometric
        
        if use_geometric:
            st.info("🔬 Geometric model uses Shannon ionic radii and coordination numbers")
        
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
            
            st.divider()
            
            # Geometric Model Parameters (NEW)
            if use_geometric:
                st.subheader("Geometric Model Parameters")
                st.markdown("Select cations and their charges")
                
                # A-site selection
                ion_A_options = ['Ba', 'Sr', 'Ca', 'La']
                ion_A = st.selectbox(
                    "A-site cation",
                    ion_A_options,
                    index=ion_A_options.index(st.session_state.geometric_params.get('ion_A', 'Ba')),
                    key="ion_A_select"
                )
                charge_A = 2 if ion_A != 'La' else 3
                
                # B-site selection based on A-site
                if ion_A == 'La':
                    ion_B_options = ['In', 'Sc', 'Y', 'Yb']
                else:
                    ion_B_options = ['Ce', 'Zr', 'Ti', 'Hf', 'Sn']
                
                ion_B = st.selectbox(
                    "B-site cation",
                    ion_B_options,
                    index=ion_B_options.index(st.session_state.geometric_params.get('ion_B', 'Zr')) 
                    if st.session_state.geometric_params.get('ion_B', 'Zr') in ion_B_options else 0,
                    key="ion_B_select"
                )
                charge_B = 4
                
                # Acceptor selection
                if ion_A == 'La':
                    acc_options = ['Mg', 'Zn']
                else:
                    acc_options = ['Al', 'Ga', 'In', 'Sc', 'Y', 'Yb', 'Ho', 'Dy', 'Gd', 'Sm', 'Nd', 'La']
                
                ion_Acc = st.selectbox(
                    "Acceptor cation (Acc)",
                    acc_options,
                    index=acc_options.index(st.session_state.geometric_params.get('ion_Acc', 'Y')) 
                    if st.session_state.geometric_params.get('ion_Acc', 'Y') in acc_options else 0,
                    key="ion_Acc_select"
                )
                charge_Acc = 2 if ion_A == 'La' else 3
                
                # Geometric model type
                geo_model_type = st.selectbox(
                    "Geometric model formula",
                    ['B_only', 'A_only', 'full'],
                    index=['B_only', 'A_only', 'full'].index(st.session_state.geometric_params.get('geometric_model_type', 'B_only')),
                    key="geo_model_type_select",
                    help="B_only: a = 2(r_B + r_anion); A_only: a = 2(r_A + r_anion); full: a = 2/√2(r_A + r_B + r_anion)"
                )
                
                # r_V and r_OH settings
                col_r1, col_r2 = st.columns(2)
                with col_r1:
                    r_V_init = st.number_input(
                        "Initial r_V (Å)",
                        value=st.session_state.geometric_params.get('r_V', 1.2),
                        step=0.05,
                        format="%.3f",
                        key="r_V_init"
                    )
                    fit_r_V = st.checkbox("Fit r_V", value=True, key="fit_r_V")
                
                with col_r2:
                    r_OH_init = st.number_input(
                        "Initial r_OH (Å)",
                        value=st.session_state.geometric_params.get('r_OH', 1.35),
                        step=0.05,
                        format="%.3f",
                        key="r_OH_init"
                    )
                    fit_r_OH = st.checkbox("Fit r_OH", value=False, key="fit_r_OH")
                
                # Update geometric params
                st.session_state.geometric_params['ion_A'] = ion_A
                st.session_state.geometric_params['ion_B'] = ion_B
                st.session_state.geometric_params['ion_Acc'] = ion_Acc
                st.session_state.geometric_params['charge_A'] = charge_A
                st.session_state.geometric_params['charge_B'] = charge_B
                st.session_state.geometric_params['charge_Acc'] = charge_Acc
                st.session_state.geometric_params['geometric_model_type'] = geo_model_type
                st.session_state.geometric_params['r_V'] = r_V_init
                st.session_state.geometric_params['r_OH'] = r_OH_init
                st.session_state.geometric_params['fit_r_V'] = fit_r_V
                st.session_state.geometric_params['fit_r_OH'] = fit_r_OH
            
            # Fitting button
            fit_button = st.form_submit_button("🚀 Fit Model and Create Plots", type="primary", use_container_width=True)
            
            if fit_button:
                if st.session_state.experimental_data is None:
                    st.error("Please load data first!")
                else:
                    # Update parameter values
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
                    
                    # Prepare parameters for fitting
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
                        
                        # Set bounds
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
                    
                    # Perform phenomenological fitting first
                    with st.spinner("Performing phenomenological fitting..."):
                        start_time = time.time()
                        st.session_state.fit_results = fit_model_cached(
                            st.session_state.experimental_data, 
                            fixed_params, 
                            initial_guess
                        )
                        end_time = time.time()
                        
                        if st.session_state.fit_results is not None:
                            st.session_state.fitting_complete = True
                            
                            # Update fixed params with fitted values
                            for param_name in ['Acc', 'alpha_1e6', 'beta', 'dH', 'dS', 'pH2O', 'residue']:
                                if not st.session_state.model_params[param_name]['fixed']:
                                    fitted_value = st.session_state.fit_results['params'][param_name]
                                    st.session_state.model_params[param_name]['value'] = fitted_value
                                    fixed_params[param_name] = fitted_value
                            
                            st.success(f"Phenomenological fitting completed in {end_time - start_time:.2f} seconds")
                            
                            # Perform geometric fitting if selected
                            if use_geometric:
                                st.info("Performing geometric model fitting...")
                                
                                geo_initial_guess = {
                                    'r_V': st.session_state.geometric_params['r_V'],
                                    'r_OH': st.session_state.geometric_params['r_OH']
                                }
                                
                                geo_fixed_params = {
                                    'Acc': fixed_params['Acc'] if fixed_params['Acc'] is not None else st.session_state.model_params['Acc']['value'],
                                    'alpha_1e6': fixed_params['alpha_1e6'] if fixed_params['alpha_1e6'] is not None else st.session_state.model_params['alpha_1e6']['value'],
                                    'residue': fixed_params['residue'] if fixed_params['residue'] is not None else st.session_state.model_params['residue']['value'],
                                    'dH': fixed_params['dH'] if fixed_params['dH'] is not None else st.session_state.model_params['dH']['value'],
                                    'dS': fixed_params['dS'] if fixed_params['dS'] is not None else st.session_state.model_params['dS']['value'],
                                    'pH2O': fixed_params['pH2O'] if fixed_params['pH2O'] is not None else st.session_state.model_params['pH2O']['value']
                                }
                                
                                geo_start_time = time.time()
                                st.session_state.geometric_fit_results = fit_geometric_model_cached(
                                    st.session_state.experimental_data,
                                    geo_fixed_params,
                                    geo_initial_guess,
                                    st.session_state.geometric_params['ion_A'],
                                    st.session_state.geometric_params['ion_B'],
                                    st.session_state.geometric_params['ion_Acc'],
                                    st.session_state.geometric_params['charge_A'],
                                    st.session_state.geometric_params['charge_B'],
                                    st.session_state.geometric_params['charge_Acc'],
                                    1.4,  # r_O
                                    st.session_state.geometric_params['r_OH'],
                                    st.session_state.geometric_params['geometric_model_type'],
                                    fit_r_V=st.session_state.geometric_params['fit_r_V'],
                                    fit_r_OH=st.session_state.geometric_params['fit_r_OH']
                                )
                                geo_end_time = time.time()
                                
                                if st.session_state.geometric_fit_results is not None:
                                    st.session_state.geometric_fitting_complete = True
                                    st.success(f"Geometric fitting completed in {geo_end_time - geo_start_time:.2f} seconds")
                                    
                                    # Update r_V and r_OH in session state
                                    if 'r_V' in st.session_state.geometric_fit_results['params']:
                                        st.session_state.geometric_params['r_V'] = st.session_state.geometric_fit_results['params']['r_V']
                                    if 'r_OH' in st.session_state.geometric_fit_results['params']:
                                        st.session_state.geometric_params['r_OH'] = st.session_state.geometric_fit_results['params']['r_OH']
                                else:
                                    st.warning("Geometric fitting failed, using phenomenological results only")
                            
                            st.rerun()
                        else:
                            st.error("Fitting failed. Please check your parameters.")
        
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
    
    # Display loaded data
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
    
    # Display phenomenological fitting results
    if st.session_state.fit_results is not None and st.session_state.fitting_complete:
        st.header("Phenomenological Model Results")
        
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
        
        st.subheader("Fitted Parameters")
        
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
        st.header("Plots")
        
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "📈 Basic Plots", 
            "🔍 Advanced Analysis", 
            "📊 Statistical Analysis", 
            "🧪 Model Insights",
            "🔬 Geometric Analysis"
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
        
        with tab5:
            st.subheader("Geometric Model Analysis")
            
            if st.session_state.geometric_fit_results is not None:
                st.success("Geometric model fitting completed successfully!")
                
                # Display geometric fitting metrics
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Geometric MSE", f"{st.session_state.geometric_fit_results['mse']:.3e}")
                with col2:
                    st.metric("Geometric RMSE", f"{st.session_state.geometric_fit_results['rmse']:.3e}")
                with col3:
                    st.metric("Geometric R²", f"{st.session_state.geometric_fit_results['r2']:.6f}")
                
                # Display fitted radii
                st.subheader("Fitted Geometric Parameters")
                
                geo_params_data = []
                if 'r_V' in st.session_state.geometric_fit_results['params']:
                    geo_params_data.append({
                        "Parameter": "r_V (Vacancy radius)",
                        "Value (Å)": f"{st.session_state.geometric_fit_results['params']['r_V']:.4f}",
                        "Status": "Fitted"
                    })
                else:
                    geo_params_data.append({
                        "Parameter": "r_V (Vacancy radius)",
                        "Value (Å)": f"{st.session_state.geometric_params['r_V']:.4f}",
                        "Status": "Fixed"
                    })
                
                if 'r_OH' in st.session_state.geometric_fit_results['params']:
                    geo_params_data.append({
                        "Parameter": "r_OH (OH group radius)",
                        "Value (Å)": f"{st.session_state.geometric_fit_results['params']['r_OH']:.4f}",
                        "Status": "Fitted"
                    })
                else:
                    geo_params_data.append({
                        "Parameter": "r_OH (OH group radius)",
                        "Value (Å)": f"{st.session_state.geometric_params['r_OH']:.4f}",
                        "Status": "Fixed"
                    })
                
                geo_params_df = pd.DataFrame(geo_params_data)
                st.dataframe(geo_params_df, use_container_width=True)
                
                # Display ion selection
                st.subheader("Selected Ions and Charges")
                ion_data = [
                    {"Site": "A", "Ion": st.session_state.geometric_params['ion_A'], "Charge": st.session_state.geometric_params['charge_A']},
                    {"Site": "B", "Ion": st.session_state.geometric_params['ion_B'], "Charge": st.session_state.geometric_params['charge_B']},
                    {"Site": "Acceptor", "Ion": st.session_state.geometric_params['ion_Acc'], "Charge": st.session_state.geometric_params['charge_Acc']}
                ]
                ion_df = pd.DataFrame(ion_data)
                st.dataframe(ion_df, use_container_width=True)
                
                # Create geometric plots
                st.markdown("---")
                st.subheader("Geometric Model Plots")
                
                # Plot 13: Model comparison
                plot13 = create_plot13_cached(
                    st.session_state.geometric_fit_results,
                    st.session_state.fit_results,
                    st.session_state.plot_style
                )
                st.markdown("**Model Comparison: Phenomenological vs Geometric**")
                st.pyplot(plot13)
                
                # Plot 14: Effective radii evolution
                plot14 = create_plot14_cached(
                    st.session_state.geometric_fit_results,
                    st.session_state.plot_style
                )
                st.markdown("**Effective Radii Evolution**")
                st.pyplot(plot14)
                
                # Plot 15: Lattice parameter vs OH
                plot15 = create_plot15_cached(
                    st.session_state.geometric_fit_results,
                    st.session_state.plot_style
                )
                st.markdown("**Lattice Parameter vs Proton Concentration**")
                st.pyplot(plot15)
                
                col1, col2 = st.columns(2)
                
                with col1:
                    # Plot 16: Coordination numbers
                    plot16 = create_plot16_cached(
                        st.session_state.geometric_fit_results,
                        st.session_state.plot_style
                    )
                    st.markdown("**Coordination Numbers Evolution**")
                    st.pyplot(plot16)
                
                with col2:
                    # Plot 17: Goldschmidt factor
                    plot17 = create_plot17_cached(
                        st.session_state.geometric_fit_results,
                        st.session_state.plot_style
                    )
                    st.markdown("**Goldschmidt Tolerance Factor**")
                    st.pyplot(plot17)
                
                # Plot 18: r_V sensitivity
                plot18 = create_plot18_cached(
                    st.session_state.geometric_fit_results,
                    st.session_state.plot_style
                )
                st.markdown("**Sensitivity to r_V (Vacancy Radius)**")
                st.pyplot(plot18)
                
                # Plot 19: CN_B vs OH
                plot19 = create_plot19_cached(
                    st.session_state.geometric_fit_results,
                    st.session_state.plot_style
                )
                st.markdown("**CN_B vs OH Concentration Correlation**")
                st.pyplot(plot19)
                
                st.info("""
                **Geometric Model Insights:**
                
                - **Model Comparison**: Shows agreement between phenomenological and geometric approaches
                - **Radii Evolution**: Reveals how B-site and anion radii change with temperature
                - **Lattice Parameter vs [OH]**: Indicates the dependence of expansion on proton concentration
                - **Coordination Numbers**: Shows the evolution of CN_B and CN_A with temperature (CN_B = 6 - [Acc] + [OH], CN_A = 12 - 2[Acc] + 2[OH])
                - **Goldschmidt Factor**: Indicates structural stability (t ~ 0.89-1.02 for cubic perovskite)
                - **r_V Sensitivity**: Shows how variations in vacancy radius affect the model predictions
                - **CN_B vs [OH]**: Validates the linear relationship from the task formulation
                """)
                
            else:
                st.info("Geometric model results not available. Run the fitting with geometric model enabled to see these plots.")
        
        # Download section
        st.divider()
        st.subheader("Export Results")
        
        download_tab1, download_tab2, download_tab3, download_tab4, download_tab5 = st.tabs([
            "📥 Basic Plots", 
            "📥 Advanced Plots", 
            "📥 Statistical Plots", 
            "📥 Insight Plots",
            "📥 Geometric Plots"
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
        
        with download_tab5:
            if st.session_state.geometric_fit_results is not None:
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("📥 Download Plot 13 (Model Comparison)", key="dl_plot13"):
                        buf = io.BytesIO()
                        plot13.savefig(buf, format='png', dpi=600)
                        st.download_button(
                            label="Download PNG (600 DPI)",
                            data=buf.getvalue(),
                            file_name="plot13_model_comparison.png",
                            mime="image/png",
                            key="dl_btn_plot13"
                        )
                    
                    if st.button("📥 Download Plot 14 (Radii Evolution)", key="dl_plot14"):
                        buf = io.BytesIO()
                        plot14.savefig(buf, format='png', dpi=600)
                        st.download_button(
                            label="Download PNG (600 DPI)",
                            data=buf.getvalue(),
                            file_name="plot14_radii_evolution.png",
                            mime="image/png",
                            key="dl_btn_plot14"
                        )
                    
                    if st.button("📥 Download Plot 15 (a vs [OH])", key="dl_plot15"):
                        buf = io.BytesIO()
                        plot15.savefig(buf, format='png', dpi=600)
                        st.download_button(
                            label="Download PNG (600 DPI)",
                            data=buf.getvalue(),
                            file_name="plot15_a_vs_oh.png",
                            mime="image/png",
                            key="dl_btn_plot15"
                        )
                
                with col2:
                    if st.button("📥 Download Plot 16 (Coordination Numbers)", key="dl_plot16"):
                        buf = io.BytesIO()
                        plot16.savefig(buf, format='png', dpi=600)
                        st.download_button(
                            label="Download PNG (600 DPI)",
                            data=buf.getvalue(),
                            file_name="plot16_coordination_numbers.png",
                            mime="image/png",
                            key="dl_btn_plot16"
                        )
                    
                    if st.button("📥 Download Plot 17 (Goldschmidt)", key="dl_plot17"):
                        buf = io.BytesIO()
                        plot17.savefig(buf, format='png', dpi=600)
                        st.download_button(
                            label="Download PNG (600 DPI)",
                            data=buf.getvalue(),
                            file_name="plot17_goldschmidt.png",
                            mime="image/png",
                            key="dl_btn_plot17"
                        )
                    
                    if st.button("📥 Download Plot 18 (r_V Sensitivity)", key="dl_plot18"):
                        buf = io.BytesIO()
                        plot18.savefig(buf, format='png', dpi=600)
                        st.download_button(
                            label="Download PNG (600 DPI)",
                            data=buf.getvalue(),
                            file_name="plot18_rV_sensitivity.png",
                            mime="image/png",
                            key="dl_btn_plot18"
                        )
            else:
                st.info("Geometric plots not available. Run geometric fitting first.")
        
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
                params_text = f"""PHENOMENOLOGICAL MODEL FITTING RESULTS
========================================
MSE: {st.session_state.fit_results['mse']:.6e}
RMSE: {st.session_state.fit_results['rmse']:.6e}
R²: {st.session_state.fit_results['r2']:.6f}
χ²: {st.session_state.fit_results['chi2']:.6f}
N points: {st.session_state.fit_results['N_points']}
Fitted parameters: {st.session_state.fit_results['n_free_params']}

MODEL PARAMETERS
========================================
[Acc] = {st.session_state.fit_results['params']['Acc']:.6f}
α·10⁶ = {st.session_state.fit_results['params']['alpha_1e6']:.6f}
β = {st.session_state.fit_results['params']['beta']:.6f}
ΔH = {st.session_state.fit_results['params']['dH']:.6f} kJ/mol
ΔS = {st.session_state.fit_results['params']['dS']:.6f} J/mol·K
pH₂O = {st.session_state.fit_results['params']['pH2O']:.6f}
Residue = {st.session_state.fit_results['params']['residue']:.6f}

Fitted parameters: {', '.join(st.session_state.fit_results['vary_params'])}
Fixed parameters: {', '.join([k for k, v in st.session_state.fit_results['fixed_params'].items() if v is not None])}

"""
                if st.session_state.geometric_fit_results is not None:
                    params_text += f"""

GEOMETRIC MODEL FITTING RESULTS
========================================
MSE: {st.session_state.geometric_fit_results['mse']:.6e}
RMSE: {st.session_state.geometric_fit_results['rmse']:.6e}
R²: {st.session_state.geometric_fit_results['r2']:.6f}
χ²: {st.session_state.geometric_fit_results['chi2']:.6f}
N points: {st.session_state.geometric_fit_results['N_points']}
Fitted parameters: {st.session_state.geometric_fit_results['n_free_params']}

GEOMETRIC PARAMETERS
========================================
Ion A: {st.session_state.geometric_params['ion_A']} (charge {st.session_state.geometric_params['charge_A']})
Ion B: {st.session_state.geometric_params['ion_B']} (charge {st.session_state.geometric_params['charge_B']})
Acceptor: {st.session_state.geometric_params['ion_Acc']} (charge {st.session_state.geometric_params['charge_Acc']})
Geometric model: {st.session_state.geometric_params['geometric_model_type']}
r_V = {st.session_state.geometric_fit_results['params'].get('r_V', st.session_state.geometric_params['r_V']):.6f} Å
r_OH = {st.session_state.geometric_fit_results['params'].get('r_OH', st.session_state.geometric_params['r_OH']):.6f} Å
r_O = 1.4000 Å (fixed)

COORDINATION NUMBERS FORMULAS
========================================
CN_B = 6 - [Acc] + [OH] = 6 - {st.session_state.fit_results['params']['Acc']:.4f} + [OH]
CN_A = 12 - 2[Acc] + 2[OH] = 12 - {2 * st.session_state.fit_results['params']['Acc']:.4f} + 2[OH]
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
    
    # Footer
    st.divider()
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center'>
            <p>Thermo-Mechanical Expansion Modeling Tool | For scientific publications | 600 DPI export</p>
            <p>Includes 19 comprehensive plots for detailed analysis (12 phenomenological + 7 geometric)</p>
            <p><strong>Geometric model</strong>: a = 2(r_B + r_anion) | CN_B = 6 - [Acc] + [OH] | CN_A = 12 - 2[Acc] + 2[OH]</p>
        </div>
        """,
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()
