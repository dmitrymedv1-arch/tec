import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit, fsolve
from scipy.stats import norm, probplot
import pandas as pd
import io
import warnings
import matplotlib.cm as cm
from matplotlib.colors import Normalize, LinearSegmentedColormap
from typing import Dict, Any, Optional, Tuple, List
import time
from statsmodels.graphics.tsaplots import plot_acf
from datetime import datetime
import json
import base64
from pathlib import Path
import re
import zipfile
from io import BytesIO

# Suppress warnings
warnings.filterwarnings('ignore')

# ============================================
# CONFIGURATION AND INITIALIZATION
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
    page_title="Thermo-Mechanical Expansion Modeling Suite",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern scientific design
st.markdown("""
<style>
    /* Main container styling */
    .main > div {
        padding: 2rem 1rem;
    }
    
    /* Card styling for sections */
    .card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 10px;
        padding: 20px;
        margin-bottom: 20px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        transition: transform 0.3s ease;
    }
    
    .card:hover {
        transform: translateY(-5px);
        box-shadow: 0 6px 12px rgba(0, 0, 0, 0.15);
    }
    
    /* Stage indicators */
    .stage-indicator {
        display: inline-block;
        padding: 8px 16px;
        border-radius: 20px;
        font-weight: bold;
        margin-right: 10px;
    }
    
    .stage-active {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
    }
    
    .stage-inactive {
        background: #f0f0f0;
        color: #666;
    }
    
    /* Progress bar styling */
    .stProgress > div > div > div > div {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    
    /* Metric cards */
    .metric-card {
        background: white;
        border-radius: 8px;
        padding: 15px;
        text-align: center;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
    }
    
    .metric-value {
        font-size: 24px;
        font-weight: bold;
        color: #667eea;
    }
    
    .metric-label {
        font-size: 14px;
        color: #666;
        margin-top: 5px;
    }
    
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 2px;
    }
    
    .stTabs [data-baseweb="tab"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 4px 4px 0 0;
        color: white;
        font-weight: bold;
        padding: 10px 20px;
        border: none;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #764ba2 0%, #667eea 100%);
        color: white;
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        font-weight: bold;
        border: none;
        padding: 10px 24px;
        border-radius: 25px;
        transition: all 0.3s ease;
        width: 100%;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(102, 126, 234, 0.4);
        color: white;
    }
    
    /* Download buttons */
    .stDownloadButton > button {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        color: white;
        font-weight: bold;
        border: none;
        padding: 8px 16px;
        border-radius: 20px;
        transition: all 0.3s ease;
    }
    
    .stDownloadButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(17, 153, 142, 0.4);
    }
    
    /* Info boxes */
    .info-box {
        background: linear-gradient(135deg, #f6f9fc 0%, #e6f0f5 100%);
        border-left: 4px solid #667eea;
        padding: 15px;
        border-radius: 0 8px 8px 0;
        margin: 10px 0;
    }
    
    /* Success box */
    .success-box {
        background: linear-gradient(135deg, #f0fff4 0%, #dcfce7 100%);
        border-left: 4px solid #10b981;
        padding: 15px;
        border-radius: 0 8px 8px 0;
        margin: 10px 0;
    }
    
    /* Warning box */
    .warning-box {
        background: linear-gradient(135deg, #fffaf0 0%, #ffedd5 100%);
        border-left: 4px solid #f59e0b;
        padding: 15px;
        border-radius: 0 8px 8px 0;
        margin: 10px 0;
    }
    
    /* Element selector styling */
    .element-selector {
        background: white;
        border-radius: 8px;
        padding: 15px;
        margin: 10px 0;
        border: 1px solid #e0e0e0;
    }
    
    .element-selector h4 {
        color: #667eea;
        margin-bottom: 10px;
    }
    
    /* Footer styling */
    .footer {
        text-align: center;
        padding: 20px;
        color: #666;
        font-size: 12px;
        border-top: 1px solid #e0e0e0;
        margin-top: 40px;
    }
    
    /* Tooltip */
    .tooltip {
        position: relative;
        display: inline-block;
        cursor: help;
    }
    
    .tooltip .tooltiptext {
        visibility: hidden;
        width: 200px;
        background-color: #555;
        color: #fff;
        text-align: center;
        border-radius: 6px;
        padding: 5px;
        position: absolute;
        z-index: 1;
        bottom: 125%;
        left: 50%;
        margin-left: -100px;
        opacity: 0;
        transition: opacity 0.3s;
    }
    
    .tooltip:hover .tooltiptext {
        visibility: visible;
        opacity: 1;
    }
    
    /* Loading animation */
    @keyframes pulse {
        0% { opacity: 0.6; }
        50% { opacity: 1; }
        100% { opacity: 0.6; }
    }
    
    .loading {
        animation: pulse 1.5s ease-in-out infinite;
        text-align: center;
        padding: 20px;
        color: #667eea;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# ============================================
# IONIC RADII DATABASE (from provided tables)
# ============================================
# Crystal radii in Angstroms
# Source: Provided tables with Key: R - recommended, * - primary value

IONIC_RADII = {
    # A-site (2+, coordination 12 - base for cubic perovskite)
    'Ca': {2: {12: 1.48}},
    'Sr': {2: {12: 1.58}},
    'Ba': {2: {12: 1.75}},
    
    # B-site (4+, coordination 6 - base for octahedron)
    'Ce': {4: {6: 1.01, 8: 1.11}},
    'Zr': {4: {6: 0.86, 8: 0.98}},
    'Sn': {4: {6: 0.83, 8: 0.95}},
    'Ti': {4: {6: 0.745, 5: 0.65, 8: 0.88}},
    'Hf': {4: {6: 0.85, 8: 0.97}},
    
    # M-site (3+, acceptor, coordination 6 - base)
    'Al': {3: {6: 0.675, 4: 0.53}},
    'Ga': {3: {6: 0.76, 4: 0.61}},
    'In': {3: {6: 0.94, 4: 0.76}},
    'Sc': {3: {6: 0.885, 8: 1.01}},
    'Y': {3: {6: 1.04, 8: 1.159}},
    'Yb': {3: {6: 1.008, 7: 1.065, 8: 1.125}},
    'Ho': {3: {6: 1.041, 8: 1.155}},
    'Dy': {3: {6: 1.052, 7: 1.11, 8: 1.167, 9: 1.223}},
    'Gd': {3: {6: 1.078, 8: 1.193}},
    'Sm': {3: {6: 1.098, 8: 1.219}},
    'Nd': {3: {6: 1.123, 8: 1.249}},
    'La': {3: {6: 1.172, 8: 1.3, 12: 1.5}},
    
    # Anions
    'O2-': {-2: {2: 1.21, 3: 1.22, 4: 1.24, 6: 1.26}},  # Crystal radius O2-
    'OH-': {-1: {2: 1.18, 3: 1.20, 4: 1.21, 6: 1.23}}   # Crystal radius OH-
}

# Shannon's original radii for comparison (ionic radii)
SHANNON_RADII = {
    'Ca': {2: {12: 1.34}},
    'Sr': {2: {12: 1.44}},
    'Ba': {2: {12: 1.61}},
    'Ce': {4: {6: 0.87, 8: 0.97}},
    'Zr': {4: {6: 0.72, 8: 0.84}},
    'Sn': {4: {6: 0.69, 8: 0.81}},
    'Ti': {4: {6: 0.605, 5: 0.51, 8: 0.74}},
    'Hf': {4: {6: 0.71, 8: 0.83}},
    'Al': {3: {6: 0.535, 4: 0.39}},
    'Ga': {3: {6: 0.62, 4: 0.47}},
    'In': {3: {6: 0.8, 4: 0.62}},
    'Sc': {3: {6: 0.745, 8: 0.87}},
    'Y': {3: {6: 0.9, 8: 1.019}},
    'Yb': {3: {6: 0.868, 7: 0.925, 8: 0.985}},
    'Ho': {3: {6: 0.901, 8: 1.015}},
    'Dy': {3: {6: 0.912, 7: 0.97, 8: 1.027, 9: 1.083}},
    'Gd': {3: {6: 0.938, 8: 1.053}},
    'Sm': {3: {6: 0.958, 8: 1.079}},
    'Nd': {3: {6: 0.983, 8: 1.109}},
    'La': {3: {6: 1.032, 8: 1.16, 12: 1.36}},
    'O2-': {-2: {2: 1.35, 3: 1.36, 4: 1.38, 6: 1.4}},
    'OH-': {-1: {2: 1.32, 3: 1.34, 4: 1.35, 6: 1.37}}
}

# ============================================
# IONIC RADII HELPER FUNCTIONS
# ============================================

def get_radius(element: str, charge: int, coordination: int, use_shannon: bool = False) -> float:
    """
    Get ionic/crystal radius for an element with given charge and coordination number.
    
    Parameters:
    -----------
    element : str
        Element symbol
    charge : int
        Ionic charge
    coordination : int
        Coordination number
    use_shannon : bool
        If True, use Shannon's ionic radii, otherwise use crystal radii
    
    Returns:
    --------
    float
        Radius in Angstroms
    """
    db = SHANNON_RADII if use_shannon else IONIC_RADII
    
    if element not in db:
        raise ValueError(f"Element '{element}' not found in database.")
    
    charge_dict = db[element]
    if charge not in charge_dict:
        # Try to find closest charge
        available_charges = sorted(charge_dict.keys())
        closest_charge = min(available_charges, key=lambda x: abs(x - charge))
        print(f"  Warning: Charge {charge} for {element} not found. Using charge {closest_charge}.")
        charge = closest_charge
    
    coord_dict = charge_dict[charge]
    if coordination not in coord_dict:
        # Find closest available coordination
        available_coords = sorted(coord_dict.keys())
        closest_coord = min(available_coords, key=lambda x: abs(x - coordination))
        print(f"  Warning: CN {coordination} for {element}^{charge} not found. Using CN {closest_coord}.")
        coordination = closest_coord
    
    return coord_dict[coordination]

def get_radius_interp(element: str, charge: int, coordination: float, 
                     use_shannon: bool = False, 
                     coord_points: Optional[Tuple[int, int]] = None) -> float:
    """
    Interpolate ionic radius between two coordination numbers.
    
    Parameters:
    -----------
    element : str
        Element symbol
    charge : int
        Ionic charge
    coordination : float
        Desired coordination number (can be fractional)
    use_shannon : bool
        If True, use Shannon's ionic radii
    coord_points : Tuple[int, int], optional
        Coordination numbers to interpolate between
    
    Returns:
    --------
    float
        Interpolated radius in Angstroms
    """
    db = SHANNON_RADII if use_shannon else IONIC_RADII
    
    if element not in db:
        raise ValueError(f"Element '{element}' not found in database.")
    
    # If coordination is integer and available, return exact value
    if abs(coordination - round(coordination)) < 1e-6:
        coord_int = int(round(coordination))
        try:
            return get_radius(element, charge, coord_int, use_shannon)
        except:
            pass
    
    # Need to interpolate
    if coord_points is None:
        # Get all available coordination numbers for this element/charge
        charge_dict = db[element]
        if charge not in charge_dict:
            available_charges = sorted(charge_dict.keys())
            closest_charge = min(available_charges, key=lambda x: abs(x - charge))
            charge = closest_charge
        
        coord_dict = charge_dict[charge]
        available_coords = sorted([c for c in coord_dict.keys() if c >= 4])
        
        if len(available_coords) >= 2:
            # Find two closest coordination numbers
            idx = np.searchsorted(available_coords, coordination)
            if idx == 0:
                coord_points = (available_coords[0], available_coords[1])
            elif idx >= len(available_coords):
                coord_points = (available_coords[-2], available_coords[-1])
            else:
                coord_points = (available_coords[idx-1], available_coords[idx])
        else:
            # Only one coordination number available
            return get_radius(element, charge, available_coords[0], use_shannon)
    
    # Linear interpolation
    r1 = get_radius(element, charge, coord_points[0], use_shannon)
    r2 = get_radius(element, charge, coord_points[1], use_shannon)
    
    # Avoid division by zero
    if coord_points[1] - coord_points[0] == 0:
        return r1
    
    radius = r1 + (r2 - r1) * (coordination - coord_points[0]) / (coord_points[1] - coord_points[0])
    return radius

# ============================================
# COMPOSITION VALIDATION AND PARSING
# ============================================

def validate_composition(A: str, B: str, M: str, x: float) -> Tuple[bool, str]:
    """
    Validate the chemical composition.
    
    Returns:
    --------
    Tuple[bool, str]
        (is_valid, error_message)
    """
    if A not in ['Ca', 'Sr', 'Ba']:
        return False, f"A-site element '{A}' not allowed. Choose from Ca, Sr, Ba."
    
    if B not in ['Ce', 'Zr', 'Ti', 'Sn', 'Hf']:
        return False, f"B-site element '{B}' not allowed. Choose from Ce, Zr, Ti, Sn, Hf."
    
    allowed_M = ['Al', 'Ga', 'In', 'Sc', 'Y', 'Yb', 'Ho', 'Dy', 'Gd', 'Sm', 'Nd', 'La']
    if M not in allowed_M:
        return False, f"M-site element '{M}' not allowed. Choose from {', '.join(allowed_M)}."
    
    if x <= 0 or x >= 1:
        return False, f"x must be between 0.01 and 0.99. Current value: {x}"
    
    return True, ""

def format_composition(A: str, B: str, M: str, x: float) -> str:
    """
    Format composition as LaTeX string.
    """
    if x == 0:
        return f"{A}{B}O₃"
    elif x == 1:
        return f"{A}{M}O₂.₅"
    else:
        # Format with proper subscripts
        b_sub = f"{1-x:.2f}".replace('.', '').lstrip('0')
        m_sub = f"{x:.2f}".replace('.', '').lstrip('0')
        return f"{A}{B} $_{{{b_sub}}}$ {M} $_{{{m_sub}}}$ O $_{{3-x/2}}$"

# ============================================
# INVERSE PROBLEM CALCULATIONS
# ============================================

class InverseProblemSolver:
    """
    Solver for the inverse problem: from fitted parameters to microscopic properties.
    """
    
    def __init__(self, A: str, B: str, M: str, x: float, 
                 fit_results: Dict[str, Any],
                 use_shannon: bool = False):
        """
        Initialize the inverse problem solver.
        
        Parameters:
        -----------
        A, B, M : str
            Element symbols
        x : float
            Acceptor concentration
        fit_results : Dict
            Results from the fitting procedure (Stage 1)
        use_shannon : bool
            If True, use Shannon's ionic radii for comparison
        """
        self.A = A
        self.B = B
        self.M = M
        self.x = x
        self.fit_results = fit_results
        self.use_shannon = use_shannon
        
        # Extract fitted parameters
        self.params = fit_results['params']
        self.beta_chem_exp = self.params['beta']
        self.alpha_exp = self.params['alpha_1e6'] * 1e-6
        self.dH_exp = self.params['dH']
        self.dS_exp = self.params['dS']
        self.pH2O = self.params['pH2O']
        self.Acc = self.params['Acc']
        
        # Constants
        self.r_O = get_radius('O2-', -2, 4, use_shannon)
        self.r_OH_table = get_radius('OH-', -1, 4, use_shannon)
        self.N_sites = 6  # Number of oxygen sites per formula unit
        
        # Calculate initial state (dry, maximum vacancies)
        self.delta_dry = x / 2.0
        self._calculate_dry_state()
        
        # Calculate fully hydrated state (all vacancies filled)
        self._calculate_hydrated_state()
        
        # Results dictionary
        self.results = {}
    
    def _calculate_dry_state(self):
        """Calculate parameters for dry state (maximum vacancies)."""
        # Coordination numbers in dry state (Equation 35 from Zuev et al.)
        self.CN_A_dry = max(12 - 4 * self.delta_dry, 6)
        self.CN_B_dry = max(6 - 2 * self.delta_dry, 4)
        self.CN_M_dry = self.CN_B_dry
        
        # Ionic radii in dry state
        self.r_A_dry = get_radius_interp(self.A, 2, self.CN_A_dry, self.use_shannon, coord_points=(8, 12))
        self.r_B_dry = get_radius_interp(self.B, 4, self.CN_B_dry, self.use_shannon, coord_points=(6, 8))
        self.r_M_dry = get_radius_interp(self.M, 3, self.CN_M_dry, self.use_shannon, coord_points=(6, 8))
        
        # Lattice contribution in dry state (proportional to unit cell size)
        self.L_dry = (self.r_A_dry + 
                     (1 - self.x) * self.r_B_dry + 
                     self.x * self.r_M_dry + 
                     (3 - self.delta_dry) * self.r_O)
    
    def _calculate_hydrated_state(self):
        """Calculate parameters for fully hydrated state (no vacancies, all protons)."""
        # Coordination numbers in hydrated state (delta = 0)
        self.CN_A_wet = 12
        self.CN_B_wet = 6
        self.CN_M_wet = 6
        
        # Ionic radii in hydrated state
        self.r_A_wet = get_radius_interp(self.A, 2, self.CN_A_wet, self.use_shannon, coord_points=(8, 12))
        self.r_B_wet = get_radius_interp(self.B, 4, self.CN_B_wet, self.use_shannon, coord_points=(6, 8))
        self.r_M_wet = get_radius_interp(self.M, 3, self.CN_M_wet, self.use_shannon, coord_points=(6, 8))
        
        # Lattice contribution from cations only
        self.L_cat_wet = self.r_A_wet + (1 - self.x) * self.r_B_wet + self.x * self.r_M_wet
    
    def calculate_effective_OH_radius(self) -> Dict[str, Any]:
        """
        Calculate the effective radius of OH- group in the lattice.
        
        From experimental data, we have:
        L_wet_exp = L_dry * (1 + beta_chem_exp * x)
        
        And theoretically:
        L_wet = L_cat_wet + (3 - x) * r_O + x * r_OH_eff
        
        Therefore:
        r_OH_eff = [L_dry * (1 + beta_chem_exp * x) - L_cat_wet - (3 - x) * r_O] / x
        """
        # Experimental wet state lattice contribution
        L_wet_exp = self.L_dry * (1 + self.beta_chem_exp * self.x)
        
        # Calculate effective OH radius
        self.r_OH_eff = (L_wet_exp - self.L_cat_wet - (3 - self.x) * self.r_O) / self.x
        
        # Compare with tabulated value
        deviation_pct = (self.r_OH_eff - self.r_OH_table) / self.r_OH_table * 100
        
        result = {
            'r_OH_table': self.r_OH_table,
            'r_OH_eff': self.r_OH_eff,
            'deviation_pct': deviation_pct,
            'L_dry': self.L_dry,
            'L_wet_exp': L_wet_exp,
            'L_cat_wet': self.L_cat_wet
        }
        self.results['oh_radius'] = result
        return result
    
    def calculate_vacancy_radius(self) -> Dict[str, Any]:
        """
        Calculate the effective radius of oxygen vacancy.
        
        In dry state: L_dry = L_cat_dry + (3 - delta_dry) * r_O + delta_dry * r_V_eff
        
        Therefore:
        r_V_eff = [L_dry - L_cat_dry - (3 - delta_dry) * r_O] / delta_dry
        """
        L_cat_dry = self.r_A_dry + (1 - self.x) * self.r_B_dry + self.x * self.r_M_dry
        
        # Calculate effective vacancy radius
        if self.delta_dry > 0:
            self.r_V_eff = (self.L_dry - L_cat_dry - (3 - self.delta_dry) * self.r_O) / self.delta_dry
        else:
            self.r_V_eff = self.r_O  # No vacancies, radius equals oxygen radius
        
        # Literature range for perovskites
        lit_min, lit_max = 1.16, 1.24
        lit_ratio_min, lit_ratio_max = 0.92, 0.98
        
        result = {
            'r_V_eff': self.r_V_eff,
            'r_O': self.r_O,
            'ratio_rV_rO': self.r_V_eff / self.r_O,
            'literature_range': (lit_min, lit_max),
            'literature_ratio_range': (lit_ratio_min, lit_ratio_max),
            'within_literature': (lit_min <= self.r_V_eff <= lit_max)
        }
        self.results['vacancy_radius'] = result
        return result
    
    def calculate_cation_radius_changes(self) -> Dict[str, Any]:
        """
        Calculate the changes in cation radii upon hydration.
        
        The total change from dry to wet state is:
        ΔL_total = L_dry * beta_chem_exp * x = ΔL_cat + x * (r_OH_eff - r_O) - (x/2) * (r_O - r_V_eff)
        
        Where ΔL_cat = (r_A_wet - r_A_dry) + (1-x)(r_B_wet - r_B_dry) + x(r_M_wet - r_M_dry)
        """
        # Total change from experiment
        delta_L_total = self.L_dry * self.beta_chem_exp * self.x
        
        # Contribution from anions (OH replacing O and V_O disappearing)
        delta_L_anion = self.x * (self.r_OH_eff - self.r_O) - (self.x / 2) * (self.r_O - self.r_V_eff)
        
        # Therefore, cation contribution
        delta_L_cation = delta_L_total - delta_L_anion
        
        # Individual cation radius changes
        delta_r_A = self.r_A_wet - self.r_A_dry
        delta_r_B = self.r_B_wet - self.r_B_dry
        delta_r_M = self.r_M_wet - self.r_M_dry
        
        # Sum should equal delta_L_cation
        delta_r_sum = delta_r_A + (1 - self.x) * delta_r_B + self.x * delta_r_M
        
        # Calculate relative changes
        rel_delta_r_A = delta_r_A / self.r_A_dry * 100
        rel_delta_r_B = delta_r_B / self.r_B_dry * 100
        rel_delta_r_M = delta_r_M / self.r_M_dry * 100
        
        result = {
            'delta_L_total': delta_L_total,
            'delta_L_anion': delta_L_anion,
            'delta_L_cation': delta_L_cation,
            'delta_r_A': delta_r_A,
            'delta_r_B': delta_r_B,
            'delta_r_M': delta_r_M,
            'rel_delta_r_A': rel_delta_r_A,
            'rel_delta_r_B': rel_delta_r_B,
            'rel_delta_r_M': rel_delta_r_M,
            'delta_r_sum': delta_r_sum,
            'consistency_check': abs(delta_r_sum - delta_L_cation) < 1e-6
        }
        self.results['cation_changes'] = result
        return result
    
    def calculate_coordination_dependence_coeffs(self) -> Dict[str, Any]:
        """
        Calculate the coefficients k_A and k_B that describe how radius changes with CN.
        
        From Zuev et al., Equation 36: r(CN) = a + m * CN
        The coefficient k = m / r_base represents the fractional change per CN unit.
        """
        # For A-site, interpolate between CN=8 and CN=12
        r_A_8 = get_radius_interp(self.A, 2, 8, self.use_shannon, coord_points=(8, 12))
        r_A_12 = get_radius_interp(self.A, 2, 12, self.use_shannon, coord_points=(8, 12))
        
        # Calculate slope m_A and base radius at CN=12
        self.m_A = (r_A_12 - r_A_8) / 4  # Δr per unit CN
        r_A_base = r_A_12
        
        # Experimental slope based on observed change
        delta_CN_A = self.CN_A_wet - self.CN_A_dry
        if abs(delta_CN_A) > 1e-6:
            self.m_A_exp = (self.r_A_wet - self.r_A_dry) / delta_CN_A
        else:
            self.m_A_exp = self.m_A
        
        # For B-site, interpolate between CN=6 and CN=8
        r_B_6 = get_radius_interp(self.B, 4, 6, self.use_shannon, coord_points=(6, 8))
        r_B_8 = get_radius_interp(self.B, 4, 8, self.use_shannon, coord_points=(6, 8))
        
        self.m_B = (r_B_8 - r_B_6) / 2
        r_B_base = r_B_6
        
        delta_CN_B = self.CN_B_wet - self.CN_B_dry
        if abs(delta_CN_B) > 1e-6:
            self.m_B_exp = (self.r_B_wet - self.r_B_dry) / delta_CN_B
        else:
            self.m_B_exp = self.m_B
        
        # For M-site (acceptor)
        r_M_6 = get_radius_interp(self.M, 3, 6, self.use_shannon, coord_points=(6, 8))
        r_M_8 = get_radius_interp(self.M, 3, 8, self.use_shannon, coord_points=(6, 8))
        
        self.m_M = (r_M_8 - r_M_6) / 2
        r_M_base = r_M_6
        
        delta_CN_M = self.CN_M_wet - self.CN_M_dry
        if abs(delta_CN_M) > 1e-6:
            self.m_M_exp = (self.r_M_wet - self.r_M_dry) / delta_CN_M
        else:
            self.m_M_exp = self.m_M
        
        # Calculate coefficients k = m / r_base
        self.k_A_theory = self.m_A / r_A_base
        self.k_B_theory = self.m_B / r_B_base
        self.k_M_theory = self.m_M / r_M_base
        
        self.k_A_exp = self.m_A_exp / r_A_base
        self.k_B_exp = self.m_B_exp / r_B_base
        self.k_M_exp = self.m_M_exp / r_M_base
        
        result = {
            'k_A_theory': self.k_A_theory,
            'k_A_exp': self.k_A_exp,
            'k_B_theory': self.k_B_theory,
            'k_B_exp': self.k_B_exp,
            'k_M_theory': self.k_M_theory,
            'k_M_exp': self.k_M_exp,
            'm_A_theory': self.m_A,
            'm_A_exp': self.m_A_exp,
            'm_B_theory': self.m_B,
            'm_B_exp': self.m_B_exp,
            'delta_CN_A': delta_CN_A,
            'delta_CN_B': delta_CN_B,
            'delta_CN_M': delta_CN_M,
            'r_A_base': r_A_base,
            'r_B_base': r_B_base,
            'r_M_base': r_M_base,
            # Add element symbols
            'A_element': self.A,
            'B_element': self.B,
            'M_element': self.M
        }
        self.results['cn_coefficients'] = result
        return result
    
    def calculate_association_energy(self) -> Dict[str, Any]:
        """
        Estimate defect association energy from temperature dependence of beta_chem.
        
        If beta_chem shows significant variation at low T, it may indicate defect association.
        """
        # This would require temperature-dependent beta_chem data
        # For now, provide a placeholder
        result = {
            'estimated': False,
            'message': 'Requires temperature-dependent beta_chem data from multi-temperature fits.'
        }
        self.results['association_energy'] = result
        return result
    
    def calculate_tolerance_factor(self) -> Dict[str, Any]:
        """
        Calculate Goldschmidt tolerance factor for the composition.
        
        t = (r_A + r_O) / sqrt(2) * (r_B + r_O)
        """
        r_A_12 = get_radius(self.A, 2, 12, self.use_shannon)
        r_B_6 = get_radius(self.B, 4, 6, self.use_shannon)
        r_O_6 = get_radius('O2-', -2, 6, self.use_shannon)
        
        t = (r_A_12 + r_O_6) / (np.sqrt(2) * (r_B_6 + r_O_6))
        
        # Interpretation
        if t > 1:
            stability = "Hexagonal or tetragonal (t > 1)"
        elif 0.9 <= t <= 1:
            stability = "Cubic perovskite (ideal)"
        elif 0.71 <= t < 0.9:
            stability = "Orthorhombic/rhombohedral distorted perovskite"
        else:
            stability = "Other structures (ilmenite, etc.)"
        
        result = {
            'tolerance_factor': t,
            'stability': stability,
            'r_A_12': r_A_12,
            'r_B_6': r_B_6,
            'r_O_6': r_O_6
        }
        self.results['tolerance_factor'] = result
        return result
    
    def calculate_theoretical_beta_chem(self) -> Dict[str, Any]:
        """
        Calculate theoretical beta_chem based on ionic radii model.
        
        beta_chem_theory = [L_wet_theory - L_dry] / (L_dry * x)
        where L_wet_theory uses tabulated radii.
        """
        L_wet_theory = (self.r_A_wet + 
                       (1 - self.x) * self.r_B_wet + 
                       self.x * self.r_M_wet + 
                       (3 - self.x) * self.r_O + 
                       self.x * self.r_OH_table)
        
        beta_chem_theory = (L_wet_theory - self.L_dry) / (self.L_dry * self.x)
        
        # Compare with experimental
        deviation_pct = (beta_chem_theory - self.beta_chem_exp) / self.beta_chem_exp * 100
        
        result = {
            'beta_chem_theory': beta_chem_theory,
            'beta_chem_exp': self.beta_chem_exp,
            'deviation_pct': deviation_pct,
            'L_wet_theory': L_wet_theory,
            'L_wet_exp': self.L_dry * (1 + self.beta_chem_exp * self.x)
        }
        self.results['theoretical_beta'] = result
        return result
    
    def run_all_calculations(self) -> Dict[str, Any]:
        """
        Run all inverse problem calculations.
        """
        self.calculate_effective_OH_radius()
        self.calculate_vacancy_radius()
        self.calculate_cation_radius_changes()
        self.calculate_coordination_dependence_coeffs()
        self.calculate_association_energy()
        self.calculate_tolerance_factor()
        self.calculate_theoretical_beta_chem()
        
        # Add composition information to results
        self.results['A_element'] = self.A
        self.results['B_element'] = self.B
        self.results['M_element'] = self.M
        self.results['x'] = self.x
        
        return self.results

# ============================================
# INVERSE PROBLEM PLOTTING FUNCTIONS
# ============================================

@st.cache_data(ttl=3600, show_spinner=False)
def create_inverse_plot1_cached(inverse_results: Dict[str, Any], 
                                composition: str,
                                style: Dict[str, Any]) -> plt.Figure:
    """
    Create plot 13: Ionic radii comparison (tabulated vs experimental/effective)
    """
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(10, 8))
    
    # Extract results
    oh_res = inverse_results.get('oh_radius', {})
    vac_res = inverse_results.get('vacancy_radius', {})
    cation_res = inverse_results.get('cation_changes', {})
    cn_res = inverse_results.get('cn_coefficients', {})
    
    # Plot 1: OH- radius comparison
    if oh_res:
        categories = ['Tabulated OH⁻', 'Effective OH⁻']
        values = [oh_res.get('r_OH_table', 0), oh_res.get('r_OH_eff', 0)]
        colors = [style.get('bar_thermal_color', '#1f77b4'), 
                  style.get('bar_chemical_color', '#d62728')]
        
        bars = ax1.bar(categories, values, color=colors, alpha=0.7, edgecolor='black')
        ax1.axhline(y=oh_res.get('r_OH_table', 0), color='gray', linestyle='--', alpha=0.5)
        ax1.set_ylabel('Radius (Å)', fontweight='bold')
        ax1.set_title('OH⁻ Radius: Tabulated vs Effective', fontweight='bold')
        
        # Add value labels
        for bar, val in zip(bars, values):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{val:.3f} Å', ha='center', va='bottom', fontweight='bold')
        
        # Add deviation annotation
        dev = oh_res.get('deviation_pct', 0)
        ax1.text(0.5, 0.95, f'Deviation: {dev:+.1f}%',
                transform=ax1.transAxes, ha='center', va='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Plot 2: Vacancy radius comparison
    if vac_res:
        r_V = vac_res.get('r_V_eff', 0)
        r_O = vac_res.get('r_O', 0)
        ratio = vac_res.get('ratio_rV_rO', 0)
        
        categories = ['O²⁻', 'V_O•• (effective)']
        values = [r_O, r_V]
        colors = [style.get('thermal_line_color', '#1f77b4'), 
                  style.get('chemical_line_color', '#d62728')]
        
        bars = ax2.bar(categories, values, color=colors, alpha=0.7, edgecolor='black')
        ax2.set_ylabel('Radius (Å)', fontweight='bold')
        ax2.set_title(f'Oxygen Vacancy Radius (r_V/r_O = {ratio:.3f})', fontweight='bold')
        
        # Add value labels
        for bar, val in zip(bars, values):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{val:.3f} Å', ha='center', va='bottom', fontweight='bold')
        
        # Literature range
        lit_min, lit_max = vac_res.get('literature_range', (1.16, 1.24))
        ax2.axhspan(lit_min, lit_max, alpha=0.2, color='green', label='Literature range')
        ax2.legend(loc='upper right')
    
    # Plot 3: Cation radius changes
    if cation_res:
        cations = ['A', 'B', 'M']
        delta_r = [cation_res.get('delta_r_A', 0) * 1000,
                   cation_res.get('delta_r_B', 0) * 1000,
                   cation_res.get('delta_r_M', 0) * 1000]
        
        colors = [style.get('thermal_line_color', '#1f77b4'),
                  style.get('model_line_color', '#000000'),
                  style.get('chemical_line_color', '#d62728')]
        
        bars = ax3.bar(cations, delta_r, color=colors, alpha=0.7, edgecolor='black')
        ax3.set_ylabel('Δr (10⁻³ Å)', fontweight='bold')
        ax3.set_xlabel('Cation Site', fontweight='bold')
        ax3.set_title('Cation Radius Change upon Hydration', fontweight='bold')
        
        # Add value labels
        for bar, val in zip(bars, delta_r):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., 
                    height + 0.1 if height > 0 else height - 0.2,
                    f'{val:.2f}', ha='center', va='bottom' if height > 0 else 'top',
                    fontweight='bold')
        
        # Add total
        total_delta = cation_res.get('delta_L_cation', 0) * 1000
        ax3.text(0.5, 0.95, f'Total ΔL_cation = {total_delta:.2f} ×10⁻³',
                transform=ax3.transAxes, ha='center', va='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Plot 4: Coordination dependence coefficients
    if cn_res:
        sites = ['A-site', 'B-site', 'M-site']
        k_theory = [cn_res.get('k_A_theory', 0) * 1000,
                    cn_res.get('k_B_theory', 0) * 1000,
                    cn_res.get('k_M_theory', 0) * 1000]
        k_exp = [cn_res.get('k_A_exp', 0) * 1000,
                 cn_res.get('k_B_exp', 0) * 1000,
                 cn_res.get('k_M_exp', 0) * 1000]
        
        x = np.arange(len(sites))
        width = 0.35
        
        bars1 = ax4.bar(x - width/2, k_theory, width, label='Theoretical',
                        color=style.get('thermal_line_color', '#1f77b4'),
                        alpha=0.7, edgecolor='black')
        bars2 = ax4.bar(x + width/2, k_exp, width, label='Experimental',
                        color=style.get('chemical_line_color', '#d62728'),
                        alpha=0.7, edgecolor='black')
        
        ax4.set_ylabel('k × 10³ (Δr/r per CN)', fontweight='bold')
        ax4.set_xlabel('Cation Site', fontweight='bold')
        ax4.set_title('Coordination Dependence Coefficients', fontweight='bold')
        ax4.set_xticks(x)
        ax4.set_xticklabels(sites)
        ax4.legend(loc='upper right')
        ax4.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
    
    plt.suptitle(f'Ionic Radii Analysis: {composition}', fontweight='bold', fontsize=14)
    plt.tight_layout()
    fig.set_dpi(600)
    return fig

@st.cache_data(ttl=3600, show_spinner=False)
def create_inverse_plot2_cached(inverse_results: Dict[str, Any],
                                composition: str,
                                style: Dict[str, Any],
                                A: str, B: str, M: str) -> plt.Figure:
    """
    Create plot 14: Defect parameters comparison
    """
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(10, 8))
    
    theo_res = inverse_results.get('theoretical_beta', {})
    tol_res = inverse_results.get('tolerance_factor', {})
    
    # Plot 1: Beta chemical comparison
    if theo_res:
        categories = ['Experimental β', 'Theoretical β']
        values = [theo_res.get('beta_chem_exp', 0) * 1000,
                  theo_res.get('beta_chem_theory', 0) * 1000]
        
        bars = ax1.bar(categories, values, color=[style.get('point_color', '#1f77b4'),
                                                   style.get('model_line_color', '#000000')],
                       alpha=0.7, edgecolor='black')
        ax1.set_ylabel('β × 10³', fontweight='bold')
        ax1.set_title('Chemical Expansion Coefficient: Experiment vs Theory', fontweight='bold')
        
        for bar, val in zip(bars, values):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                    f'{val:.3f}', ha='center', va='bottom', fontweight='bold')
        
        dev = theo_res.get('deviation_pct', 0)
        ax1.text(0.5, 0.95, f'Deviation: {dev:+.1f}%',
                transform=ax1.transAxes, ha='center', va='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Plot 2: Lattice contributions
    if theo_res:
        L_dry = theo_res.get('L_dry', 0)
        categories = ['Dry state', 'Hydrated (theory)', 'Hydrated (exp)']
        values = [L_dry,
                  theo_res.get('L_wet_theory', 0),
                  theo_res.get('L_wet_exp', 0)]
        
        bars = ax2.bar(categories, values, color=[style.get('thermal_line_color', '#1f77b4'),
                                                   style.get('model_line_color', '#000000'),
                                                   style.get('point_color', '#1f77b4')],
                       alpha=0.7, edgecolor='black')
        ax2.set_ylabel('Lattice Contribution (Å)', fontweight='bold')
        ax2.set_title('Lattice Sum (Σ r_i)', fontweight='bold')
        
        for bar, val in zip(bars, values):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{val:.3f} Å', ha='center', va='bottom', fontweight='bold', fontsize=8)
    
    # Plot 3: Tolerance factor visualization
    if tol_res:
        t = tol_res.get('tolerance_factor', 1)
        r_A = tol_res.get('r_A_12', 0)
        r_B = tol_res.get('r_B_6', 0)
        r_O = tol_res.get('r_O_6', 0)
        
        from matplotlib.patches import Circle
        
        # Draw circles representing ions
        ax3.add_patch(Circle((0.2, 0.6), r_A/5, color=style.get('thermal_line_color', '#1f77b4'),
                             alpha=0.8, label=f'A ({A})'))
        ax3.add_patch(Circle((0.5, 0.3), r_B/5, color=style.get('chemical_line_color', '#d62728'),
                              alpha=0.8, label=f'B ({B}/{M})'))
        ax3.add_patch(Circle((0.8, 0.6), r_O/5, color='red', alpha=0.8, label='O'))
        
        ax3.set_xlim(0, 1)
        ax3.set_ylim(0, 1)
        ax3.set_aspect('equal')
        ax3.axis('off')
        ax3.set_title(f'Tolerance Factor t = {t:.3f}', fontweight='bold')
        ax3.legend(loc='upper right', fontsize=8)
        
        stability = tol_res.get('stability', 'Unknown')
        ax3.text(0.5, 0.05, stability, ha='center', va='center',
                transform=ax3.transAxes, fontsize=9,
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Plot 4: Comparison with literature
    vac_res = inverse_results.get('vacancy_radius', {})
    if vac_res:
        r_V = vac_res.get('r_V_eff', 0)
        lit_min, lit_max = vac_res.get('literature_range', (1.16, 1.24))
        
        categories = ['Literature min', 'This work', 'Literature max']
        values = [lit_min, r_V, lit_max]
        colors = ['gray', style.get('chemical_line_color', '#d62728'), 'gray']
        
        bars = ax4.bar(categories, values, color=colors, alpha=0.7, edgecolor='black')
        ax4.set_ylabel('r_V_O (Å)', fontweight='bold')
        ax4.set_title('Vacancy Radius: Comparison with Literature', fontweight='bold')
        
        for bar, val in zip(bars, values):
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{val:.3f} Å', ha='center', va='bottom', fontweight='bold')
        
        within = vac_res.get('within_literature', False)
        status = "✓ Within literature range" if within else "✗ Outside literature range"
        ax4.text(0.5, 0.95, status,
                transform=ax4.transAxes, ha='center', va='top',
                color='green' if within else 'red', fontweight='bold',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.suptitle(f'Defect Parameters: {composition}', fontweight='bold', fontsize=14)
    plt.tight_layout()
    fig.set_dpi(600)
    return fig

@st.cache_data(ttl=3600, show_spinner=False)
def create_inverse_plot3_cached(inverse_results: Dict[str, Any],
                                composition: str,
                                style: Dict[str, Any]) -> plt.Figure:
    """
    Create plot 15: Structural insights and correlations
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Get the actual element symbols from the results
    # They should be stored in the results dictionary
    A_element = inverse_results.get('A_element', 'Ba')
    B_element = inverse_results.get('B_element', 'Zr')
    M_element = inverse_results.get('M_element', 'Y')
    x_value = inverse_results.get('x', 0.1)
    
    cn_res = inverse_results.get('cn_coefficients', {})
    cation_res = inverse_results.get('cation_changes', {})
    
    # Plot 1: Radius vs Coordination Number
    if cn_res:
        # A-site
        cn_range = np.linspace(6, 12, 50)
        r_A = [get_radius_interp(A_element, 2, cn, False, coord_points=(8, 12)) 
               for cn in cn_range]
        ax1.plot(cn_range, r_A, '-', color=style.get('thermal_line_color', '#1f77b4'),
                linewidth=2, label=f'A-site ({A_element})')
        
        # B-site
        cn_range_b = np.linspace(4, 8, 50)
        r_B = [get_radius_interp(B_element, 4, cn, False, coord_points=(6, 8)) 
               for cn in cn_range_b]
        ax1.plot(cn_range_b, r_B, '-', color=style.get('chemical_line_color', '#d62728'),
                linewidth=2, label=f'B-site ({B_element})')
        
        # M-site
        r_M = [get_radius_interp(M_element, 3, cn, False, coord_points=(6, 8)) 
               for cn in cn_range_b]
        ax1.plot(cn_range_b, r_M, '--', color=style.get('model_line_color', '#000000'),
                linewidth=2, label=f'M-site ({M_element})')
        
        # Mark experimental points if available
        if 'CN_A_dry' in cn_res and 'r_A_dry' in cn_res:
            ax1.plot(cn_res.get('CN_A_dry', 0), cn_res.get('r_A_dry', 0), 'o',
                    color=style.get('point_color', '#1f77b4'), markersize=8,
                    markeredgecolor='black', label='Dry state')
        if 'CN_A_wet' in cn_res and 'r_A_wet' in cn_res:
            ax1.plot(cn_res.get('CN_A_wet', 0), cn_res.get('r_A_wet', 0), 's',
                    color=style.get('point_color', '#1f77b4'), markersize=8,
                    markeredgecolor='black', label='Hydrated state')
        
        ax1.set_xlabel('Coordination Number', fontweight='bold')
        ax1.set_ylabel('Ionic Radius (Å)', fontweight='bold')
        ax1.set_title('Radius Dependence on Coordination Number', fontweight='bold')
        ax1.legend(loc='best')
        ax1.grid(True, alpha=0.3)
    
    # Plot 2: Radial distribution
    if cation_res:
        # Create a pie chart of contributions
        labels = [f'A-site ({A_element})', f'B-site ({B_element})', 
                  f'M-site ({M_element})', 'Oxygen/Anions']
        sizes = [abs(cation_res.get('delta_r_A', 0)),
                 abs((1 - x_value) * cation_res.get('delta_r_B', 0)),
                 abs(x_value * cation_res.get('delta_r_M', 0)),
                 abs(cation_res.get('delta_L_anion', 0))]
        
        # Filter out zero values
        non_zero = [(l, s) for l, s in zip(labels, sizes) if s > 1e-10]
        if non_zero:
            labels, sizes = zip(*non_zero)
            
            colors = [style.get('thermal_line_color', '#1f77b4'),
                      style.get('chemical_line_color', '#d62728'),
                      style.get('model_line_color', '#000000'),
                      'gray']
            # Use only as many colors as we have non-zero sizes
            colors = colors[:len(labels)]
            
            wedges, texts, autotexts = ax2.pie(sizes, labels=labels, colors=colors,
                                               autopct='%1.1f%%', startangle=90,
                                               wedgeprops={'edgecolor': 'black', 'linewidth': 1})
            for autotext in autotexts:
                autotext.set_color('white')
                autotext.set_fontweight('bold')
        else:
            ax2.text(0.5, 0.5, 'No significant\ncontributions', 
                    ha='center', va='center', transform=ax2.transAxes)
        
        ax2.set_title('Contribution to Chemical Expansion', fontweight='bold')
    
    plt.suptitle(f'Structural Insights: {composition}', fontweight='bold', fontsize=14)
    plt.tight_layout()
    fig.set_dpi(600)
    return fig

@st.cache_data(ttl=3600, show_spinner=False)
def create_inverse_report_cached(inverse_results: Dict[str, Any],
                                 composition: str,
                                 params: Dict[str, Any]) -> str:
    """
    Generate a text report of inverse problem results.
    """
    # Get element symbols from inverse_results
    A_element = inverse_results.get('A_element', 'A')
    B_element = inverse_results.get('B_element', 'B')
    M_element = inverse_results.get('M_element', 'M')
    x_value = inverse_results.get('x', 0.1)
    
    report = []
    report.append("=" * 80)
    report.append("INVERSE PROBLEM ANALYSIS REPORT")
    report.append("=" * 80)
    report.append(f"Composition: {composition}")
    report.append(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append("")
    
    # Experimental parameters
    report.append("-" * 40)
    report.append("EXPERIMENTAL PARAMETERS (from fitting)")
    report.append("-" * 40)
    report.append(f"Acceptor concentration [Acc] = {params.get('Acc', 0):.6f}")
    report.append(f"Thermal expansion coefficient α = {params.get('alpha_1e6', 0):.3f} ×10⁻⁶ K⁻¹")
    report.append(f"Chemical expansion coefficient β = {params.get('beta', 0):.6f}")
    report.append(f"Enthalpy ΔH = {params.get('dH', 0):.2f} kJ/mol")
    report.append(f"Entropy ΔS = {params.get('dS', 0):.2f} J/mol·K")
    report.append(f"Water vapor pressure pH₂O = {params.get('pH2O', 0):.4f} atm")
    report.append("")
    
    # OH- radius
    oh_res = inverse_results.get('oh_radius', {})
    if oh_res:
        report.append("-" * 40)
        report.append("OH- RADIUS ANALYSIS")
        report.append("-" * 40)
        report.append(f"Tabulated OH- radius: {oh_res.get('r_OH_table', 0):.4f} Å")
        report.append(f"Effective OH- radius in lattice: {oh_res.get('r_OH_eff', 0):.4f} Å")
        report.append(f"Deviation: {oh_res.get('deviation_pct', 0):+.2f}%")
        report.append("")
    
    # Vacancy radius
    vac_res = inverse_results.get('vacancy_radius', {})
    if vac_res:
        report.append("-" * 40)
        report.append("OXYGEN VACANCY RADIUS")
        report.append("-" * 40)
        report.append(f"O²- radius (CN=4): {vac_res.get('r_O', 0):.4f} Å")
        report.append(f"Effective V_O•• radius: {vac_res.get('r_V_eff', 0):.4f} Å")
        report.append(f"Ratio r_V/r_O: {vac_res.get('ratio_rV_rO', 0):.4f}")
        
        lit_min, lit_max = vac_res.get('literature_range', (1.16, 1.24))
        report.append(f"Literature range: {lit_min:.2f} - {lit_max:.2f} Å")
        report.append(f"Within literature range: {vac_res.get('within_literature', False)}")
        report.append("")
    
    # Cation changes
    cat_res = inverse_results.get('cation_changes', {})
    if cat_res:
        report.append("-" * 40)
        report.append("CATION RADIUS CHANGES UPON HYDRATION")
        report.append("-" * 40)
        report.append(f"A-site ({A_element}) change: {cat_res.get('delta_r_A', 0)*1000:.3f} ×10⁻³ Å ({cat_res.get('rel_delta_r_A', 0):+.2f}%)")
        report.append(f"B-site ({B_element}) change: {cat_res.get('delta_r_B', 0)*1000:.3f} ×10⁻³ Å ({cat_res.get('rel_delta_r_B', 0):+.2f}%)")
        report.append(f"M-site ({M_element}) change: {cat_res.get('delta_r_M', 0)*1000:.3f} ×10⁻³ Å ({cat_res.get('rel_delta_r_M', 0):+.2f}%)")
        report.append(f"Total cation contribution: {cat_res.get('delta_L_cation', 0)*1000:.3f} ×10⁻³ Å")
        report.append(f"Total anion contribution: {cat_res.get('delta_L_anion', 0)*1000:.3f} ×10⁻³ Å")
        report.append(f"Consistency check: {cat_res.get('consistency_check', False)}")
        report.append("")
    
    # Coordination dependence
    cn_res = inverse_results.get('cn_coefficients', {})
    if cn_res:
        report.append("-" * 40)
        report.append("COORDINATION DEPENDENCE COEFFICIENTS")
        report.append("-" * 40)
        report.append(f"A-site: k_theory = {cn_res.get('k_A_theory', 0)*1000:.3f} ×10⁻³, k_exp = {cn_res.get('k_A_exp', 0)*1000:.3f} ×10⁻³")
        report.append(f"B-site: k_theory = {cn_res.get('k_B_theory', 0)*1000:.3f} ×10⁻³, k_exp = {cn_res.get('k_B_exp', 0)*1000:.3f} ×10⁻³")
        report.append(f"M-site: k_theory = {cn_res.get('k_M_theory', 0)*1000:.3f} ×10⁻³, k_exp = {cn_res.get('k_M_exp', 0)*1000:.3f} ×10⁻³")
        report.append("")
    
    # Theoretical beta
    theo_res = inverse_results.get('theoretical_beta', {})
    if theo_res:
        report.append("-" * 40)
        report.append("THEORETICAL VS EXPERIMENTAL β")
        report.append("-" * 40)
        report.append(f"Experimental β: {theo_res.get('beta_chem_exp', 0)*1000:.3f} ×10⁻³")
        report.append(f"Theoretical β (ionic radii model): {theo_res.get('beta_chem_theory', 0)*1000:.3f} ×10⁻³")
        report.append(f"Deviation: {theo_res.get('deviation_pct', 0):+.2f}%")
        report.append("")
    
    # Tolerance factor
    tol_res = inverse_results.get('tolerance_factor', {})
    if tol_res:
        report.append("-" * 40)
        report.append("GOLDSCHMIDT TOLERANCE FACTOR")
        report.append("-" * 40)
        report.append(f"t = {tol_res.get('tolerance_factor', 0):.4f}")
        report.append(f"Structural stability: {tol_res.get('stability', 'Unknown')}")
        report.append("")
    
    report.append("=" * 80)
    report.append("END OF REPORT")
    report.append("=" * 80)
    
    return "\n".join(report)

# ============================================
# OPTIMIZED CACHED FUNCTIONS (STAGE 1)
# ============================================

@st.cache_data(ttl=3600, show_spinner=False)
def parse_data_cached(data_string: str) -> np.ndarray:
    """Parse data with various separators (cached)"""
    lines = data_string.strip().split('\n')
    data = []
    
    for line in lines:
        if not line.strip() or line.strip().startswith(('#', '%', '//')):
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
    oh = -A/2 + np.sqrt(A * Acc + (A/2)**2)
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
        
        # Calculate TEC for both experimental and model data
        tec_exp = calculate_tec_cached(T_data, dl_data)
        tec_model = calculate_tec_cached(T_data, dl_model)
        
        # Calculate proton concentration
        oh, _ = calculate_oh_cached(T_data, result_params['Acc'], result_params['dH'], 
                                  result_params['dS'], result_params['pH2O'])
        
        # Calculate individual contributions
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

# ============================================
# PLOTTING FUNCTIONS (STAGE 1)
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
    
    # Determine heating or cooling direction
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
    """Create plot 7: Fractional contribution of thermal and chemical components (cached)"""
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
    """Create plot 8: Sensitivity to key parameter variations (cached)"""
    T = fit_results['T_data']
    base_dl = fit_results['dl_model']
    params = fit_results['params']
    T_start = fit_results['T_start']
    oh_start = fit_results['oh_start']
    
    fig, axes = plt.subplots(2, 2, figsize=(8, 6))
    axes = axes.flatten()
    
    # Variation of α (±10%, ±5%, 0%, +5%, +10%)
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
    
    # Variation of β (±10%, ±5%, 0%, +5%, +10%)
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
    
    # Variation of ΔH (±5, ±2, 0, +2, +5 kJ/mol)
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
    
    # Variation of pH₂O (logarithmic scale)
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
    """Create plot 9: Phase portrait of the system (cached)"""
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
    """Create plot 10: Statistical analysis of residuals (cached)"""
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
    """Create plot 11: OH concentration dynamics and derivatives (cached)"""
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
    """Create plot 12: Parameter correlation matrix (cached)"""
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
# HELPER FUNCTIONS
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

def initialize_session_state():
    """Initialize all session state variables"""
    defaults = {
        'experimental_data': None,
        'fit_results': None,
        'last_fit_params': None,
        'inverse_results': None,
        'current_stage': 1,
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
        'composition': {
            'A': 'Ba',
            'B': 'Zr',
            'M': 'Y',
            'x': 0.1
        },
        'data_loaded': False,
        'plots_generated': False,
        'fitting_complete': False,
        'inverse_complete': False,
        'fit_timestamp': 0
    }
    
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

# Initialize session state
initialize_session_state()

# ============================================
# MAIN APPLICATION INTERFACE
# ============================================

def main():
    # Header with stage indicators
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.title("🔬 Thermo-Mechanical Expansion Modeling Suite")
        st.markdown("#### For Proton-Conducting Perovskites")
    
    # Stage indicators
    col1, col2, col3, col4 = st.columns([1, 2, 2, 1])
    with col2:
        stage1_class = "stage-active" if st.session_state.current_stage == 1 else "stage-inactive"
        st.markdown(f'<span class="stage-indicator {stage1_class}">Stage 1: Data Fitting</span>', 
                   unsafe_allow_html=True)
    with col3:
        stage2_class = "stage-active" if st.session_state.current_stage == 2 else "stage-inactive"
        disabled = "" if st.session_state.fitting_complete else " (requires Stage 1)"
        st.markdown(f'<span class="stage-indicator {stage2_class}">Stage 2: Inverse Analysis{disabled}</span>', 
                   unsafe_allow_html=True)
    
    st.divider()
    
    # Sidebar
    with st.sidebar:
        st.markdown("### ⚙️ Control Panel")
        
        # Stage 1: Data Input
        with st.expander("📁 **Stage 1: Data Input**", expanded=True):
            data_option = st.radio(
                "Data input method:",
                ["Manual entry", "File upload", "Example data"],
                key="data_option"
            )
            
            if data_option == "Manual entry":
                data_text = st.text_area(
                    "Enter data (Temperature ΔL/L₀):",
                    value="20\t0.0045\n40\t0.004787988\n60\t0.005075916\n80\t0.005363555\n100\t0.005650042\n120\t0.005932612\n140\t0.006203565",
                    height=150
                )
                if st.button("📥 Load Data", type="primary", use_container_width=True):
                    try:
                        st.session_state.experimental_data = parse_data_cached(data_text)
                        st.session_state.data_loaded = True
                        st.session_state.fitting_complete = False
                        st.session_state.fit_results = None
                        st.session_state.inverse_complete = False
                        st.success(f"✅ Loaded {len(st.session_state.experimental_data)} data points")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error: {str(e)}")
            
            elif data_option == "File upload":
                uploaded_file = st.file_uploader("Choose a file", type=['txt', 'csv', 'dat'])
                if uploaded_file is not None:
                    try:
                        data_text = uploaded_file.getvalue().decode()
                        st.session_state.experimental_data = parse_data_cached(data_text)
                        st.session_state.data_loaded = True
                        st.session_state.fitting_complete = False
                        st.session_state.fit_results = None
                        st.session_state.inverse_complete = False
                        st.success(f"✅ Loaded {len(st.session_state.experimental_data)} data points")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error: {str(e)}")
            
            else:  # Example data
                if st.button("📥 Load Example Data", type="primary", use_container_width=True):
                    example_data = """20\t0.0045
40\t0.004787988
60\t0.005075916
80\t0.005363555
100\t0.005650042
120\t0.005932612
140\t0.006203565"""
                    st.session_state.experimental_data = parse_data_cached(example_data)
                    st.session_state.data_loaded = True
                    st.session_state.fitting_complete = False
                    st.session_state.fit_results = None
                    st.session_state.inverse_complete = False
                    st.success("✅ Loaded example data")
                    st.rerun()
        
        # Stage 1: Model Parameters
        if st.session_state.data_loaded:
            with st.expander("🔧 **Stage 1: Model Parameters**", expanded=True):
                st.markdown("Check 'Fix' to keep parameter constant during fitting")
                
                # Use a unique form key based on session state to prevent duplicates
                form_key = f"model_params_form_{hash(str(st.session_state.model_params))}_{st.session_state.get('fit_timestamp', 0)}"
                
                with st.form(form_key):
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        acc_value = st.number_input(
                            "[Acc]", 
                            value=st.session_state.model_params['Acc']['value'],
                            step=0.01, format="%.4f",
                            key=f"acc_input_{st.session_state.model_params['Acc']['value']}_{st.session_state.get('fit_timestamp', 0)}"
                        )
                        acc_fixed = st.checkbox(
                            "Fix", 
                            value=st.session_state.model_params['Acc']['fixed'],
                            key=f"acc_fix_{st.session_state.model_params['Acc']['fixed']}_{st.session_state.get('fit_timestamp', 0)}"
                        )
                        
                        alpha_value = st.number_input(
                            "α·10⁶", 
                            value=st.session_state.model_params['alpha_1e6']['value'],
                            step=0.1, format="%.4f",
                            key=f"alpha_input_{st.session_state.model_params['alpha_1e6']['value']}_{st.session_state.get('fit_timestamp', 0)}"
                        )
                        alpha_fixed = st.checkbox(
                            "Fix", 
                            value=st.session_state.model_params['alpha_1e6']['fixed'],
                            key=f"alpha_fix_{st.session_state.model_params['alpha_1e6']['fixed']}_{st.session_state.get('fit_timestamp', 0)}"
                        )
                        
                        beta_value = st.number_input(
                            "β", 
                            value=st.session_state.model_params['beta']['value'],
                            step=0.001, format="%.4f",
                            key=f"beta_input_{st.session_state.model_params['beta']['value']}_{st.session_state.get('fit_timestamp', 0)}"
                        )
                        beta_fixed = st.checkbox(
                            "Fix", 
                            value=st.session_state.model_params['beta']['fixed'],
                            key=f"beta_fix_{st.session_state.model_params['beta']['fixed']}_{st.session_state.get('fit_timestamp', 0)}"
                        )
                        
                        dH_value = st.number_input(
                            "ΔH (kJ/mol)", 
                            value=st.session_state.model_params['dH']['value'],
                            step=1.0, format="%.2f",
                            key=f"dH_input_{st.session_state.model_params['dH']['value']}_{st.session_state.get('fit_timestamp', 0)}"
                        )
                        dH_fixed = st.checkbox(
                            "Fix", 
                            value=st.session_state.model_params['dH']['fixed'],
                            key=f"dH_fix_{st.session_state.model_params['dH']['fixed']}_{st.session_state.get('fit_timestamp', 0)}"
                        )
                    
                    with col2:
                        dS_value = st.number_input(
                            "ΔS (J/mol·K)", 
                            value=st.session_state.model_params['dS']['value'],
                            step=1.0, format="%.2f",
                            key=f"dS_input_{st.session_state.model_params['dS']['value']}_{st.session_state.get('fit_timestamp', 0)}"
                        )
                        dS_fixed = st.checkbox(
                            "Fix", 
                            value=st.session_state.model_params['dS']['fixed'],
                            key=f"dS_fix_{st.session_state.model_params['dS']['fixed']}_{st.session_state.get('fit_timestamp', 0)}"
                        )
                        
                        pH2O_value = st.number_input(
                            "pH₂O", 
                            value=st.session_state.model_params['pH2O']['value'],
                            step=0.001, format="%.4f",
                            key=f"pH2O_input_{st.session_state.model_params['pH2O']['value']}_{st.session_state.get('fit_timestamp', 0)}"
                        )
                        pH2O_fixed = st.checkbox(
                            "Fix", 
                            value=st.session_state.model_params['pH2O']['fixed'],
                            key=f"pH2O_fix_{st.session_state.model_params['pH2O']['fixed']}_{st.session_state.get('fit_timestamp', 0)}"
                        )
                        
                        residue_value = st.number_input(
                            "Residue", 
                            value=st.session_state.model_params['residue']['value'],
                            step=0.0001, format="%.6f",
                            key=f"residue_input_{st.session_state.model_params['residue']['value']}_{st.session_state.get('fit_timestamp', 0)}"
                        )
                        residue_fixed = st.checkbox(
                            "Fix", 
                            value=st.session_state.model_params['residue']['fixed'],
                            key=f"residue_fix_{st.session_state.model_params['residue']['fixed']}_{st.session_state.get('fit_timestamp', 0)}"
                        )
                    
                    fit_button = st.form_submit_button("🚀 Fit Model", type="primary", use_container_width=True)
                    
                    if fit_button:
                        if st.session_state.experimental_data is None:
                            st.error("Please load data first!")
                        else:
                            # Обновление параметров
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
                                st.session_state.model_params[param_name]['value'] = value
                                st.session_state.model_params[param_name]['fixed'] = is_fixed
                            
                            fixed_params = {}
                            initial_guess = {}
                            
                            param_config = {
                                'Acc': (acc_fixed, acc_value),
                                'alpha_1e6': (alpha_fixed, alpha_value),
                                'beta': (beta_fixed, beta_value),
                                'dH': (dH_fixed, dH_value),
                                'dS': (dS_fixed, dS_value),
                                'pH2O': (pH2O_fixed, pH2O_value),
                                'residue': (residue_fixed, residue_value)
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
                                    st.session_state.current_stage = 1
                                    st.session_state.inverse_complete = False
                                    
                                    for param_name in ['Acc', 'alpha_1e6', 'beta', 'dH', 'dS', 'pH2O', 'residue']:
                                        if not st.session_state.model_params[param_name]['fixed']:
                                            fitted_value = st.session_state.fit_results['params'][param_name]
                                            st.session_state.model_params[param_name]['value'] = fitted_value
                                    
                                    # Обновляем timestamp для принудительного обновления формы
                                    st.session_state.fit_timestamp += 1
                                    
                                    st.success(f"✅ Fitting completed in {end_time - start_time:.2f} seconds")
                                    st.rerun()
                                else:
                                    st.error("Fitting failed. Please check your parameters.")
        
        # Stage 2: Composition Input (only if fitting completed)
        if st.session_state.fitting_complete:
            with st.expander("🧪 **Stage 2: Composition Input**", expanded=True):
                st.markdown("Enter the chemical composition for inverse analysis")
                
                col1, col2 = st.columns(2)
                with col1:
                    A_site = st.selectbox(
                        "A-site (2+)",
                        options=['Ba', 'Sr', 'Ca'],
                        index=['Ba', 'Sr', 'Ca'].index(st.session_state.composition['A'])
                    )
                    
                    B_site = st.selectbox(
                        "B-site (4+)",
                        options=['Ce', 'Zr', 'Ti', 'Sn', 'Hf'],
                        index=['Ce', 'Zr', 'Ti', 'Sn', 'Hf'].index(st.session_state.composition['B'])
                    )
                
                with col2:
                    M_site = st.selectbox(
                        "M-site (3+, acceptor)",
                        options=['Al', 'Ga', 'In', 'Sc', 'Y', 'Yb', 'Ho', 'Dy', 'Gd', 'Sm', 'Nd', 'La'],
                        index=['Al', 'Ga', 'In', 'Sc', 'Y', 'Yb', 'Ho', 'Dy', 'Gd', 'Sm', 'Nd', 'La'].index(st.session_state.composition['M'])
                    )
                    
                    acc_value = st.session_state.model_params['Acc']['value']
                    x_value = st.number_input(
                        "x (acceptor concentration) = [Acc]",
                        value=acc_value,
                        format="%.4f",
                        disabled=True,
                        help="Значение автоматически берется из параметра [Acc] в Stage 1"
                    )
                    st.session_state.composition['x'] = acc_value
                
                use_shannon = st.checkbox("Use Shannon's ionic radii for comparison", value=False)
                
                if st.button("🔬 Run Inverse Analysis", type="primary", use_container_width=True):
                    # Validate composition
                    is_valid, error_msg = validate_composition(A_site, B_site, M_site, x_value)
                    if not is_valid:
                        st.error(f"Invalid composition: {error_msg}")
                    else:
                        # Update session state
                        st.session_state.composition['A'] = A_site
                        st.session_state.composition['B'] = B_site
                        st.session_state.composition['M'] = M_site
                        st.session_state.composition['x'] = x_value
                        
                        # Run inverse calculations
                        with st.spinner("Solving inverse problem..."):
                            solver = InverseProblemSolver(
                                A=A_site, B=B_site, M=M_site, x=x_value,
                                fit_results=st.session_state.fit_results,
                                use_shannon=use_shannon
                            )
                            st.session_state.inverse_results = solver.run_all_calculations()
                            st.session_state.inverse_complete = True
                            st.session_state.current_stage = 2
                            st.success("✅ Inverse analysis completed")
                            st.rerun()
        
        # Plot Settings
        with st.expander("🎨 **Plot Settings**", expanded=False):
            col1, col2 = st.columns(2)
            
            with col1:
                new_point_color = st.color_picker("Point color", st.session_state.plot_style['point_color'])
                if new_point_color != st.session_state.plot_style['point_color']:
                    update_plot_style('point_color', new_point_color)
                
                new_point_alpha = st.slider("Point transparency", 0.1, 1.0, st.session_state.plot_style['point_alpha'])
                if new_point_alpha != st.session_state.plot_style['point_alpha']:
                    update_plot_style('point_alpha', new_point_alpha)
                
                new_model_line_color = st.color_picker("Model line color", st.session_state.plot_style['model_line_color'])
                if new_model_line_color != st.session_state.plot_style['model_line_color']:
                    update_plot_style('model_line_color', new_model_line_color)
            
            with col2:
                new_cmap = st.selectbox("Colormap", ['viridis', 'plasma', 'inferno', 'magma', 'cividis', 'coolwarm'],
                                       index=['viridis', 'plasma', 'inferno', 'magma', 'cividis', 'coolwarm'].index(st.session_state.plot_style['cmap_style']))
                if new_cmap != st.session_state.plot_style['cmap_style']:
                    update_plot_style('cmap_style', new_cmap)
                
                new_point_size = st.slider("Point size", 10, 100, st.session_state.plot_style['point_size'])
                if new_point_size != st.session_state.plot_style['point_size']:
                    update_plot_style('point_size', new_point_size)
                
                new_line_width = st.slider("Line width", 1.0, 5.0, float(st.session_state.plot_style['line_width']), step=0.1)
                if new_line_width != st.session_state.plot_style['line_width']:
                    update_plot_style('line_width', new_line_width)
    
    # ============================================
    # MAIN CONTENT AREA
    # ============================================
    
    # Stage 1: Data Fitting Results
    if st.session_state.current_stage == 1:
        if st.session_state.experimental_data is not None:
            st.markdown("### 📊 Loaded Data Preview")
            
            df = pd.DataFrame(st.session_state.experimental_data, columns=['Temperature (°C)', 'ΔL/L₀'])
            st.dataframe(df, use_container_width=True, height=150)
            
            fig_raw, ax_raw = plt.subplots(figsize=(8, 3))
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
            st.markdown("### 📈 Fitting Results")
            
            # Metrics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("MSE", f"{st.session_state.fit_results['mse']:.3e}")
            with col2:
                st.metric("RMSE", f"{st.session_state.fit_results['rmse']:.3e}")
            with col3:
                st.metric("R²", f"{st.session_state.fit_results['r2']:.6f}")
            with col4:
                chi2_value = st.session_state.fit_results['chi2']
                st.metric("χ²_red", f"{chi2_value:.6f}" if not np.isnan(chi2_value) else "N/A")
            
            with st.expander("📊 Metric Explanations (for scientific paper)"):
                explanations = get_metric_explanation()
                for metric, info in explanations.items():
                    st.markdown(f"**{info['title']}**")
                    st.latex(info['formula'])
                    st.markdown(f"*{info['explanation']}*")
                    st.markdown(f"**Units:** {info['units']}")
                    st.markdown("---")
            
            # Parameters
            st.markdown("#### Model Parameters")
            params_data = []
            for param_name, display_name in [('Acc', '[Acc]'), ('alpha_1e6', 'α·10⁶'), 
                                            ('beta', 'β'), ('dH', 'ΔH (kJ/mol)'),
                                            ('dS', 'ΔS (J/mol·K)'), ('pH2O', 'pH₂O'),
                                            ('residue', 'Residue')]:
                if st.session_state.model_params[param_name]['fixed']:
                    value = st.session_state.model_params[param_name]['value']
                    status = "Fixed"
                else:
                    value = st.session_state.fit_results['params'][param_name]
                    status = "Fitted"
                
                params_data.append({"Parameter": display_name, "Value": value, "Status": status})
            
            params_df = pd.DataFrame(params_data)
            st.dataframe(params_df.style.format({"Value": "{:.6f}"}), use_container_width=True, height=200)
            
            # Plots
            st.markdown("#### Visualization")
            
            tab1, tab2, tab3, tab4 = st.tabs([
                "📈 Basic Plots", "🔍 Advanced Analysis", "📊 Statistical Analysis", "🧪 Model Insights"
            ])
            
            with tab1:
                col1, col2 = st.columns(2)
                
                plot1 = create_plot1_cached(st.session_state.fit_results, st.session_state.plot_style)
                plot2 = create_plot2_cached(st.session_state.fit_results, st.session_state.plot_style)
                plot3 = create_plot3_cached(st.session_state.fit_results, st.session_state.plot_style)
                plot4 = create_plot4_cached(st.session_state.fit_results, st.session_state.plot_style)
                
                with col1:
                    st.pyplot(plot1)
                    st.pyplot(plot3)
                
                with col2:
                    st.pyplot(plot2)
                    st.pyplot(plot4)
            
            with tab2:
                col1, col2 = st.columns(2)
                
                plot5 = create_plot5_cached(st.session_state.fit_results, st.session_state.plot_style)
                plot6 = create_plot6_cached(st.session_state.fit_results, st.session_state.plot_style)
                plot9 = create_plot9_cached(st.session_state.fit_results, st.session_state.plot_style)
                
                with col1:
                    st.pyplot(plot5)
                    st.pyplot(plot6)
                
                with col2:
                    st.pyplot(plot9)
                    
                    st.info("""
                    **Phase Portrait Insights:**
                    - **Left plot**: Shows expansion rate vs expansion
                    - **Right plot**: Shows acceleration vs velocity
                    - Black line: model trajectory
                    - Points colored by temperature
                    """)
            
            with tab3:
                col1, col2 = st.columns(2)
                
                plot10 = create_plot10_cached(st.session_state.fit_results, st.session_state.plot_style)
                plot12 = create_plot12_cached(st.session_state.fit_results, st.session_state.plot_style)
                
                with col1:
                    st.pyplot(plot10)
                
                with col2:
                    st.pyplot(plot12)
            
            with tab4:
                plot7 = create_plot7_cached(st.session_state.fit_results, st.session_state.plot_style)
                plot8 = create_plot8_cached(st.session_state.fit_results, st.session_state.plot_style)
                plot11 = create_plot11_cached(st.session_state.fit_results, st.session_state.plot_style)
                
                st.pyplot(plot7)
                st.pyplot(plot8)
                st.pyplot(plot11)
            
            # Export section
            st.markdown("#### 📥 Export Results")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if st.button("📊 Download Fitted Data (CSV)", use_container_width=True):
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
                        mime="text/csv"
                    )
            
            with col2:
                if st.button("⚙️ Download Parameters (TXT)", use_container_width=True):
                    params_text = f"""FITTING RESULTS
================
MSE: {st.session_state.fit_results['mse']:.6e}
RMSE: {st.session_state.fit_results['rmse']:.6e}
R²: {st.session_state.fit_results['r2']:.6f}
χ²_red: {st.session_state.fit_results['chi2']:.6f}
N points: {st.session_state.fit_results['N_points']}
Fitted parameters: {st.session_state.fit_results['n_free_params']}

MODEL PARAMETERS
================
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
                    st.download_button(
                        label="Download Parameters",
                        data=params_text,
                        file_name="model_parameters.txt",
                        mime="text/plain"
                    )
            
            with col3:
                if st.button("📈 Download All Plots (ZIP)", use_container_width=True):
                    # Create a zip file with all plots
                    import zipfile
                    from io import BytesIO
                    
                    zip_buffer = BytesIO()
                    with zipfile.ZipFile(zip_buffer, 'a', zipfile.ZIP_DEFLATED) as zf:
                        for i, plot_func in enumerate([create_plot1_cached, create_plot2_cached, create_plot3_cached,
                                                       create_plot4_cached, create_plot5_cached, create_plot6_cached,
                                                       create_plot7_cached, create_plot8_cached, create_plot9_cached,
                                                       create_plot10_cached, create_plot11_cached, create_plot12_cached]):
                            try:
                                fig = plot_func(st.session_state.fit_results, st.session_state.plot_style)
                                img_buf = BytesIO()
                                fig.savefig(img_buf, format='png', dpi=600)
                                img_buf.seek(0)
                                zf.writestr(f'plot_{i+1:02d}.png', img_buf.getvalue())
                                plt.close(fig)
                            except:
                                pass
                    
                    zip_buffer.seek(0)
                    st.download_button(
                        label="Download ZIP",
                        data=zip_buffer,
                        file_name="all_plots.zip",
                        mime="application/zip"
                    )
    
    # Stage 2: Inverse Analysis Results
    elif st.session_state.current_stage == 2 and st.session_state.inverse_complete:
        st.markdown("### 🔬 Inverse Problem Analysis")
        
        composition = format_composition(
            st.session_state.composition['A'],
            st.session_state.composition['B'],
            st.session_state.composition['M'],
            st.session_state.composition['x']
        )
        st.markdown(f"#### Composition: {composition}")
        
        # Create inverse plots
        plot13 = create_inverse_plot1_cached(
            st.session_state.inverse_results,
            composition,
            st.session_state.plot_style
        )
        
        plot14 = create_inverse_plot2_cached(
            st.session_state.inverse_results,
            composition,
            st.session_state.plot_style,
            st.session_state.composition['A'],
            st.session_state.composition['B'],
            st.session_state.composition['M']
        )
        
        plot15 = create_inverse_plot3_cached(
            st.session_state.inverse_results,
            composition,
            st.session_state.plot_style
        )
        
        # Display inverse plots
        tab1, tab2, tab3, tab4 = st.tabs([
            "🔬 Ionic Radii Analysis",
            "⚛️ Defect Parameters",
            "📐 Structural Insights",
            "📄 Report"
        ])
        
        with tab1:
            st.pyplot(plot13)
            
            # Add detailed explanations
            st.markdown("""
            <div class="info-box">
            <h4>📌 Ionic Radii Analysis</h4>
            <ul>
                <li><b>OH⁻ Radius:</b> Comparison between tabulated value and effective radius in the lattice. Deviation indicates lattice relaxation effects.</li>
                <li><b>Vacancy Radius:</b> Effective radius of oxygen vacancy calculated from dry state. Literature range for perovskites: 1.16-1.24 Å.</li>
                <li><b>Cation Changes:</b> Change in cation radii upon hydration due to coordination number change.</li>
                <li><b>Coordination Dependence:</b> Coefficients k = (Δr/r)/ΔCN showing how strongly radius depends on coordination.</li>
            </ul>
            </div>
            """, unsafe_allow_html=True)
        
        with tab2:
            st.pyplot(plot14)
            
            st.markdown("""
            <div class="info-box">
            <h4>📌 Defect Parameters</h4>
            <ul>
                <li><b>β Comparison:</b> Experimental vs theoretical chemical expansion coefficient. Deviation indicates limitations of simple ionic model.</li>
                <li><b>Lattice Sum:</b> Total contribution of ionic radii to unit cell size in different states.</li>
                <li><b>Tolerance Factor:</b> Goldschmidt factor indicating perovskite stability.</li>
                <li><b>Literature Comparison:</b> How our calculated vacancy radius compares with literature values.</li>
            </ul>
            </div>
            """, unsafe_allow_html=True)
        
        with tab3:
            st.pyplot(plot15)
            
            st.markdown("""
            <div class="info-box">
            <h4>📌 Structural Insights</h4>
            <ul>
                <li><b>Radius vs Coordination:</b> Shows how ionic radius changes with coordination number for each site.</li>
                <li><b>Contribution Pie:</b> Relative contributions of different ions to total chemical expansion.</li>
                <li>Markers show dry and hydrated states for each cation.</li>
            </ul>
            </div>
            """, unsafe_allow_html=True)
        
        with tab4:
            st.markdown("#### 📄 Inverse Analysis Report")
            
            report_text = create_inverse_report_cached(
                st.session_state.inverse_results,
                composition,
                st.session_state.fit_results['params']
            )
            
            st.text_area("Report", report_text, height=400)
            
            col1, col2 = st.columns(2)
            with col1:
                if st.button("📥 Download Report (TXT)", use_container_width=True):
                    st.download_button(
                        label="Download Report",
                        data=report_text,
                        file_name=f"inverse_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                        mime="text/plain"
                    )
            
            with col2:
                if st.button("📥 Download Inverse Plots (ZIP)", use_container_width=True):
                    import zipfile
                    from io import BytesIO
                    
                    zip_buffer = BytesIO()
                    with zipfile.ZipFile(zip_buffer, 'a', zipfile.ZIP_DEFLATED) as zf:
                        for i, (plot_func, name) in enumerate([
                            (create_inverse_plot1_cached, 'ionic_radii'),
                            (create_inverse_plot2_cached, 'defect_params'),
                            (create_inverse_plot3_cached, 'structural_insights')
                        ]):
                            try:
                                fig = plot_func(st.session_state.inverse_results, composition, st.session_state.plot_style)
                                img_buf = BytesIO()
                                fig.savefig(img_buf, format='png', dpi=600)
                                img_buf.seek(0)
                                zf.writestr(f'inverse_plot_{name}.png', img_buf.getvalue())
                                plt.close(fig)
                            except Exception as e:
                                st.warning(f"Could not save {name}: {e}")
                    
                    zip_buffer.seek(0)
                    st.download_button(
                        label="Download ZIP",
                        data=zip_buffer,
                        file_name="inverse_plots.zip",
                        mime="application/zip"
                    )
    
    else:
        # Welcome screen
        st.markdown("""
        <div style='text-align: center; padding: 50px;'>
            <h2>Welcome to the Thermo-Mechanical Expansion Modeling Suite</h2>
            <p>This application allows you to:</p>
            <ul style='list-style-type: none; padding: 0;'>
                <li>✅ Fit dilatometry/HT-XRD data to extract thermal and chemical expansion parameters</li>
                <li>✅ Solve the inverse problem to determine microscopic properties from fitted parameters</li>
                <li>✅ Generate publication-quality plots at 600 DPI</li>
                <li>✅ Export results for scientific papers</li>
            </ul>
            <p style='margin-top: 30px;'>👈 Please load data in the sidebar to begin</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Footer
    st.divider()
    st.markdown("""
    <div class='footer'>
        <p>Thermo-Mechanical Expansion Modeling Suite v2.0 | For scientific publications | 600 DPI export</p>
        <p>Based on models from Løken et al. (2018) and Zuev et al. (2022)</p>
        <p>Stage 1: Data fitting | Stage 2: Inverse problem analysis with ionic radii</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
