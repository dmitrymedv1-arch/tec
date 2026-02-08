import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import pandas as pd
import io
import warnings
import matplotlib.cm as cm
from matplotlib.colors import Normalize
warnings.filterwarnings('ignore')

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

# Initialize session state
if 'experimental_data' not in st.session_state:
    st.session_state.experimental_data = None
if 'fit_results' not in st.session_state:
    st.session_state.fit_results = None
if 'plot_style' not in st.session_state:
    st.session_state.plot_style = {
        'point_color': '#1f77b4',      # matplotlib blue
        'point_alpha': 0.8,
        'model_line_color': '#000000', # black
        'thermal_line_color': '#1f77b4', # blue
        'chemical_line_color': '#d62728', # red
        'tec_exp_color': '#1f77b4',    # blue
        'tec_model_color': '#d62728',  # red
        'oh_color': '#2ca02c',         # green
        'bar_thermal_color': '#1f77b4', # blue
        'bar_chemical_color': '#d62728', # red
        'cmap_style': 'viridis',
        'point_size': 50,
        'line_width': 2
    }

def parse_data(data_string):
    """Parse data with various separators"""
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

def calculate_oh(T, Acc, dH, dS, pH2O):
    """Calculate [OH] for given temperatures"""
    T_K = T + 273.15
    R = 8.314
    Khydr = np.exp(-dH * 1000 / (R * T_K) + dS / R)
    
    A = Khydr * pH2O
    oh = -A/2 + np.sqrt(A * Acc + (A/2)**2)
    return oh, Khydr

def model_func(T, Acc, alpha_1e6, beta, dH, dS, pH2O, residue, T_start, oh_start):
    """Model function for fitting"""
    oh, _ = calculate_oh(T, Acc, dH, dS, pH2O)
    
    dl_dl0 = (alpha_1e6/1e6) * (T - T_start) + beta * (oh - oh_start) + residue
    
    return dl_dl0

def calculate_tec(T, dl):
    """Calculate Thermal Expansion Coefficient (dΔL/Lo/dT)"""
    tec = np.zeros_like(T)
    for i in range(1, len(T)-1):
        tec[i] = (dl[i+1] - dl[i-1]) / (T[i+1] - T[i-1])
    
    # Handle boundaries
    if len(T) > 1:
        tec[0] = (dl[1] - dl[0]) / (T[1] - T[0])
        tec[-1] = (dl[-1] - dl[-2]) / (T[-1] - T[-2])
    
    return tec

def fit_model(data, fixed_params, initial_guess):
    """Perform model fitting"""
    T_data = data[:, 0]
    dl_data = data[:, 1]
    
    T_start = T_data[0]
    oh_start, _ = calculate_oh(np.array([T_start]), 
                              fixed_params['Acc'] if fixed_params['Acc'] is not None else initial_guess['Acc'],
                              fixed_params['dH'] if fixed_params['dH'] is not None else initial_guess['dH'],
                              fixed_params['dS'] if fixed_params['dS'] is not None else initial_guess['dS'],
                              fixed_params['pH2O'] if fixed_params['pH2O'] is not None else initial_guess['pH2O'])
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
        
        return model_func(T, all_params['Acc'], all_params['alpha_1e6'], 
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
        chi2 = np.sum(residuals**2) / mse if mse > 0 else 0
        
        # Calculate TEC for both experimental and model data
        tec_exp = calculate_tec(T_data, dl_data)
        tec_model = calculate_tec(T_data, dl_model)
        
        # Calculate proton concentration
        oh, _ = calculate_oh(T_data, result_params['Acc'], result_params['dH'], 
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
            'T_start': T_start,
            'oh_start': oh_start,
            'vary_params': vary_params,
            'tec_exp': tec_exp,
            'tec_model': tec_model,
            'oh_concentration': oh,
            'thermal_contrib': thermal_contrib,
            'chem_contrib': chem_contrib,
            'T_data': T_data,
            'dl_data': dl_data
        }
    
    except Exception as e:
        st.error(f"Fitting error: {str(e)}")
        return None

def adjust_color_brightness(color, factor=0.7):
    """Adjust color brightness"""
    import colorsys
    try:
        if color.startswith('#'):
            color = color.lstrip('#')
            r, g, b = tuple(int(color[i:i+2], 16) / 255.0 for i in (0, 2, 4))
        else:
            import matplotlib.colors as mcolors
            r, g, b = mcolors.to_rgb(color)
        
        h, l, s = colorsys.rgb_to_hls(r, g, b)
        l = max(0, min(1, l * factor))
        r, g, b = colorsys.hls_to_rgb(h, l, s)
        
        return f'#{int(r*255):02x}{int(g*255):02x}{int(b*255):02x}'
    except:
        return color

def create_plot1(fit_results, style):
    """Create plot 1: Experimental data and model with residual plot"""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(5, 6), 
                                   height_ratios=[3, 1],
                                   sharex=True)
    fig.subplots_adjust(hspace=0.05)
    
    T = fit_results['T_data']
    dl_exp = fit_results['dl_data']
    dl_model = fit_results['dl_model']
    residuals = fit_results['residuals']
    
    edge_color = adjust_color_brightness(style['point_color'], 0.7)
    
    # First plot model line, then experimental points
    ax1.plot(T, dl_model, '-', color=style['model_line_color'], 
            linewidth=style['line_width'], label='Model', zorder=3)
    ax1.scatter(T, dl_exp, s=style['point_size'], color=style['point_color'], 
               edgecolor=edge_color, linewidth=1.5, 
               label='Experimental', zorder=4, alpha=style['point_alpha'])
    
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

def create_plot2(fit_results, style):
    """Create plot 2: Model contributions"""
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
    ax.plot(T, chem_contrib + residue, '--', color=style['chemical_line_color'], 
           linewidth=style['line_width'], label='Chemical contribution')
    
    ax.set_xlabel('Temperature (°C)', fontweight='bold', fontsize=11)
    ax.set_ylabel('ΔL/L₀', fontweight='bold', fontsize=11)
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3, linestyle='--')
    
    fig.set_dpi(600)
    return fig

def create_plot3(fit_results, style):
    """Create plot 3: Histograms of changes"""
    fig, ax = plt.subplots(figsize=(5, 3.5))
    
    T = fit_results['T_data']
    thermal_contrib = fit_results['thermal_contrib']
    chem_contrib = fit_results['chem_contrib']
    residue = fit_results['params']['residue']
    
    thermal_start = thermal_contrib[0] + residue
    thermal_changes = thermal_start - (thermal_contrib + residue)
    
    chem_end = chem_contrib[-1] + residue
    chem_changes = (chem_contrib + residue) - chem_end
    
    bar_width = (T[1] - T[0]) * 0.7 if len(T) > 1 else 10
    ax.bar(T - bar_width/2, thermal_changes, width=bar_width, 
           color=style['bar_thermal_color'], alpha=0.7, label='Δ Thermal')
    ax.bar(T + bar_width/2, chem_changes, width=bar_width, 
           color=style['bar_chemical_color'], alpha=0.7, label='Δ Chemical')
    
    ax.set_xlabel('Temperature (°C)', fontweight='bold', fontsize=11)
    ax.set_ylabel('Change in ΔL/L₀', fontweight='bold', fontsize=11)
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3, linestyle='--', axis='y')
    
    fig.set_dpi(600)
    return fig

def create_plot4(fit_results, style):
    """Create plot 4: TEC and Proton concentration"""
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
    
    # Combine legends in one box on the right
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color=style['tec_exp_color'], 
               label='Experimental TEC', markersize=6, linewidth=style['line_width']-0.5),
        Line2D([0], [0], color=style['tec_model_color'], 
               label='Model TEC', linewidth=style['line_width']),
        Line2D([0], [0], color=style['oh_color'], linestyle='--',
               label='[OH] concentration', linewidth=style['line_width'])
    ]
    
    # Position legend on the right side, centered vertically
    ax1.legend(handles=legend_elements, loc='center left', 
              bbox_to_anchor=(1.02, 0.5), frameon=True, framealpha=0.9)
    
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

def create_plot5(fit_results, style):
    """Create plot 5: (ΔL/Lo)exp vs (ΔL/Lo)model with temperature color scale"""
    fig, ax = plt.subplots(figsize=(5, 3.5))
    
    dl_exp = fit_results['dl_data']
    dl_model = fit_results['dl_model']
    T = fit_results['T_data']
    
    # Get colormap
    cmap = getattr(cm, style['cmap_style'])
    norm = Normalize(vmin=T.min(), vmax=T.max())
    
    sc = ax.scatter(dl_model, dl_exp, c=T, cmap=cmap, norm=norm,
                   s=style['point_size'], edgecolor='black', linewidth=0.5)
    
    # Add colorbar
    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label('Temperature (°C)', fontweight='bold', fontsize=10)
    
    # Add diagonal line for perfect fit
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

def create_plot6(fit_results, style):
    """Create plot 6: α_exp vs α_model with temperature color scale"""
    fig, ax = plt.subplots(figsize=(5, 3.5))
    
    tec_exp = fit_results['tec_exp'] * 1e6  # Convert to 10⁻⁶ K⁻¹
    tec_model = fit_results['tec_model'] * 1e6
    T = fit_results['T_data']
    
    # Get colormap
    cmap = getattr(cm, style['cmap_style'])
    norm = Normalize(vmin=T.min(), vmax=T.max())
    
    sc = ax.scatter(tec_model, tec_exp, c=T, cmap=cmap, norm=norm,
                   s=style['point_size'], edgecolor='black', linewidth=0.5)
    
    # Add colorbar
    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label('Temperature (°C)', fontweight='bold', fontsize=10)
    
    # Add diagonal line for perfect fit
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

def get_metric_explanation():
    """Return explanations for the fitting metrics"""
    explanations = {
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
            'title': 'Chi-Squared Statistic (χ²)',
            'formula': r'$\chi^2 = \sum_{i=1}^{n} \frac{(y_i - \hat{y}_i)^2}{\sigma_i^2}$',
            'explanation': 'Weighted sum of squared residuals. Useful when measurement errors are known. Lower values indicate better fit assuming correct error estimates.',
            'units': 'dimensionless'
        }
    }
    return explanations

# Main app
st.title("📈 Thermo-Mechanical Expansion Modeling")
st.markdown("Modeling of proton-conducting oxides thermal expansion")

# Sidebar for inputs
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
                st.session_state.experimental_data = parse_data(data_text)
                st.success(f"Loaded {len(st.session_state.experimental_data)} data points")
            except Exception as e:
                st.error(f"Error: {str(e)}")
    
    elif data_option == "File upload":
        uploaded_file = st.file_uploader("Choose a file", type=['txt', 'csv', 'dat'])
        if uploaded_file is not None:
            try:
                data_text = uploaded_file.getvalue().decode()
                st.session_state.experimental_data = parse_data(data_text)
                st.success(f"Loaded {len(st.session_state.experimental_data)} data points")
            except Exception as e:
                st.error(f"Error: {str(e)}")
    
    else:  # Example data
        if st.button("Load Example Data"):
            example_data = """20\t0.0045
40\t0.004787988
60\t0.005075916
80\t0.005363555
100\t0.005650042
120\t0.005932612
140\t0.006203565"""
            st.session_state.experimental_data = parse_data(example_data)
            st.success(f"Loaded {len(st.session_state.experimental_data)} example data points")
    
    st.divider()
    
    st.header("Model Parameters")
    st.markdown("Check 'Fix' to keep parameter constant during fitting")
    
    # Create two columns for parameter inputs
    col1, col2 = st.columns(2)
    
    with col1:
        acc_fixed = st.checkbox("Fix", key="acc_fix", value=False)
        acc_value = st.number_input("[Acc]", value=0.6, step=0.01, format="%.4f")
        
        alpha_fixed = st.checkbox("Fix", key="alpha_fix", value=False)
        alpha_value = st.number_input("α·10⁶", value=14.4, step=0.1, format="%.4f")
        
        beta_fixed = st.checkbox("Fix", key="beta_fix", value=False)
        beta_value = st.number_input("β", value=0.017, step=0.001, format="%.4f")
        
        dH_fixed = st.checkbox("Fix", key="dH_fix", value=False)
        dH_value = st.number_input("ΔH (kJ/mol)", value=-81.0, step=1.0, format="%.2f")
    
    with col2:
        dS_fixed = st.checkbox("Fix", key="dS_fix", value=False)
        dS_value = st.number_input("ΔS (J/mol·K)", value=-131.0, step=1.0, format="%.2f")
        
        pH2O_fixed = st.checkbox("Fix", key="pH2O_fix", value=False)
        pH2O_value = st.number_input("pH₂O", value=0.083, step=0.001, format="%.4f")
        
        residue_fixed = st.checkbox("Fix", key="residue_fix", value=False)
        residue_value = st.number_input("Residue", value=0.0, step=0.0001, format="%.6f")
    
    st.divider()
    
    st.header("Plot Settings")
    
    # Color settings
    col1, col2 = st.columns(2)
    with col1:
        st.session_state.plot_style['point_color'] = st.color_picker(
            "Point color", 
            st.session_state.plot_style['point_color']
        )
        st.session_state.plot_style['point_alpha'] = st.slider(
            "Point transparency", 
            0.1, 1.0, 
            st.session_state.plot_style['point_alpha']
        )
        st.session_state.plot_style['model_line_color'] = st.color_picker(
            "Model line color", 
            st.session_state.plot_style['model_line_color']
        )
        st.session_state.plot_style['thermal_line_color'] = st.color_picker(
            "Thermal line color", 
            st.session_state.plot_style['thermal_line_color']
        )
        st.session_state.plot_style['chemical_line_color'] = st.color_picker(
            "Chemical line color", 
            st.session_state.plot_style['chemical_line_color']
        )
    
    with col2:
        st.session_state.plot_style['tec_exp_color'] = st.color_picker(
            "Exp. TEC color", 
            st.session_state.plot_style['tec_exp_color']
        )
        st.session_state.plot_style['tec_model_color'] = st.color_picker(
            "Model TEC color", 
            st.session_state.plot_style['tec_model_color']
        )
        st.session_state.plot_style['oh_color'] = st.color_picker(
            "[OH] line color", 
            st.session_state.plot_style['oh_color']
        )
        st.session_state.plot_style['bar_thermal_color'] = st.color_picker(
            "Thermal bar color", 
            st.session_state.plot_style['bar_thermal_color']
        )
        st.session_state.plot_style['bar_chemical_color'] = st.color_picker(
            "Chemical bar color", 
            st.session_state.plot_style['bar_chemical_color']
        )
    
    # Additional plot settings
    col1, col2 = st.columns(2)
    with col1:
        st.session_state.plot_style['point_size'] = st.slider(
            "Point size", 
            10, 100, 
            st.session_state.plot_style['point_size'],
            step=1
        )
        st.session_state.plot_style['line_width'] = st.slider(
            "Line width", 
            1.0, 5.0, 
            st.session_state.plot_style['line_width'],
            step=0.1  # Добавляем float step
        )
    
    with col2:
        cmap_options = ['viridis', 'plasma', 'inferno', 'magma', 'cividis', 
                       'coolwarm', 'RdYlBu', 'Spectral', 'rainbow', 'jet']
        st.session_state.plot_style['cmap_style'] = st.selectbox(
            "Temperature colormap", 
            cmap_options, 
            index=cmap_options.index(st.session_state.plot_style['cmap_style'])
        )
    
    st.divider()
    
    if st.button("🚀 Fit Model and Create Plots", type="primary", use_container_width=True):
        if st.session_state.experimental_data is None:
            st.error("Please load data first!")
        else:
            # Prepare parameters
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
                
                # Set bounds for each parameter
                if name == 'Acc':
                    initial_guess[f'{name}_bounds'] = (0.01, 10.0)
                elif name == 'alpha_1e6':
                    initial_guess[f'{name}_bounds'] = (1.0, 100.0)
                elif name == 'beta':
                    initial_guess[f'{name}_bounds'] = (0.0001, 1.0)
                elif name == 'dH':
                    initial_guess[f'{name}_bounds'] = (-200.0, -10.0)
                elif name == 'dS':
                    initial_guess[f'{name}_bounds'] = (-300.0, -50.0)
                elif name == 'pH2O':
                    initial_guess[f'{name}_bounds'] = (0.001, 1.0)
                elif name == 'residue':
                    initial_guess[f'{name}_bounds'] = (-0.01, 0.01)
            
            # Perform fitting
            with st.spinner("Fitting model..."):
                st.session_state.fit_results = fit_model(
                    st.session_state.experimental_data, 
                    fixed_params, 
                    initial_guess
                )

# Main content area
if st.session_state.experimental_data is not None:
    st.header("Loaded Data Preview")
    
    # Display data table
    df = pd.DataFrame(st.session_state.experimental_data, columns=['Temperature (°C)', 'ΔL/L₀'])
    st.dataframe(df, use_container_width=True)
    
    # Quick plot of raw data
    fig_raw, ax_raw = plt.subplots(figsize=(6, 3))
    ax_raw.scatter(df['Temperature (°C)'], df['ΔL/L₀'], s=40, 
                  color=st.session_state.plot_style['point_color'], 
                  edgecolor='black',
                  alpha=st.session_state.plot_style['point_alpha'])
    ax_raw.set_xlabel('Temperature (°C)', fontweight='bold')
    ax_raw.set_ylabel('ΔL/L₀', fontweight='bold')
    ax_raw.set_title('Raw Experimental Data', fontweight='bold')
    ax_raw.grid(True, alpha=0.3)
    fig_raw.set_dpi(600)
    st.pyplot(fig_raw)

if st.session_state.fit_results is not None:
    st.header("Fitting Results")
    
    # Display metrics with expandable explanations
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("MSE", f"{st.session_state.fit_results['mse']:.3e}")
    with col2:
        st.metric("RMSE", f"{st.session_state.fit_results['rmse']:.3e}")
    with col3:
        st.metric("R²", f"{st.session_state.fit_results['r2']:.6f}")
    with col4:
        st.metric("χ²", f"{st.session_state.fit_results['chi2']:.6f}")
    
    # Metric explanations
    with st.expander("📊 Metric Explanations (for scientific paper)"):
        explanations = get_metric_explanation()
        for metric, info in explanations.items():
            st.markdown(f"**{info['title']}**")
            st.latex(info['formula'])
            st.markdown(f"*{info['explanation']}*")
            st.markdown(f"**Units:** {info['units']}")
            st.markdown("---")
    
    # Display parameters
    st.subheader("Model Parameters")
    params_df = pd.DataFrame([
        {"Parameter": "[Acc]", "Value": st.session_state.fit_results['params']['Acc'], "Status": "Fixed" if 'Acc' not in st.session_state.fit_results['vary_params'] else "Fitted"},
        {"Parameter": "α·10⁶", "Value": st.session_state.fit_results['params']['alpha_1e6'], "Status": "Fixed" if 'alpha_1e6' not in st.session_state.fit_results['vary_params'] else "Fitted"},
        {"Parameter": "β", "Value": st.session_state.fit_results['params']['beta'], "Status": "Fixed" if 'beta' not in st.session_state.fit_results['vary_params'] else "Fitted"},
        {"Parameter": "ΔH (kJ/mol)", "Value": st.session_state.fit_results['params']['dH'], "Status": "Fixed" if 'dH' not in st.session_state.fit_results['vary_params'] else "Fitted"},
        {"Parameter": "ΔS (J/mol·K)", "Value": st.session_state.fit_results['params']['dS'], "Status": "Fixed" if 'dS' not in st.session_state.fit_results['vary_params'] else "Fitted"},
        {"Parameter": "pH₂O", "Value": st.session_state.fit_results['params']['pH2O'], "Status": "Fixed" if 'pH2O' not in st.session_state.fit_results['vary_params'] else "Fitted"},
        {"Parameter": "Residue", "Value": st.session_state.fit_results['params']['residue'], "Status": "Fixed" if 'residue' not in st.session_state.fit_results['vary_params'] else "Fitted"},
    ])
    st.dataframe(params_df.style.format({"Value": "{:.6f}"}), use_container_width=True)
    
    st.divider()
    st.header("Plots")
    
    # Create all plots
    plot1 = create_plot1(st.session_state.fit_results, st.session_state.plot_style)
    plot2 = create_plot2(st.session_state.fit_results, st.session_state.plot_style)
    plot3 = create_plot3(st.session_state.fit_results, st.session_state.plot_style)
    plot4 = create_plot4(st.session_state.fit_results, st.session_state.plot_style)
    plot5 = create_plot5(st.session_state.fit_results, st.session_state.plot_style)
    plot6 = create_plot6(st.session_state.fit_results, st.session_state.plot_style)
    
    # Display plots in columns
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Experimental Data and Model")
        st.pyplot(plot1)
        
        st.subheader("Histograms of Changes")
        st.pyplot(plot3)
        
        st.subheader("ΔL/L₀: Model vs Experiment")
        st.pyplot(plot5)
    
    with col2:
        st.subheader("Model Contributions")
        st.pyplot(plot2)
        
        st.subheader("TEC and Proton Concentration")
        st.pyplot(plot4)
        
        st.subheader("TEC: Model vs Experiment")
        st.pyplot(plot6)
    
    # Download section for individual plots
    st.divider()
    st.subheader("Export Results")
    
    # Create columns for download buttons
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("📥 Download Plot 1 (Data & Model)"):
            buf = io.BytesIO()
            plot1.savefig(buf, format='png', dpi=600)
            st.download_button(
                label="Download PNG (600 DPI)",
                data=buf.getvalue(),
                file_name="plot1_data_and_model.png",
                mime="image/png"
            )
        
        if st.button("📥 Download Plot 2 (Contributions)"):
            buf = io.BytesIO()
            plot2.savefig(buf, format='png', dpi=600)
            st.download_button(
                label="Download PNG (600 DPI)",
                data=buf.getvalue(),
                file_name="plot2_contributions.png",
                mime="image/png"
            )
    
    with col2:
        if st.button("📥 Download Plot 3 (Histograms)"):
            buf = io.BytesIO()
            plot3.savefig(buf, format='png', dpi=600)
            st.download_button(
                label="Download PNG (600 DPI)",
                data=buf.getvalue(),
                file_name="plot3_histograms.png",
                mime="image/png"
            )
        
        if st.button("📥 Download Plot 4 (TEC & [OH])"):
            buf = io.BytesIO()
            plot4.savefig(buf, format='png', dpi=600)
            st.download_button(
                label="Download PNG (600 DPI)",
                data=buf.getvalue(),
                file_name="plot4_tec_and_oh.png",
                mime="image/png"
            )
    
    with col3:
        if st.button("📥 Download Plot 5 (ΔL/L₀ Correlation)"):
            buf = io.BytesIO()
            plot5.savefig(buf, format='png', dpi=600)
            st.download_button(
                label="Download PNG (600 DPI)",
                data=buf.getvalue(),
                file_name="plot5_dl_correlation.png",
                mime="image/png"
            )
        
        if st.button("📥 Download Plot 6 (TEC Correlation)"):
            buf = io.BytesIO()
            plot6.savefig(buf, format='png', dpi=600)
            st.download_button(
                label="Download PNG (600 DPI)",
                data=buf.getvalue(),
                file_name="plot6_tec_correlation.png",
                mime="image/png"
            )
    
    # Download data
    st.divider()
    st.subheader("Download Data")
    
    col1, col2 = st.columns(2)
    with col1:
        # Download fitted data as CSV
        if st.button("📊 Download Fitted Data (CSV)"):
            fitted_df = pd.DataFrame({
                'Temperature_C': st.session_state.fit_results['T_data'],
                'DeltaL_L0_exp': st.session_state.fit_results['dl_data'],
                'DeltaL_L0_model': st.session_state.fit_results['dl_model'],
                'Residual': st.session_state.fit_results['residuals'],
                'TEC_exp_1e6K-1': st.session_state.fit_results['tec_exp'] * 1e6,
                'TEC_model_1e6K-1': st.session_state.fit_results['tec_model'] * 1e6,
                'OH_concentration': st.session_state.fit_results['oh_concentration']
            })
            csv = fitted_df.to_csv(index=False)
            st.download_button(
                label="Download CSV",
                data=csv,
                file_name="fitted_data.csv",
                mime="text/csv"
            )
    
    with col2:
        # Download parameters as text
        if st.button("⚙️ Download Parameters (TXT)"):
            params_text = f"""FITTING RESULTS
================
MSE: {st.session_state.fit_results['mse']:.6e}
RMSE: {st.session_state.fit_results['rmse']:.6e}
R²: {st.session_state.fit_results['r2']:.6f}
χ²: {st.session_state.fit_results['chi2']:.6f}

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
"""
            st.download_button(
                label="Download Parameters",
                data=params_text,
                file_name="model_parameters.txt",
                mime="text/plain"
            )

else:
    # Instructions
    st.info("👈 Please load data and configure parameters in the sidebar, then click 'Fit Model and Create Plots'")

# Footer
st.divider()
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center'>
        <p>Thermo-Mechanical Expansion Modeling Tool | For scientific publications | 600 DPI export</p>
    </div>
    """,
    unsafe_allow_html=True
)


