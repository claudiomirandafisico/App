import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy.optimize import fsolve
from datetime import datetime

# --- CONFIGURACION DE PAGINA ---
st.set_page_config(
    page_title="Gravitational Dynamics Research Suite",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Inyeccion de CSS para tema blanco cientifico (estilo publicacion academica)
st.markdown("""
<style>
    .stApp {
        background-color: #FFFFFF;
        color: #0F172A;
    }
    section[data-testid="stSidebar"] {
        background-color: #F8FAFC;
        border-right: 1px solid #E2E8F0;
    }
    div[data-testid="stMetric"] {
        background-color: #F8FAFC;
        border: 1px solid #CBD5E1;
        padding: 12px 16px;
        border-radius: 4px;
    }
    div[data-testid="stMetricLabel"] {
        color: #475569;
        font-family: 'Courier New', monospace;
        font-size: 0.8rem;
        font-weight: 600;
    }
    div[data-testid="stMetricValue"] {
        color: #0F172A;
        font-family: 'Courier New', monospace;
        font-size: 1.35rem;
        font-weight: 700;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: #FFFFFF;
        border-bottom: 1px solid #CBD5E1;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: #F1F5F9;
        border: 1px solid #CBD5E1;
        border-bottom: none;
        color: #475569;
        padding: 8px 20px;
        font-family: monospace;
    }
    .stTabs [aria-selected="true"] {
        background-color: #FFFFFF;
        color: #1D4ED8;
        border-top: 2px solid #1D4ED8;
        font-weight: 600;
    }
    h1, h2, h3, h4 {
        font-family: 'Inter', 'Segoe UI', sans-serif;
        letter-spacing: -0.02em;
        color: #0F172A;
    }
</style>
""", unsafe_allow_html=True)

st.title("SIMULADOR DE DINAMICA GRAVITACIONAL Y ESTABILIDAD LAGRANGIANA")
st.caption("Plataforma de integracion numerica de N-cuerpos, analisis de invariantes canonicos y topologia del problema circular restringido de tres cuerpos (CR3BP).")

# --- NUCLEO MATEMATICO Y FISICO ---

def compute_accelerations(pos, masses, G=1.0, eps=1e-4):
    """Calcula el campo de aceleracion gravitacional con parametro de suavizado de Plummer (eps)."""
    diff = pos[np.newaxis, :, :] - pos[:, np.newaxis, :]
    dist_sq = np.sum(diff**2, axis=-1) + eps**2
    np.fill_diagonal(dist_sq, np.inf)
    inv_dist_cubed = dist_sq**(-1.5)
    acc = G * np.sum(diff * masses[np.newaxis, :, np.newaxis] * inv_dist_cubed[:, :, np.newaxis], axis=1)
    return acc

def compute_invariants(pos, vel, masses, G=1.0, eps=1e-4):
    """Calcula Energia Cinetica (K), Potencial (U), Total (E), Momento Angular (Lz) y Momento Lineal (P)."""
    v_sq = np.sum(vel**2, axis=1)
    K = 0.5 * np.sum(masses * v_sq)
    
    N = len(masses)
    U = 0.0
    for i in range(N):
        for j in range(i + 1, N):
            r_ij = np.linalg.norm(pos[j] - pos[i])
            U -= G * masses[i] * masses[j] / np.sqrt(r_ij**2 + eps**2)
            
    Lz = np.sum(masses * (pos[:, 0] * vel[:, 1] - pos[:, 1] * vel[:, 0]))
    P_total = np.linalg.norm(np.sum(masses[:, np.newaxis] * vel, axis=0))
    return K, U, K + U, Lz, P_total

def step_euler(pos, vel, masses, dt, G, eps):
    acc = compute_accelerations(pos, masses, G, eps)
    return pos + vel * dt, vel + acc * dt

def step_verlet(pos, vel, masses, dt, G, eps):
    """Integrador simplectico Velocity-Verlet de 2do orden."""
    acc_0 = compute_accelerations(pos, masses, G, eps)
    pos_new = pos + vel * dt + 0.5 * acc_0 * (dt**2)
    acc_1 = compute_accelerations(pos_new, masses, G, eps)
    vel_new = vel + 0.5 * (acc_0 + acc_1) * dt
    return pos_new, vel_new

def step_rk4(pos, vel, masses, dt, G, eps):
    """Integrador Runge-Kutta clasico de 4to orden."""
    k1_r = vel
    k1_v = compute_accelerations(pos, masses, G, eps)
    
    k2_r = vel + 0.5 * dt * k1_v
    k2_v = compute_accelerations(pos + 0.5 * dt * k1_r, masses, G, eps)
    
    k3_r = vel + 0.5 * dt * k2_v
    k3_v = compute_accelerations(pos + 0.5 * dt * k2_r, masses, G, eps)
    
    k4_r = vel + dt * k3_v
    k4_v = compute_accelerations(pos + dt * k3_r, masses, G, eps)
    
    pos_new = pos + (dt / 6.0) * (k1_r + 2*k2_r + 2*k3_r + k4_r)
    vel_new = vel + (dt / 6.0) * (k1_v + 2*k2_v + 2*k3_v + k4_v)
    return pos_new, vel_new

def step_yoshida4(pos, vel, masses, dt, G, eps):
    """Integrador simplectico de 4to orden de Yoshida."""
    cbrt2 = 2.0**(1.0 / 3.0)
    w0 = -cbrt2 / (2.0 - cbrt2)
    w1 = 1.0 / (2.0 - cbrt2)
    c = [w1 / 2.0, (w0 + w1) / 2.0, (w0 + w1) / 2.0, w1 / 2.0]
    d = [w1, w0, w1]
    
    r = pos.copy()
    v = vel.copy()
    for i in range(3):
        r = r + c[i] * v * dt
        a = compute_accelerations(r, masses, G, eps)
        v = v + d[i] * a * dt
    r = r + c[3] * v * dt
    return r, v

def get_initial_conditions(preset_name):
    if preset_name == "Coreografia Figura-8 (Chenciner-Montgomery)":
        x1, y1 = 0.97000436, -0.24308753
        vx3, vy3 = -0.93240737, -0.86473146
        pos = np.array([[x1, y1], [-x1, -y1], [0.0, 0.0]], dtype=float)
        vel = np.array([[-vx3/2, -vy3/2], [-vx3/2, -vy3/2], [vx3, vy3]], dtype=float)
        masses = np.array([1.0, 1.0, 1.0], dtype=float)
    elif preset_name == "Sistema Jerarquico Kepleriano (Estrella-Planeta-Satelite)":
        pos = np.array([[0.0, 0.0], [2.0, 0.0], [2.15, 0.0]], dtype=float)
        vel = np.array([[0.0, -0.015], [0.0, 0.707], [0.0, 1.065]], dtype=float)
        masses = np.array([1.0, 0.02, 0.0005], dtype=float)
    elif preset_name == "Problema de Burrau (3-4-5 Pitagorico Caotico)":
        pos = np.array([[1.0, 3.0], [-2.0, -1.0], [1.0, -1.0]], dtype=float)
        vel = np.zeros((3, 2), dtype=float)
        masses = np.array([3.0, 4.0, 5.0], dtype=float)
    else:
        pos = np.array([[1.0, 0.5], [-1.0, -0.5], [0.0, -1.0]], dtype=float)
        vel = np.array([[-0.35, 0.2], [0.35, 0.1], [0.0, -0.3]], dtype=float)
        masses = np.array([1.0, 0.85, 0.65], dtype=float)
    return pos, vel, masses

def solve_lagrange_and_stability(mu):
    """Calcula coordenadas de L1-L5, potencial de Jacobi y autovalores de estabilidad lineal."""
    x1, x2 = -mu, 1.0 - mu
    
    def collinear_eq(x):
        r1 = np.abs(x - x1)
        r2 = np.abs(x - x2)
        return x - (1 - mu) * (x - x1) / (r1**3 + 1e-15) - mu * (x - x2) / (r2**3 + 1e-15)
    
    L1_x = fsolve(collinear_eq, x2 - (mu / 3.0)**(1/3))[0]
    L2_x = fsolve(collinear_eq, x2 + (mu / 3.0)**(1/3))[0]
    L3_x = fsolve(collinear_eq, -1.0 - (5.0 * mu / 12.0))[0]
    
    pts = {
        "L1": (L1_x, 0.0),
        "L2": (L2_x, 0.0),
        "L3": (L3_x, 0.0),
        "L4": (0.5 - mu, np.sqrt(3.0) / 2.0),
        "L5": (0.5 - mu, -np.sqrt(3.0) / 2.0)
    }
    
    results = []
    for name, (lx, ly) in pts.items():
        r1 = np.sqrt((lx - x1)**2 + ly**2)
        r2 = np.sqrt((lx - x2)**2 + ly**2)
        omega = (1.0 - mu) / r1 + mu / r2 + 0.5 * (lx**2 + ly**2)
        C_jacobi = 2.0 * omega
        
        Uxx = 1.0 - (1 - mu)/r1**3 - mu/r2**3 + 3*(1 - mu)*(lx - x1)**2/r1**5 + 3*mu*(lx - x2)**2/r2**5
        Uyy = 1.0 - (1 - mu)/r1**3 - mu/r2**3 + 3*(1 - mu)*ly**2/r1**5 + 3*mu*ly**2/r2**5
        Uxy = 3*(1 - mu)*(lx - x1)*ly/r1**5 + 3*mu*(lx - x2)*ly/r2**5
        
        J = np.array([
            [0.0, 0.0,  1.0, 0.0],
            [0.0, 0.0,  0.0, 1.0],
            [Uxx, Uxy,  0.0, 2.0],
            [Uxy, Uyy, -2.0, 0.0]
        ])
        eigvals = np.linalg.eigvals(J)
        max_real = np.max(np.real(eigvals))
        stability = "Linealmente Estable (Centro)" if max_real < 1e-7 else "Inestable (Punto de Silla)"
        
        results.append({
            "Punto": name,
            "x [adim]": lx,
            "y [adim]": ly,
            "Potencial Omega": omega,
            "Constante Jacobi (C_J)": C_jacobi,
            "Max Re(lambda)": max_real,
            "Clasificacion": stability
        })
    return pd.DataFrame(results)

# --- PANEL LATERAL DE CONTROL Y ADQUISICION ---
st.sidebar.header("PARAMETROS DE SIMULACION")

preset = st.sidebar.selectbox(
    "Condicion Inicial del Sistema",
    [
        "Coreografia Figura-8 (Chenciner-Montgomery)",
        "Sistema Jerarquico Kepleriano (Estrella-Planeta-Satelite)",
        "Problema de Burrau (3-4-5 Pitagorico Caotico)",
        "Interaccion Tripartita Inestable"
    ]
)

integrator_name = st.sidebar.selectbox(
    "Esquema de Integracion Numerica",
    [
        "Yoshida 4to Orden (Simplectico)",
        "Velocity Verlet 2do Orden (Simplectico)",
        "Runge-Kutta 4to Orden (Clasico)",
        "Euler Explicito 1er Orden (Referencia)"
    ]
)

dt = st.sidebar.number_input("Paso de Integracion (dt)", min_value=0.0005, max_value=0.0500, value=0.0050, step=0.0005, format="%.4f")
n_steps = st.sidebar.slider("Pasos de Muestreo (N)", 200, 5000, 1500, step=100)
eps_soft = st.sidebar.number_input("Suavizado de Plummer (epsilon)", min_value=1e-6, max_value=1e-1, value=1e-4, format="%.1e")

pos_init, vel_init, masses_init = get_initial_conditions(preset)

with st.sidebar.expander("Edicion Manual de Masas y Vectores", expanded=False):
    m1_val = st.number_input("Masa Cuerpo 1 (m1)", value=float(masses_init[0]), step=0.1, format="%.4f")
    m2_val = st.number_input("Masa Cuerpo 2 (m2)", value=float(masses_init[1]), step=0.1, format="%.4f")
    m3_val = st.number_input("Masa Cuerpo 3 (m3)", value=float(masses_init[2]), step=0.1, format="%.4f")
    masses = np.array([m1_val, m2_val, m3_val])

# --- EJECUCION DEL MOTOR NUMERICO ---
pos = pos_init.copy()
vel = vel_init.copy()
G = 1.0

traj_pos = np.zeros((n_steps, 3, 2))
traj_vel = np.zeros((n_steps, 3, 2))
K_arr = np.zeros(n_steps)
U_arr = np.zeros(n_steps)
E_arr = np.zeros(n_steps)
Lz_arr = np.zeros(n_steps)
P_arr = np.zeros(n_steps)
t_arr = np.arange(n_steps) * dt

for k in range(n_steps):
    traj_pos[k] = pos
    traj_vel[k] = vel
    K_arr[k], U_arr[k], E_arr[k], Lz_arr[k], P_arr[k] = compute_invariants(pos, vel, masses, G, eps_soft)
    
    if "Yoshida" in integrator_name:
        pos, vel = step_yoshida4(pos, vel, masses, dt, G, eps_soft)
    elif "Verlet" in integrator_name:
        pos, vel = step_verlet(pos, vel, masses, dt, G, eps_soft)
    elif "Runge-Kutta" in integrator_name:
        pos, vel = step_rk4(pos, vel, masses, dt, G, eps_soft)
    else:
        pos, vel = step_euler(pos, vel, masses, dt, G, eps_soft)

E0 = E_arr[0]
Lz0 = Lz_arr[0]
rel_dE = (E_arr - E0) / (np.abs(E0) + 1e-15)
abs_dLz = Lz_arr - Lz0
virial_ratio = 2.0 * np.mean(K_arr) / (np.abs(np.mean(U_arr)) + 1e-15)

# Construccion del DataFrame de Telemetria para Investigacion
df_telemetry = pd.DataFrame({
    "step": np.arange(n_steps),
    "time": t_arr,
    "x1": traj_pos[:, 0, 0], "y1": traj_pos[:, 0, 1], "vx1": traj_vel[:, 0, 0], "vy1": traj_vel[:, 0, 1],
    "x2": traj_pos[:, 1, 0], "y2": traj_pos[:, 1, 1], "vx2": traj_vel[:, 1, 0], "vy2": traj_vel[:, 1, 1],
    "x3": traj_pos[:, 2, 0], "y3": traj_pos[:, 2, 1], "vx3": traj_vel[:, 2, 0], "vy3": traj_vel[:, 2, 1],
    "K_kinetic": K_arr,
    "U_potential": U_arr,
    "E_total": E_arr,
    "rel_energy_error": rel_dE,
    "Lz_angular_momentum": Lz_arr,
    "Lz_drift": abs_dLz,
    "linear_momentum_norm": P_arr
})

# Plantilla comun de Plotly para Fondo Blanco Cientifico (Estilo Publicacion)
LIGHT_LAYOUT = dict(
    template="plotly_white",
    paper_bgcolor="#FFFFFF",
    plot_bgcolor="#FFFFFF",
    font=dict(family="Courier New, monospace", color="#0F172A", size=12),
    xaxis=dict(gridcolor="#E2E8F0", zerolinecolor="#94A3B8", showline=True, linecolor="#0F172A", mirror=True, ticks="outside"),
    yaxis=dict(gridcolor="#E2E8F0", zerolinecolor="#94A3B8", showline=True, linecolor="#0F172A", mirror=True, ticks="outside"),
    margin=dict(l=55, r=25, t=45, b=45)
)

# --- CREACION DE PESTAÑAS ---
tab1, tab2, tab3 = st.tabs([
    "01 // DINAMICA ORBITAL Y ESPACIO DE FASES",
    "02 // TOPOLOGIA CR3BP Y ESTABILIDAD DE LAGRANGE",
    "03 // ADQUISICION DE DATOS Y EXPORTACION"
])

# ==============================================================================
# TAB 1: DINAMICA ORBITAL Y AUDITORIA DE INVARIANTES
# ==============================================================================
with tab1:
    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("ENERGIA INICIAL (E0)", f"{E0:.6f}")
    k2.metric("ERROR RELATIVO MAX |dE/E0|", f"{np.max(np.abs(rel_dE)):.2e}")
    k3.metric("DERIVA MOMENTO ANGULAR |dLz|", f"{np.max(np.abs(abs_dLz)):.2e}")
    k4.metric("COCIENTE VIRIAL (2<K>/|<U>|)", f"{virial_ratio:.4f}")
    k5.metric("TIEMPO TOTAL INTEGRADO", f"{t_arr[-1]:.2f} u.t.")

    col_orb, col_phase = st.columns([1.2, 1])

    with col_orb:
        fig_orb = go.Figure()
        # Paleta de alto contraste sobre blanco (Azul Cobalto, Carmesi, Verde Esmeralda Oscuro)
        palette = ["#1D4ED8", "#DC2626", "#059669"]
        
        for i in range(3):
            fig_orb.add_trace(go.Scatter(
                x=traj_pos[:, i, 0],
                y=traj_pos[:, i, 1],
                mode="lines",
                line=dict(color=palette[i], width=2.0),
                name=f"Cuerpo {i+1} (m={masses[i]:.3f})"
            ))
            fig_orb.add_trace(go.Scatter(
                x=[traj_pos[0, i, 0]],
                y=[traj_pos[0, i, 1]],
                mode="markers",
                marker=dict(size=7, color="#FFFFFF", line=dict(width=2, color=palette[i])),
                showlegend=False
            ))
            fig_orb.add_trace(go.Scatter(
                x=[traj_pos[-1, i, 0]],
                y=[traj_pos[-1, i, 1]],
                mode="markers",
                marker=dict(size=10, color=palette[i], line=dict(width=1.5, color="#0F172A")),
                showlegend=False
            ))

        fig_orb.update_layout(**LIGHT_LAYOUT)
        fig_orb.update_layout(
            title="Proyeccion en el Espacio de Configuracion (x, y)",
            xaxis_title="Coordenada x [u.a.]",
            yaxis_title="Coordenada y [u.a.]",
            yaxis=dict(scaleanchor="x", scaleratio=1, gridcolor="#E2E8F0", zerolinecolor="#94A3B8", showline=True, linecolor="#0F172A", mirror=True),
            height=460,
            legend=dict(x=0.02, y=0.98, bgcolor="rgba(255,255,255,0.9)", bordercolor="#CBD5E1", borderwidth=1)
        )
        st.plotly_chart(fig_orb, use_container_width=True)

    with col_phase:
        body_idx = st.selectbox("Seleccionar Cuerpo para Retrato de Fase", [1, 2, 3], index=0) - 1
        r_norm = np.linalg.norm(traj_pos[:, body_idx, :], axis=1)
        v_rad = np.sum(traj_pos[:, body_idx, :] * traj_vel[:, body_idx, :], axis=1) / (r_norm + 1e-12)
        
        fig_phase = go.Figure()
        fig_phase.add_trace(go.Scatter(
            x=r_norm,
            y=v_rad,
            mode="lines",
            line=dict(color=palette[body_idx], width=1.6),
            name=f"Espacio de Fases C{body_idx+1}"
        ))
        fig_phase.update_layout(**LIGHT_LAYOUT)
        fig_phase.update_layout(
            title=f"Retrato de Fase Radial (r vs. v_r) - Cuerpo {body_idx+1}",
            xaxis_title="Distancia Radial al Origen r [u.a.]",
            yaxis_title="Velocidad Radial v_r [u.v.]",
            height=405
        )
        st.plotly_chart(fig_phase, use_container_width=True)

    col_err1, col_err2 = st.columns(2)
    with col_err1:
        fig_e = go.Figure()
        fig_e.add_trace(go.Scatter(x=t_arr, y=rel_dE, mode="lines", line=dict(color="#0F172A", width=1.5), name="dE/E0"))
        fig_e.update_layout(**LIGHT_LAYOUT)
        fig_e.update_layout(
            title="Evolucion Temporal del Error Relativo de Energia (E(t) - E0) / |E0|",
            xaxis_title="Tiempo [t]",
            yaxis_title="Error Relativo Adimensional",
            height=280
        )
        st.plotly_chart(fig_e, use_container_width=True)

    with col_err2:
        fig_ek = go.Figure()
        fig_ek.add_trace(go.Scatter(x=t_arr, y=K_arr, mode="lines", line=dict(color="#DC2626", width=1.5), name="Cinetica (K)"))
        fig_ek.add_trace(go.Scatter(x=t_arr, y=U_arr, mode="lines", line=dict(color="#1D4ED8", width=1.5), name="Potencial (U)"))
        fig_ek.add_trace(go.Scatter(x=t_arr, y=E_arr, mode="lines", line=dict(color="#0F172A", width=2.2), name="Total (E)"))
        fig_ek.update_layout(**LIGHT_LAYOUT)
        fig_ek.update_layout(
            title="Particion Energetica del Sistema (K, U, E)",
            xaxis_title="Tiempo [t]",
            yaxis_title="Energia [u.e.]",
            height=280,
            legend=dict(orientation="h", y=1.15, x=0.0)
        )
        st.plotly_chart(fig_ek, use_container_width=True)

# ==============================================================================
# TAB 2: TOPOLOGIA DE ROCHE Y ESTABILIDAD LINEAL EN CR3BP
# ==============================================================================
with tab2:
    col_l_ctrl, col_l_vis = st.columns([1, 1.8])
    
    with col_l_ctrl:
        st.subheader("Parametro de Masa del Sistema Binario")
        mu_val = st.slider(
            "Fraccion de Masa Reducida mu = M2 / (M1 + M2)",
            min_value=0.005, max_value=0.450, value=0.038, step=0.001, format="%.3f"
        )
        
        r_outh = 0.5 * (1.0 - np.sqrt(23.0 / 27.0))
        st.caption(f"Valor critico de estabilidad de Routh (mu_crit): {r_outh:.6f}")
        
        df_lagrange = solve_lagrange_and_stability(mu_val)
        st.markdown("#### Matriz de Puntos de Equilibrio y Estabilidad Lineal")
        st.dataframe(
            df_lagrange.style.format({
                "x [adim]": "{:.5f}",
                "y [adim]": "{:.5f}",
                "Potencial Omega": "{:.5f}",
                "Constante Jacobi (C_J)": "{:.5f}",
                "Max Re(lambda)": "{:.4e}"
            }),
            use_container_width=True,
            hide_index=True
        )
        
        st.latex(r"\mathbf{J} = \begin{bmatrix} \mathbf{0}_{2\times2} & \mathbf{I}_{2\times2} \\ \nabla^2 \Omega & 2\mathbf{\Theta} \end{bmatrix}, \quad \mathbf{\Theta} = \begin{bmatrix} 0 & 1 \\ -1 & 0 \end{bmatrix}")

    with col_l_vis:
        grid_n = 220
        gx = np.linspace(-1.65, 1.65, grid_n)
        gy = np.linspace(-1.65, 1.65, grid_n)
        GX, GY = np.meshgrid(gx, gy)
        
        r1_grid = np.sqrt((GX + mu_val)**2 + GY**2 + 1e-4)
        r2_grid = np.sqrt((GX - (1.0 - mu_val))**2 + GY**2 + 1e-4)
        Omega_grid = (1.0 - mu_val)/r1_grid + mu_val/r2_grid + 0.5*(GX**2 + GY**2)
        Omega_clip = np.clip(Omega_grid, 1.38, 2.75)

        fig_roche = go.Figure()
        fig_roche.add_trace(go.Contour(
            z=Omega_clip,
            x=gx,
            y=gy,
            colorscale="Blues",
            contours=dict(start=1.40, end=2.35, size=0.03, showlabels=False),
            colorbar=dict(title="Omega(x,y)"),
            name="Superficie Equipotencial"
        ))

        fig_roche.add_trace(go.Scatter(
            x=[-mu_val, 1.0 - mu_val],
            y=[0.0, 0.0],
            mode="markers+text",
            marker=dict(size=[14, 10], color=["#0F172A", "#1D4ED8"], line=dict(width=1.5, color="#FFFFFF")),
            text=["M1", "M2"],
            textposition="bottom center",
            textfont=dict(color="#0F172A", size=12),
            name="Cuerpos Primarios"
        ))

        fig_roche.add_trace(go.Scatter(
            x=df_lagrange["x [adim]"],
            y=df_lagrange["y [adim]"],
            mode="markers+text",
            marker=dict(size=10, symbol="x", color="#DC2626", line=dict(width=1.5, color="#FFFFFF")),
            text=df_lagrange["Punto"],
            textposition="top center",
            textfont=dict(color="#DC2626", size=12),
            name="Puntos Criticos (L1-L5)"
        ))

        fig_roche.update_layout(**LIGHT_LAYOUT)
        fig_roche.update_layout(
            title="Topologia del Potencial de Roche y Curvas de Velocidad Cero (Marco Corrotante)",
            xaxis_title="x [Longitud Caracteristica L]",
            yaxis_title="y [Longitud Caracteristica L]",
            yaxis=dict(scaleanchor="x", scaleratio=1, gridcolor="#E2E8F0", showline=True, linecolor="#0F172A", mirror=True),
            height=580
        )
        st.plotly_chart(fig_roche, use_container_width=True)

# ==============================================================================
# TAB 3: ADQUISICION DE DATOS Y EXPORTACION DE DOCUMENTOS CIENTIFICOS
# ==============================================================================
with tab3:
    st.subheader("Exportacion de Datos Adquiridos y Generacion de Informe")
    st.markdown(
        "Descarga la matriz de estados completa en formato tabular (**CSV**) para post-procesamiento "
        "o genera un **Documento Tecnico de Investigacion (Markdown / LaTeX)** con el resumen estadistico "
        "de invariantes, parametros del integrador y analisis de estabilidad."
    )

    timestamp_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    report_md = rf"""# INFORME DE SIMULACION: DINAMICA GRAVITACIONAL DE N-CUERPOS Y ESTABILIDAD CR3BP
**Fecha de Adquisicion:** {timestamp_str}  
**Configuracion Analizada:** {preset}  
**Esquema de Integracion:** {integrator_name}  

---

## 1. Parametros de Configuracion e Integracion Numerica
- **Paso de integracion ($\Delta t$):** `{dt:.5f}`
- **Numero de pasos muestreados ($N$):** `{n_steps}`
- **Tiempo total simulado ($T_{{max}}$):** `{t_arr[-1]:.4f}`
- **Parametro de suavizado de Plummer ($\epsilon$):** `{eps_soft:.2e}`
- **Vector de masas ($m_1, m_2, m_3$):** `[{masses[0]:.4f}, {masses[1]:.4f}, {masses[2]:.4f}]`

---

## 2. Auditoria de Invariantes Canonicos y Conservacion
| Magnitud Fisica | Valor Inicial ($t = 0$) | Valor Final ($t = T_{{max}}$) | Desviacion Maxima Absoluta / Relativa |
| :--- | :---: | :---: | :---: |
| **Energia Cinetica ($K$)** | `{K_arr[0]:.6f}` | `{K_arr[-1]:.6f}` | `{np.max(K_arr) - np.min(K_arr):.6e}` (Rango) |
| **Energia Potencial ($U$)** | `{U_arr[0]:.6f}` | `{U_arr[-1]:.6f}` | `{np.max(U_arr) - np.min(U_arr):.6e}` (Rango) |
| **Energia Mecanica Total ($E$)** | `{E0:.8f}` | `{E_arr[-1]:.8f}` | `{np.max(np.abs(rel_dE)):.4e}` ($\max \lvert \Delta E / E_0 \rvert$) |
| **Momento Angular Total ($L_z$)** | `{Lz0:.8f}` | `{Lz_arr[-1]:.8f}` | `{np.max(np.abs(abs_dLz)):.4e}` ($\max \lvert \Delta L_z \rvert$) |
| **Cociente del Virial ($2\langle K \rangle / \lvert \langle U \rangle \rvert$)** | - | `{virial_ratio:.6f}` | Referencia de equilibrio ligado: `1.0000` |

---

## 3. Analisis de Puntos de Lagrange (CR3BP para $\mu = {mu_val:.4f}$)
| Punto | Coordenada $x$ | Coordenada $y$ | Potencial Efectivo $\Omega$ | Constante de Jacobi $C_J$ | $\max \operatorname{{Re}}(\lambda)$ | Estado de Estabilidad |
| :---: | :---: | :---: | :---: | :---: | :---: | :--- |
"""
    for _, row in df_lagrange.iterrows():
        report_md += (
            f"| **{row['Punto']}** | `{row['x [adim]']:.6f}` | `{row['y [adim]']:.6f}` | "
            f"`{row['Potencial Omega']:.6f}` | `{row['Constante Jacobi (C_J)']:.6f}` | "
            f"`{row['Max Re(lambda)']:.4e}` | {row['Clasificacion']} |\n"
        )

    report_md += r"""
---

## 4. Formulacion Matematica Empleada
El sistema dinamico se integra resolviendo las ecuaciones diferenciales ordinarias acopladas:

$$m_i \frac{d^2\mathbf{r}_i}{dt^2} = \sum_{j \neq i} \frac{G m_i m_j (\mathbf{r}_j - \mathbf{r}_i)}{\left(\Vert{}\mathbf{r}_j - \mathbf{r}_i\Vert{}^2 + \epsilon^2\right)^{3/2}}$$
"""

    col_dl1, col_dl2, col_dl3 = st.columns(3)
    
    csv_bytes = df_telemetry.to_csv(index=False).encode("utf-8")
    col_dl1.download_button(
        label="DESCARGAR SERIES TEMPORALES (CSV)",
        data=csv_bytes,
        file_name=f"nbody_telemetry_{n_steps}steps.csv",
        mime="text/csv",
        use_container_width=True
    )

    csv_lag_bytes = df_lagrange.to_csv(index=False).encode("utf-8")
    col_dl2.download_button(
        label="DESCARGAR MATRIZ DE LAGRANGE (CSV)",
        data=csv_lag_bytes,
        file_name=f"cr3bp_lagrange_mu_{mu_val:.3f}.csv",
        mime="text/csv",
        use_container_width=True
    )

    col_dl3.download_button(
        label="EXPORTAR INFORME DE INVESTIGACION (.MD)",
        data=report_md.encode("utf-8"),
        file_name="informe_dinamica_gravitacional.md",
        mime="text/markdown",
        use_container_width=True
    )

    st.markdown("#### Vista Previa de la Matriz de Telemetria Adquirida (Primeros 100 registros)")
    st.dataframe(df_telemetry.head(100), use_container_width=True, height=320)
