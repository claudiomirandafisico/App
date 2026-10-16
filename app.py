import streamlit as st
import numpy as np
import plotly.graph_objects as go

# Configuración de la página
st.set_page_config(
    page_title="Simulador Cinemático 3-DOF",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 Simulador de Brazo Robótico y Matrices de Transformación (3-DOF)")
st.markdown(
    "Ajusta los ángulos articulares y longitudes de los eslabones en el panel lateral "
    "para observar la **Cinemática Directa** y la actualización en tiempo real de las **Matrices Homogéneas $4\\times 4$**."
)

# --- FUNCIONES MATEMÁTICAS ---
def matriz_dh(theta, d, a, alpha):
    """
    Calcula la matriz de transformación homogénea 4x4 usando parámetros Denavit-Hartenberg (DH).
    theta y alpha en radianes.
    """
    ct, st_ = np.cos(theta), np.sin(theta)
    ca, sa = np.cos(alpha), np.sin(alpha)
    
    return np.array([
        [ct, -st_ * ca,  st_ * sa, a * ct],
        [st_,  ct * ca, -ct * sa, a * st_],
        [0.0,       sa,       ca,       d],
        [0.0,      0.0,      0.0,     1.0]
    ])

def array_to_latex(M, nombre="T"):
    """Convierte un array 4x4 de NumPy a una cadena en formato LaTeX."""
    filas = []
    for fila in M:
        valores = [f"{val:6.3f}" for val in fila]
        filas.append(" & ".join(valores))
    cuerpo = " \\\\ ".join(filas)
    return f"{nombre} = \\begin{{bmatrix}} {cuerpo} \\end{{bmatrix}}"

# --- PANEL LATERAL (CONTROLES) ---
st.sidebar.header("⚙️ Parámetros del Robot")

st.sidebar.subheader("Ángulos Articulares")
theta1_deg = st.sidebar.slider("θ₁ - Base (Cintura) [°]", -180, 180, 35, step=1)
theta2_deg = st.sidebar.slider("θ₂ - Hombro [°]", -90, 90, 40, step=1)
theta3_deg = st.sidebar.slider("θ₃ - Codo [°]", -135, 135, -30, step=1)

st.sidebar.subheader("Longitudes de Eslabones [m]")
L1 = st.sidebar.number_input("L₁ (Altura Base)", min_value=0.5, max_value=5.0, value=2.0, step=0.1)
L2 = st.sidebar.number_input("L₂ (Brazo)", min_value=0.5, max_value=5.0, value=2.5, step=0.1)
L3 = st.sidebar.number_input("L₃ (Antebrazo)", min_value=0.5, max_value=5.0, value=1.8, step=0.1)

# Conversión a radianes
t1 = np.radians(theta1_deg)
t2 = np.radians(theta2_deg)
t3 = np.radians(theta3_deg)

# --- CÁLCULO DE MATRICES (Parámetros DH) ---
# Eslabón 1: Rotación en Z0, altura L1, giro de +90° en X1 para levantar el plano del hombro
T01 = matriz_dh(theta=t1, d=L1, a=0.0, alpha=np.pi/2)
# Eslabón 2: Movimiento en el plano vertical, longitud L2
T12 = matriz_dh(theta=t2, d=0.0, a=L2, alpha=0.0)
# Eslabón 3: Movimiento en el plano vertical, longitud L3
T23 = matriz_dh(theta=t3, d=0.0, a=L3, alpha=0.0)

# Matrices acumuladas respecto a la base {0}
T02 = T01 @ T12
T03 = T02 @ T23  # Matriz Total del Efector Final

# Coordenadas de cada articulación para el gráfico 3D
P0 = np.array([0.0, 0.0, 0.0])      # Origen en el suelo
P1 = T01[:3, 3]                     # Articulación Hombro
P2 = T02[:3, 3]                     # Articulación Codo
P3 = T03[:3, 3]                     # Efector Final (Pinza)

# --- DISEÑO DE LA INTERFAZ ---
col_graf, col_mat = st.columns([1.2, 1])

with col_graf:
    st.subheader("🌐 Visualización 3D en Tiempo Real")
    
    fig = go.Figure()

    # Coordenadas X, Y, Z de la estructura del brazo
    xs = [P0[0], P1[0], P2[0], P3[0]]
    ys = [P0[1], P1[1], P2[1], P3[1]]
    zs = [P0[2], P1[2], P2[2], P3[2]]

    # Dibujar Eslabones y Articulaciones
    fig.add_trace(go.Scatter3d(
        x=xs, y=ys, z=zs,
        mode='lines+markers',
        line=dict(color='#00b4d8', width=12),
        marker=dict(size=[10, 8, 8, 10], color=['#333333', '#f77f00', '#f77f00', '#e63946']),
        name='Estructura Robótica'
    ))

    # Dibujar la tríada de orientación (Ejes X, Y, Z locales) en el Efector Final
    escala_ejes = 0.6
    colores_ejes = ['red', 'green', 'blue']
    nombres_ejes = ['Eje X₃ (Normal)', 'Eje Y₃ (Orientación)', 'Eje Z₃ (Aproximación)']

    for i in range(3):
        vector_eje = T03[:3, i] * escala_ejes
        fig.add_trace(go.Scatter3d(
            x=[P3[0], P3[0] + vector_eje[0]],
            y=[P3[1], P3[1] + vector_eje[1]],
            z=[P3[2], P3[2] + vector_eje[2]],
            mode='lines',
            line=dict(color=colores_ejes[i], width=5),
            name=nombres_ejes[i]
        ))

    # Ajuste del volumen de visualización
    alcance_max = L1 + L2 + L3
    fig.update_layout(
        scene=dict(
            xaxis=dict(range=[-alcance_max, alcance_max], title='X [m]'),
            yaxis=dict(range=[-alcance_max, alcance_max], title='Y [m]'),
            zaxis=dict(range=[0, alcance_max], title='Z [m]'),
            aspectmode='cube'
        ),
        margin=dict(l=0, r=0, b=0, t=10),
        height=550,
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
    )
    st.plotly_chart(fig, use_container_width=True)

with col_mat:
    st.subheader("📐 Matrices de Transformación Homogénea")
    
    # Métricas rápidas de posición del efector final
    c1, c2, c3 = st.columns(3)
    c1.metric("Posición X (pₓ)", f"{P3[0]:.3f} m")
    c2.metric("Posición Y (pᵧ)", f"{P3[1]:.3f} m")
    c3.metric("Posición Z (p_z)", f"{P3[2]:.3f} m")

    st.markdown("#### Matriz Compuesta Total ($^0T_3 = ^0T_1 \\cdot ^1T_2 \\cdot ^2T_3$)")
    st.latex(array_to_latex(T03, "^0T_3"))
    st.caption(
        "📌 **Lectura:** La submatriz superior izquierda de $3\\times 3$ contiene los cosenos directores "
        "(orientación del efector final), mientras que la cuarta columna muestra el vector de posición $(p_x, p_y, p_z)^T$."
    )

    with st.expander("🔍 Ver Matrices Individuales por Articulación ($^0T_1, ^1T_2, ^2T_3$)", expanded=False):
        st.latex(array_to_latex(T01, "^0T_1(\\theta_1)"))
        st.latex(array_to_latex(T12, "^1T_2(\\theta_2)"))
        st.latex(array_to_latex(T23, "^2T_3(\\theta_3)"))

    with st.expander("📋 Ver Tabla de Parámetros Denavit-Hartenberg (DH)", expanded=False):
        st.markdown(f"""
        | Articulación $i$ | $\\theta_i$ (Ángulo) | $d_i$ (Desplazamiento Z) | $a_i$ (Longitud X) | $\\alpha_i$ (Torsión X) |
        | :---: | :---: | :---: | :---: | :---: |
        | **1 (Base)** | ${theta1_deg}^\\circ$ | ${L1:.2f}\\text{{ m}}$ | $0.00\\text{{ m}}$ | $+90^\\circ$ |
        | **2 (Hombro)** | ${theta2_deg}^\\circ$ | $0.00\\text{{ m}}$ | ${L2:.2f}\\text{{ m}}$ | $0^\\circ$ |
        | **3 (Codo)** | ${theta3_deg}^\\circ$ | $0.00\\text{{ m}}$ | ${L3:.2f}\\text{{ m}}$ | $0^\\circ$ |
        """)