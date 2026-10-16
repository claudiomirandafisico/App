# 🤖 Simulador de Cinemática Directa y Matrices Homogéneas (3-DOF)

Simulador interactivo en Python que calcula y visualiza en tiempo real la **cinemática directa** de un brazo robótico antropomórfico de 3 grados de libertad utilizando el convenio de **Denavit-Hartenberg (DH)**.

## ✨ Características
- **Visualización 3D Interactiva:** Renderizado espacial con Plotly, incluyendo la tríada ortonormal $(\hat{x}, \hat{y}, \hat{z})$ del efector final.
- **Matrices en Tiempo Real:** Cálculo instantáneo en $\LaTeX$ de las matrices homogéneas individuales ($^0T_1, ^1T_2, ^2T_3$) y la matriz de transformación total ($^0T_3$).
- **Tabla de Denavit-Hartenberg Dinámica:** Seguimiento en vivo de los parámetros $(\theta_i, d_i, a_i, \alpha_i)$.

## 🧮 Fundamento Matemático

Cada articulación se modela mediante la matriz de transformación homogénea de Denavit-Hartenberg:

$$^{i-1}T_i = \begin{bmatrix} \cos\theta_i & -\sin\theta_i\cos\alpha_i & \sin\theta_i\sin\alpha_i & a_i\cos\theta_i \\ \sin\theta_i & \cos\theta_i\cos\alpha_i & -\cos\theta_i\sin\alpha_i & a_i\sin\theta_i \\ 0 & \sin\alpha_i & \cos\alpha_i & d_i \\ 0 & 0 & 0 & 1 \end{bmatrix}$$

## 🚀 Instalación y Ejecución Local

1. Clona este repositorio:
   ```bash
   git clone [https://github.com/TU_USUARIO/robotic-arm-matrix-simulator.git](https://github.com/TU_USUARIO/robotic-arm-matrix-simulator.git)
   cd robotic-arm-matrix-simulator