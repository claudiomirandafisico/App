# Gravitational Dynamics and CR3BP Stability Research Suite

Entorno computacional en Python para la integración numérica del problema gravitacional de $N$-cuerpos, auditoría de invariantes canónicos y análisis espectral de estabilidad lineal en el Problema Circular Restringido de los Tres Cuerpos (CR3BP).

## Capacidades para Investigación

1. **Integradores Simplécticos de Alto Orden:**
   - Implementación del esquema **Yoshida de 4.º orden**, **Velocity-Verlet de 2.º orden**, **Runge-Kutta de 4.º orden (RK4)** y **Euler Explícito**.
   - Monitoreo en tiempo real del error relativo de la energía mecánica total $|(E(t) - E_0)/E_0|$, deriva del momento angular canónico $|\Delta L_z|$ y evaluación estadística del **Teorema del Virial** ($2\langle K \rangle / |\langle U \rangle|$).
2. **Espacio de Configuración y Retrato de Fase:**
   - Visualización simultánea de trayectorias cartesianas $(x, y)$ y retratos de fase radiales $(r, \dot{r})$ sobre lienzo de alto contraste.
3. **Análisis Espectral en Puntos de Lagrange ($L_1 - L_5$):**
   - Resolución numérica de puntos estacionarios mediante Newton-Raphson.
   - Cálculo de la matriz Jacobiana $4\times 4$ del flujo en el espacio de fases corrotante y extracción de sus autovalores $\lambda_i$ para determinar la estabilidad lineal (criterio de Routh $\mu_{\text{crit}} \approx 0.03852$).
4. **Adquisición y Exportación de Telemetría:**
   - Exportación directa de las series temporales completas de vectores de estado $(x_i, y_i, v_{xi}, v_{yi})$ e invariantes dinámicos en formato **CSV**.
   - Generación automatizada de un **Informe Técnico en Markdown/LaTeX** con los resultados de la corrida experimental.

## Ejecución Local

```bash
git clone [https://github.com/TU_USUARIO/gravitational-dynamics-lab.git](https://github.com/TU_USUARIO/gravitational-dynamics-lab.git)
cd gravitational-dynamics-lab
pip install -r requirements.txt
streamlit run app.py