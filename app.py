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
    
    # Construccion del Documento de Investigacion usando raw f-string (rf) y sin llaves vacias
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
