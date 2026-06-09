# =============================================================================
# app.py
# Dashboard de Segmentación de Botellas PET en Cuerpos de Agua — V1
# UI construida con Streamlit. Toda la lógica de procesado está en
# procesamiento.py (sin imports de Streamlit).
#
# Fase 1 : carga y visualización de imagen original.
# Fase 2 : conversión a escala de grises.
# Fase 3 : motor de filtrado espacial acumulativo (7 filtros).
# =============================================================================

import numpy as np
import cv2
import streamlit as st

from procesamiento import (
    # Constante de configuración
    MAX_ANCHO_PX,
    # Fase 1 y 2
    redimensionar_imagen,
    bgr_a_rgb,
    convertir_a_gris,
    calcular_histograma,
    # Fase 3
    aplicar_filtro,
    descripcion_filtro,
    NOMBRES_FILTROS,
    # Fase 3.5 — FFT
    fft_filter,
    crear_espectro_con_mascara,
    # Fase 4 — Enhancement
    aplicar_mejora,
    calcular_histograma_comparativo,
    MEJORAS_OPCIONES,
    # Fase 5 — Segmentación
    aplicar_umbral,
    aplicar_cierre,
    aplicar_apertura,
    aplicar_relleno_huecos,
    METODOS_UMBRALIZACION,
    METODOS_AUTOMATICOS,
)


# =============================================================================
# 1. CONFIGURACIÓN GLOBAL DE LA PÁGINA
# =============================================================================
st.set_page_config(
    page_title="Dashboard · Botellas PET",
    page_icon="🍶",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS personalizado ─────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Tarjetas del historial de filtros en el sidebar */
    .filtro-card {
        background: rgba(255,255,255,0.04);
        border: 1px solid rgba(255,255,255,0.09);
        border-radius: 8px;
        padding: 0.45rem 0.8rem;
        margin-bottom: 0.35rem;
        font-size: 0.82rem;
        line-height: 1.5;
    }
    /* Badge de dimensiones debajo de las imágenes */
    .badge {
        display: inline-block;
        padding: 2px 10px;
        border-radius: 999px;
        font-size: 0.78rem;
        font-weight: 600;
        background: rgba(59,130,246,0.18);
        color: #93c5fd;
        margin-right: 6px;
    }
    /* Badge naranja para imagen de entrada de mejora */
    .badge-entrada {
        display: inline-block;
        padding: 2px 10px;
        border-radius: 999px;
        font-size: 0.78rem;
        font-weight: 600;
        background: rgba(251,146,60,0.2);
        color: #fb923c;
        margin-right: 6px;
    }
    /* Badge morado para imagen de salida de mejora */
    .badge-salida {
        display: inline-block;
        padding: 2px 10px;
        border-radius: 999px;
        font-size: 0.78rem;
        font-weight: 600;
        background: rgba(167,139,250,0.2);
        color: #a78bfa;
        margin-right: 6px;
    }
    /* Separadores entre pasos */
    hr.paso-sep {
        border: none;
        border-top: 1px solid rgba(255,255,255,0.1);
        margin: 1.6rem 0;
    }
</style>
""", unsafe_allow_html=True)


# =============================================================================
# 2. INICIALIZACIÓN DE SESSION STATE
# =============================================================================
# Historial de filtros: lista de dicts {"tipo": str, <params>}
if "historial_filtros" not in st.session_state:
    st.session_state["historial_filtros"] = []

# Historial de mejoras: lista de dicts {"tipo": str, <params>}
if "historial_mejoras" not in st.session_state:
    st.session_state["historial_mejoras"] = []


# =============================================================================
# 3. BARRA LATERAL (SIDEBAR)
# =============================================================================
with st.sidebar:
    st.title("🍶 Botellas PET")
    st.caption("Dashboard de Segmentación — V1")
    st.divider()

    # =========================================================================
    # § 1 — Carga de imagen (Fase 1)
    # =========================================================================
    st.markdown("#### 📂 Fase 1 — Imagen")
    archivo = st.file_uploader(
        "Sube una imagen de botella PET",
        type=["png", "jpg", "jpeg"],
        help=(
            f"Formatos: PNG, JPG, JPEG. "
            f"Si el ancho supera {MAX_ANCHO_PX}px se redimensiona automáticamente."
        ),
    )

    st.divider()

    # =========================================================================
    # § 2 — Preprocesado visual (Fase 2)
    # =========================================================================
    st.markdown("#### ⬛ Fase 2 — Preprocesado")
    modo_visualizacion = st.selectbox(
        "Modo de visualización",
        options=["Sin preprocesado (color)", "Escala de grises"],
        index=0,
        key="sel_modo_visual",
        help=(
            "El pipeline SIEMPRE trabaja internamente en escala de grises. "
            "Esta opción solo afecta cómo se muestra el Paso 2 en pantalla."
        ),
    )

    st.divider()

    # =========================================================================
    # § 3 — Filtros espaciales acumulativos (Fase 3)
    # =========================================================================
    st.markdown("#### 🧩 Fase 3 — Filtros espaciales")
    st.caption("Cada filtro opera sobre la salida del anterior.")

    # ── Selector del tipo de filtro ───────────────────────────────────────────
    filtro_elegido = st.selectbox(
        "Tipo de filtro",
        options=NOMBRES_FILTROS,
        index=0,
        key="sel_filtro_tipo",
    )

    # ── Controles dinámicos según el filtro elegido ───────────────────────────
    config_nuevo = {"tipo": filtro_elegido}   # se construye progresivamente

    KERNELS_IMPARES = [3, 5, 7, 9]

    if filtro_elegido == "Gaussiano":
        # Gaussiano necesita ksize y sigma
        ksize_g = st.selectbox(
            "Tamaño del kernel (ksize)",
            options=KERNELS_IMPARES,
            index=0,
            key="sel_ksize_gaussiano",
            help="El kernel debe ser cuadrado e impar.",
        )
        sigma_g = st.slider(
            "Sigma (σ)",
            min_value=0.0,
            max_value=5.0,
            value=0.0,
            step=0.5,
            key="slider_sigma_gaussiano",
            help="σ=0 → OpenCV calcula el valor automáticamente desde ksize.",
        )
        config_nuevo["ksize"] = ksize_g
        config_nuevo["sigma"] = sigma_g
        st.caption(
            f"σ={sigma_g if sigma_g > 0 else 'auto'} · "
            f"Mayor ksize = más suavizado."
        )

    elif filtro_elegido == "Bilateral":
        # Bilateral necesita d, sigma_color y sigma_space
        d_bil = st.selectbox(
            "Diámetro de vecindad (d)",
            options=[5, 7, 9, 11, 15],
            index=2,
            key="sel_d_bilateral",
            help="d más grande = mayor área considerada (más lento).",
        )
        sc_bil = st.slider(
            "σ Color (sigma_color)",
            min_value=1,
            max_value=150,
            value=75,
            step=5,
            key="slider_sc_bilateral",
            help=(
                "Rango de intensidades 'similares'. "
                "Bajo → bordes muy preservados. "
                "Alto → más suavizado."
            ),
        )
        ss_bil = st.slider(
            "σ Espacio (sigma_space)",
            min_value=1,
            max_value=150,
            value=75,
            step=5,
            key="slider_ss_bilateral",
            help="Radio de influencia espacial. Alto → vecindad más amplia.",
        )
        config_nuevo["d"]           = d_bil
        config_nuevo["sigma_color"] = float(sc_bil)
        config_nuevo["sigma_space"] = float(ss_bil)
        st.caption(
            f"d={d_bil} · σC={sc_bil} · σS={ss_bil} · "
            f"{'Bordes muy preservados' if sc_bil < 50 else 'Suavizado moderado'}"
        )

    elif filtro_elegido == "Mediana":
        # Mediana solo necesita ksize
        ksize_med = st.selectbox(
            "Tamaño del kernel (ksize)",
            options=KERNELS_IMPARES,
            index=0,
            key="sel_ksize_mediana",
        )
        config_nuevo["ksize"] = ksize_med
        st.caption("Excelente contra brillos puntuales del agua.")

    else:
        # Resto de filtros (Paso Bajas, Promediador, Max, Min): solo ksize
        ksize_gen = st.selectbox(
            "Tamaño del kernel (ksize)",
            options=KERNELS_IMPARES,
            index=0,
            key="sel_ksize_general",
        )
        config_nuevo["ksize"] = ksize_gen

    # ── Botones de acción ─────────────────────────────────────────────────────
    col_añadir, col_limpiar = st.columns(2)

    with col_añadir:
        if st.button("➕ Añadir filtro", key="btn_add_filtro"):
            st.session_state["historial_filtros"].append(
                dict(config_nuevo)   # copia del dict para evitar referencias
            )
            st.toast(f"Filtro '{filtro_elegido}' añadido ✅", icon="🧩")

    with col_limpiar:
        if st.button("🗑️ Limpiar filtros", key="btn_clear_filtros"):
            st.session_state["historial_filtros"] = []
            st.toast("Pipeline de filtros limpiado.", icon="🗑️")

    # ── Historial visual del pipeline ─────────────────────────────────────────
    st.divider()
    if st.session_state["historial_filtros"]:
        st.markdown("**Pipeline activo**")
        for i, cfg in enumerate(st.session_state["historial_filtros"]):
            # Construimos el string de parámetros (excluimos "tipo")
            params_str = " · ".join(
                f"{k}={v}" for k, v in cfg.items() if k != "tipo"
            )
            col_info, col_del = st.columns([4, 1])
            with col_info:
                st.markdown(
                    f'<div class="filtro-card">'
                    f'<b>{i+1}. {cfg["tipo"]}</b>'
                    + (f'<br><span style="color:#94a3b8">{params_str}</span>'
                       if params_str else "")
                    + '</div>',
                    unsafe_allow_html=True,
                )
            with col_del:
                # Botón individual para eliminar solo este filtro
                if st.button("✕", key=f"btn_del_filtro_{i}",
                             help="Eliminar este filtro del pipeline"):
                    st.session_state["historial_filtros"].pop(i)
                    st.rerun()   # refresca el sidebar para actualizar la lista
    else:
        st.caption("Sin filtros en el pipeline todavía.")

    st.divider()

    # =========================================================================
    # § 3.5 — Filtro FFT (entre Fase 3 y Fase 4)
    # =========================================================================
    st.markdown("#### 🌀 Fase 3.5 — Filtro FFT")
    st.caption("Filtra la imagen en el dominio de la frecuencia.")

    # Checkbox de activación del filtro FFT
    fft_activo = st.checkbox(
        "Aplicar filtro FFT",
        value=False,
        key="chk_fft_activo",
        help="Aplica un filtro ideal paso bajas o paso altas en el dominio de Fourier.",
    )

    # Controles FFT (siempre visibles para pre-configurar antes de activar)
    tipo_fft = st.radio(
        "Tipo de filtro",
        options=["lowpass", "highpass"],
        index=0,
        key="radio_fft_tipo",
        horizontal=True,
    )
    cutoff_fft = st.slider(
        "Cutoff (fracción del radio)",
        min_value=0.01,
        max_value=1.00,
        value=0.15,
        step=0.01,
        key="slider_fft_cutoff",
        help="Fracción del radio mínimo de la imagen que define la frecuencia de corte.",
    )
    # Caption explicativo según el tipo elegido
    if tipo_fft == "lowpass":
        st.caption(
            f"🟢 **Paso Bajas** — cutoff={cutoff_fft:.2f}. "
            "Conserva las bajas frecuencias (formas gruesas, colores planos). "
            "Suaviza y elimina ruido de alta frecuencia."
        )
    else:
        st.caption(
            f"🔴 **Paso Altas** — cutoff={cutoff_fft:.2f}. "
            "Conserva las altas frecuencias (bordes, texturas). "
            "Elimina la iluminación de fondo (baja frecuencia)."
        )

    st.divider()

    # =========================================================================
    # § 4 — Mejora de contraste y brillo (Fase 4, acumulativa)
    # =========================================================================
    st.markdown("#### ✨ Fase 4 — Mejora de Contraste")
    st.caption("Cada mejora opera sobre la salida de la anterior.")

    # Selector de mejora
    mejora_elegida = st.selectbox(
        "Técnica de mejora",
        options=MEJORAS_OPCIONES,
        index=0,
        key="sel_mejora",
    )

    # Controles dinámicos según la mejora elegida
    config_mejora = {"tipo": mejora_elegida}

    if mejora_elegida == "Corrección Gamma":
        gamma_val = st.slider(
            "Gamma (γ)",
            min_value=0.1,
            max_value=4.0,
            value=1.0,
            step=0.05,
            key="slider_gamma",
            help="γ<1 aclara, γ=1 identidad, γ>1 oscurece.",
        )
        config_mejora["gamma"] = float(gamma_val)
        st.caption(
            f"γ={gamma_val:.2f} — "
            + ("Aclarando sombras" if gamma_val < 1.0
               else "Identidad" if gamma_val == 1.0
               else "Oscureciendo luces")
        )

    elif mejora_elegida == "Desplazamiento (Brillo)":
        delta_val = st.slider(
            "Delta (Δ brillo)",
            min_value=-120,
            max_value=120,
            value=0,
            step=5,
            key="slider_delta",
            help="+Δ aclara la imagen. -Δ la oscurece.",
        )
        config_mejora["delta"] = int(delta_val)
        st.caption(
            f"Δ={delta_val:+d} — "
            + ("Aclarando" if delta_val > 0
               else "Sin cambio" if delta_val == 0
               else "Oscureciendo")
        )

    elif mejora_elegida == "Contracción / Expansión":
        # Cuatro number_inputs para los rangos de entrada y salida
        col_a, col_b = st.columns(2)
        with col_a:
            a_in = st.number_input(
                "a_in", min_value=0, max_value=254,
                value=50, step=5, key="num_a_in",
                help="Límite inferior del rango de entrada.",
            )
            a_out = st.number_input(
                "a_out", min_value=0, max_value=255,
                value=0, step=5, key="num_a_out",
                help="Límite inferior del rango de salida.",
            )
        with col_b:
            b_in = st.number_input(
                "b_in", min_value=1, max_value=255,
                value=200, step=5, key="num_b_in",
                help="Límite superior del rango de entrada.",
            )
            b_out = st.number_input(
                "b_out", min_value=0, max_value=255,
                value=255, step=5, key="num_b_out",
                help="Límite superior del rango de salida.",
            )
        # Validación: a_in debe ser menor que b_in
        if int(a_in) >= int(b_in):
            st.warning("⚠️ a_in debe ser menor que b_in.", icon="⚠️")
        config_mejora["a_in"]  = int(a_in)
        config_mejora["b_in"]  = int(b_in)
        config_mejora["a_out"] = int(a_out)
        config_mejora["b_out"] = int(b_out)
        st.caption(f"[{int(a_in)},{int(b_in)}] → [{int(a_out)},{int(b_out)}]")

    else:
        # Ecualizaciones: sin parámetros adicionales
        st.caption(f"ℹ️ {mejora_elegida} no requiere parámetros adicionales.")

    # Botones Añadir / Limpiar mejoras
    col_add_m, col_clr_m = st.columns(2)
    with col_add_m:
        if st.button("➕ Añadir mejora", key="btn_add_mejora"):
            st.session_state["historial_mejoras"].append(dict(config_mejora))
            st.toast(f"Mejora '{mejora_elegida}' añadida ✅", icon="✨")
    with col_clr_m:
        if st.button("🗑️ Limpiar mejoras", key="btn_clr_mejoras"):
            st.session_state["historial_mejoras"] = []
            st.toast("Pipeline de mejoras limpiado.", icon="🗑️")

    # Historial visual de mejoras con botón ✕ individual
    st.divider()
    if st.session_state["historial_mejoras"]:
        st.markdown("**Pipeline de mejoras activo**")
        for j, cfg_m in enumerate(st.session_state["historial_mejoras"]):
            params_m = " · ".join(
                f"{k}={v}" for k, v in cfg_m.items() if k != "tipo"
            )
            col_info_m, col_del_m = st.columns([4, 1])
            with col_info_m:
                st.markdown(
                    f'<div class="filtro-card">'
                    f'<b>{j+1}. {cfg_m["tipo"]}</b>'
                    + (f'<br><span style="color:#94a3b8">{params_m}</span>'
                       if params_m else "")
                    + '</div>',
                    unsafe_allow_html=True,
                )
            with col_del_m:
                if st.button("✕", key=f"btn_del_mejora_{j}",
                             help="Eliminar esta mejora del pipeline"):
                    st.session_state["historial_mejoras"].pop(j)
                    st.rerun()
    else:
        st.caption("Sin mejoras en el pipeline todavía.")

    st.divider()

    # =========================================================================
    # § 5 — Segmentación (Fase 5)
    # =========================================================================
    st.markdown("#### 🎯 Fase 5 — Segmentación")
    st.caption("Opera sobre `img_mejorada` (salida de Fase 4).")

    # ── Selector del método de umbralización ─────────────────────────────────
    metodo_umbral = st.selectbox(
        "Método de umbralización",
        options=METODOS_UMBRALIZACION,
        index=0,
        key="sel_metodo_umbral",
        help=(
            "Otsu/Kapur/Media → automático. "
            "Banda → rango [T1,T2] ideal para botellas. "
            "Manual → umbral fijo."
        ),
    )

    # ── Controles dinámicos según el método elegido ───────────────────────────
    if metodo_umbral == "Manual":
        umbral_manual_val = st.slider(
            "Umbral manual",
            min_value=0,
            max_value=255,
            value=127,
            step=1,
            key="slider_umbral_manual",
            help="Píxeles con intensidad > umbral → blanco (objeto).",
        )

    elif metodo_umbral == "Banda":
        banda_t1 = st.slider(
            "T1 — límite inferior",
            min_value=0,
            max_value=253,
            value=80,
            step=1,
            key="slider_banda_t1",
            help="Intensidad mínima del rango de la botella.",
        )
        banda_t2 = st.slider(
            "T2 — límite superior",
            min_value=banda_t1 + 1,
            max_value=255,
            value=max(banda_t1 + 1, 200),
            step=1,
            key="slider_banda_t2",
            help="Intensidad máxima del rango de la botella.",
        )
        st.caption(
            f"🟦 Rango activo: **[{banda_t1}, {banda_t2}]** "
            f"— amplitud = {banda_t2 - banda_t1} niveles."
        )

    else:
        # Otsu, Kapur, Media
        st.info(
            "Umbral calculado automáticamente al procesar la imagen.",
            icon="🤖",
        )

    # ── Inversión de la máscara ───────────────────────────────────────────────
    invertir_mascara = st.checkbox(
        "Invertir máscara",
        value=False,
        key="chk_invertir",
        help="Intercambia fondo y objeto en la máscara binaria.",
    )

    st.markdown("**Morfología post-umbralización**")

    # ── Cierre morfológico ────────────────────────────────────────────────────
    aplicar_cierre_chk = st.checkbox(
        "Aplicar cierre (rellena huecos)",
        value=True,
        key="chk_cierre",
        help="Dilatación + Erosión. Sella discontinuidades dentro de la botella.",
    )
    ksize_cierre = st.slider(
        "Kernel cierre",
        min_value=3,
        max_value=15,
        value=7,
        step=2,
        key="slider_kcierre",
        help="Tamaño del elemento estructurante elíptico para el cierre.",
    )

    # ── Apertura morfológica ──────────────────────────────────────────────────
    aplicar_apertura_chk = st.checkbox(
        "Aplicar apertura (elimina ruido)",
        value=False,
        key="chk_apertura",
        help="Erosión + Dilatación. Elimina manchas de ruido pequeñas.",
    )
    ksize_apertura = st.slider(
        "Kernel apertura",
        min_value=3,
        max_value=11,
        value=3,
        step=2,
        key="slider_kapertura",
        help="Tamaño del elemento estructurante elíptico para la apertura.",
    )

    # ── Relleno de huecos internos ────────────────────────────────────────────
    aplicar_relleno_chk = st.checkbox(
        "Rellenar huecos internos",
        value=False,
        key="chk_relleno",
        help="Flood-fill desde el exterior para cerrar cavidades internas.",
    )

    st.divider()

    # ── Resumen del pipeline ──────────────────────────────────────────────────
    n_filtros  = len(st.session_state["historial_filtros"])
    n_mejoras  = len(st.session_state["historial_mejoras"])
    fft_estado = "✅ activa" if st.session_state.get("chk_fft_activo", False) else "❌ inactiva"
    st.markdown(
        f"**Resumen del pipeline**\n\n"
        f"1. 📥 Carga + redimensión\n"
        f"2. ⬛ Escala de grises\n"
        f"3. 🧩 Filtros espaciales: _{n_filtros} aplicado(s)_\n"
        f"3.5 🌀 FFT: _{fft_estado}_\n"
        f"4. ✨ Mejoras: _{n_mejoras} aplicada(s)_\n"
        f"5. 🎯 Segmentación: _{metodo_umbral}_"
    )


# =============================================================================
# 4. FUNCIÓN AUXILIAR: mostrar_paso_ui
#    Muestra un paso del pipeline con imagen a la izquierda e
#    histograma a la derecha en proporción [2, 1].
# =============================================================================
def mostrar_paso_ui(
    titulo: str,
    imagen: np.ndarray,
    descripcion: str,
    es_gris: bool = False,
    key_hist: str = "",
) -> None:
    """
    Renderiza un paso del pipeline: título, descripción,
    imagen e histograma en dos columnas (proporción 2:1).

    REGLAS aplicadas aquí:
    · st.image usa width="stretch"  (NO use_container_width)
    · st.plotly_chart usa width="stretch" con key único
    · st.image NO lleva key

    Parámetros
    ----------
    titulo      : str          — encabezado del paso.
    imagen      : np.ndarray   — array RGB (color) o 2D (grises).
    descripcion : str          — texto descriptivo en Markdown.
    es_gris     : bool         — True si la imagen es monocromática.
    key_hist    : str          — key único para el st.plotly_chart.
    """
    st.markdown(f"### {titulo}")
    st.caption(descripcion)

    col_img, col_hist = st.columns([2, 1])

    with col_img:
        # st.image: width="stretch", SIN key
        st.image(
            imagen,
            width="stretch",
            clamp=True,
            channels="GRAY" if es_gris else "RGB",
        )
        # Badge con dimensiones
        h, w = imagen.shape[:2]
        st.markdown(
            f'<span class="badge">📐 {w} × {h} px</span>',
            unsafe_allow_html=True,
        )

    with col_hist:
        # Para calcular el histograma en grises pasamos el array 2D directamente.
        # Para color pasamos el RGB (el histograma espera RGB, no BGR).
        entrada_hist = imagen   # ya es 2D si es_gris, o RGB si color
        if not es_gris and imagen.ndim == 3:
            # La imagen ya está en RGB aquí, calcular_histograma espera RGB
            entrada_hist = imagen

        fig = calcular_histograma(entrada_hist)

        # st.plotly_chart: width="stretch" y key único obligatorio
        st.plotly_chart(
            fig,
            width="stretch",
            key=f"chart_hist_{key_hist}",
        )

    # Separador entre pasos
    st.markdown('<hr class="paso-sep">', unsafe_allow_html=True)


# =============================================================================
# 5. FUNCIÓN CACHEADA: cargar y preprocesar imagen
#    @st.cache_data evita repetir la decodificación y conversión
#    cada vez que Streamlit re-renderiza el script.
#    La clave de caché son los bytes crudos → si el archivo no cambia,
#    no se reprocesa.
# =============================================================================
@st.cache_data(show_spinner="Cargando imagen…")
def cargar_y_preprocesar(bytes_imagen: bytes):
    """
    Decodifica los bytes del archivo subido, redimensiona si es necesario
    y genera la versión en escala de grises.

    Parámetros
    ----------
    bytes_imagen : bytes — contenido binario del archivo subido.

    Retorna
    -------
    (imagen_rgb, imagen_gris) como arrays NumPy.

    Raises
    ------
    ValueError si OpenCV no puede decodificar la imagen.
    """
    # Convertimos bytes → array NumPy → decodificamos como BGR
    arr = np.frombuffer(bytes_imagen, dtype=np.uint8)
    imagen_bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)

    if imagen_bgr is None:
        raise ValueError(
            "No se pudo decodificar la imagen. "
            "Asegúrate de que el archivo sea PNG, JPG o JPEG válido."
        )

    # Redimensionar si es necesario (mantiene relación de aspecto)
    imagen_bgr = redimensionar_imagen(imagen_bgr)

    # Convertir a RGB para Streamlit
    imagen_rgb = bgr_a_rgb(imagen_bgr)

    # Convertir a escala de grises para el pipeline de procesado
    imagen_gris = convertir_a_gris(imagen_bgr)

    return imagen_rgb, imagen_gris


# =============================================================================
# 6. FUNCIÓN CACHEADA: aplicar un filtro con caché
#    Se cachea por (imagen_gris_bytes, config_tuple).
#    config se convierte a tuple de pares porque los dicts no son hasheables.
# =============================================================================
@st.cache_data(show_spinner=False)
def aplicar_filtro_cacheado(
    imagen_gris: np.ndarray,
    config_tuple: tuple,
) -> np.ndarray:
    """
    Versión cacheada del dispatcher de filtros.

    Parámetros
    ----------
    imagen_gris  : np.ndarray — imagen de entrada.
    config_tuple : tuple      — tuple(sorted(config.items())), hasheable.

    Retorna
    -------
    np.ndarray — imagen filtrada.
    """
    config = dict(config_tuple)
    return aplicar_filtro(imagen_gris, config)


# =============================================================================
# 6b. FUNCIÓN CACHEADA: aplicar una mejora con caché
#     Se cachea por (imagen_gris_bytes, config_tuple).
# =============================================================================
@st.cache_data(show_spinner=False)
def aplicar_mejora_cacheada(
    imagen_gris: np.ndarray,
    config_tuple: tuple,
) -> np.ndarray:
    """
    Versión cacheada del dispatcher de mejoras.

    Parámetros
    ----------
    imagen_gris  : np.ndarray — imagen de entrada.
    config_tuple : tuple      — tuple(sorted(config.items())), hasheable.

    Retorna
    -------
    np.ndarray — imagen mejorada.
    """
    config = dict(config_tuple)
    return aplicar_mejora(imagen_gris, config)


# =============================================================================
# 7. FLUJO PRINCIPAL — función main()
# =============================================================================
def main() -> None:
    """
    Función principal que controla el flujo de renderizado del dashboard.
    Se ejecuta en cada re-render de Streamlit.
    """
    # ── Título y descripción ──────────────────────────────────────────────────
    st.title("🍶 Dashboard · Segmentación de Botellas PET")
    st.markdown(
        "Pipeline de procesamiento digital de imágenes para detectar y segmentar "
        "botellas PET en cuerpos de agua usando **solo técnicas clásicas de PDI** "
        "(sin machine learning). "
        "Sube una imagen desde la barra lateral para comenzar."
    )

    # ── Sin imagen cargada: mostrar instrucciones ─────────────────────────────
    if archivo is None:
        st.info(
            "👈 **Empieza subiendo una imagen** desde la barra lateral.\n\n"
            "**Consejos para mejores resultados:**\n"
            "- Busca imágenes con contraste claro entre la botella y el agua\n"
            "- Botellas oscuras sobre agua clara o botellas claras sobre agua oscura\n"
            "- Términos de búsqueda: *plastic bottle dark water top view*, "
            "*PET bottle river bank*, *water pollution plastic bottle close up*",
            icon="📂",
        )
        return

    # ── Cargar y preprocesar ──────────────────────────────────────────────────
    try:
        bytes_imagen = archivo.read()
        imagen_rgb, imagen_gris = cargar_y_preprocesar(bytes_imagen)
    except ValueError as e:
        st.error(f"❌ Error al procesar la imagen: {e}")
        return

    # Guardamos en session_state para que fases futuras puedan acceder
    st.session_state["img_original"] = imagen_rgb
    st.session_state["img_gris"]     = imagen_gris

    # ── Notificación de redimensión ───────────────────────────────────────────
    alto, ancho = imagen_rgb.shape[:2]
    if ancho == MAX_ANCHO_PX:
        st.warning(
            f"⚠️ Imagen redimensionada a **{ancho} × {alto} px** "
            f"(ancho máximo: {MAX_ANCHO_PX}px).",
            icon="📐",
        )
    else:
        st.success(
            f"✅ Imagen cargada correctamente · **{ancho} × {alto} px**",
            icon="🖼️",
        )

    st.divider()

    # ─────────────────────────────────────────────────────────────────────────
    # PASO 1 · Imagen Original (color, RGB)
    # ─────────────────────────────────────────────────────────────────────────
    mostrar_paso_ui(
        titulo="Paso 1 · Imagen Original",
        imagen=st.session_state["img_original"],
        descripcion=(
            "Imagen cargada en color. El histograma RGB muestra la distribución "
            "de intensidades por canal. **Observa si hay separación clara** entre "
            "el pico de la botella y el pico del fondo (agua): eso indica que "
            "la segmentación será más sencilla."
        ),
        es_gris=False,
        key_hist="paso1_original",
    )

    # ─────────────────────────────────────────────────────────────────────────
    # PASO 2 · Escala de grises
    # ─────────────────────────────────────────────────────────────────────────
    # El pipeline siempre trabaja en grises. Este paso lo hace visible.
    mostrar_paso_ui(
        titulo="Paso 2 · Escala de Grises",
        imagen=st.session_state["img_gris"],
        descripcion=(
            "Conversión perceptual: **0.299·R + 0.587·G + 0.114·B** "
            "(ponderación ITU-R BT.601). "
            "A partir de aquí, todos los filtros y la segmentación "
            "operan sobre esta imagen en grises. "
            "El histograma muestra la distribución de luminancia."
        ),
        es_gris=True,
        key_hist="paso2_gris",
    )

    # ─────────────────────────────────────────────────────────────────────────
    # PASOS 3..N · Fase 3 — Pipeline de filtros espaciales acumulativos
    # ─────────────────────────────────────────────────────────────────────────
    historial = st.session_state["historial_filtros"]

    if not historial:
        # Sin filtros: avisamos al usuario y guardamos img_filtrada = img_gris
        # NO hacemos return: las Fases 3.5 y 4 deben ejecutarse igualmente
        st.info(
            "🧩 **Sin filtros en el pipeline.** "
            "Usa la sección 'Fase 3 — Filtros espaciales' de la barra lateral "
            "para añadir filtros. Cada filtro se aplica sobre el resultado del anterior.\n\n"
            "**Estrategia recomendada para botellas PET:**\n"
            "1. Mediana (ksize=5) → elimina brillos puntuales del agua\n"
            "2. Bilateral (d=9) → suaviza el fondo preservando el borde de la botella\n"
            "3. Gaussiano (ksize=3) → suavizado final suave",
            icon="ℹ️",
        )
        st.session_state["img_filtrada"] = imagen_gris
        # Continuar al bloque de Fase 3.5 y Fase 4

    # Si hay filtros: mostramos el encabezado y ejecutamos el pipeline
    if historial:
        st.markdown("---")
        st.markdown("## 🧩 Fase 3 · Filtros Espaciales")
        st.caption(
            f"Aplicando {len(historial)} filtro(s) de forma acumulativa. "
            "Cada paso muestra el resultado y su histograma."
        )

        # Imagen de entrada al pipeline: la imagen en grises del Paso 2
        img_actual = imagen_gris.copy()

        for paso_idx, config in enumerate(historial):
            numero_paso = paso_idx + 3   # Pasos 1 y 2 ya se mostraron
            tipo_f = config["tipo"]

            # Convertimos config a tuple para que sea hasheable en la caché
            config_tuple = tuple(sorted(config.items()))

            # Aplicamos el filtro (con caché: si no cambió, no recalcula)
            with st.spinner(
                f"Aplicando filtro {paso_idx+1}/{len(historial)}: {tipo_f}…"
            ):
                img_actual = aplicar_filtro_cacheado(img_actual, config_tuple)

            # Construimos el string de parámetros para el título
            params_str = ", ".join(
                f"{k}={v}" for k, v in config.items() if k != "tipo"
            )

            mostrar_paso_ui(
                titulo=(
                    f"Paso {numero_paso} · Filtro {paso_idx+1}: {tipo_f}"
                    + (f" ({params_str})" if params_str else "")
                ),
                imagen=img_actual,
                descripcion=descripcion_filtro(tipo_f, config),
                es_gris=True,
                # Key único usando el índice del paso y el tipo de filtro
                key_hist=f"paso{numero_paso}_{tipo_f.lower()}_{paso_idx}",
            )

        # Guardamos el resultado final del pipeline de filtros
        st.session_state["img_filtrada"] = img_actual

        st.success(
            f"✅ Fase 3 completada · {len(historial)} filtro(s) aplicado(s). "
            "Resultado guardado en `img_filtrada`.",
            icon="🏁",
        )


    # ─────────────────────────────────────────────────────────────────────────
    # FASE 3.5 · Filtrado FFT (si está activo)
    # ─────────────────────────────────────────────────────────────────────────
    fft_activo = st.session_state.get("chk_fft_activo", False)
    tipo_fft   = st.session_state.get("radio_fft_tipo", "lowpass")
    cutoff_fft = st.session_state.get("slider_fft_cutoff", 0.15)

    if fft_activo:
        st.markdown("---")
        st.markdown("## 🌀 Fase 3.5 · Filtrado FFT")
        st.caption(
            f"Tipo: **{tipo_fft}** · Cutoff: **{cutoff_fft:.2f}** · "
            "Dominio de frecuencia (Transformada de Fourier Discreta 2D)."
        )

        with st.spinner("Calculando FFT…"):
            img_fft, mascara_fft, espectro_fft = fft_filter(
                st.session_state["img_filtrada"],
                cutoff=cutoff_fft,
                tipo=tipo_fft,
            )
            # Espectro con el borde de la máscara superpuesto en rojo
            espectro_con_borde = crear_espectro_con_mascara(espectro_fft, mascara_fft)

        col_esp, col_fft = st.columns(2)

        with col_esp:
            st.markdown("**Espectro de magnitud (log)** con máscara")
            # Calcular porcentaje de frecuencias conservadas
            pct_conservadas = mascara_fft.mean() * 100.0
            st.image(espectro_con_borde, width="stretch")
            st.caption(
                f"🔴 Línea roja = límite de corte (cutoff={cutoff_fft:.2f}). "
                f"Se conservan el **{pct_conservadas:.1f} %** de las frecuencias."
            )

        with col_fft:
            st.markdown("**Imagen resultante (dominio espacial)**")
            st.image(img_fft, width="stretch", clamp=True, channels="GRAY")
            # Métricas antes / después
            img_antes_fft = st.session_state["img_filtrada"]
            mu_antes  = float(img_antes_fft.mean())
            mu_despues = float(img_fft.mean())
            sig_antes  = float(img_antes_fft.std())
            sig_despues = float(img_fft.std())
            m1, m2 = st.columns(2)
            m1.metric(
                label="μ (media)",
                value=f"{mu_despues:.1f}",
                delta=f"{mu_despues - mu_antes:+.1f}",
            )
            m2.metric(
                label="σ (desviación)",
                value=f"{sig_despues:.1f}",
                delta=f"{sig_despues - sig_antes:+.1f}",
            )

        # La imagen FFT reemplaza la imagen filtrada para la siguiente fase
        st.session_state["img_filtrada"] = img_fft
        st.markdown('<hr class="paso-sep">', unsafe_allow_html=True)

    # ─────────────────────────────────────────────────────────────────────────
    # FASE 4 · Mejora de Contraste y Brillo (acumulativa)
    # ─────────────────────────────────────────────────────────────────────────
    historial_mejoras = st.session_state["historial_mejoras"]

    if historial_mejoras:
        st.markdown("---")
        st.markdown("## ✨ Fase 4 · Mejora de Contraste y Brillo")
        st.caption(
            f"Aplicando {len(historial_mejoras)} mejora(s) de forma acumulativa "
            "sobre el resultado de la Fase 3/3.5."
        )

        # Imagen de entrada al pipeline de mejoras
        img_mejora_actual = st.session_state["img_filtrada"].copy()

        for m_idx, cfg_m in enumerate(historial_mejoras):
            tipo_m = cfg_m["tipo"]
            params_m_str = ", ".join(
                f"{k}={v}" for k, v in cfg_m.items() if k != "tipo"
            )

            st.markdown(
                f"### Mejora {m_idx+1}: {tipo_m}"
                + (f" ({params_m_str})" if params_m_str else "")
            )

            # Imagen antes de la mejora (entrada)
            img_entrada_m = img_mejora_actual.copy()

            # Aplicar mejora (con caché)
            with st.spinner(f"Aplicando {tipo_m}…"):
                config_m_tuple = tuple(sorted(cfg_m.items()))
                img_salida_m = aplicar_mejora_cacheada(
                    img_entrada_m, config_m_tuple
                )

            # Columnas: entrada (izquierda) / salida (derecha)
            col_ent, col_sal = st.columns(2)

            with col_ent:
                st.markdown('<span class="badge-entrada">Entrada</span>',
                            unsafe_allow_html=True)
                st.image(
                    img_entrada_m, width="stretch",
                    clamp=True, channels="GRAY",
                )

            with col_sal:
                st.markdown('<span class="badge-salida">Salida</span>',
                            unsafe_allow_html=True)
                st.image(
                    img_salida_m, width="stretch",
                    clamp=True, channels="GRAY",
                )

            # Histograma comparativo debajo de las imágenes
            fig_comp = calcular_histograma_comparativo(
                img_entrada_m, img_salida_m
            )
            st.plotly_chart(
                fig_comp,
                width="stretch",
                key=f"chart_comp_{m_idx}_{tipo_m.lower().replace(' ', '_')}",
            )

            # Cuatro métricas comparativas
            delta_mu  = float(img_salida_m.mean())  - float(img_entrada_m.mean())
            delta_sig = float(img_salida_m.std())   - float(img_entrada_m.std())
            delta_min = int(img_salida_m.min())     - int(img_entrada_m.min())
            delta_max = int(img_salida_m.max())     - int(img_entrada_m.max())

            mc1, mc2, mc3, mc4 = st.columns(4)
            mc1.metric(
                "Δμ (media)",
                f"{float(img_salida_m.mean()):.1f}",
                f"{delta_mu:+.1f}",
            )
            mc2.metric(
                "Δσ (desviación)",
                f"{float(img_salida_m.std()):.1f}",
                f"{delta_sig:+.1f}",
            )
            mc3.metric(
                "Δmín",
                f"{int(img_salida_m.min())}",
                f"{delta_min:+d}",
            )
            mc4.metric(
                "Δmáx",
                f"{int(img_salida_m.max())}",
                f"{delta_max:+d}",
            )

            st.markdown('<hr class="paso-sep">', unsafe_allow_html=True)

            # La salida de esta mejora es la entrada de la siguiente
            img_mejora_actual = img_salida_m

        # Guardar resultado final de las mejoras
        st.session_state["img_mejorada"] = img_mejora_actual

        st.success(
            f"✅ Fase 4 completada · {len(historial_mejoras)} mejora(s) aplicada(s). "
            "Resultado guardado en `img_mejorada`.",
            icon="✨",
        )

    else:
        # Si no hay mejoras, guardar la imagen filtrada como base para futuras fases
        st.session_state["img_mejorada"] = st.session_state["img_filtrada"]
        st.info(
            "✨ **Sin mejoras en el pipeline.** "
            "Usa la sección 'Fase 4 — Mejora de Contraste' de la barra lateral "
            "para añadir mejoras.\n\n"
            "**Estrategia recomendada:**\n"
            "1. Ecualizar Rayleigh → aclarar imágenes subexpuestas\n"
            "2. Corrección Gamma (γ=0.7) → realzar sombras\n"
            "3. Contracción/Expansión → ajustar rango dinámico",
            icon="ℹ️",
        )


    # =========================================================================
    # FASE 5 · Segmentación
    # =========================================================================
    st.markdown("---")
    st.markdown("## 🎯 Fase 5 · Segmentación")

    # Recuperar la imagen mejorada del session state
    img_mejorada_f5 = st.session_state.get("img_mejorada")

    if img_mejorada_f5 is None:
        st.warning(
            "⚠️ No hay imagen disponible para la Fase 5. "
            "Sube una imagen y procesa las fases anteriores primero.",
            icon="⚠️",
        )
    else:
        # ── Leer parámetros de umbralización desde session_state ──────────────
        metodo_f5  = st.session_state.get("sel_metodo_umbral", "Otsu")
        invertir   = st.session_state.get("chk_invertir", False)

        # Parámetros opcionales según el método
        kwargs_umbral = {}
        if metodo_f5 == "Manual":
            kwargs_umbral["umbral"] = int(
                st.session_state.get("slider_umbral_manual", 127)
            )
        elif metodo_f5 == "Banda":
            t1_val = int(st.session_state.get("slider_banda_t1", 80))
            t2_val = int(st.session_state.get("slider_banda_t2", 200))
            kwargs_umbral["t1"] = t1_val
            kwargs_umbral["t2"] = t2_val

        with st.spinner("Calculando umbralización…"):
            # 1. Umbralizar
            mascara_f5, info_umbral = aplicar_umbral(
                img_mejorada_f5,
                metodo=metodo_f5,
                invertir=invertir,
                **kwargs_umbral,
            )

            # 2. Cierre morfológico (opcional)
            if st.session_state.get("chk_cierre", True):
                kc = int(st.session_state.get("slider_kcierre", 7))
                mascara_f5 = aplicar_cierre(mascara_f5, kc)

            # 3. Apertura morfológica (opcional)
            if st.session_state.get("chk_apertura", False):
                ka = int(st.session_state.get("slider_kapertura", 3))
                mascara_f5 = aplicar_apertura(mascara_f5, ka)

            # 4. Relleno de huecos internos (opcional)
            if st.session_state.get("chk_relleno", False):
                mascara_f5 = aplicar_relleno_huecos(mascara_f5)

        # 5. Guardar en session_state para fases futuras
        st.session_state["img_binarizada"] = mascara_f5

        # ── Métrica del umbral (solo para métodos automáticos) ────────────────
        if metodo_f5 in METODOS_AUTOMATICOS:
            umbral_calc = info_umbral.get("umbral", "—")
            col_met1, col_met2, col_met3 = st.columns(3)
            col_met1.metric("Umbral calculado", f"{umbral_calc}")
            if metodo_f5 == "Kapur":
                col_met2.metric(
                    "Entropía máx.",
                    f"{info_umbral.get('entropia_max', '—')}"
                )
            elif metodo_f5 == "Media":
                col_met2.metric(
                    "Media global",
                    f"{info_umbral.get('media', '—')}"
                )
            col_met3.metric(
                "Píxeles objeto",
                f"{int((mascara_f5 > 0).sum()):,}",
            )

        # ── Construir overlay: imagen mejorada + máscara verde al 50 % ────────
        # Convertir la imagen mejorada a RGB de 3 canales para el overlay
        if img_mejorada_f5.ndim == 2:
            img_rgb_base = cv2.cvtColor(img_mejorada_f5, cv2.COLOR_GRAY2RGB)
        else:
            img_rgb_base = img_mejorada_f5.copy()

        # Crear una capa verde sólida donde la máscara es positiva
        capa_verde = np.zeros_like(img_rgb_base, dtype=np.uint8)
        capa_verde[mascara_f5 > 0] = [0, 220, 80]   # verde brillante

        # Mezclar imagen base con la capa verde al 50 %
        img_overlay = cv2.addWeighted(img_rgb_base, 0.6, capa_verde, 0.4, 0)

        # ── Histograma de la máscara binaria ──────────────────────────────────
        hist_bin, bins_bin = np.histogram(
            mascara_f5.flatten(), bins=2, range=(0, 256)
        )
        import plotly.graph_objects as go  # ya importado, pero por claridad
        fig_hist_bin = go.Figure()
        fig_hist_bin.add_bar(
            x=["Fondo (0)", "Objeto (255)"],
            y=[int(hist_bin[0]), int(hist_bin[1])],
            marker_color=["#334155", "#38bdf8"],
        )
        fig_hist_bin.update_layout(
            title=dict(text="Distribución binaria", font=dict(size=13)),
            xaxis_title="Clase",
            yaxis_title="Píxeles",
            margin=dict(l=10, r=10, t=40, b=30),
            height=300,
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
        )

        # ── Tres columnas iguales ─────────────────────────────────────────────
        col_bin, col_ov, col_hist_bin = st.columns(3)

        with col_bin:
            st.markdown("**Máscara binaria**")
            # st.image: width="stretch", SIN key
            st.image(
                mascara_f5,
                width="stretch",
                clamp=True,
                channels="GRAY",
            )
            n_obj = int((mascara_f5 > 0).sum())
            pct   = round(100.0 * n_obj / mascara_f5.size, 1)
            st.caption(f"🟦 {n_obj:,} px objeto · {pct} % del total")

        with col_ov:
            st.markdown("**Overlay (verde = objeto)**")
            # st.image: width="stretch", SIN key
            st.image(
                img_overlay,
                width="stretch",
                clamp=True,
                channels="RGB",
            )
            st.caption("Máscara superpuesta en verde al 40 %")

        with col_hist_bin:
            st.markdown("**Histograma binario**")
            st.plotly_chart(
                fig_hist_bin,
                width="stretch",
                key="chart_hist_bin_fase5",
            )

        # ── Caption explicativo del método y parámetros ───────────────────────
        if metodo_f5 == "Otsu":
            desc_metodo = (
                f"🤖 **Otsu** — umbral automático = **{info_umbral['umbral']}**. "
                "Maximiza la varianza entre fondo y objeto."
            )
        elif metodo_f5 == "Kapur":
            desc_metodo = (
                f"🤖 **Kapur** — umbral = **{info_umbral['umbral']}** "
                f"(entropía máx. = {info_umbral['entropia_max']}). "
                "Maximiza la entropía conjunta de las dos regiones."
            )
        elif metodo_f5 == "Media":
            desc_metodo = (
                f"🤖 **Media** — umbral = media global = **{info_umbral['media']}**. "
                "Simple y rápido, funciona bien con iluminación uniforme."
            )
        elif metodo_f5 == "Manual":
            desc_metodo = (
                f"🔧 **Manual** — umbral fijo = **{info_umbral['umbral']}**. "
                "Control total del usuario sobre el punto de corte."
            )
        else:  # Banda
            desc_metodo = (
                f"🎯 **Banda** — rango **[{info_umbral['t1']}, {info_umbral['t2']}]**. "
                "Aisla el rango de intensidad específico de la botella PET. "
                "Es el método más efectivo cuando la botella tiene intensidad "
                "diferenciada del cuerpo de agua."
            )

        morf_aplicada = []
        if st.session_state.get("chk_cierre", True):
            morf_aplicada.append(
                f"Cierre k={st.session_state.get('slider_kcierre', 7)}"
            )
        if st.session_state.get("chk_apertura", False):
            morf_aplicada.append(
                f"Apertura k={st.session_state.get('slider_kapertura', 3)}"
            )
        if st.session_state.get("chk_relleno", False):
            morf_aplicada.append("Relleno de huecos")
        if invertir:
            morf_aplicada.append("Máscara invertida")

        desc_morf = (
            " → ".join(morf_aplicada) if morf_aplicada
            else "Sin operaciones morfológicas."
        )

        st.caption(
            f"{desc_metodo}  \n"
            f"🔩 Morfología: {desc_morf}"
        )

        st.success(
            "✅ Fase 5 completada · Máscara guardada en `img_binarizada`.",
            icon="🎯",
        )


# =============================================================================
# 8. PUNTO DE ENTRADA
# =============================================================================
if __name__ == "__main__":
    main()