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
        if st.button("➕ Añadir filtro", use_container_width=True):
            st.session_state["historial_filtros"].append(
                dict(config_nuevo)   # copia del dict para evitar referencias
            )
            st.toast(f"Filtro '{filtro_elegido}' añadido ✅", icon="🧩")

    with col_limpiar:
        if st.button("🗑️ Limpiar filtros", use_container_width=True):
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

    # ── Resumen del pipeline ──────────────────────────────────────────────────
    n_filtros = len(st.session_state["historial_filtros"])
    st.markdown(
        f"**Resumen del pipeline**\n\n"
        f"1. 📥 Carga + redimensión\n"
        f"2. ⬛ Escala de grises\n"
        f"3. 🧩 Filtros: _{n_filtros} aplicado(s)_\n"
        f"4. 🎯 Segmentación _(próxima versión)_\n"
        f"5. 🔬 Extracción _(próxima versión)_"
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
        return

    # Mostramos el encabezado de la sección de filtros
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
        "Resultado guardado en `img_filtrada`. "
        "La próxima versión (V2) añadirá FFT y Enhancement.",
        icon="🏁",
    )


# =============================================================================
# 8. PUNTO DE ENTRADA
# =============================================================================
if __name__ == "__main__":
    main()