# =============================================================================
# procesamiento.py
# Módulo de lógica de procesamiento de imágenes para el Dashboard de
# Segmentación de Botellas PET en cuerpos de agua.
#
# REGLA: Este archivo NO importa Streamlit. Solo OpenCV, NumPy y Plotly.
#
# Fase 1 : carga, redimensión y conversión de color.
# Fase 2 : conversión a escala de grises.
# Fase 3 : filtros espaciales acumulativos (7 filtros).
# =============================================================================

import cv2
import numpy as np
import plotly.graph_objects as go
from scipy.ndimage import maximum_filter, minimum_filter


# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTES GLOBALES
# ─────────────────────────────────────────────────────────────────────────────

# Ancho máximo permitido antes de redimensionar (en píxeles)
MAX_ANCHO_PX = 800


# =============================================================================
# FASE 1 — Carga y preprocesado básico
# =============================================================================

def redimensionar_imagen(imagen_bgr: np.ndarray) -> np.ndarray:
    """
    Redimensiona la imagen si su ancho supera MAX_ANCHO_PX,
    manteniendo la relación de aspecto original.

    Se usa INTER_AREA porque es la interpolación óptima al REDUCIR
    imágenes: minimiza el aliasing y produce mejor calidad que
    INTER_LINEAR al hacer downscaling.

    Parámetros
    ----------
    imagen_bgr : np.ndarray
        Imagen en formato BGR (tal como la carga OpenCV).

    Retorna
    -------
    np.ndarray
        Imagen original si ancho <= MAX_ANCHO_PX,
        o imagen redimensionada si ancho > MAX_ANCHO_PX.
    """
    alto_original, ancho_original = imagen_bgr.shape[:2]

    # Si la imagen ya cabe dentro del límite, la devolvemos sin cambios
    if ancho_original <= MAX_ANCHO_PX:
        return imagen_bgr

    # Calculamos la escala proporcional
    escala = MAX_ANCHO_PX / ancho_original
    nuevo_alto = int(alto_original * escala)

    return cv2.resize(
        imagen_bgr,
        (MAX_ANCHO_PX, nuevo_alto),
        interpolation=cv2.INTER_AREA,
    )


def bgr_a_rgb(imagen_bgr: np.ndarray) -> np.ndarray:
    """
    Convierte una imagen del formato BGR (OpenCV) al formato RGB (Streamlit).

    OpenCV carga imágenes en BGR por razones históricas.
    Streamlit y Matplotlib esperan RGB.
    Esta conversión es necesaria antes de cualquier visualización.

    Parámetros
    ----------
    imagen_bgr : np.ndarray
        Imagen en formato BGR.

    Retorna
    -------
    np.ndarray
        Imagen en formato RGB.
    """
    return cv2.cvtColor(imagen_bgr, cv2.COLOR_BGR2RGB)


def bgr_a_gris_directo(imagen_bgr: np.ndarray) -> np.ndarray:
    """
    Convierte directamente una imagen BGR a escala de grises.
    Función auxiliar para el pipeline de carga.

    Parámetros
    ----------
    imagen_bgr : np.ndarray
        Imagen en formato BGR.

    Retorna
    -------
    np.ndarray
        Imagen en escala de grises (2D, uint8).
    """
    return cv2.cvtColor(imagen_bgr, cv2.COLOR_BGR2GRAY)


# =============================================================================
# FASE 2 — Conversión a escala de grises
# =============================================================================

def convertir_a_gris(imagen_bgr: np.ndarray) -> np.ndarray:
    """
    Convierte una imagen BGR a escala de grises usando ponderación perceptual.

    Fórmula ITU-R BT.601 (estándar televisión):
        I_gris = 0.299·R + 0.587·G + 0.114·B

    Los coeficientes reflejan la sensibilidad del ojo humano:
    - Verde recibe mayor peso (0.587) por ser el más visible
    - Azul recibe menor peso (0.114) por ser el menos visible
    Esta ponderación produce una percepción de brillo más natural
    que el promedio simple (R+G+B)/3.

    Parámetros
    ----------
    imagen_bgr : np.ndarray
        Imagen en formato BGR.

    Retorna
    -------
    np.ndarray
        Imagen en escala de grises (2D, uint8).
    """
    return cv2.cvtColor(imagen_bgr, cv2.COLOR_BGR2GRAY)


# =============================================================================
# HISTOGRAMA — Visualización con Plotly
# =============================================================================

def calcular_histograma(imagen: np.ndarray) -> go.Figure:
    """
    Calcula y devuelve un gráfico Plotly con el histograma de la imagen.

    Comportamiento según el tipo de imagen:
    - imagen 2D (escala de grises) → un solo canal en gris
    - imagen 3D RGB (H×W×3)       → tres canales superpuestos R, G, B

    Se usa fill="tozeroy" para crear áreas rellenas bajo las curvas,
    facilitando la comparación visual entre canales.

    Parámetros
    ----------
    imagen : np.ndarray
        Imagen 2D (grises) o 3D RGB. En formato RGB para visualización.

    Retorna
    -------
    plotly.graph_objects.Figure
        Figura Plotly lista para renderizar con st.plotly_chart.
    """
    # Determinamos si es imagen en escala de grises o color
    es_gris = (imagen.ndim == 2) or (
        imagen.ndim == 3 and imagen.shape[2] == 1
    )

    fig = go.Figure()

    if es_gris:
        # ── Canal único (escala de grises) ────────────────────────────────
        canal = imagen.flatten() if imagen.ndim == 2 \
                else imagen[:, :, 0].flatten()
        hist, bins = np.histogram(canal, bins=256, range=(0, 256))

        fig.add_trace(go.Scatter(
            x=bins[:-1],
            y=hist,
            mode="lines",
            fill="tozeroy",
            line=dict(color="#a0aec0", width=1.5),
            fillcolor="rgba(160,174,192,0.3)",
            name="Gris",
        ))

    else:
        # ── Tres canales RGB superpuestos ─────────────────────────────────
        # Nota: la imagen ya está en RGB cuando llega aquí
        configuracion_canales = [
            (0, "#ef4444", "rgba(239,68,68,0.2)",  "Rojo (R)"),
            (1, "#22c55e", "rgba(34,197,94,0.2)",  "Verde (G)"),
            (2, "#3b82f6", "rgba(59,130,246,0.2)", "Azul (B)"),
        ]
        for idx, color_linea, color_relleno, nombre in configuracion_canales:
            canal = imagen[:, :, idx].flatten()
            hist, bins = np.histogram(canal, bins=256, range=(0, 256))

            fig.add_trace(go.Scatter(
                x=bins[:-1],
                y=hist,
                mode="lines",
                fill="tozeroy",
                line=dict(color=color_linea, width=1.5),
                fillcolor=color_relleno,
                name=nombre,
            ))

    # ── Estilo del gráfico ────────────────────────────────────────────────
    fig.update_layout(
        title=dict(text="Histograma", font=dict(size=13)),
        xaxis=dict(title="Intensidad (0-255)", range=[0, 255]),
        yaxis=dict(title="Frecuencia"),
        margin=dict(l=10, r=10, t=40, b=30),
        legend=dict(orientation="h", y=-0.25),
        height=260,
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )

    return fig


# =============================================================================
# FASE 3 — Filtros espaciales
# Todas las funciones:
#   · Reciben imagen en escala de grises (np.ndarray 2D, uint8)
#   · Devuelven imagen en escala de grises (np.ndarray 2D, uint8)
#   · Usan BORDER_REFLECT_101 para evitar artefactos en bordes
# =============================================================================

def aplicar_gaussiano(
    imagen_gris: np.ndarray,
    ksize: int = 3,
    sigma: float = 0.0,
) -> np.ndarray:
    """
    Filtro Gaussiano — suavizado con ponderación normal.

    Cada píxel se reemplaza por la media ponderada de su vecindad,
    donde los pesos siguen una distribución gaussiana. Los píxeles
    más cercanos al centro reciben mayor peso.

    sigma=0 → OpenCV calcula σ automáticamente:
        σ ≈ 0.3·((ksize-1)/2 - 1) + 0.8

    Ventaja sobre el box filter: mejor preservación de bordes suaves.
    Desventaja: más costoso computacionalmente.

    Parámetros
    ----------
    imagen_gris : np.ndarray 2D uint8
    ksize       : int   — tamaño del kernel (debe ser impar).
    sigma       : float — desviación estándar.

    Retorna
    -------
    np.ndarray 2D uint8
    """
    return cv2.GaussianBlur(
        imagen_gris,
        (ksize, ksize),
        sigma,
        borderType=cv2.BORDER_REFLECT_101,
    )


def aplicar_mediana(
    imagen_gris: np.ndarray,
    ksize: int = 3,
) -> np.ndarray:
    """
    Filtro de Mediana — filtro no lineal de orden.

    Reemplaza cada píxel por la mediana de su vecindad k×k.
    Al ser no lineal, es altamente efectivo contra ruido impulsivo
    (sal y pimienta) sin difuminar los bordes.

    Para botellas PET en agua: elimina brillos puntuales del sol
    sobre el agua sin destruir el contorno de la botella.

    Parámetros
    ----------
    imagen_gris : np.ndarray 2D uint8
    ksize       : int — tamaño del kernel (debe ser impar).

    Retorna
    -------
    np.ndarray 2D uint8
    """
    # cv2.medianBlur maneja internamente el padding de bordes
    return cv2.medianBlur(imagen_gris, ksize)


def aplicar_bilateral(
    imagen_gris: np.ndarray,
    d: int = 9,
    sigma_color: float = 75.0,
    sigma_space: float = 75.0,
) -> np.ndarray:
    """
    Filtro Bilateral — suavizado que PRESERVA BORDES.

    Combina dos kernels gaussianos:
    · Kernel espacial (sigma_space): pondera por distancia euclidiana.
      Equivalente a un gaussiano espacial clásico.
    · Kernel de rango (sigma_color): pondera por similitud de intensidad.
      Píxeles con intensidad similar al píxel central reciben mayor peso.

    Fórmula:
        I_out(p) = Σ_q [G_s(||p-q||) · G_r(|I_p-I_q|) · I_q]
                   ──────────────────────────────────────────────
                   Σ_q [G_s(||p-q||) · G_r(|I_p-I_q|)]

    Para botellas en agua: preserva el borde nítido entre la botella
    y el agua mientras suaviza la textura interna del agua.

    Parámetros
    ----------
    imagen_gris  : np.ndarray 2D uint8
    d            : int   — diámetro de la vecindad.
    sigma_color  : float — σ en espacio de intensidades.
    sigma_space  : float — σ en espacio cartesiano.

    Retorna
    -------
    np.ndarray 2D uint8
    """
    return cv2.bilateralFilter(
        imagen_gris,
        d=d,
        sigmaColor=sigma_color,
        sigmaSpace=sigma_space,
    )


def aplicar_paso_bajas(
    imagen_gris: np.ndarray,
    ksize: int = 3,
) -> np.ndarray:
    """
    Filtro de Paso Bajas con kernel de caja uniforme (box filter).

    Matemáticamente equivalente a la convolución con:
        K = (1/k²) · J_{k×k}
    donde J es una matriz de unos.

    Todos los píxeles de la vecindad tienen el mismo peso (1/k²).
    Elimina altas frecuencias (detalles, textura del agua) pero
    tiende a difuminar los bordes de la botella.

    Parámetros
    ----------
    imagen_gris : np.ndarray 2D uint8
    ksize       : int — tamaño del kernel (impar).

    Retorna
    -------
    np.ndarray 2D uint8
    """
    return cv2.blur(
        imagen_gris,
        (ksize, ksize),
        borderType=cv2.BORDER_REFLECT_101,
    )


def aplicar_promediador(
    imagen_gris: np.ndarray,
    ksize: int = 3,
) -> np.ndarray:
    """
    Filtro Promediador — box filter con kernel NumPy explícito.

    Conceptualmente idéntico a aplicar_paso_bajas pero el kernel
    se construye explícitamente con NumPy para mostrar la mecánica
    de la convolución con kernel personalizado.

    Kernel: K = ones(k,k) / k²

    Parámetros
    ----------
    imagen_gris : np.ndarray 2D uint8
    ksize       : int — tamaño del kernel (impar).

    Retorna
    -------
    np.ndarray 2D uint8
    """
    kernel = np.ones((ksize, ksize), dtype=np.float32) / (ksize * ksize)
    return cv2.filter2D(
        imagen_gris,
        -1,
        kernel,
        borderType=cv2.BORDER_REFLECT_101,
    )


def aplicar_max(
    imagen_gris: np.ndarray,
    ksize: int = 3,
) -> np.ndarray:
    """
    Filtro de Máximo — dilatación morfológica.

    Reemplaza cada píxel por el valor máximo en su vecindad k×k.
    Efecto: expande las zonas brillantes, elimina manchas oscuras
    pequeñas (pimienta).

    Para botellas transparentes: puede realzar los brillos sobre
    la botella y distinguirlos del fondo oscuro del agua.

    Parámetros
    ----------
    imagen_gris : np.ndarray 2D uint8
    ksize       : int — tamaño de la ventana.

    Retorna
    -------
    np.ndarray 2D uint8
    """
    return maximum_filter(
        imagen_gris,
        size=ksize,
        mode="reflect",
    ).astype(np.uint8)


def aplicar_min(
    imagen_gris: np.ndarray,
    ksize: int = 3,
) -> np.ndarray:
    """
    Filtro de Mínimo — erosión morfológica.

    Reemplaza cada píxel por el valor mínimo en su vecindad k×k.
    Efecto: expande las zonas oscuras, elimina manchas brillantes
    pequeñas (sal).

    Para botellas: puede oscurecer la textura del agua y hacer
    más homogéneo el fondo antes de la segmentación.

    Parámetros
    ----------
    imagen_gris : np.ndarray 2D uint8
    ksize       : int — tamaño de la ventana.

    Retorna
    -------
    np.ndarray 2D uint8
    """
    return minimum_filter(
        imagen_gris,
        size=ksize,
        mode="reflect",
    ).astype(np.uint8)


# ── Dispatcher central de filtros ────────────────────────────────────────────

# Mapeo nombre → función para el dispatcher
_FILTROS_DISPONIBLES = {
    "Gaussiano":  aplicar_gaussiano,
    "Mediana":    aplicar_mediana,
    "Bilateral":  aplicar_bilateral,
    "Paso Bajas": aplicar_paso_bajas,
    "Promediador":aplicar_promediador,
    "Max":        aplicar_max,
    "Min":        aplicar_min,
}

# Lista de nombres para el selectbox del sidebar (exportada a app.py)
NOMBRES_FILTROS = list(_FILTROS_DISPONIBLES.keys())


def aplicar_filtro(imagen_gris: np.ndarray, config: dict) -> np.ndarray:
    """
    Dispatcher central de la Fase 3.

    Recibe un diccionario de configuración con al menos la clave "tipo"
    y delega a la función correspondiente pasando el resto del dict
    como kwargs.

    Parámetros
    ----------
    imagen_gris : np.ndarray 2D uint8
        Imagen de entrada en escala de grises.
    config : dict
        Debe contener {"tipo": str, <param>: val, ...}
        Ejemplo: {"tipo": "Gaussiano", "ksize": 5, "sigma": 1.5}

    Retorna
    -------
    np.ndarray 2D uint8
        Imagen filtrada.

    Raises
    ------
    ValueError si el tipo no existe en _FILTROS_DISPONIBLES.
    """
    tipo = config.get("tipo", "")
    if tipo not in _FILTROS_DISPONIBLES:
        raise ValueError(
            f"Filtro desconocido: '{tipo}'. "
            f"Opciones válidas: {list(_FILTROS_DISPONIBLES.keys())}"
        )

    # Extraemos todos los parámetros excepto "tipo"
    kwargs = {k: v for k, v in config.items() if k != "tipo"}

    return _FILTROS_DISPONIBLES[tipo](imagen_gris, **kwargs)


def descripcion_filtro(tipo: str, config: dict) -> str:
    """
    Devuelve una descripción didáctica del filtro aplicado para
    mostrar debajo de cada paso en el área principal.

    Parámetros
    ----------
    tipo   : str  — nombre del filtro.
    config : dict — configuración completa.

    Retorna
    -------
    str — descripción en formato Markdown.
    """
    k  = config.get("ksize", 3)
    d  = config.get("d", 9)
    sc = config.get("sigma_color", 75.0)
    ss = config.get("sigma_space", 75.0)
    sg = config.get("sigma", 0.0)

    descripciones = {
        "Gaussiano": (
            f"**Gaussiano** (ksize={k}×{k}, σ={sg}). "
            "Suavizado con ponderación gaussiana. Reduce la textura del agua "
            "preservando los bordes suaves de la botella. "
            "Mayor ksize = mayor suavizado pero más difuminado de bordes."
        ),
        "Mediana": (
            f"**Mediana** (ksize={k}×{k}). Filtro no lineal. "
            "Elimina brillos puntuales del sol sobre el agua sin afectar "
            "significativamente el borde de la botella. "
            "Ideal como primer filtro para imágenes con reflejos."
        ),
        "Bilateral": (
            f"**Bilateral** (d={d}, σ_color={sc}, σ_space={ss}). "
            "Suaviza manteniendo bordes nítidos. Excelente para homogenizar "
            "la textura del agua sin difuminar el contorno de la botella. "
            f"σ_color={sc}: {'bordes muy preservados' if sc < 50 else 'suavizado moderado'}."
        ),
        "Paso Bajas": (
            f"**Paso Bajas / Box filter** ({k}×{k}). "
            "Todos los píxeles de la vecindad tienen el mismo peso (1/k²). "
            "Reduce variaciones de alta frecuencia (textura del agua). "
            "Puede difuminar los bordes de la botella."
        ),
        "Promediador": (
            f"**Promediador** ({k}×{k}). "
            "Kernel explícito ones(k,k)/k². Equivalente al Paso Bajas. "
            "Útil para homogenizar regiones del agua antes de umbralizar."
        ),
        "Max": (
            f"**Máximo / Dilatación** ({k}×{k}). "
            "Expande zonas brillantes. Puede realzar la botella clara "
            "sobre fondo oscuro. Elimina manchas oscuras pequeñas."
        ),
        "Min": (
            f"**Mínimo / Erosión** ({k}×{k}). "
            "Expande zonas oscuras. Homogeniza el fondo de agua "
            "antes de la segmentación. Elimina brillos puntuales pequeños."
        ),
    }
    return descripciones.get(tipo, "Filtro aplicado.")