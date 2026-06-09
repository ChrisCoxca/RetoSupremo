# =============================================================================
# procesamiento.py
# Módulo de lógica de procesamiento de imágenes para el Dashboard de
# Segmentación de Botellas PET en cuerpos de agua.
#
# REGLA: Este archivo NO importa Streamlit. Solo OpenCV, NumPy y Plotly.
#
# Fase 1   : carga, redimensión y conversión de color.
# Fase 2   : conversión a escala de grises.
# Fase 3   : filtros espaciales acumulativos (7 filtros).
# Fase 3.5 : filtrado en frecuencia (FFT lowpass / highpass).
# Fase 4   : mejora de contraste y brillo (6 técnicas con LUT).
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


# =============================================================================
# FASE 3.5 — Filtrado en frecuencia (FFT)
# =============================================================================

def fft_filter(
    imagen_gris: np.ndarray,
    cutoff: float = 0.15,
    tipo: str = "lowpass",
) -> tuple:
    """
    Aplica un filtro ideal de paso bajas o paso altas en el dominio
    de la frecuencia usando la Transformada de Fourier Discreta 2D.

    Pasos internos:
    1. np.fft.fft2  → transforma la imagen al dominio de frecuencias.
    2. np.fft.fftshift → desplaza la componente DC al centro.
    3. Máscara circular: radio = cutoff × min(H, W) / 2.
       lowpass  → M=1 dentro del círculo (pasa bajas).
       highpass → M=0 dentro del círculo (pasa altas).
    4. Multiplicar espectro × máscara.
    5. np.fft.ifftshift + np.fft.ifft2 → volver al dominio espacial.
    6. Tomar parte real, recortar a [0,255] y convertir a uint8.

    Espectro de magnitud:
        S = log(1 + |F_shifted|), normalizado a [0, 255] uint8.

    Parámetros
    ----------
    imagen_gris : np.ndarray 2D uint8
        Imagen en escala de grises.
    cutoff : float
        Fracción del radio mínimo (0.01–1.00) que define el corte.
        cutoff=0.15 → radio = 15 % del semiancho más pequeño.
    tipo : str
        "lowpass"  → elimina altas frecuencias (suaviza).
        "highpass" → elimina bajas frecuencias (realza bordes).

    Retorna
    -------
    (img_filtrada, mascara, espectro_mag) : tuple de np.ndarray
        img_filtrada : np.ndarray 2D uint8  — resultado espacial.
        mascara      : np.ndarray 2D float  — máscara binaria [0,1].
        espectro_mag : np.ndarray 2D uint8  — espectro de magnitud.
    """
    H, W = imagen_gris.shape

    # ── 1. Transformada de Fourier y desplazamiento al centro ─────────────────
    F = np.fft.fft2(imagen_gris.astype(np.float64))
    F_shifted = np.fft.fftshift(F)

    # ── 2. Espectro de magnitud (para visualización) ──────────────────────────
    magnitud = np.abs(F_shifted)
    espectro_log = np.log1p(magnitud)   # log(1 + |F|)
    # Normalizar a [0, 255]
    espectro_min, espectro_max = espectro_log.min(), espectro_log.max()
    if espectro_max > espectro_min:
        espectro_norm = (espectro_log - espectro_min) / (espectro_max - espectro_min)
    else:
        espectro_norm = espectro_log * 0.0
    espectro_mag = (espectro_norm * 255).astype(np.uint8)

    # ── 3. Máscara circular ───────────────────────────────────────────────────
    centro_y, centro_x = H // 2, W // 2
    radio = cutoff * min(H, W) / 2.0

    # Rejilla de coordenadas
    ys = np.arange(H) - centro_y
    xs = np.arange(W) - centro_x
    XX, YY = np.meshgrid(xs, ys)
    distancia = np.sqrt(XX**2 + YY**2)

    if tipo == "lowpass":
        mascara = (distancia <= radio).astype(np.float64)   # 1 dentro, 0 fuera
    else:  # highpass
        mascara = (distancia > radio).astype(np.float64)    # 0 dentro, 1 fuera

    # ── 4-5. Filtrado y transformada inversa ──────────────────────────────────
    F_filtrado = F_shifted * mascara
    F_inv = np.fft.ifftshift(F_filtrado)
    img_reconstruida = np.fft.ifft2(F_inv)

    # ── 6. Convertir a uint8 ──────────────────────────────────────────────────
    img_real = np.real(img_reconstruida)
    img_filtrada = np.clip(img_real, 0, 255).astype(np.uint8)

    return img_filtrada, mascara, espectro_mag


def crear_espectro_con_mascara(
    espectro_mag: np.ndarray,
    mascara: np.ndarray,
) -> np.ndarray:
    """
    Convierte el espectro de magnitud a RGB y dibuja el borde de la
    máscara circular en rojo sobre él para facilitar la interpretación.

    El borde se extrae con una operación morfológica MORPH_GRADIENT
    (dilatación - erosión) con kernel 3×3, que devuelve solo los
    píxeles de transición de la máscara.

    Parámetros
    ----------
    espectro_mag : np.ndarray 2D uint8
        Espectro de magnitud normalizado.
    mascara : np.ndarray 2D float
        Máscara binaria [0.0, 1.0] generada por fft_filter.

    Retorna
    -------
    np.ndarray 3D uint8 (H × W × 3)
        Espectro en RGB con borde de la máscara resaltado en rojo.
    """
    # Convertir espectro gris → RGB (3 canales)
    espectro_rgb = cv2.cvtColor(espectro_mag, cv2.COLOR_GRAY2RGB)

    # Borde de la máscara: MORPH_GRADIENT = dilatación - erosión
    kernel_borde = np.ones((3, 3), dtype=np.uint8)
    mascara_u8 = (mascara * 255).astype(np.uint8)
    borde = cv2.morphologyEx(mascara_u8, cv2.MORPH_GRADIENT, kernel_borde)

    # Pintar el borde en rojo (255, 60, 60) sobre el espectro RGB
    espectro_rgb[borde > 0] = [255, 60, 60]

    return espectro_rgb


# =============================================================================
# FASE 4 — Enhancement (mejora de contraste y brillo)
# Todas las funciones:
#   · Usan LUT de 256 entradas con cv2.LUT para máxima eficiencia.
#   · Reciben np.ndarray 2D uint8 y devuelven np.ndarray 2D uint8.
# =============================================================================

# Lista de opciones exportada al sidebar de app.py
MEJORAS_OPCIONES = [
    "Corrección Gamma",
    "Desplazamiento (Brillo)",
    "Contracción / Expansión",
    "Ecualización Uniforme",
    "Ecualización Rayleigh",
    "Ecualización Log. Hiperbólica",
]


def _construir_lut(valores: np.ndarray) -> np.ndarray:
    """
    Construye una LUT de 256 entradas a partir de un array de 256 flotantes.
    Recorta a [0,255] y convierte a uint8.

    Parámetros
    ----------
    valores : np.ndarray de forma (256,) con floats en [0, 255].

    Retorna
    -------
    np.ndarray de forma (256,) uint8.
    """
    return np.clip(valores, 0, 255).astype(np.uint8)


def mejora_gamma(
    imagen_gris: np.ndarray,
    gamma: float = 1.5,
) -> np.ndarray:
    """
    Corrección gamma:
        I_out = (I / 255)^γ × 255

    γ < 1 → aclarar (realza sombras).
    γ > 1 → oscurecer (comprime luces).
    γ = 1 → identidad.

    Parámetros
    ----------
    imagen_gris : np.ndarray 2D uint8
    gamma       : float — exponente de corrección.

    Retorna
    -------
    np.ndarray 2D uint8
    """
    # Construir LUT con la curva gamma
    entradas = np.arange(256, dtype=np.float64)
    lut = _construir_lut((entradas / 255.0) ** gamma * 255.0)
    return cv2.LUT(imagen_gris, lut)


def mejora_desplazamiento(
    imagen_gris: np.ndarray,
    delta: int = 50,
) -> np.ndarray:
    """
    Desplazamiento de brillo (suma constante):
        I_out = clip(I + Δ, 0, 255)

    δ > 0 → imagen más brillante.
    δ < 0 → imagen más oscura.

    Parámetros
    ----------
    imagen_gris : np.ndarray 2D uint8
    delta       : int — desplazamiento en intensidad (−255 a +255).

    Retorna
    -------
    np.ndarray 2D uint8
    """
    entradas = np.arange(256, dtype=np.float64)
    lut = _construir_lut(entradas + delta)
    return cv2.LUT(imagen_gris, lut)


def mejora_contraccion_expansion(
    imagen_gris: np.ndarray,
    a_in: int = 50,
    b_in: int = 200,
    a_out: int = 0,
    b_out: int = 255,
) -> np.ndarray:
    """
    Mapeo lineal por tramos (contracción / expansión de contraste):

      I < a_in               → a_out
      a_in <= I <= b_in      → lineal: a_out + (I - a_in) × (b_out - a_out) / (b_in - a_in)
      I > b_in               → b_out

    Permite expandir un rango de interés [a_in, b_in] hacia [a_out, b_out],
    aumentando el contraste dentro de ese rango.

    Parámetros
    ----------
    imagen_gris : np.ndarray 2D uint8
    a_in, b_in  : int — rango de entrada (a_in < b_in).
    a_out, b_out: int — rango de salida.

    Retorna
    -------
    np.ndarray 2D uint8
    """
    entradas = np.arange(256, dtype=np.float64)
    rango_in = float(b_in - a_in) if b_in != a_in else 1.0
    rango_out = float(b_out - a_out)

    # Mapeo lineal por tramos
    valores = np.where(
        entradas < a_in,
        float(a_out),
        np.where(
            entradas > b_in,
            float(b_out),
            a_out + (entradas - a_in) * rango_out / rango_in,
        ),
    )
    lut = _construir_lut(valores)
    return cv2.LUT(imagen_gris, lut)


def mejora_ecual_uniforme(
    imagen_gris: np.ndarray,
) -> np.ndarray:
    """
    Ecualización de histograma uniforme (estándar).

    Redistribuye las intensidades para que el histograma sea lo más
    plano posible. Maximiza el contraste global pero puede amplificar
    ruido en regiones homogéneas.

    Usa cv2.equalizeHist (implementación optimizada de OpenCV).

    Parámetros
    ----------
    imagen_gris : np.ndarray 2D uint8

    Retorna
    -------
    np.ndarray 2D uint8
    """
    return cv2.equalizeHist(imagen_gris)


def mejora_ecual_rayleigh(
    imagen_gris: np.ndarray,
) -> np.ndarray:
    """
    Ecualización con distribución Rayleigh (curva raíz cuadrada):
        I_out = 255 × √(I / 255)

    Variante no lineal que aclarar la imagen de forma suave.
    Más conservadora que la ecualización uniforme: no corta las
    distribuciones bimodales abruptamente.

    Parámetros
    ----------
    imagen_gris : np.ndarray 2D uint8

    Retorna
    -------
    np.ndarray 2D uint8
    """
    entradas = np.arange(256, dtype=np.float64)
    lut = _construir_lut(255.0 * np.sqrt(entradas / 255.0))
    return cv2.LUT(imagen_gris, lut)


def mejora_ecual_log_hiperbolica(
    imagen_gris: np.ndarray,
) -> np.ndarray:
    """
    Ecualización logarítmica hiperbólica:
        I_out = 255 × log(1 + I) / log(256)

    La curva logarítmica comprime los valores altos y expande los
    bajos, realzando los detalles en zonas oscuras de la imagen.
    Útil para imágenes subexpuestas con reflejos de agua.

    Parámetros
    ----------
    imagen_gris : np.ndarray 2D uint8

    Retorna
    -------
    np.ndarray 2D uint8
    """
    entradas = np.arange(256, dtype=np.float64)
    lut = _construir_lut(255.0 * np.log1p(entradas) / np.log(256.0))
    return cv2.LUT(imagen_gris, lut)


# ── Dispatcher central de mejoras ─────────────────────────────────────────────

# Mapeo nombre → función para el dispatcher
_MEJORAS_DISPONIBLES = {
    "Corrección Gamma":           mejora_gamma,
    "Desplazamiento (Brillo)":    mejora_desplazamiento,
    "Contracción / Expansión":    mejora_contraccion_expansion,
    "Ecualización Uniforme":      mejora_ecual_uniforme,
    "Ecualización Rayleigh":      mejora_ecual_rayleigh,
    "Ecualización Log. Hiperbólica": mejora_ecual_log_hiperbolica,
}


def aplicar_mejora(imagen_gris: np.ndarray, config: dict) -> np.ndarray:
    """
    Dispatcher central de la Fase 4.

    Recibe un diccionario con al menos la clave "tipo" y delega
    a la función de mejora correspondiente.

    Parámetros
    ----------
    imagen_gris : np.ndarray 2D uint8
    config      : dict — {"tipo": str, <params>: val, ...}

    Retorna
    -------
    np.ndarray 2D uint8

    Raises
    ------
    ValueError si el tipo no existe en _MEJORAS_DISPONIBLES.
    """
    tipo = config.get("tipo", "")
    if tipo not in _MEJORAS_DISPONIBLES:
        raise ValueError(
            f"Mejora desconocida: '{tipo}'. "
            f"Opciones válidas: {list(_MEJORAS_DISPONIBLES.keys())}"
        )

    # Extraemos todos los parámetros excepto "tipo"
    kwargs = {k: v for k, v in config.items() if k != "tipo"}
    return _MEJORAS_DISPONIBLES[tipo](imagen_gris, **kwargs)


def calcular_histograma_comparativo(
    img_antes: np.ndarray,
    img_despues: np.ndarray,
) -> go.Figure:
    """
    Superpone dos histogramas en escala de grises:
    - Rojo  = imagen de entrada (antes de la mejora).
    - Azul  = imagen de salida  (después de la mejora).

    Permite comparar visualmente cómo la mejora redistribuye
    las intensidades.

    Parámetros
    ----------
    img_antes   : np.ndarray 2D uint8 — imagen sin mejora.
    img_despues : np.ndarray 2D uint8 — imagen mejorada.

    Retorna
    -------
    plotly.graph_objects.Figure
    """
    fig = go.Figure()

    # ── Canal "antes" en rojo ─────────────────────────────────────────────────
    hist_antes, bins = np.histogram(
        img_antes.flatten(), bins=256, range=(0, 256)
    )
    fig.add_trace(go.Scatter(
        x=bins[:-1],
        y=hist_antes,
        mode="lines",
        fill="tozeroy",
        line=dict(color="#f87171", width=1.5),
        fillcolor="rgba(248,113,113,0.2)",
        name="Antes",
    ))

    # ── Canal "después" en azul ───────────────────────────────────────────────
    hist_despues, _ = np.histogram(
        img_despues.flatten(), bins=256, range=(0, 256)
    )
    fig.add_trace(go.Scatter(
        x=bins[:-1],
        y=hist_despues,
        mode="lines",
        fill="tozeroy",
        line=dict(color="#60a5fa", width=1.5),
        fillcolor="rgba(96,165,250,0.2)",
        name="Después",
    ))

    # ── Estilo del gráfico ────────────────────────────────────────────────────
    fig.update_layout(
        title=dict(text="Histograma comparativo", font=dict(size=13)),
        xaxis=dict(title="Intensidad (0–255)", range=[0, 255]),
        yaxis=dict(title="Frecuencia"),
        margin=dict(l=10, r=10, t=40, b=30),
        legend=dict(orientation="h", y=-0.3),
        height=300,
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )

    return fig


# =============================================================================
# FASE 5 — Segmentación por umbralización y morfología
# =============================================================================

# Constantes exportadas para la UI
METODOS_UMBRALIZACION = ["Otsu", "Kapur", "Media", "Banda", "Manual"]
METODOS_AUTOMATICOS   = {"Otsu", "Kapur", "Media"}


def umbralizar_otsu(imagen_gris: np.ndarray) -> tuple:
    """
    Umbraliza la imagen usando el método de Otsu.

    Otsu maximiza la varianza inter-clase buscando el umbral óptimo
    de manera automática a partir del histograma de la imagen.

    Parámetros
    ----------
    imagen_gris : np.ndarray 2D uint8
        Imagen en escala de grises.

    Retorna
    -------
    (binaria, info) : tuple
        binaria : np.ndarray 2D uint8 — máscara binaria 0/255.
        info    : dict — {"umbral": int} con el valor encontrado por Otsu.
    """
    thresh_val, binaria = cv2.threshold(
        imagen_gris,
        0,           # ignorado cuando se usa THRESH_OTSU
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU,
    )
    info = {"umbral": int(thresh_val)}
    return binaria, info


def umbralizar_kapur(imagen_gris: np.ndarray) -> tuple:
    """
    Umbraliza la imagen usando la entropía de Kapur (máxima entropía).

    El método itera sobre todos los umbrales candidatos (1–254) y elige
    el valor t* que maximiza la entropía total de la imagen dividida en
    dos regiones: fondo (0..t*-1) y objeto (t*..255).

    H_total(t) = p1 * (−Σ p1n·log(p1n+ε)) + p2 * (−Σ p2n·log(p2n+ε))

    Parámetros
    ----------
    imagen_gris : np.ndarray 2D uint8

    Retorna
    -------
    (binaria, info) : tuple
        binaria  : np.ndarray 2D uint8
        info     : dict — {"umbral": int, "entropia_max": float}
    """
    # Histograma normalizado (probabilidades)
    hist, _ = np.histogram(imagen_gris.flatten(), bins=256, range=(0, 256))
    prob = hist.astype(np.float64) / float(imagen_gris.size)

    mejor_H    = -np.inf
    umbral_opt = 127

    for t in range(1, 255):
        # Región 1: [0, t-1]
        p1_vec = prob[:t]
        P1     = p1_vec.sum()
        if P1 <= 0:
            continue
        p1_norm  = p1_vec / P1
        H1 = -np.sum(p1_norm * np.log(p1_norm + 1e-12))

        # Región 2: [t, 255]
        p2_vec = prob[t:]
        P2     = p2_vec.sum()
        if P2 <= 0:
            continue
        p2_norm  = p2_vec / P2
        H2 = -np.sum(p2_norm * np.log(p2_norm + 1e-12))

        H_total = P1 * H1 + P2 * H2
        if H_total > mejor_H:
            mejor_H    = H_total
            umbral_opt = t

    _, binaria = cv2.threshold(
        imagen_gris,
        umbral_opt,
        255,
        cv2.THRESH_BINARY,
    )
    info = {"umbral": int(umbral_opt), "entropia_max": round(float(mejor_H), 4)}
    return binaria, info


def umbralizar_media(imagen_gris: np.ndarray) -> tuple:
    """
    Umbraliza usando la media global de intensidad como umbral.

    Umbral = round(mean(imagen_gris)).

    Parámetros
    ----------
    imagen_gris : np.ndarray 2D uint8

    Retorna
    -------
    (binaria, info) : tuple
        binaria : np.ndarray 2D uint8
        info    : dict — {"umbral": int, "media": float}
    """
    media   = float(np.mean(imagen_gris))
    umbral  = int(round(media))
    _, binaria = cv2.threshold(imagen_gris, umbral, 255, cv2.THRESH_BINARY)
    info = {"umbral": umbral, "media": round(media, 2)}
    return binaria, info


def umbralizar_manual(imagen_gris: np.ndarray, umbral: int = 127) -> tuple:
    """
    Umbraliza usando un umbral definido manualmente por el usuario.

    Parámetros
    ----------
    imagen_gris : np.ndarray 2D uint8
    umbral      : int  — valor entre 0 y 255.

    Retorna
    -------
    (binaria, info) : tuple
        binaria : np.ndarray 2D uint8
        info    : dict — {"umbral": int}
    """
    _, binaria = cv2.threshold(imagen_gris, umbral, 255, cv2.THRESH_BINARY)
    info = {"umbral": int(umbral)}
    return binaria, info


def umbralizar_banda(
    imagen_gris: np.ndarray,
    t1: int = 80,
    t2: int = 200,
) -> tuple:
    """
    Umbralización por banda de intensidad.

    Los píxeles cuya intensidad cae en [t1, t2] se marcan como 255 (objeto);
    el resto queda en 0 (fondo). Este método es el más útil para aislar
    botellas PET en cuerpos de agua, porque permite ajustar precisamente
    el rango de intensidad correspondiente al plástico transparente/blanco.

    Parámetros
    ----------
    imagen_gris : np.ndarray 2D uint8
    t1          : int — límite inferior del rango (inclusive).
    t2          : int — límite superior del rango (inclusive), t2 > t1.

    Retorna
    -------
    (binaria, info) : tuple
        binaria : np.ndarray 2D uint8 — máscara de banda.
        info    : dict — {"t1": int, "t2": int}
    """
    binaria = np.zeros_like(imagen_gris, dtype=np.uint8)
    # Píxeles dentro del rango [t1, t2] → 255
    mascara_banda = (imagen_gris >= t1) & (imagen_gris <= t2)
    binaria[mascara_banda] = 255
    info = {"t1": int(t1), "t2": int(t2)}
    return binaria, info


# -----------------------------------------------------------------------------
# MORFOLOGÍA POST-UMBRALIZACIÓN — operaciones para limpiar la máscara binaria
# -----------------------------------------------------------------------------

def aplicar_cierre(mascara: np.ndarray, ksize: int = 5) -> np.ndarray:
    """
    Aplica cierre morfológico (MORPH_CLOSE) con kernel elíptico.

    El cierre (dilatación seguida de erosión) rellena pequeños huecos
    y discontinuidades dentro de los objetos segmentados.

    Parámetros
    ----------
    mascara : np.ndarray 2D uint8 — máscara binaria (0/255).
    ksize   : int — tamaño del kernel cuadrado (debe ser impar).

    Retorna
    -------
    np.ndarray 2D uint8 — máscara con huecos rellenados.
    """
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (int(ksize), int(ksize))
    )
    return cv2.morphologyEx(mascara, cv2.MORPH_CLOSE, kernel)


def aplicar_apertura(mascara: np.ndarray, ksize: int = 3) -> np.ndarray:
    """
    Aplica apertura morfológica (MORPH_OPEN) con kernel elíptico.

    La apertura (erosión seguida de dilatación) elimina pequeñas manchas
    de ruido aisladas fuera del objeto principal (botellas).

    Parámetros
    ----------
    mascara : np.ndarray 2D uint8 — máscara binaria (0/255).
    ksize   : int — tamaño del kernel (debe ser impar).

    Retorna
    -------
    np.ndarray 2D uint8 — máscara sin ruido pequeño.
    """
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (int(ksize), int(ksize))
    )
    return cv2.morphologyEx(mascara, cv2.MORPH_OPEN, kernel)


def aplicar_relleno_huecos(mascara: np.ndarray) -> np.ndarray:
    """
    Rellena completamente los huecos internos de los objetos binarios.

    Algoritmo:
    1. Hacer flood-fill desde la esquina superior izquierda (0, 0)
       sobre una copia con borde de 2px — esto pinta el fondo exterior.
    2. Invertir esa imagen → solo el interior de los objetos queda blanco.
    3. OR con la máscara original → objetos sólidos sin huecos internos.

    Parámetros
    ----------
    mascara : np.ndarray 2D uint8 — máscara binaria (0/255).

    Retorna
    -------
    np.ndarray 2D uint8 — máscara con huecos internos rellenados.
    """
    h, w = mascara.shape[:2]
    # Canvas más grande (borde de 2px para que flood-fill no se escape)
    canvas = np.zeros((h + 4, w + 4), dtype=np.uint8)
    canvas[2:h + 2, 2:w + 2] = mascara

    # Flood-fill desde la esquina (0,0) → pinta el fondo exterior
    mascara_flood = canvas.copy()
    cv2.floodFill(mascara_flood, None, (0, 0), 255)

    # Invertir → solo quedan blancos los huecos internos
    exterior_invertido = cv2.bitwise_not(mascara_flood)

    # Recortar el borde que añadimos
    exterior_recortado = exterior_invertido[2:h + 2, 2:w + 2]

    # OR con la máscara original: objetos + huecos rellenados
    resultado = cv2.bitwise_or(mascara, exterior_recortado)
    return resultado


def aplicar_umbral(
    imagen_gris: np.ndarray,
    metodo: str,
    invertir: bool = False,
    **kwargs,
) -> tuple:
    """
    Dispatcher central de umbralización.

    Enruta la solicitud al método correcto según el parámetro `metodo`
    y opcionalmente invierte la máscara resultante.

    Parámetros
    ----------
    imagen_gris : np.ndarray 2D uint8
    metodo      : str — uno de METODOS_UMBRALIZACION.
    invertir    : bool — si True, devuelve (255 − binaria).
    **kwargs    : parámetros adicionales según el método:
                  umbral (Manual), t1/t2 (Banda).

    Retorna
    -------
    (binaria, info) : tuple
        binaria : np.ndarray 2D uint8
        info    : dict con los parámetros usados.

    Raises
    ------
    ValueError si `metodo` no está en METODOS_UMBRALIZACION.
    """
    if metodo == "Otsu":
        binaria, info = umbralizar_otsu(imagen_gris)
    elif metodo == "Kapur":
        binaria, info = umbralizar_kapur(imagen_gris)
    elif metodo == "Media":
        binaria, info = umbralizar_media(imagen_gris)
    elif metodo == "Manual":
        umbral = int(kwargs.get("umbral", 127))
        binaria, info = umbralizar_manual(imagen_gris, umbral)
    elif metodo == "Banda":
        t1 = int(kwargs.get("t1", 80))
        t2 = int(kwargs.get("t2", 200))
        binaria, info = umbralizar_banda(imagen_gris, t1, t2)
    else:
        raise ValueError(
            f"Método desconocido: '{metodo}'. "
            f"Opciones válidas: {METODOS_UMBRALIZACION}"
        )

    # Inversión opcional de la máscara
    if invertir:
        binaria = cv2.bitwise_not(binaria)

    return binaria, info