# =============================================================================
# procesamiento.py
# MÃ³dulo de lÃ³gica de procesamiento de imÃ¡genes para el Dashboard de
# SegmentaciÃ³n de Botellas PET en cuerpos de agua.
#
# REGLA: Este archivo NO importa Streamlit. Solo OpenCV, NumPy y Plotly.
#
# Fase 1   : carga, redimensiÃ³n y conversiÃ³n de color.
# Fase 2   : conversiÃ³n a escala de grises.
# Fase 3   : filtros espaciales acumulativos (7 filtros).
# Fase 3.5 : filtrado en frecuencia (FFT lowpass / highpass).
# Fase 4   : mejora de contraste y brillo (6 tÃ©cnicas con LUT).
# =============================================================================

import cv2
import numpy as np
import plotly.graph_objects as go
from scipy.ndimage import maximum_filter, minimum_filter, gaussian_filter1d
from scipy.signal import find_peaks


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# CONSTANTES GLOBALES
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

# Ancho mÃ¡ximo permitido antes de redimensionar (en pÃ­xeles)
MAX_ANCHO_PX = 800


# =============================================================================
# FASE 1 â€” Carga y preprocesado bÃ¡sico
# =============================================================================

def redimensionar_imagen(imagen_bgr: np.ndarray) -> np.ndarray:
    """
    Redimensiona la imagen si su ancho supera MAX_ANCHO_PX,
    manteniendo la relaciÃ³n de aspecto original.

    Se usa INTER_AREA porque es la interpolaciÃ³n Ã³ptima al REDUCIR
    imÃ¡genes: minimiza el aliasing y produce mejor calidad que
    INTER_LINEAR al hacer downscaling.

    ParÃ¡metros
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

    # Si la imagen ya cabe dentro del lÃ­mite, la devolvemos sin cambios
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

    OpenCV carga imÃ¡genes en BGR por razones histÃ³ricas.
    Streamlit y Matplotlib esperan RGB.
    Esta conversiÃ³n es necesaria antes de cualquier visualizaciÃ³n.

    ParÃ¡metros
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
    FunciÃ³n auxiliar para el pipeline de carga.

    ParÃ¡metros
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
# FASE 2 â€” ConversiÃ³n a escala de grises
# =============================================================================

def convertir_a_gris(imagen_bgr: np.ndarray) -> np.ndarray:
    """
    Convierte una imagen BGR a escala de grises usando ponderaciÃ³n perceptual.

    FÃ³rmula ITU-R BT.601 (estÃ¡ndar televisiÃ³n):
        I_gris = 0.299Â·R + 0.587Â·G + 0.114Â·B

    Los coeficientes reflejan la sensibilidad del ojo humano:
    - Verde recibe mayor peso (0.587) por ser el mÃ¡s visible
    - Azul recibe menor peso (0.114) por ser el menos visible
    Esta ponderaciÃ³n produce una percepciÃ³n de brillo mÃ¡s natural
    que el promedio simple (R+G+B)/3.

    ParÃ¡metros
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
# HISTOGRAMA â€” VisualizaciÃ³n con Plotly
# =============================================================================

def calcular_histograma(imagen: np.ndarray) -> go.Figure:
    """
    Calcula y devuelve un grÃ¡fico Plotly con el histograma de la imagen.

    Comportamiento segÃºn el tipo de imagen:
    - imagen 2D (escala de grises) â†’ un solo canal en gris
    - imagen 3D RGB (HÃ—WÃ—3)       â†’ tres canales superpuestos R, G, B

    Se usa fill="tozeroy" para crear Ã¡reas rellenas bajo las curvas,
    facilitando la comparaciÃ³n visual entre canales.

    ParÃ¡metros
    ----------
    imagen : np.ndarray
        Imagen 2D (grises) o 3D RGB. En formato RGB para visualizaciÃ³n.

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
        # â”€â”€ Canal Ãºnico (escala de grises) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
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
        # â”€â”€ Tres canales RGB superpuestos â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        # Nota: la imagen ya estÃ¡ en RGB cuando llega aquÃ­
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

    # â”€â”€ Estilo del grÃ¡fico â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
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
# FASE 3 â€” Filtros espaciales
# Todas las funciones:
#   Â· Reciben imagen en escala de grises (np.ndarray 2D, uint8)
#   Â· Devuelven imagen en escala de grises (np.ndarray 2D, uint8)
#   Â· Usan BORDER_REFLECT_101 para evitar artefactos en bordes
# =============================================================================

def aplicar_gaussiano(
    imagen_gris: np.ndarray,
    ksize: int = 3,
    sigma: float = 0.0,
) -> np.ndarray:
    """
    Filtro Gaussiano â€” suavizado con ponderaciÃ³n normal.

    Cada pÃ­xel se reemplaza por la media ponderada de su vecindad,
    donde los pesos siguen una distribuciÃ³n gaussiana. Los pÃ­xeles
    mÃ¡s cercanos al centro reciben mayor peso.

    sigma=0 â†’ OpenCV calcula Ïƒ automÃ¡ticamente:
        Ïƒ â‰ˆ 0.3Â·((ksize-1)/2 - 1) + 0.8

    Ventaja sobre el box filter: mejor preservaciÃ³n de bordes suaves.
    Desventaja: mÃ¡s costoso computacionalmente.

    ParÃ¡metros
    ----------
    imagen_gris : np.ndarray 2D uint8
    ksize       : int   â€” tamaÃ±o del kernel (debe ser impar).
    sigma       : float â€” desviaciÃ³n estÃ¡ndar.

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
    Filtro de Mediana â€” filtro no lineal de orden.

    Reemplaza cada pÃ­xel por la mediana de su vecindad kÃ—k.
    Al ser no lineal, es altamente efectivo contra ruido impulsivo
    (sal y pimienta) sin difuminar los bordes.

    Para botellas PET en agua: elimina brillos puntuales del sol
    sobre el agua sin destruir el contorno de la botella.

    ParÃ¡metros
    ----------
    imagen_gris : np.ndarray 2D uint8
    ksize       : int â€” tamaÃ±o del kernel (debe ser impar).

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
    Filtro Bilateral â€” suavizado que PRESERVA BORDES.

    Combina dos kernels gaussianos:
    Â· Kernel espacial (sigma_space): pondera por distancia euclidiana.
      Equivalente a un gaussiano espacial clÃ¡sico.
    Â· Kernel de rango (sigma_color): pondera por similitud de intensidad.
      PÃ­xeles con intensidad similar al pÃ­xel central reciben mayor peso.

    FÃ³rmula:
        I_out(p) = Î£_q [G_s(||p-q||) Â· G_r(|I_p-I_q|) Â· I_q]
                   â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
                   Î£_q [G_s(||p-q||) Â· G_r(|I_p-I_q|)]

    Para botellas en agua: preserva el borde nÃ­tido entre la botella
    y el agua mientras suaviza la textura interna del agua.

    ParÃ¡metros
    ----------
    imagen_gris  : np.ndarray 2D uint8
    d            : int   â€” diÃ¡metro de la vecindad.
    sigma_color  : float â€” Ïƒ en espacio de intensidades.
    sigma_space  : float â€” Ïƒ en espacio cartesiano.

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

    MatemÃ¡ticamente equivalente a la convoluciÃ³n con:
        K = (1/kÂ²) Â· J_{kÃ—k}
    donde J es una matriz de unos.

    Todos los pÃ­xeles de la vecindad tienen el mismo peso (1/kÂ²).
    Elimina altas frecuencias (detalles, textura del agua) pero
    tiende a difuminar los bordes de la botella.

    ParÃ¡metros
    ----------
    imagen_gris : np.ndarray 2D uint8
    ksize       : int â€” tamaÃ±o del kernel (impar).

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
    Filtro Promediador â€” box filter con kernel NumPy explÃ­cito.

    Conceptualmente idÃ©ntico a aplicar_paso_bajas pero el kernel
    se construye explÃ­citamente con NumPy para mostrar la mecÃ¡nica
    de la convoluciÃ³n con kernel personalizado.

    Kernel: K = ones(k,k) / kÂ²

    ParÃ¡metros
    ----------
    imagen_gris : np.ndarray 2D uint8
    ksize       : int â€” tamaÃ±o del kernel (impar).

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
    Filtro de MÃ¡ximo â€” dilataciÃ³n morfolÃ³gica.

    Reemplaza cada pÃ­xel por el valor mÃ¡ximo en su vecindad kÃ—k.
    Efecto: expande las zonas brillantes, elimina manchas oscuras
    pequeÃ±as (pimienta).

    Para botellas transparentes: puede realzar los brillos sobre
    la botella y distinguirlos del fondo oscuro del agua.

    ParÃ¡metros
    ----------
    imagen_gris : np.ndarray 2D uint8
    ksize       : int â€” tamaÃ±o de la ventana.

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
    Filtro de MÃ­nimo â€” erosiÃ³n morfolÃ³gica.

    Reemplaza cada pÃ­xel por el valor mÃ­nimo en su vecindad kÃ—k.
    Efecto: expande las zonas oscuras, elimina manchas brillantes
    pequeÃ±as (sal).

    Para botellas: puede oscurecer la textura del agua y hacer
    mÃ¡s homogÃ©neo el fondo antes de la segmentaciÃ³n.

    ParÃ¡metros
    ----------
    imagen_gris : np.ndarray 2D uint8
    ksize       : int â€” tamaÃ±o de la ventana.

    Retorna
    -------
    np.ndarray 2D uint8
    """
    return minimum_filter(
        imagen_gris,
        size=ksize,
        mode="reflect",
    ).astype(np.uint8)


# â”€â”€ Dispatcher central de filtros â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

# Mapeo nombre â†’ funciÃ³n para el dispatcher
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

    Recibe un diccionario de configuraciÃ³n con al menos la clave "tipo"
    y delega a la funciÃ³n correspondiente pasando el resto del dict
    como kwargs.

    ParÃ¡metros
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
            f"Opciones vÃ¡lidas: {list(_FILTROS_DISPONIBLES.keys())}"
        )

    # Extraemos todos los parÃ¡metros excepto "tipo"
    kwargs = {k: v for k, v in config.items() if k != "tipo"}

    return _FILTROS_DISPONIBLES[tipo](imagen_gris, **kwargs)


def descripcion_filtro(tipo: str, config: dict) -> str:
    """
    Devuelve una descripciÃ³n didÃ¡ctica del filtro aplicado para
    mostrar debajo de cada paso en el Ã¡rea principal.

    ParÃ¡metros
    ----------
    tipo   : str  â€” nombre del filtro.
    config : dict â€” configuraciÃ³n completa.

    Retorna
    -------
    str â€” descripciÃ³n en formato Markdown.
    """
    k  = config.get("ksize", 3)
    d  = config.get("d", 9)
    sc = config.get("sigma_color", 75.0)
    ss = config.get("sigma_space", 75.0)
    sg = config.get("sigma", 0.0)

    descripciones = {
        "Gaussiano": (
            f"**Gaussiano** (ksize={k}Ã—{k}, Ïƒ={sg}). "
            "Suavizado con ponderaciÃ³n gaussiana. Reduce la textura del agua "
            "preservando los bordes suaves de la botella. "
            "Mayor ksize = mayor suavizado pero mÃ¡s difuminado de bordes."
        ),
        "Mediana": (
            f"**Mediana** (ksize={k}Ã—{k}). Filtro no lineal. "
            "Elimina brillos puntuales del sol sobre el agua sin afectar "
            "significativamente el borde de la botella. "
            "Ideal como primer filtro para imÃ¡genes con reflejos."
        ),
        "Bilateral": (
            f"**Bilateral** (d={d}, Ïƒ_color={sc}, Ïƒ_space={ss}). "
            "Suaviza manteniendo bordes nÃ­tidos. Excelente para homogenizar "
            "la textura del agua sin difuminar el contorno de la botella. "
            f"Ïƒ_color={sc}: {'bordes muy preservados' if sc < 50 else 'suavizado moderado'}."
        ),
        "Paso Bajas": (
            f"**Paso Bajas / Box filter** ({k}Ã—{k}). "
            "Todos los pÃ­xeles de la vecindad tienen el mismo peso (1/kÂ²). "
            "Reduce variaciones de alta frecuencia (textura del agua). "
            "Puede difuminar los bordes de la botella."
        ),
        "Promediador": (
            f"**Promediador** ({k}Ã—{k}). "
            "Kernel explÃ­cito ones(k,k)/kÂ². Equivalente al Paso Bajas. "
            "Ãštil para homogenizar regiones del agua antes de umbralizar."
        ),
        "Max": (
            f"**MÃ¡ximo / DilataciÃ³n** ({k}Ã—{k}). "
            "Expande zonas brillantes. Puede realzar la botella clara "
            "sobre fondo oscuro. Elimina manchas oscuras pequeÃ±as."
        ),
        "Min": (
            f"**MÃ­nimo / ErosiÃ³n** ({k}Ã—{k}). "
            "Expande zonas oscuras. Homogeniza el fondo de agua "
            "antes de la segmentaciÃ³n. Elimina brillos puntuales pequeÃ±os."
        ),
    }
    return descripciones.get(tipo, "Filtro aplicado.")


# =============================================================================
# FASE 3.5 â€” Filtrado en frecuencia (FFT)
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
    1. np.fft.fft2  â†’ transforma la imagen al dominio de frecuencias.
    2. np.fft.fftshift â†’ desplaza la componente DC al centro.
    3. MÃ¡scara circular: radio = cutoff Ã— min(H, W) / 2.
       lowpass  â†’ M=1 dentro del cÃ­rculo (pasa bajas).
       highpass â†’ M=0 dentro del cÃ­rculo (pasa altas).
    4. Multiplicar espectro Ã— mÃ¡scara.
    5. np.fft.ifftshift + np.fft.ifft2 â†’ volver al dominio espacial.
    6. Tomar parte real, recortar a [0,255] y convertir a uint8.

    Espectro de magnitud:
        S = log(1 + |F_shifted|), normalizado a [0, 255] uint8.

    ParÃ¡metros
    ----------
    imagen_gris : np.ndarray 2D uint8
        Imagen en escala de grises.
    cutoff : float
        FracciÃ³n del radio mÃ­nimo (0.01â€“1.00) que define el corte.
        cutoff=0.15 â†’ radio = 15 % del semiancho mÃ¡s pequeÃ±o.
    tipo : str
        "lowpass"  â†’ elimina altas frecuencias (suaviza).
        "highpass" â†’ elimina bajas frecuencias (realza bordes).

    Retorna
    -------
    (img_filtrada, mascara, espectro_mag) : tuple de np.ndarray
        img_filtrada : np.ndarray 2D uint8  â€” resultado espacial.
        mascara      : np.ndarray 2D float  â€” mÃ¡scara binaria [0,1].
        espectro_mag : np.ndarray 2D uint8  â€” espectro de magnitud.
    """
    H, W = imagen_gris.shape

    # â”€â”€ 1. Transformada de Fourier y desplazamiento al centro â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    F = np.fft.fft2(imagen_gris.astype(np.float64))
    F_shifted = np.fft.fftshift(F)

    # â”€â”€ 2. Espectro de magnitud (para visualizaciÃ³n) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    magnitud = np.abs(F_shifted)
    espectro_log = np.log1p(magnitud)   # log(1 + |F|)
    # Normalizar a [0, 255]
    espectro_min, espectro_max = espectro_log.min(), espectro_log.max()
    if espectro_max > espectro_min:
        espectro_norm = (espectro_log - espectro_min) / (espectro_max - espectro_min)
    else:
        espectro_norm = espectro_log * 0.0
    espectro_mag = (espectro_norm * 255).astype(np.uint8)

    # â”€â”€ 3. MÃ¡scara circular â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
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

    # â”€â”€ 4-5. Filtrado y transformada inversa â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    F_filtrado = F_shifted * mascara
    F_inv = np.fft.ifftshift(F_filtrado)
    img_reconstruida = np.fft.ifft2(F_inv)

    # â”€â”€ 6. Convertir a uint8 â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    img_real = np.real(img_reconstruida)
    img_filtrada = np.clip(img_real, 0, 255).astype(np.uint8)

    return img_filtrada, mascara, espectro_mag


def crear_espectro_con_mascara(
    espectro_mag: np.ndarray,
    mascara: np.ndarray,
) -> np.ndarray:
    """
    Convierte el espectro de magnitud a RGB y dibuja el borde de la
    mÃ¡scara circular en rojo sobre Ã©l para facilitar la interpretaciÃ³n.

    El borde se extrae con una operaciÃ³n morfolÃ³gica MORPH_GRADIENT
    (dilataciÃ³n - erosiÃ³n) con kernel 3Ã—3, que devuelve solo los
    pÃ­xeles de transiciÃ³n de la mÃ¡scara.

    ParÃ¡metros
    ----------
    espectro_mag : np.ndarray 2D uint8
        Espectro de magnitud normalizado.
    mascara : np.ndarray 2D float
        MÃ¡scara binaria [0.0, 1.0] generada por fft_filter.

    Retorna
    -------
    np.ndarray 3D uint8 (H Ã— W Ã— 3)
        Espectro en RGB con borde de la mÃ¡scara resaltado en rojo.
    """
    # Convertir espectro gris â†’ RGB (3 canales)
    espectro_rgb = cv2.cvtColor(espectro_mag, cv2.COLOR_GRAY2RGB)

    # Borde de la mÃ¡scara: MORPH_GRADIENT = dilataciÃ³n - erosiÃ³n
    kernel_borde = np.ones((3, 3), dtype=np.uint8)
    mascara_u8 = (mascara * 255).astype(np.uint8)
    borde = cv2.morphologyEx(mascara_u8, cv2.MORPH_GRADIENT, kernel_borde)

    # Pintar el borde en rojo (255, 60, 60) sobre el espectro RGB
    espectro_rgb[borde > 0] = [255, 60, 60]

    return espectro_rgb


# =============================================================================
# FASE 4 â€” Enhancement (mejora de contraste y brillo)
# Todas las funciones:
#   Â· Usan LUT de 256 entradas con cv2.LUT para mÃ¡xima eficiencia.
#   Â· Reciben np.ndarray 2D uint8 y devuelven np.ndarray 2D uint8.
# =============================================================================

# Lista de opciones exportada al sidebar de app.py
MEJORAS_OPCIONES = [
    "CorrecciÃ³n Gamma",
    "Desplazamiento (Brillo)",
    "ContracciÃ³n / ExpansiÃ³n",
    "EcualizaciÃ³n Uniforme",
    "EcualizaciÃ³n Rayleigh",
    "EcualizaciÃ³n Log. HiperbÃ³lica",
]


def _construir_lut(valores: np.ndarray) -> np.ndarray:
    """
    Construye una LUT de 256 entradas a partir de un array de 256 flotantes.
    Recorta a [0,255] y convierte a uint8.

    ParÃ¡metros
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
    CorrecciÃ³n gamma:
        I_out = (I / 255)^Î³ Ã— 255

    Î³ < 1 â†’ aclarar (realza sombras).
    Î³ > 1 â†’ oscurecer (comprime luces).
    Î³ = 1 â†’ identidad.

    ParÃ¡metros
    ----------
    imagen_gris : np.ndarray 2D uint8
    gamma       : float â€” exponente de correcciÃ³n.

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
        I_out = clip(I + Î”, 0, 255)

    Î´ > 0 â†’ imagen mÃ¡s brillante.
    Î´ < 0 â†’ imagen mÃ¡s oscura.

    ParÃ¡metros
    ----------
    imagen_gris : np.ndarray 2D uint8
    delta       : int â€” desplazamiento en intensidad (âˆ’255 a +255).

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
    Mapeo lineal por tramos (contracciÃ³n / expansiÃ³n de contraste):

      I < a_in               â†’ a_out
      a_in <= I <= b_in      â†’ lineal: a_out + (I - a_in) Ã— (b_out - a_out) / (b_in - a_in)
      I > b_in               â†’ b_out

    Permite expandir un rango de interÃ©s [a_in, b_in] hacia [a_out, b_out],
    aumentando el contraste dentro de ese rango.

    ParÃ¡metros
    ----------
    imagen_gris : np.ndarray 2D uint8
    a_in, b_in  : int â€” rango de entrada (a_in < b_in).
    a_out, b_out: int â€” rango de salida.

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
    EcualizaciÃ³n de histograma uniforme (estÃ¡ndar).

    Redistribuye las intensidades para que el histograma sea lo mÃ¡s
    plano posible. Maximiza el contraste global pero puede amplificar
    ruido en regiones homogÃ©neas.

    Usa cv2.equalizeHist (implementaciÃ³n optimizada de OpenCV).

    ParÃ¡metros
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
    EcualizaciÃ³n con distribuciÃ³n Rayleigh (curva raÃ­z cuadrada):
        I_out = 255 Ã— âˆš(I / 255)

    Variante no lineal que aclarar la imagen de forma suave.
    MÃ¡s conservadora que la ecualizaciÃ³n uniforme: no corta las
    distribuciones bimodales abruptamente.

    ParÃ¡metros
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
    EcualizaciÃ³n logarÃ­tmica hiperbÃ³lica:
        I_out = 255 Ã— log(1 + I) / log(256)

    La curva logarÃ­tmica comprime los valores altos y expande los
    bajos, realzando los detalles en zonas oscuras de la imagen.
    Ãštil para imÃ¡genes subexpuestas con reflejos de agua.

    ParÃ¡metros
    ----------
    imagen_gris : np.ndarray 2D uint8

    Retorna
    -------
    np.ndarray 2D uint8
    """
    entradas = np.arange(256, dtype=np.float64)
    lut = _construir_lut(255.0 * np.log1p(entradas) / np.log(256.0))
    return cv2.LUT(imagen_gris, lut)


# â”€â”€ Dispatcher central de mejoras â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

# Mapeo nombre â†’ funciÃ³n para el dispatcher
_MEJORAS_DISPONIBLES = {
    "CorrecciÃ³n Gamma":           mejora_gamma,
    "Desplazamiento (Brillo)":    mejora_desplazamiento,
    "ContracciÃ³n / ExpansiÃ³n":    mejora_contraccion_expansion,
    "EcualizaciÃ³n Uniforme":      mejora_ecual_uniforme,
    "EcualizaciÃ³n Rayleigh":      mejora_ecual_rayleigh,
    "EcualizaciÃ³n Log. HiperbÃ³lica": mejora_ecual_log_hiperbolica,
}


def aplicar_mejora(imagen_gris: np.ndarray, config: dict) -> np.ndarray:
    """
    Dispatcher central de la Fase 4.

    Recibe un diccionario con al menos la clave "tipo" y delega
    a la funciÃ³n de mejora correspondiente.

    ParÃ¡metros
    ----------
    imagen_gris : np.ndarray 2D uint8
    config      : dict â€” {"tipo": str, <params>: val, ...}

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
            f"Opciones vÃ¡lidas: {list(_MEJORAS_DISPONIBLES.keys())}"
        )

    # Extraemos todos los parÃ¡metros excepto "tipo"
    kwargs = {k: v for k, v in config.items() if k != "tipo"}
    return _MEJORAS_DISPONIBLES[tipo](imagen_gris, **kwargs)


def calcular_histograma_comparativo(
    img_antes: np.ndarray,
    img_despues: np.ndarray,
) -> go.Figure:
    """
    Superpone dos histogramas en escala de grises:
    - Rojo  = imagen de entrada (antes de la mejora).
    - Azul  = imagen de salida  (despuÃ©s de la mejora).

    Permite comparar visualmente cÃ³mo la mejora redistribuye
    las intensidades.

    ParÃ¡metros
    ----------
    img_antes   : np.ndarray 2D uint8 â€” imagen sin mejora.
    img_despues : np.ndarray 2D uint8 â€” imagen mejorada.

    Retorna
    -------
    plotly.graph_objects.Figure
    """
    fig = go.Figure()

    # â”€â”€ Canal "antes" en rojo â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
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

    # â”€â”€ Canal "despuÃ©s" en azul â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
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
        name="DespuÃ©s",
    ))

    # â”€â”€ Estilo del grÃ¡fico â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    fig.update_layout(
        title=dict(text="Histograma comparativo", font=dict(size=13)),
        xaxis=dict(title="Intensidad (0â€“255)", range=[0, 255]),
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
# FASE 5 â€” SegmentaciÃ³n por umbralizaciÃ³n y morfologÃ­a
# =============================================================================

# Constantes exportadas para la UI
METODOS_UMBRALIZACION = ["Otsu", "Kapur", "Media", "Banda", "Manual"]
METODOS_AUTOMATICOS   = {"Otsu", "Kapur", "Media"}


def umbralizar_otsu(imagen_gris: np.ndarray) -> tuple:
    """
    Umbraliza la imagen usando el mÃ©todo de Otsu.

    Otsu maximiza la varianza inter-clase buscando el umbral Ã³ptimo
    de manera automÃ¡tica a partir del histograma de la imagen.

    ParÃ¡metros
    ----------
    imagen_gris : np.ndarray 2D uint8
        Imagen en escala de grises.

    Retorna
    -------
    (binaria, info) : tuple
        binaria : np.ndarray 2D uint8 â€” mÃ¡scara binaria 0/255.
        info    : dict â€” {"umbral": int} con el valor encontrado por Otsu.
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
    Umbraliza la imagen usando la entropÃ­a de Kapur (mÃ¡xima entropÃ­a).

    El mÃ©todo itera sobre todos los umbrales candidatos (1â€“254) y elige
    el valor t* que maximiza la entropÃ­a total de la imagen dividida en
    dos regiones: fondo (0..t*-1) y objeto (t*..255).

    H_total(t) = p1 * (âˆ’Î£ p1nÂ·log(p1n+Îµ)) + p2 * (âˆ’Î£ p2nÂ·log(p2n+Îµ))

    ParÃ¡metros
    ----------
    imagen_gris : np.ndarray 2D uint8

    Retorna
    -------
    (binaria, info) : tuple
        binaria  : np.ndarray 2D uint8
        info     : dict â€” {"umbral": int, "entropia_max": float}
    """
    # Histograma normalizado (probabilidades)
    hist, _ = np.histogram(imagen_gris.flatten(), bins=256, range=(0, 256))
    prob = hist.astype(np.float64) / float(imagen_gris.size)

    mejor_H    = -np.inf
    umbral_opt = 127

    for t in range(1, 255):
        # RegiÃ³n 1: [0, t-1]
        p1_vec = prob[:t]
        P1     = p1_vec.sum()
        if P1 <= 0:
            continue
        p1_norm  = p1_vec / P1
        H1 = -np.sum(p1_norm * np.log(p1_norm + 1e-12))

        # RegiÃ³n 2: [t, 255]
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

    ParÃ¡metros
    ----------
    imagen_gris : np.ndarray 2D uint8

    Retorna
    -------
    (binaria, info) : tuple
        binaria : np.ndarray 2D uint8
        info    : dict â€” {"umbral": int, "media": float}
    """
    media   = float(np.mean(imagen_gris))
    umbral  = int(round(media))
    _, binaria = cv2.threshold(imagen_gris, umbral, 255, cv2.THRESH_BINARY)
    info = {"umbral": umbral, "media": round(media, 2)}
    return binaria, info


def umbralizar_manual(imagen_gris: np.ndarray, umbral: int = 127) -> tuple:
    """
    Umbraliza usando un umbral definido manualmente por el usuario.

    ParÃ¡metros
    ----------
    imagen_gris : np.ndarray 2D uint8
    umbral      : int  â€” valor entre 0 y 255.

    Retorna
    -------
    (binaria, info) : tuple
        binaria : np.ndarray 2D uint8
        info    : dict â€” {"umbral": int}
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
    UmbralizaciÃ³n por banda de intensidad.

    Los pÃ­xeles cuya intensidad cae en [t1, t2] se marcan como 255 (objeto);
    el resto queda en 0 (fondo). Este mÃ©todo es el mÃ¡s Ãºtil para aislar
    botellas PET en cuerpos de agua, porque permite ajustar precisamente
    el rango de intensidad correspondiente al plÃ¡stico transparente/blanco.

    ParÃ¡metros
    ----------
    imagen_gris : np.ndarray 2D uint8
    t1          : int â€” lÃ­mite inferior del rango (inclusive).
    t2          : int â€” lÃ­mite superior del rango (inclusive), t2 > t1.

    Retorna
    -------
    (binaria, info) : tuple
        binaria : np.ndarray 2D uint8 â€” mÃ¡scara de banda.
        info    : dict â€” {"t1": int, "t2": int}
    """
    binaria = np.zeros_like(imagen_gris, dtype=np.uint8)
    # PÃ­xeles dentro del rango [t1, t2] â†’ 255
    mascara_banda = (imagen_gris >= t1) & (imagen_gris <= t2)
    binaria[mascara_banda] = 255
    info = {"t1": int(t1), "t2": int(t2)}
    return binaria, info


# -----------------------------------------------------------------------------
# MORFOLOGÃA POST-UMBRALIZACIÃ“N â€” operaciones para limpiar la mÃ¡scara binaria
# -----------------------------------------------------------------------------

def aplicar_cierre(mascara: np.ndarray, ksize: int = 5) -> np.ndarray:
    """
    Aplica cierre morfolÃ³gico (MORPH_CLOSE) con kernel elÃ­ptico.

    El cierre (dilataciÃ³n seguida de erosiÃ³n) rellena pequeÃ±os huecos
    y discontinuidades dentro de los objetos segmentados.

    ParÃ¡metros
    ----------
    mascara : np.ndarray 2D uint8 â€” mÃ¡scara binaria (0/255).
    ksize   : int â€” tamaÃ±o del kernel cuadrado (debe ser impar).

    Retorna
    -------
    np.ndarray 2D uint8 â€” mÃ¡scara con huecos rellenados.
    """
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (int(ksize), int(ksize))
    )
    return cv2.morphologyEx(mascara, cv2.MORPH_CLOSE, kernel)


def aplicar_apertura(mascara: np.ndarray, ksize: int = 3) -> np.ndarray:
    """
    Aplica apertura morfolÃ³gica (MORPH_OPEN) con kernel elÃ­ptico.

    La apertura (erosiÃ³n seguida de dilataciÃ³n) elimina pequeÃ±as manchas
    de ruido aisladas fuera del objeto principal (botellas).

    ParÃ¡metros
    ----------
    mascara : np.ndarray 2D uint8 â€” mÃ¡scara binaria (0/255).
    ksize   : int â€” tamaÃ±o del kernel (debe ser impar).

    Retorna
    -------
    np.ndarray 2D uint8 â€” mÃ¡scara sin ruido pequeÃ±o.
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
       sobre una copia con borde de 2px â€” esto pinta el fondo exterior.
    2. Invertir esa imagen â†’ solo el interior de los objetos queda blanco.
    3. OR con la mÃ¡scara original â†’ objetos sÃ³lidos sin huecos internos.

    ParÃ¡metros
    ----------
    mascara : np.ndarray 2D uint8 â€” mÃ¡scara binaria (0/255).

    Retorna
    -------
    np.ndarray 2D uint8 â€” mÃ¡scara con huecos internos rellenados.
    """
    h, w = mascara.shape[:2]
    # Canvas mÃ¡s grande (borde de 2px para que flood-fill no se escape)
    canvas = np.zeros((h + 4, w + 4), dtype=np.uint8)
    canvas[2:h + 2, 2:w + 2] = mascara

    # Flood-fill desde la esquina (0,0) â†’ pinta el fondo exterior
    mascara_flood = canvas.copy()
    cv2.floodFill(mascara_flood, None, (0, 0), 255)

    # Invertir â†’ solo quedan blancos los huecos internos
    exterior_invertido = cv2.bitwise_not(mascara_flood)

    # Recortar el borde que aÃ±adimos
    exterior_recortado = exterior_invertido[2:h + 2, 2:w + 2]

    # OR con la mÃ¡scara original: objetos + huecos rellenados
    resultado = cv2.bitwise_or(mascara, exterior_recortado)
    return resultado


def aplicar_umbral(
    imagen_gris: np.ndarray,
    metodo: str,
    invertir: bool = False,
    **kwargs,
) -> tuple:
    """
    Dispatcher central de umbralizaciÃ³n.

    Enruta la solicitud al mÃ©todo correcto segÃºn el parÃ¡metro `metodo`
    y opcionalmente invierte la mÃ¡scara resultante.

    ParÃ¡metros
    ----------
    imagen_gris : np.ndarray 2D uint8
    metodo      : str â€” uno de METODOS_UMBRALIZACION.
    invertir    : bool â€” si True, devuelve (255 âˆ’ binaria).
    **kwargs    : parÃ¡metros adicionales segÃºn el mÃ©todo:
                  umbral (Manual), t1/t2 (Banda).

    Retorna
    -------
    (binaria, info) : tuple
        binaria : np.ndarray 2D uint8
        info    : dict con los parÃ¡metros usados.

    Raises
    ------
    ValueError si `metodo` no estÃ¡ en METODOS_UMBRALIZACION.
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
            f"MÃ©todo desconocido: '{metodo}'. "
            f"Opciones vÃ¡lidas: {METODOS_UMBRALIZACION}"
        )

    # InversiÃ³n opcional de la mÃ¡scara
    if invertir:
        binaria = cv2.bitwise_not(binaria)

    return binaria, info


# =============================================================================
# FASE 6 â€” AnÃ¡lisis de Componentes Conexos (CCL) y ExtracciÃ³n
# =============================================================================

def aplicar_ccl(imagen_binaria: np.ndarray, conectividad: int = 8) -> tuple:
    """
    Aplica el anÃ¡lisis de componentes conexos (CCL) con estadÃ­sticas.

    ParÃ¡metros
    ----------
    imagen_binaria : np.ndarray 2D uint8
        Imagen binaria (mÃ¡scara de entrada).
    conectividad : int
        Tipo de conectividad a evaluar: 4 u 8.

    Retorna
    -------
    (n_etiquetas, etiquetas, stats, centroides) : tuple
        n_etiquetas : int - nÃºmero total de etiquetas encontradas (incluye fondo).
        etiquetas   : np.ndarray 2D de enteros (H x W) - mapa de etiquetas.
        stats       : np.ndarray (N x 5) - estadÃ­sticas de cada etiqueta.
        centroides  : np.ndarray (N x 2) - coordenadas (x, y) de los centroides.
    """
    # Asegurar que sea binaria de un solo canal y uint8
    if imagen_binaria.ndim == 3:
        imagen_binaria = cv2.cvtColor(imagen_binaria, cv2.COLOR_BGR2GRAY)
    
    # Forzar binarizaciÃ³n estricta (pÃ­xeles > 0 se vuelven 255)
    _, bin_u8 = cv2.threshold(imagen_binaria, 0, 255, cv2.THRESH_BINARY)
    
    n_etiquetas, etiquetas, stats, centroides = cv2.connectedComponentsWithStats(
        bin_u8,
        connectivity=conectividad,
    )
    
    return n_etiquetas, etiquetas, stats, centroides


def generar_mapa_color_ccl(
    etiquetas: np.ndarray,
    n_etiquetas: int,
    semilla: int = 42,
) -> np.ndarray:
    """
    Genera una imagen en color RGB a partir de la matriz de etiquetas (label map).
    Asigna un color aleatorio brillante a cada componente y negro (0, 0, 0) al fondo.

    ParÃ¡metros
    ----------
    etiquetas : np.ndarray 2D de enteros
        Matriz donde cada pÃ­xel tiene la etiqueta de su componente.
    n_etiquetas : int
        NÃºmero total de etiquetas (incluyendo el fondo 0).
    semilla : int
        Semilla para el generador pseudo-aleatorio.

    Retorna
    -------
    np.ndarray 3D uint8 (H x W x 3)
        Imagen RGB con los componentes coloreados.
    """
    if n_etiquetas <= 1:
        # Solo hay fondo (o vacÃ­o)
        h, w = etiquetas.shape[:2]
        return np.zeros((h, w, 3), dtype=np.uint8)

    # Generar colores RGB en rango [60, 255] para evitar colores muy oscuros
    rng = np.random.default_rng(semilla)
    colores = rng.integers(60, 256, size=(n_etiquetas, 3), dtype=np.uint8)
    
    # El fondo (etiqueta 0) debe ser negro
    colores[0] = [0, 0, 0]

    # Mapear las etiquetas a los colores generados
    return colores[etiquetas]


def calcular_descriptores(stats: np.ndarray, idx: int) -> dict:
    """
    Calcula los descriptores de forma clÃ¡sicos para un componente especÃ­fico.

    ParÃ¡metros
    ----------
    stats : np.ndarray
        Matriz de estadÃ­sticas retornada por cv2.connectedComponentsWithStats.
    idx : int
        Ãndice del componente conexo.

    Retorna
    -------
    dict
        Diccionario con claves: 'area', 'ancho', 'alto', 'elongacion', 'circularidad'.
    """
    area = int(stats[idx, cv2.CC_STAT_AREA])
    ancho = int(stats[idx, cv2.CC_STAT_WIDTH])
    alto = int(stats[idx, cv2.CC_STAT_HEIGHT])
    
    # ElongaciÃ³n: max(ancho, alto) / min(ancho, alto)
    menor = min(ancho, alto)
    elongacion = float(max(ancho, alto) / menor) if menor > 0 else 0.0

    return {
        "area": area,
        "ancho": ancho,
        "alto": alto,
        "elongacion": round(elongacion, 4) if elongacion > 0 else 0.0,
        "circularidad": None,  # Requiere perÃ­metro, no calculado en esta fase
    }


def es_probable_botella(descriptores: dict) -> tuple:
    """
    EvalÃºa si un componente conexo podrÃ­a ser una botella PET segÃºn sus descriptores.
    HeurÃ­sticas clÃ¡sicas basadas en el Ã¡rea y la elongaciÃ³n.

    ParÃ¡metros
    ----------
    descriptores : dict
        Resultado de calcular_descriptores.

    Retorna
    -------
    (es_botella, razon) : (bool, str)
        es_botella : True si cumple las condiciones de botella PET, False de lo contrario.
        razon : Texto explicativo sobre la clasificaciÃ³n.
    """
    area = descriptores["area"]
    elongacion = descriptores["elongacion"]

    # 1. Descartar ruido pequeÃ±o
    if area <= 500:
        return False, f"Ruido o componente pequeÃ±o (Ãrea: {area} pxÂ² <= 500 pxÂ²)"

    # 2. Verificar elongaciÃ³n
    # Las botellas de plÃ¡stico suelen tener una relaciÃ³n de aspecto (elongaciÃ³n) de entre 1.2 y 6.0
    if not (1.2 <= elongacion <= 6.0):
        return False, f"Forma no elongada (ElongaciÃ³n: {elongacion} fuera del rango [1.2, 6.0])"

    return True, f"Cumple con Ã¡rea ({area} pxÂ²) y elongaciÃ³n ({elongacion} en [1.2, 6.0])"


def extraer_componente_por_indice(
    etiquetas: np.ndarray,
    stats: np.ndarray,
    imagen_rgb: np.ndarray,
    idx: int,
) -> tuple:
    """
    Aisla un componente conexo por su etiqueta y lo extrae sobre fondo negro.

    ParÃ¡metros
    ----------
    etiquetas : np.ndarray 2D
        Matriz de etiquetas.
    stats : np.ndarray
        Matriz de estadÃ­sticas.
    imagen_rgb : np.ndarray 3D
        Imagen a color de referencia.
    idx : int
        Ãndice del componente a extraer.

    Retorna
    -------
    (mascara, objeto_color, area_px) : tuple
        mascara : np.ndarray 2D uint8 (0/255)
        objeto_color : np.ndarray 3D uint8 (RGB)
        area_px : int - Ã¡rea del componente
    """
    # Crear mÃ¡scara binaria del componente (0 o 255)
    mascara = (etiquetas == idx).astype(np.uint8) * 255
    
    # Extraer el objeto a color usando la mÃ¡scara de 3 canales
    mascara_3ch = cv2.merge([mascara, mascara, mascara])
    objeto_color = cv2.bitwise_and(imagen_rgb, mascara_3ch)
    
    area_px = int(stats[idx, cv2.CC_STAT_AREA])
    
    return mascara, objeto_color, area_px


def extraer_componente_mayor(
    etiquetas: np.ndarray,
    stats: np.ndarray,
    imagen_rgb: np.ndarray,
) -> tuple:
    """
    Encuentra y extrae el componente conexo con mayor Ã¡rea en la imagen,
    excluyendo el fondo (etiqueta 0).

    ParÃ¡metros
    ----------
    etiquetas : np.ndarray 2D
        Matriz de etiquetas.
    stats : np.ndarray
        Matriz de estadÃ­sticas.
    imagen_rgb : np.ndarray 3D
        Imagen de entrada a color.

    Retorna
    -------
    (mascara, objeto_color, idx_principal, area_px) : tuple
        mascara : np.ndarray 2D uint8 (0/255)
        objeto_color : np.ndarray 3D uint8
        idx_principal : int - Ã­ndice de la etiqueta del componente mayor
        area_px : int - Ã¡rea del componente mayor
    """
    n_etiquetas = stats.shape[0]
    
    # Si solo hay fondo (etiqueta 0), retornar imÃ¡genes vacÃ­as
    if n_etiquetas <= 1:
        h, w = etiquetas.shape[:2]
        mascara = np.zeros((h, w), dtype=np.uint8)
        objeto_color = np.zeros((h, w, 3), dtype=np.uint8)
        return mascara, objeto_color, 0, 0

    # Encontrar la etiqueta con mayor Ã¡rea excluyendo la 0 (fondo)
    # stats[1:, cv2.CC_STAT_AREA] nos da el Ã¡rea de las etiquetas 1 a n-1
    areas_objeto = stats[1:, cv2.CC_STAT_AREA]
    if len(areas_objeto) == 0:
        h, w = etiquetas.shape[:2]
        mascara = np.zeros((h, w), dtype=np.uint8)
        objeto_color = np.zeros((h, w, 3), dtype=np.uint8)
        return mascara, objeto_color, 0, 0
        
    idx_principal = int(np.argmax(areas_objeto) + 1)
    
    mascara, objeto_color, area_px = extraer_componente_por_indice(
        etiquetas, stats, imagen_rgb, idx_principal
    )
    
    return mascara, objeto_color, idx_principal, area_px


def dibujar_contorno(
    imagen_rgb: np.ndarray,
    mascara: np.ndarray,
    color: tuple = (0, 255, 0),
    grosor: int = 2,
) -> np.ndarray:
    """
    Dibuja el contorno exterior de una mÃ¡scara binaria sobre la imagen a color.

    ParÃ¡metros
    ----------
    imagen_rgb : np.ndarray 3D
        Imagen a color de base (RGB).
    mascara : np.ndarray 2D uint8
        MÃ¡scara binaria del objeto (0/255).
    color : tuple
        Color del contorno en formato RGB (R, G, B).
    grosor : int
        Grosor de la lÃ­nea del contorno en pÃ­xeles.

    Retorna
    -------
    np.ndarray 3D
        Copia de la imagen original con el contorno dibujado.
    """
    copia = imagen_rgb.copy()
    
    # Encontrar contornos externos
    # cv2.RETR_EXTERNAL: solo contornos exteriores extremos
    # cv2.CHAIN_APPROX_SIMPLE: comprime segmentos horizontales, verticales y diagonales
    contornos, _ = cv2.findContours(
        mascara,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )
    
    # Dibujar contornos
    # -1 dibuja todos los contornos encontrados
    cv2.drawContours(copia, contornos, -1, color, grosor)
    
    return copia


# =============================================================================
# FASE 7 â€” AnÃ¡lisis de SimetrÃ­a Bilateral
# Confirma matemÃ¡ticamente si el objeto segmentado es una botella PET
# mediante el Ã­ndice de simetrÃ­a IoU (IntersecciÃ³n sobre UniÃ³n).
# =============================================================================

def calcular_eje_simetria(mascara_binaria: np.ndarray) -> tuple:
    """
    Calcula el centroide y el eje principal de simetrÃ­a de una mÃ¡scara binaria
    usando momentos invariantes de imagen.

    Pasos:
    1. cv2.moments(mascara) â†’ M00, M10, M01 para el centroide.
    2. Momentos centrales Î¼20, Î¼02, Î¼11 â†’ orientaciÃ³n del eje principal.
    3. theta = 0.5 Â· arctan2(2Â·Î¼11, Î¼20 âˆ’ Î¼02) en grados.

    ParÃ¡metros
    ----------
    mascara_binaria : np.ndarray 2D uint8
        MÃ¡scara binaria (0/255) del componente.

    Retorna
    -------
    (cx, cy, theta) : tuple
        cx    : float â€” coordenada X del centroide.
        cy    : float â€” coordenada Y del centroide.
        theta : float â€” Ã¡ngulo del eje principal en grados (âˆ’90Â° a +90Â°).
    """
    # Calcular momentos de la imagen binaria
    momentos = cv2.moments(mascara_binaria)

    # Evitar divisiÃ³n por cero si la mÃ¡scara estÃ¡ vacÃ­a
    M00 = momentos["m00"]
    if M00 == 0:
        h, w = mascara_binaria.shape[:2]
        return float(w / 2), float(h / 2), 0.0

    # Centroide geomÃ©trico
    cx = momentos["m10"] / M00
    cy = momentos["m01"] / M00

    # Momentos centrales normalizados (varianza e covarianza espacial)
    mu20 = momentos["m20"] / M00 - cx ** 2   # varianza en X
    mu02 = momentos["m02"] / M00 - cy ** 2   # varianza en Y
    mu11 = momentos["m11"] / M00 - cx * cy   # covarianza XY

    # Ãngulo del eje principal (fÃ³rmula de la elipse equivalente)
    theta = 0.5 * np.degrees(np.arctan2(2.0 * mu11, mu20 - mu02))

    return float(cx), float(cy), float(theta)


def calcular_indice_simetria(
    mascara_binaria: np.ndarray,
    eje: str = "vertical",
) -> float:
    """
    Calcula el Ã­ndice de simetrÃ­a IoU (IntersecciÃ³n / UniÃ³n) de la mÃ¡scara
    respecto a un eje de reflexiÃ³n.

    Procedimiento:
    1. Voltear la mÃ¡scara respecto al eje indicado.
    2. Calcular la intersecciÃ³n binaria (AND) y la uniÃ³n binaria (OR).
    3. Ãndice = |IntersecciÃ³n| / |UniÃ³n|   (Jaccard / IoU de simetrÃ­a).

    InterpretaciÃ³n orientativa para botellas PET:
      > 0.85 â†’ simetrÃ­a alta (muy probable botella cilÃ­ndrica)
      0.70â€“0.85 â†’ simetrÃ­a media (botella deformada o parcialmente oculta)
      < 0.70  â†’ simetrÃ­a baja (probablemente no es botella)

    ParÃ¡metros
    ----------
    mascara_binaria : np.ndarray 2D uint8
        MÃ¡scara binaria (0/255).
    eje : str
        "vertical"   â†’ reflexiÃ³n respecto al eje vertical (flip horizontal, code=1).
        "horizontal" â†’ reflexiÃ³n respecto al eje horizontal (flip vertical, code=0).

    Retorna
    -------
    float en [0, 1] â€” Ã­ndice de simetrÃ­a IoU.
    """
    # Seleccionar el cÃ³digo de flip segÃºn el eje
    # cv2.flip(src, flipCode):
    #   flipCode=1  â†’ espejo horizontal (refleja sobre el eje vertical central)
    #   flipCode=0  â†’ espejo vertical   (refleja sobre el eje horizontal central)
    flip_code = 1 if eje == "vertical" else 0
    mascara_volteada = cv2.flip(mascara_binaria, flip_code)

    # IntersecciÃ³n (pÃ­xeles que coinciden en ambas mitades)
    interseccion = cv2.bitwise_and(mascara_binaria, mascara_volteada)
    # UniÃ³n (pÃ­xeles presentes en cualquiera de las dos)
    union = cv2.bitwise_or(mascara_binaria, mascara_volteada)

    n_inter = float(np.sum(interseccion > 0))
    n_union = float(np.sum(union > 0))

    # Evitar divisiÃ³n por cero si la mÃ¡scara estÃ¡ vacÃ­a
    if n_union == 0:
        return 0.0

    return n_inter / n_union


def normalizar_mascara_simetria(
    mascara_binaria: np.ndarray,
    cx: float,
    cy: float,
    theta: float,
) -> np.ndarray:
    """
    Normaliza la mascara binaria centrando su centroide en el centro del lienzo
    y rotandola por -theta para alinear su eje principal de simetria
    horizontalmente (0 grados).
    """
    h, w = mascara_binaria.shape[:2]

    # 1. Centrar el objeto en el lienzo (w/2, h/2)
    dx = w / 2.0 - cx
    dy = h / 2.0 - cy
    M_trans = np.float32([[1, 0, dx], [0, 1, dy]])
    mascara_centrada = cv2.warpAffine(
        mascara_binaria,
        M_trans,
        (w, h),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )

    # 2. Rotar por -theta alrededor del centro de la imagen para alinear horizontalmente
    centro_x, centro_y = w / 2.0, h / 2.0
    M_rot = cv2.getRotationMatrix2D((centro_x, centro_y), -theta, 1.0)
    mascara_normalizada = cv2.warpAffine(
        mascara_centrada,
        M_rot,
        (w, h),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )

    return mascara_normalizada


def calcular_ambos_ejes_simetria(mascara_binaria: np.ndarray) -> tuple:
    """
    Calcula los indices de simetria IoU para el eje vertical y horizontal,
    normalizando primero la mascara (centrado y alineacion horizontal por momentos).

    Tras la normalizacion (centrar + rotar por -theta), el eje longitudinal
    de la botella queda alineado horizontalmente. Por ello la simetria de
    mayor valor puede ser la horizontal (flip vertical) o la vertical (flip
    horizontal) segun la forma exacta del objeto.

    Se retornan ambos valores para que la clasificacion use
    max(simetria_v, simetria_h) como el indice principal de simetria.

    Parametros
    ----------
    mascara_binaria : np.ndarray 2D uint8

    Retorna
    -------
    (simetria_v, simetria_h) : tuple de float
        simetria_v : indice IoU respecto al eje vertical (reflexion horizontal).
        simetria_h : indice IoU respecto al eje horizontal (reflexion vertical).
    """
    cx, cy, theta = calcular_eje_simetria(mascara_binaria)
    mascara_norm = normalizar_mascara_simetria(mascara_binaria, cx, cy, theta)

    simetria_v = calcular_indice_simetria(mascara_norm, eje="vertical")
    simetria_h = calcular_indice_simetria(mascara_norm, eje="horizontal")
    return simetria_v, simetria_h


def visualizar_simetria(
    mascara_binaria: np.ndarray,
    cx: float,
    cy: float,
    theta: float,
) -> np.ndarray:
    """
    Genera una imagen de diagnÃ³stico RGB que muestra la mÃ¡scara binaria
    con el centroide y el eje de simetrÃ­a superpuestos.

    Elementos dibujados:
    Â· MÃ¡scara en blanco sobre fondo negro.
    Â· Centroide: cÃ­rculo verde (radio = 6 px, grosor = âˆ’1 para relleno).
    Â· Eje de simetrÃ­a: lÃ­nea roja que cruza toda la imagen pasando por
      (cx, cy) con la orientaciÃ³n estimada por el Ã¡ngulo theta.

    ParÃ¡metros
    ----------
    mascara_binaria : np.ndarray 2D uint8
        MÃ¡scara binaria (0/255).
    cx, cy : float
        Coordenadas del centroide.
    theta : float
        Ãngulo del eje principal en grados.

    Retorna
    -------
    np.ndarray 3D uint8 (H Ã— W Ã— 3) en formato RGB.
    """
    h, w = mascara_binaria.shape[:2]

    # Crear fondo negro y pintar la mÃ¡scara en gris claro
    canvas = np.zeros((h, w, 3), dtype=np.uint8)
    canvas[mascara_binaria > 0] = [200, 200, 200]  # gris claro para la silueta

    # â”€â”€ Eje de simetrÃ­a (lÃ­nea roja) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    # Convertir Ã¡ngulo a radianes y calcular el vector director
    rad = np.radians(theta)
    # Longitud suficiente para cruzar toda la imagen
    longitud = int(max(h, w) * 1.5)
    cos_t = np.cos(rad)
    sin_t = np.sin(rad)

    # Dos puntos extremos de la lÃ­nea centrada en (cx, cy)
    x1 = int(cx - longitud * cos_t)
    y1 = int(cy - longitud * sin_t)
    x2 = int(cx + longitud * cos_t)
    y2 = int(cy + longitud * sin_t)

    cv2.line(canvas, (x1, y1), (x2, y2), (220, 30, 30), 2)  # rojo

    # â”€â”€ Centroide (cÃ­rculo verde relleno) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    cv2.circle(canvas, (int(cx), int(cy)), 6, (0, 220, 80), -1)  # verde
    # Anillo exterior blanco para mejorar visibilidad
    cv2.circle(canvas, (int(cx), int(cy)), 8, (255, 255, 255), 1)

    return canvas


def clasificar_por_simetria(
    simetria_v: float,
    simetria_h: float,
    area_px: int,
    elongacion: float,
) -> tuple:
    """
    Clasifica el objeto segmentado combinando el Ã­ndice de simetrÃ­a
    con el Ã¡rea y la elongaciÃ³n calculadas en la Fase 6.

    Reglas (en orden de prioridad):
    1. simetria_v > 0.80 AND area_px > 1000 AND elongacion > 1.3
       â†’ "âœ… Probable Botella PET" (success)
    2. simetria_v > 0.65
       â†’ "âš ï¸ Posible Botella (verificar)" (warning)
    3. En cualquier otro caso
       â†’ "âŒ No parece una botella" (error)

    ParÃ¡metros
    ----------
    simetria_v  : float â€” Ã­ndice de simetrÃ­a vertical (IoU).
    simetria_h  : float â€” Ã­ndice de simetrÃ­a horizontal (IoU).
    area_px     : int   â€” Ã¡rea del componente en pÃ­xeles.
    elongacion  : float â€” razÃ³n max(ancho,alto)/min(ancho,alto).

    Retorna
    -------
    (etiqueta, tipo) : (str, str)
        etiqueta : texto descriptivo con emoji.
        tipo     : "success" | "warning" | "error" (para st.success/warning/error).
    """
    # El eje de mayor simetria es el relevante: la botella puede estar
    # en cualquier orientacion antes de la normalizacion por momentos.
    simetria_principal = max(simetria_v, simetria_h)

    if simetria_principal > 0.70 and area_px > 1000 and elongacion > 1.3:
        return '✅ Probable Botella PET', 'success'
    elif simetria_principal > 0.55:
        return '⚠️ Posible Botella (verificar)', 'warning'
    else:
        return '❌ No parece una botella', 'error'


# =============================================================================
# ANÁLISIS DE HISTOGRAMA Y CLASIFICACIÓN DE ESCENA
# =============================================================================

def analizar_histograma(imagen_rgb, imagen_gris):
    """Analiza los histogramas en grises y RGB de la imagen.

    Calcula métricas estadísticas del histograma en escala de grises
    (media, desviación estándar, entropía, picos, zonas de distribución)
    y métricas de color por canal RGB (medias, varianzas, saturación
    cromática, tipo de agua dominante y balance de blancos aproximado).
    Estos valores alimentan la función clasificar_escena() para determinar
    el pipeline de segmentación más adecuado.

    Parameters
    ----------
    imagen_rgb : np.ndarray
        Imagen en espacio de color RGB con forma (H, W, 3) y dtype uint8.
    imagen_gris : np.ndarray
        Imagen en escala de grises con forma (H, W) y dtype uint8.

    Returns
    -------
    dict
        Diccionario con todas las métricas calculadas de ambos bloques.
    """
    # ── Bloque A: métricas del histograma en GRISES ──────────────────────────

    media      = float(np.mean(imagen_gris))
    std        = float(np.std(imagen_gris))
    mediana_px = float(np.median(imagen_gris))
    min_px     = int(imagen_gris.min())
    max_px     = int(imagen_gris.max())
    rango_din  = max_px - min_px

    # Asimetría de Fisher: (media - mediana) / std
    # Positiva → cola hacia valores altos (brillos dominantes)
    # Negativa → cola hacia valores bajos (sombras dominantes)
    asimetria = (media - mediana_px) / std if std > 0 else 0.0

    # Curtosis en exceso: mide el «apuntamiento» del histograma
    img_f    = imagen_gris.astype(float)
    curtosis = (float(np.mean((img_f - media) ** 4)) / std ** 4 - 3) if std > 0 else 0.0

    # Entropía de Shannon sobre histograma normalizado de 256 bins
    # Rango teórico [0, 8]. Alta → imagen compleja con muchos tonos.
    hist, _ = np.histogram(imagen_gris.flatten(), bins=256, range=(0, 256))
    prob     = hist / hist.sum()
    entropia = float(-np.sum(prob * np.log2(prob + 1e-12)))

    # Detección de picos en el histograma suavizado con gaussiana
    hist_suave   = gaussian_filter1d(hist.astype(float), sigma=4)
    picos, props = find_peaks(
        hist_suave,
        height=hist_suave.max() * 0.05,
        distance=20,
    )
    n_picos = len(picos)

    if n_picos > 0:
        # Ordenar por altura descendente para identificar picos más relevantes
        orden          = np.argsort(props["peak_heights"])[::-1]
        pico_dominante = int(picos[orden[0]])
        segundo_pico   = int(picos[orden[1]]) if n_picos >= 2 else None
    else:
        pico_dominante = int(media)
        segundo_pico   = None

    separacion_picos = abs(pico_dominante - segundo_pico) if segundo_pico is not None else 0.0

    # Distribución por zonas de intensidad
    total       = imagen_gris.size
    zona_oscura = float((imagen_gris < 85).sum() / total)
    zona_media  = float(((imagen_gris >= 85) & (imagen_gris <= 170)).sum() / total)
    zona_clara  = float((imagen_gris > 170).sum() / total)

    # ── Bloque B: métricas del histograma RGB ────────────────────────────────

    # Medias por canal
    media_r = float(np.mean(imagen_rgb[:, :, 0]))
    media_g = float(np.mean(imagen_rgb[:, :, 1]))
    media_b = float(np.mean(imagen_rgb[:, :, 2]))

    # Varianzas por canal: cuál canal lleva más información
    var_r           = float(np.var(imagen_rgb[:, :, 0]))
    var_g           = float(np.var(imagen_rgb[:, :, 1]))
    var_b           = float(np.var(imagen_rgb[:, :, 2]))
    canal_dominante = ["R", "G", "B"][int(np.argmax([var_r, var_g, var_b]))]

    # Diferencias entre medias de canales
    diff_gr = media_g - media_r   # positivo → predomina el verde
    diff_gb = media_g - media_b   # positivo → más verde que azul
    diff_rb = media_r - media_b   # positivo → más rojo que azul

    # Saturación cromática media aproximada sin conversión HSV completa
    # < 0.10  → escena sin color (gris, blanco, negro)
    # 0.10-0.30 → colores desaturados (agua turbia, plástico blanco)
    # > 0.30  → colores vivos (agua turquesa, botella de color)
    max_c            = imagen_rgb.max(axis=2).astype(np.float32)
    min_c            = imagen_rgb.min(axis=2).astype(np.float32)
    # np.divide con where evita el RuntimeWarning por division entre cero:
    # np.where evalua ambas ramas antes de seleccionar, pero np.divide
    # solo opera donde la condicion es True.
    sat_map          = np.divide(
        max_c - min_c, max_c,
        out=np.zeros_like(max_c),
        where=max_c > 0,
    )
    saturacion_media = float(sat_map.mean())

    # Clasificación del tipo de fondo hídrico por color dominante
    if diff_gb > 15 and diff_gr > 10:
        tipo_agua = "verde"         # río con vegetación, algas
    elif diff_gb < -15 and diff_rb < -10:
        tipo_agua = "azul"          # agua clara, piscina, océano
    elif saturacion_media < 0.12:
        tipo_agua = "turbia_cafe"   # agua sucia, río fangoso
    else:
        tipo_agua = "indefinido"    # iluminación mixta o imagen compleja

    # Balance de blancos aproximado: dispersión entre medias de los 3 canales
    # < 8  → imagen casi gris (poco color) → botella blanca probable
    # > 20 → imagen con color marcado → botella de color o fondo colorido
    balance_rgb = float(np.std([media_r, media_g, media_b]))

    return {
        # ── Grises ──
        "media":            media,
        "std":              std,
        "mediana_px":       mediana_px,
        "min_px":           min_px,
        "max_px":           max_px,
        "rango_din":        rango_din,
        "asimetria":        asimetria,
        "curtosis":         curtosis,
        "entropia":         entropia,
        "hist_suave":       hist_suave,
        "picos":            picos,
        "n_picos":          n_picos,
        "pico_dominante":   pico_dominante,
        "segundo_pico":     segundo_pico,
        "separacion_picos": separacion_picos,
        "zona_oscura":      zona_oscura,
        "zona_media":       zona_media,
        "zona_clara":       zona_clara,
        # ── RGB ──
        "media_r":          media_r,
        "media_g":          media_g,
        "media_b":          media_b,
        "var_r":            var_r,
        "var_g":            var_g,
        "var_b":            var_b,
        "canal_dominante":  canal_dominante,
        "diff_gr":          diff_gr,
        "diff_gb":          diff_gb,
        "diff_rb":          diff_rb,
        "saturacion_media": saturacion_media,
        "tipo_agua":        tipo_agua,
        "balance_rgb":      balance_rgb,
    }


def clasificar_escena(metricas):
    """Clasifica el tipo de escena fotográfica a partir del dict de métricas.

    Aplica un conjunto de reglas ordenadas sobre las métricas devueltas
    por analizar_histograma() para determinar la naturaleza de la escena
    (contraste, color de fondo, distribución tonal) y sugerir el pipeline
    de segmentación más adecuado.

    Las reglas se evalúan en orden estricto: la primera que se cumpla
    determina el resultado final (no se evalúan las siguientes).

    Parameters
    ----------
    metricas : dict
        Diccionario devuelto por analizar_histograma().

    Returns
    -------
    tuple[str, str, str]
        (tipo_escena, descripcion, emoji)
        tipo_escena : identificador de cadena del tipo de escena.
        descripcion : texto explicativo con pipeline recomendado.
        emoji       : emoji representativo de la escena.
    """
    rango_din      = metricas["rango_din"]
    std            = metricas["std"]
    tipo_agua      = metricas["tipo_agua"]
    zona_oscura    = metricas["zona_oscura"]
    zona_clara     = metricas["zona_clara"]
    pico_dominante = metricas["pico_dominante"]

    # REGLA 1 — BAJO CONTRASTE
    # Histograma muy concentrado: expansión de contraste imprescindible.
    if rango_din < 80 or std < 22:
        return (
            "bajo_contraste",
            (
                f"Contraste muy bajo (σ={metricas['std']:.1f}, "
                f"rango={metricas['rango_din']} niveles). "
                "El histograma está concentrado. "
                "Se necesita expansión de contraste antes de segmentar."
            ),
            "⚠️",
        )

    # REGLA 2 — BOTELLA CLARA EN AGUA VERDE
    # Fondo con dominante verde y objeto claro destacado.
    if tipo_agua == "verde" and zona_oscura > 0.35 and zona_clara > 0.08:
        return (
            "botella_clara_agua_verde",
            (
                f"Agua con tonalidad verde (diff_gr={metricas['diff_gr']:.1f}). "
                "Botella clara detectada sobre fondo oscuro verdoso. "
                "Pipeline: Bilateral → Gamma alto → Otsu."
            ),
            "🌿",
        )

    # REGLA 3 — BOTELLA EN AGUA AZUL CLARA
    # Fondo azul predominantemente claro.
    if tipo_agua == "azul" and zona_clara > 0.40:
        return (
            "botella_agua_azul_clara",
            (
                f"Agua de tono azul claro (diff_gb={metricas['diff_gb']:.1f}). "
                "Fondo predominantemente claro. "
                "Pipeline: Gaussiano → Ecualización → Kapur invertido."
            ),
            "🌊",
        )

    # REGLA 4 — BOTELLA EN AGUA TURBIA O CAFÉ
    # Imagen casi acromática por baja saturación cromática.
    if tipo_agua == "turbia_cafe":
        return (
            "botella_agua_turbia",
            (
                "Agua turbia o contaminada (baja saturación cromática). "
                "Imagen casi acromática. "
                "Pipeline: Mediana → Expansión de contraste → Media."
            ),
            "🟫",
        )

    # REGLA 5 — BOTELLA CLARA SOBRE FONDO OSCURO (genérico)
    # Fondo mayoritariamente oscuro con objeto claro destacado.
    if zona_oscura > 0.45 and zona_clara > 0.10 and pico_dominante < 110:
        return (
            "botella_clara_fondo_oscuro",
            (
                f"Fondo oscuro dominante ({metricas['zona_oscura']:.0%} "
                "de píxeles bajo 85). Objeto claro destacado. "
                "Pipeline: Mediana → Bilateral → Gamma → Otsu."
            ),
            "🔦",
        )

    # REGLA 6 — BOTELLA OSCURA SOBRE FONDO CLARO (genérico)
    # Fondo mayoritariamente claro con objeto oscuro.
    if zona_clara > 0.45 and zona_oscura > 0.08 and pico_dominante > 135:
        return (
            "botella_oscura_fondo_claro",
            (
                f"Fondo claro dominante ({metricas['zona_clara']:.0%} "
                "de píxeles sobre 170). Objeto oscuro. "
                "Pipeline: Gaussiano → Ecualización Uniforme → Kapur invertido."
            ),
            "☀️",
        )

    # REGLA 7 — CONTRASTE MODERADO (caso por defecto)
    # Distribución equilibrada sin patrón dominante claro.
    return (
        "contraste_moderado",
        (
            f"Distribución equilibrada (σ={metricas['std']:.1f}, "
            f"separación de picos={metricas['separacion_picos']:.0f}). "
            "Se aplica enhancement previo a la segmentación."
        ),
        "📊",
    )


# =============================================================================
# SUGERENCIA Y EJECUCIÓN AUTOMÁTICA DE PIPELINE
# =============================================================================

def sugerir_pipeline(tipo_escena, metricas):
    """Devuelve un pipeline de procesamiento adaptado al tipo de escena.

    Construye un diccionario con filtros, parámetros FFT, mejoras de
    contraste y método de umbralización optimizados para cada tipo de
    escena detectado por clasificar_escena().

    Los dicts de filtros son compatibles con aplicar_filtro().
    Los dicts de mejoras son compatibles con aplicar_mejora().

    Parameters
    ----------
    tipo_escena : str
        Identificador devuelto por clasificar_escena(), por ejemplo
        "botella_clara_agua_verde", "bajo_contraste", etc.
    metricas : dict
        Diccionario de métricas devuelto por analizar_histograma().

    Returns
    -------
    dict con claves:
        "filtros"  : list[dict]  — configuraciones para aplicar_filtro()
        "fft"      : dict        — {"activo", "tipo", "cutoff"}
        "mejoras"  : list[dict]  — configuraciones para aplicar_mejora()
        "umbral"   : dict        — {"metodo", "invertir", "params"}
        "razon"    : str         — explicación del pipeline elegido
    """

    # ── BOTELLA CLARA EN AGUA VERDE ──────────────────────────────────────────
    # Mediana elimina brillos de hojas/algas; Bilateral preserva borde plástico.
    # FFT lowpass elimina textura periódica de la vegetación.
    # Gamma > 1 oscurece el fondo verde y separa el pico de la botella clara.
    if tipo_escena == "botella_clara_agua_verde":
        return {
            "filtros": [
                {"tipo": "Mediana", "ksize": 5},
                {"tipo": "Bilateral", "d": 9,
                 "sigma_color": 65.0, "sigma_space": 65.0},
            ],
            "fft": {"activo": True, "tipo": "lowpass", "cutoff": 0.20},
            "mejoras": [
                {"tipo": "Corrección Gamma", "gamma": 2.0},
            ],
            "umbral": {"metodo": "Otsu", "invertir": False, "params": {}},
            "razon": (
                f"Agua verde detectada (diff_gr={metricas['diff_gr']:.1f}). "
                f"Canal G dominante. σ={metricas['std']:.1f}. "
                "Mediana elimina ruido orgánico, Bilateral preserva borde "
                "plástico, Gamma γ=2.0 separa botella clara del fondo verdoso."
            ),
        }

    # ── BOTELLA EN AGUA AZUL CLARA ───────────────────────────────────────────
    # Fondo uniforme: gaussiano homogeniza, mediana elimina reflejos.
    # FFT inactiva (fondo sin textura periódica).
    # HE aplana histograma sesgado; gamma < 1 aclara imagen.
    # Kapur + invertir porque la botella es el objeto más oscuro.
    if tipo_escena == "botella_agua_azul_clara":
        return {
            "filtros": [
                {"tipo": "Gaussiano", "ksize": 5, "sigma": 1.5},
                {"tipo": "Mediana", "ksize": 3},
            ],
            "fft": {"activo": False, "tipo": "lowpass", "cutoff": 0.15},
            "mejoras": [
                {"tipo": "Ecualización Uniforme"},
                {"tipo": "Corrección Gamma", "gamma": 0.65},
            ],
            "umbral": {"metodo": "Kapur", "invertir": True, "params": {}},
            "razon": (
                f"Agua azul clara detectada (diff_gb={metricas['diff_gb']:.1f}). "
                f"zona_clara={metricas['zona_clara']:.0%}. "
                "Inversión de máscara necesaria: botella más oscura que el agua."
            ),
        }

    # ── BOTELLA EN AGUA TURBIA ───────────────────────────────────────────────
    # Mucho ruido de sedimentos: mediana ksize 7 los elimina.
    # Bilateral con sigma alto acepta mayor rango (imagen acromática).
    # FFT highpass muy suave resalta los pocos bordes de la botella.
    # Expansión centrada en pico dominante + log hiperbólica.
    if tipo_escena == "botella_agua_turbia":
        invertir_turbia = metricas["media"] > 127
        return {
            "filtros": [
                {"tipo": "Mediana", "ksize": 7},
                {"tipo": "Bilateral", "d": 7,
                 "sigma_color": 90.0, "sigma_space": 90.0},
            ],
            "fft": {"activo": True, "tipo": "highpass", "cutoff": 0.06},
            "mejoras": [
                {
                    "tipo": "Contracción / Expansión",
                    "a_in":  max(0,   int(metricas["pico_dominante"]) - 35),
                    "b_in":  min(255, int(metricas["pico_dominante"]) + 35),
                    "a_out": 0,
                    "b_out": 255,
                },
                {"tipo": "Ecualización Log. Hiperbólica"},
            ],
            "umbral": {
                "metodo":   "Media",
                "invertir": invertir_turbia,
                "params":   {},
            },
            "razon": (
                f"Agua turbia detectada (sat={metricas['saturacion_media']:.2f}). "
                "Imagen casi acromática. Expansión de contraste + log hiperbólica "
                "para maximizar la separación residual entre botella y agua."
            ),
        }

    # ── BOTELLA CLARA SOBRE FONDO OSCURO (genérico) ──────────────────────────
    # Mediana + Bilateral para fondo ruidoso.
    # FFT lowpass elimina textura de fondo.
    # Gamma γ=1.8 oscurece el fondo residual; Otsu binariza la zona clara.
    if tipo_escena == "botella_clara_fondo_oscuro":
        return {
            "filtros": [
                {"tipo": "Mediana", "ksize": 5},
                {"tipo": "Bilateral", "d": 9,
                 "sigma_color": 60.0, "sigma_space": 60.0},
            ],
            "fft": {"activo": True, "tipo": "lowpass", "cutoff": 0.25},
            "mejoras": [
                {"tipo": "Corrección Gamma", "gamma": 1.8},
            ],
            "umbral": {"metodo": "Otsu", "invertir": False, "params": {}},
            "razon": (
                f"Fondo oscuro genérico (zona_oscura={metricas['zona_oscura']:.0%}). "
                f"pico_dominante={metricas['pico_dominante']}. "
                "Gamma γ=1.8 oscurece el fondo residual. "
                "Otsu binariza la zona clara."
            ),
        }

    # ── BOTELLA OSCURA SOBRE FONDO CLARO (genérico) ──────────────────────────
    # Suavizado leve (gaussiano + mediana ksize 3).
    # FFT inactiva: fondo uniforme sin textura.
    # HE + gamma < 1: aclara imagen para destacar el objeto oscuro.
    # Kapur + invertir porque la botella es más oscura que el fondo.
    if tipo_escena == "botella_oscura_fondo_claro":
        return {
            "filtros": [
                {"tipo": "Gaussiano", "ksize": 3, "sigma": 0.0},
                {"tipo": "Mediana", "ksize": 3},
            ],
            "fft": {"activo": False, "tipo": "lowpass", "cutoff": 0.15},
            "mejoras": [
                {"tipo": "Ecualización Uniforme"},
                {"tipo": "Corrección Gamma", "gamma": 0.7},
            ],
            "umbral": {"metodo": "Kapur", "invertir": True, "params": {}},
            "razon": (
                f"Fondo claro genérico (zona_clara={metricas['zona_clara']:.0%}). "
                "Invertir máscara: botella más oscura que el agua."
            ),
        }

    # ── BAJO CONTRASTE ───────────────────────────────────────────────────────
    # Bilateral preserva los pocos bordes existentes.
    # FFT highpass muy suave realza bordes residuales.
    # Expansión al rango dinámico real + HE maximizan separación.
    if tipo_escena == "bajo_contraste":
        return {
            "filtros": [
                {"tipo": "Bilateral", "d": 7,
                 "sigma_color": 85.0, "sigma_space": 85.0},
                {"tipo": "Mediana", "ksize": 3},
            ],
            "fft": {"activo": True, "tipo": "highpass", "cutoff": 0.04},
            "mejoras": [
                {
                    "tipo": "Contracción / Expansión",
                    "a_in":  max(0,   int(metricas["min_px"]) + 5),
                    "b_in":  min(255, int(metricas["max_px"]) - 5),
                    "a_out": 0,
                    "b_out": 255,
                },
                {"tipo": "Ecualización Uniforme"},
            ],
            "umbral": {"metodo": "Media", "invertir": False, "params": {}},
            "razon": (
                f"Bajo contraste (σ={metricas['std']:.1f}, "
                f"rango={metricas['rango_din']} niveles). "
                "Expansión al rango dinámico real + HE para maximizar separación."
            ),
        }

    # ── CONTRASTE MODERADO (caso por defecto) ────────────────────────────────
    # Pipeline construido dinámicamente según las métricas disponibles.
    separacion_picos = metricas["separacion_picos"]
    std_val          = metricas["std"]

    # Filtros: bilateral + mediana para imágenes con poco contraste;
    # mediana + gaussiano leve para imágenes con contraste suficiente.
    if std_val < 40:
        filtros_mod = [
            {"tipo": "Bilateral", "d": 9,
             "sigma_color": 75.0, "sigma_space": 75.0},
            {"tipo": "Mediana", "ksize": 5},
        ]
    else:
        filtros_mod = [
            {"tipo": "Mediana", "ksize": 3},
            {"tipo": "Gaussiano", "ksize": 3, "sigma": 0.0},
        ]

    # Mejoras: si los picos están bien separados basta con un gamma suave;
    # si están comprimidos, primero HE para separarlos, luego gamma.
    if separacion_picos > 45:
        mejoras_mod = [{"tipo": "Corrección Gamma", "gamma": 1.4}]
    else:
        mejoras_mod = [
            {"tipo": "Ecualización Uniforme"},
            {"tipo": "Corrección Gamma", "gamma": 1.3},
        ]

    # Umbral e inversión adaptados a la distribución de zonas tonal.
    invertir_val = metricas["zona_clara"] > metricas["zona_oscura"]
    metodo_mod   = "Otsu" if separacion_picos > 55 else "Kapur"

    return {
        "filtros": filtros_mod,
        "fft":     {"activo": False, "tipo": "lowpass", "cutoff": 0.15},
        "mejoras": mejoras_mod,
        "umbral":  {
            "metodo":   metodo_mod,
            "invertir": invertir_val,
            "params":   {},
        },
        "razon": (
            f"Contraste moderado (σ={metricas['std']:.1f}, "
            f"sep_picos={metricas['separacion_picos']:.0f}, "
            f"tipo_agua={metricas['tipo_agua']}). "
            f"Pipeline adaptado. Invertir={invertir_val}."
        ),
    }


def ejecutar_pipeline_sugerido(imagen_gris, imagen_rgb, pipeline_dict):
    """Aplica el pipeline completo devuelto por sugerir_pipeline() en orden.

    Ejecuta secuencialmente: filtros espaciales → FFT opcional →
    mejoras de contraste → umbralización → cierre morfológico fijo.
    Guarda el resultado de cada etapa para visualización en el dashboard.

    Parameters
    ----------
    imagen_gris : np.ndarray
        Imagen en escala de grises (H, W) uint8. Entrada del pipeline.
    imagen_rgb : np.ndarray
        Imagen original RGB (H, W, 3) uint8. Reservada para contexto
        y posibles extensiones futuras.
    pipeline_dict : dict
        Diccionario devuelto por sugerir_pipeline() con claves
        "filtros", "fft", "mejoras", "umbral" y "razon".

    Returns
    -------
    dict con claves:
        "imgs_filtros"   : list[np.ndarray] — imagen tras cada filtro
        "img_post_fft"   : np.ndarray | None
        "mascara_fft"    : np.ndarray | None
        "espectro_fft"   : np.ndarray | None
        "imgs_mejoras"   : list[np.ndarray] — imagen tras cada mejora
        "img_mejorada"   : np.ndarray — imagen tras todas las mejoras
        "img_binarizada" : np.ndarray — máscara binaria con cierre aplicado
        "umbral_info"    : dict — parámetros usados por aplicar_umbral()
    """
    # ── Paso 1: Filtros espaciales en secuencia ──────────────────────────────
    imgs_filtros = []
    img_actual   = imagen_gris.copy()

    for cfg in pipeline_dict["filtros"]:
        img_actual = aplicar_filtro(img_actual, cfg)
        imgs_filtros.append(img_actual.copy())

    # ── Paso 2: Filtrado en frecuencia (FFT) opcional ────────────────────────
    if pipeline_dict["fft"]["activo"]:
        img_fft, mascara_fft, espectro_fft = fft_filter(
            img_actual,
            cutoff=pipeline_dict["fft"]["cutoff"],
            tipo=pipeline_dict["fft"]["tipo"],
        )
        img_actual   = img_fft
        img_post_fft = img_fft
    else:
        img_post_fft = None
        mascara_fft  = None
        espectro_fft = None

    # ── Paso 3: Mejoras de contraste en secuencia ────────────────────────────
    imgs_mejoras = []

    for cfg in pipeline_dict["mejoras"]:
        img_actual = aplicar_mejora(img_actual, cfg)
        imgs_mejoras.append(img_actual.copy())

    img_mejorada = img_actual.copy()

    # ── Paso 4: Umbralización ────────────────────────────────────────────────
    u = pipeline_dict["umbral"]
    img_bin, umbral_info = aplicar_umbral(
        img_actual,
        metodo=u["metodo"],
        invertir=u["invertir"],
        **u["params"],
    )

    # ── Paso 5: Cierre morfológico fijo para limpiar la máscara ─────────────
    # El elemento estructurante elíptico 7×7 rellena huecos pequeños
    # sin deformar el contorno de la botella.
    kernel        = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    img_binarizada = cv2.morphologyEx(img_bin, cv2.MORPH_CLOSE, kernel)

    return {
        "imgs_filtros":   imgs_filtros,
        "img_post_fft":   img_post_fft,
        "mascara_fft":    mascara_fft,
        "espectro_fft":   espectro_fft,
        "imgs_mejoras":   imgs_mejoras,
        "img_mejorada":   img_mejorada,
        "img_binarizada": img_binarizada,
        "umbral_info":    umbral_info,
    }


# =============================================================================
# ANÁLISIS MULTI-OBJETO
# =============================================================================

def analizar_multiples_objetos(mascara_binaria, conectividad=8, min_area=800):
    """Analiza todos los componentes conexos de la máscara y clasifica cada uno.

    Itera todos los componentes encontrados por CCL (excepto el fondo),
    descarta los que no alcanzan el área mínima, y para cada candidato
    calcula descriptores geométricos, simetría bilateral y clasificación.

    Parameters
    ----------
    mascara_binaria : np.ndarray
        Máscara binaria (H, W) uint8 con valores 0 y 255.
    conectividad : int
        4 u 8. Conectividad para cv2.connectedComponentsWithStats.
    min_area : int
        Área mínima en píxeles para considerar un componente como candidato.

    Returns
    -------
    list[dict]
        Lista de resultados ordenada por simetría principal descendente.
        Cada dict contiene:
            idx           : int — índice del componente en la etiqueta CCL.
            mascara       : np.ndarray — máscara binaria de ese componente.
            area          : int — área en píxeles.
            elongacion    : float — elongación (ancho/alto o alto/ancho ≥ 1).
            bbox          : tuple — (x, y, w, h) bounding box.
            centroide     : tuple — (cx, cy) en píxeles.
            simetria_v    : float — índice de simetría vertical (IoU).
            simetria_h    : float — índice de simetría horizontal (IoU).
            simetria_principal : float — max(simetria_v, simetria_h).
            theta         : float — ángulo del eje principal en grados.
            etiqueta      : str — texto con emoji de la clasificación.
            tipo          : str — "success" | "warning" | "error".
    """
    # Aplicar CCL
    etiquetas, stats, n_componentes = aplicar_ccl(mascara_binaria, conectividad)
    resultados = []

    for idx in range(1, n_componentes):  # 0 = fondo
        area = int(stats[idx, cv2.CC_STAT_AREA])
        if area < min_area:
            continue

        # Extraer máscara del componente
        mascara_comp = extraer_componente_por_indice(mascara_binaria, etiquetas, idx)

        # Descriptores geométricos
        x   = int(stats[idx, cv2.CC_STAT_LEFT])
        y   = int(stats[idx, cv2.CC_STAT_TOP])
        w   = int(stats[idx, cv2.CC_STAT_WIDTH])
        h   = int(stats[idx, cv2.CC_STAT_HEIGHT])
        elong = max(w, h) / max(min(w, h), 1)

        # Simetría bilateral en ambos ejes
        simetria_v, simetria_h, theta, cx, cy = calcular_ambos_ejes_simetria(mascara_comp)
        simetria_principal = max(simetria_v, simetria_h)

        # Clasificación
        etiqueta, tipo = clasificar_por_simetria(simetria_v, simetria_h, area, elong)

        resultados.append({
            "idx":                idx,
            "mascara":            mascara_comp,
            "area":               area,
            "elongacion":         round(elong, 2),
            "bbox":               (x, y, w, h),
            "centroide":          (round(cx), round(cy)),
            "simetria_v":         round(simetria_v, 4),
            "simetria_h":         round(simetria_h, 4),
            "simetria_principal": round(simetria_principal, 4),
            "theta":              round(theta, 1),
            "etiqueta":           etiqueta,
            "tipo":               tipo,
        })

    # Ordenar: primero probable botella, luego por simetría descendente
    resultados.sort(key=lambda r: r["simetria_principal"], reverse=True)
    return resultados


def dibujar_todos_contornos(imagen_rgb, resultados):
    """Dibuja contornos de colores sobre la imagen original por clasificación.

    Superpone el contorno de cada objeto detectado con un color según
    su clasificación:
        Verde   (0, 255, 0) → "success"  — Probable Botella PET
        Naranja (255,165, 0) → "warning" — Posible botella
        Rojo    (255,  0, 0) → "error"   — No parece botella

    El contorno se extrae con operación morfológica MORPH_GRADIENT
    (dilatación - erosión con kernel elíptico 3×3).

    Parameters
    ----------
    imagen_rgb : np.ndarray
        Imagen original RGB (H, W, 3) uint8.
    resultados : list[dict]
        Lista devuelta por analizar_multiples_objetos().

    Returns
    -------
    np.ndarray
        Copia de imagen_rgb con los contornos superpuestos.
    """
    img_out = imagen_rgb.copy()
    kernel  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

    colores = {
        "success": (0,   255, 0),    # verde
        "warning": (255, 165, 0),    # naranja
        "error":   (255, 0,   0),    # rojo
    }

    for r in resultados:
        color   = colores.get(r["tipo"], (128, 128, 128))
        contorno = cv2.morphologyEx(r["mascara"], cv2.MORPH_GRADIENT, kernel)
        img_out[contorno > 0] = color

        # Dibujar etiqueta del número en el centroide
        cx, cy = r["centroide"]
        cv2.putText(
            img_out,
            str(r["idx"]),
            (cx, cy),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2,
            cv2.LINE_AA,
        )

    return img_out
