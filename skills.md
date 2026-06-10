# Skills
## Prompt 1
Actúa como Ingeniero de Visión Computacional experto en Streamlit y OpenCV.
Vamos a construir un Dashboard de Segmentación de Botellas PET en cuerpos de 
agua usando SOLO técnicas clásicas de PDI (sin machine learning).

ARQUITECTURA OBLIGATORIA: dos archivos separados.
- procesamiento.py → toda la lógica OpenCV/NumPy, sin imports de Streamlit
- app.py → toda la UI de Streamlit, importa funciones de procesamiento.py

REGLAS ESTRICTAS DE SINTAXIS STREAMLIT:
- Prohibido usar_container_width=True → usar SIEMPRE width="stretch"
- TODOS los elementos interactivos (st.slider, st.selectbox, st.radio, 
  st.checkbox) y gráficos (st.plotly_chart) deben tener key único
- Las imágenes (st.image) NO llevan key

Requerimientos procesamiento.py — Fase 1 y 2:
1. Constante MAX_ANCHO_PX = 800
2. redimensionar_imagen(imagen_bgr): redimensiona si ancho > 800px manteniendo 
   aspecto con INTER_AREA
3. bgr_a_rgb(imagen_bgr): convierte BGR a RGB
4. convertir_a_gris(imagen_bgr): convierte a escala de grises con ponderación 
   perceptual
5. calcular_histograma(imagen): devuelve figura Plotly. Si imagen es 2D devuelve 
   histograma gris. Si es 3D RGB devuelve tres canales (R rojo, G verde, B azul) 
   superpuestos. Usar template="plotly_dark", paper_bgcolor transparente, height=260

Requerimientos procesamiento.py — Fase 3 (Filtros espaciales acumulativos):
Implementa estas funciones, todas reciben y devuelven np.ndarray uint8 2D, 
todas usan BORDER_REFLECT_101:
- aplicar_gaussiano(imagen_gris, ksize=3, sigma=0.0)
- aplicar_mediana(imagen_gris, ksize=3)
- aplicar_bilateral(imagen_gris, d=9, sigma_color=75.0, sigma_space=75.0)
- aplicar_paso_bajas(imagen_gris, ksize=3): box filter uniforme
- aplicar_promediador(imagen_gris, ksize=3): kernel ones(k,k)/k² explícito
- aplicar_max(imagen_gris, ksize=3): scipy.ndimage.maximum_filter, mode="reflect"
- aplicar_min(imagen_gris, ksize=3): scipy.ndimage.minimum_filter, mode="reflect"
- aplicar_filtro(imagen_gris, config: dict): dispatcher que lee config["tipo"] 
  y llama la función correspondiente con los kwargs del dict

Requerimientos app.py:
1. st.set_page_config con layout="wide", icon="🍶"
2. Título principal: "🍶 Dashboard · Segmentación de Botellas PET"
3. st.session_state: inicializar "historial_filtros" = [] si no existe

SIDEBAR — tres secciones:
§1 Fase 1: st.file_uploader para png/jpg/jpeg
§2 Fase 2: st.selectbox "Sin preprocesado / Escala de grises" (el pipeline 
   siempre trabaja en grises, esto es solo para mostrar el efecto visual)
§3 Fase 3 — Filtros espaciales acumulativos:
   - st.selectbox con opciones: Gaussiano, Mediana, Bilateral, Paso Bajas, 
     Promediador, Max, Min
   - Controles dinámicos según el filtro elegido (ksize selectbox con [3,5,7,9], 
     sigma slider para Gaussiano, d/sigmas para Bilateral)
   - Botón "➕ Añadir filtro" y "🗑️ Limpiar filtros"
   - Historial visual con botón ✕ por filtro individual (st.rerun al eliminar)

ÁREA PRINCIPAL — función main():
Paso 1: mostrar imagen original a color con histograma RGB en columnas [2,1]
Paso 2: mostrar imagen en escala de grises con histograma en columnas [2,1]
Paso 3..N: bucle for sobre historial_filtros, aplicar secuencialmente, mostrar 
cada resultado con su histograma en columnas [2,1] y descripción del filtro

Cada paso usa una función auxiliar mostrar_paso_ui(titulo, imagen, descripcion, 
es_gris) que muestra st.image con width="stretch" a la izquierda y 
st.plotly_chart con width="stretch" y key único a la derecha en columnas [2,1].

Guardar resultado final de los filtros en st.session_state["img_filtrada"].
Mostrar st.info si no hay imagen cargada. Comentar TODO el código en español.
Dame el código completo y comentado de ambos archivos.

## Prompt 2
Continuamos construyendo el Dashboard de Segmentación de Botellas PET.
Tenemos ya app.py y procesamiento.py con Fases 1, 2 y 3 funcionando.

REGLAS ESTRICTAS (siempre aplicar):
- width="stretch" en lugar de use_container_width=True
- key único en todos los elementos interactivos y st.plotly_chart
- st.image NO lleva key

Requerimientos procesamiento.py — agregar estas funciones nuevas:

FASE 3.5 — FFT:
fft_filter(imagen_gris, cutoff=0.15, tipo="lowpass") → tuple:
  Pasos: fft2 → fftshift → máscara circular (cutoff × min(H,W)/2) →
  multiplicar → ifftshift → ifft2 → clip uint8
  tipo="lowpass": M=1 dentro del radio, 0 fuera
  tipo="highpass": M=0 dentro del radio, 1 fuera
  Espectro: log(1 + |F_shifted|) normalizado a [0,255] uint8
  Retorna: (img_filtrada, mascara, espectro_mag)

crear_espectro_con_mascara(espectro_mag, mascara) → np.ndarray RGB:
  Convierte espectro a RGB, dibuja el borde de la máscara en rojo (255,60,60)
  usando cv2.morphologyEx con MORPH_GRADIENT kernel 3×3

FASE 4 — Enhancement (todas usan LUT de 256 entradas con cv2.LUT):
- mejora_gamma(imagen_gris, gamma=1.5): I_out = (I/255)^γ · 255
- mejora_desplazamiento(imagen_gris, delta=50): I_out = clip(I+Δ, 0,255)
- mejora_contraccion_expansion(imagen_gris, a_in=50, b_in=200, 
  a_out=0, b_out=255): mapeo lineal por tramos
- mejora_ecual_uniforme(imagen_gris): cv2.equalizeHist
- mejora_ecual_rayleigh(imagen_gris): I_out = 255·√(I/255)
- mejora_ecual_log_hiperbolica(imagen_gris): I_out=255·log(1+I)/log(256)
- aplicar_mejora(imagen_gris, config: dict): dispatcher igual que 
  aplicar_filtro pero para mejoras
- calcular_histograma_comparativo(img_antes, img_despues) → go.Figure:
  Superpone dos histogramas grises: rojo=antes, azul=después, height=300

Variables a exportar:
MEJORAS_OPCIONES = ["Corrección Gamma", "Desplazamiento (Brillo)", 
  "Contracción / Expansión", "Ecualización Uniforme", 
  "Ecualización Rayleigh", "Ecualización Log. Hiperbólica"]

Requerimientos app.py — agregar al sidebar ENTRE la Fase 3 y la Fase 4:

§3.5 Fase 3.5 — FFT (entre Fase 3 y Fase 4):
- st.checkbox "Aplicar filtro FFT" key="chk_fft_activo" default=False
- st.radio tipo: ["lowpass","highpass"] key="radio_fft_tipo" horizontal=True
- st.slider cutoff 0.01-1.00 step=0.01 default=0.15 key="slider_fft_cutoff"
- Mostrar caption explicativo según tipo elegido

§4 Fase 4 — Mejora de Contraste y Brillo (acumulativa como Fase 3):
- st.session_state["historial_mejoras"] = [] si no existe
- st.selectbox con MEJORAS_OPCIONES key="sel_mejora"
- Controles dinámicos: gamma slider 0.1-4.0 para Corrección Gamma,
  delta slider -120 a 120 para Desplazamiento,
  four number_inputs para Contracción/Expansión,
  sin controles para las ecualizaciones
- Botón "➕ Añadir mejora" y "🗑️ Limpiar mejoras"
- Historial visual con ✕ por mejora individual

Requerimientos app.py — agregar en main() en dos lugares:

LUGAR 1: Después de guardar img_filtrada (al final del bucle Fase 3),
agregar el bloque Fase 3.5:
Si fft_activo: aplicar fft_filter sobre img_filtrada, mostrar dos columnas:
  col izquierda: espectro con máscara superpuesta + caption con % frecuencias
  col derecha: imagen filtrada + métricas μ y σ antes/después
  Sobreescribir st.session_state["img_filtrada"] con la salida de la FFT
  
LUGAR 2: Después de la Fase 3.5, agregar bloque Fase 4:
Bucle acumulativo sobre historial_mejoras. Para cada mejora mostrar:
  col izquierda: imagen entrada (badge naranja "Entrada")
  col derecha: imagen salida (badge morado "Salida")
  Debajo: histograma comparativo con calcular_histograma_comparativo
  y cuatro st.metric: Δμ, Δσ, Δmín, Δmáx
Guardar resultado en st.session_state["img_mejorada"].

Comentar todo en español. Entrégame solo el código nuevo a agregar,
indicando exactamente en qué punto de cada archivo insertarlo.

## Prompt 3
Continuamos el Dashboard de Segmentación de Botellas PET.
Fases 1, 2, 3, 3.5 y 4 están completas y funcionando.
La Fase 5 toma como entrada st.session_state["img_mejorada"].

REGLAS ESTRICTAS:
- width="stretch" en st.image y st.plotly_chart
- key único en todos los elementos interactivos y st.plotly_chart
- st.image NO lleva key

Requerimientos procesamiento.py — funciones nuevas de Fase 5:

UMBRALIZACIÓN (cada función recibe imagen_gris 2D uint8, devuelve 
(binaria uint8, info dict)):

umbralizar_otsu(imagen_gris):
  cv2.threshold con THRESH_BINARY + THRESH_OTSU
  info = {"umbral": int(thresh_val)}

umbralizar_kapur(imagen_gris):
  Itera t 1-254, calcula H_total = p1·(-Σ p1n·log(p1n+1e-12)) + 
  p2·(-Σ p2n·log(p2n+1e-12)), elige t* que maximiza H_total
  info = {"umbral": umbral_opt, "entropia_max": round(mejor_H,4)}

umbralizar_media(imagen_gris):
  umbral = int(np.mean(imagen_gris))
  info = {"umbral": umbral, "media": round(float(np.mean),2)}

umbralizar_manual(imagen_gris, umbral=127):
  info = {"umbral": umbral}

umbralizar_banda(imagen_gris, t1=80, t2=200):
  Píxeles en [t1,t2] → 255, resto → 0
  ESTE ES EL MÁS IMPORTANTE para botellas:
  permite aislar el rango de intensidad específico de la botella
  info = {"t1": t1, "t2": t2}

MORFOLOGÍA POST-UMBRALIZACIÓN (para limpiar la máscara):
aplicar_cierre(mascara, ksize=5):
  cv2.morphologyEx con MORPH_CLOSE, kernel ellipse de ksize×ksize
  Rellena huecos pequeños dentro de la botella

aplicar_apertura(mascara, ksize=3):
  cv2.morphologyEx con MORPH_OPEN
  Elimina pequeñas manchas de ruido fuera de la botella

aplicar_relleno_huecos(mascara):
  cv2.floodFill desde la esquina (0,0) para pintar el exterior,
  luego invertir y hacer OR con la máscara original
  Rellena completamente el interior de la botella

aplicar_umbral(imagen_gris, metodo, invertir=False, **kwargs):
  Dispatcher central. Si invertir=True aplica 255-binaria

Constantes a exportar:
METODOS_UMBRALIZACION = ["Otsu","Kapur","Media","Banda","Manual"]
METODOS_AUTOMATICOS = {"Otsu","Kapur","Media"}

Requerimientos app.py — agregar al sidebar AL FINAL (después de Fase 4):

§5 Fase 5 — Segmentación:
st.markdown "#### 🎯 Fase 5 — Segmentación"
st.caption "Opera sobre img_mejorada"

st.selectbox método key="sel_metodo_umbral" con METODOS_UMBRALIZACION

Lógica condicional:
- Si Manual: st.slider 0-255 default=127 key="slider_umbral_manual"
- Si Banda: st.slider T1 0-253 key="slider_banda_t1" y 
            st.slider T2 (T1+1)-255 key="slider_banda_t2"
  st.caption mostrando "Rango activo: [T1, T2]"
- Si automático (Otsu/Kapur/Media): st.info "Umbral calculado automáticamente"

st.checkbox "Invertir máscara" key="chk_invertir" default=False

st.markdown "**Morfología post-umbralización**"
st.checkbox "Aplicar cierre (rellena huecos)" key="chk_cierre" default=True
ksize_cierre = st.slider "Kernel cierre" 3,15,7,2 key="slider_kcierre"
st.checkbox "Aplicar apertura (elimina ruido)" key="chk_apertura" default=False  
ksize_apertura = st.slider "Kernel apertura" 3,11,3,2 key="slider_kapertura"
st.checkbox "Rellenar huecos internos" key="chk_relleno" default=False

Requerimientos app.py — agregar en main() al final:

BLOQUE FASE 5:
st.markdown "## 🎯 Fase 5 · Segmentación"

1. Aplicar umbralización con aplicar_umbral()
2. Si chk_cierre: aplicar_cierre(mascara, ksize_cierre)
3. Si chk_apertura: aplicar_apertura(mascara, ksize_apertura)
4. Si chk_relleno: aplicar_relleno_huecos(mascara)
5. Guardar en st.session_state["img_binarizada"]

Si método automático: mostrar st.metric con el umbral calculado

Mostrar tres columnas iguales:
  col1: imagen binarizada + caption "Máscara binaria"
  col2: imagen mejorada con máscara superpuesta en verde 50% transparencia 
        (usar cv2.addWeighted para overlay) + caption "Overlay"
  col3: histograma de la binarizada con st.plotly_chart 
        width="stretch" key="chart_hist_bin_fase5"

Debajo: st.caption explicando qué método se usó y qué parámetros

Comentar todo en español.

## Prompt 4 
Continuamos el Dashboard de Segmentación de Botellas PET.
Fases 1-5 completas. La Fase 6 toma st.session_state["img_binarizada"].

REGLAS ESTRICTAS:
- width="stretch" en st.image y st.plotly_chart
- key único en todos los elementos interactivos y st.plotly_chart
- st.image NO lleva key

Requerimientos procesamiento.py — funciones nuevas de Fase 6:

CCL:
aplicar_ccl(imagen_binaria, conectividad=8):
  cv2.connectedComponentsWithStats
  Retorna: (n_etiquetas, etiquetas, stats, centroides)

generar_mapa_color_ccl(etiquetas, n_etiquetas, semilla=42):
  np.random.default_rng(semilla), colores RGB aleatorios [60,256]
  fondo (etiqueta 0) = negro (0,0,0)
  Retorna: imagen RGB uint8

DESCRIPTORES DE FORMA (para confirmar si el objeto es una botella):
calcular_descriptores(stats, idx):
  Dado el índice de un componente calcula:
  - area: stats[idx, CC_STAT_AREA]
  - ancho: stats[idx, CC_STAT_WIDTH]
  - alto: stats[idx, CC_STAT_HEIGHT]
  - elongacion: max(ancho,alto) / min(ancho,alto)
    (botellas PET típicamente 1.5 a 4.0)
  - circularidad: NO calculable sin perimetro, dejar en None por ahora
  Retorna dict con todos los valores

es_probable_botella(descriptores):
  Regla simple basada en area y elongacion:
  - area > 500 px²  (descarta ruido pequeño)
  - elongacion entre 1.2 y 6.0 (botellas son elongadas)
  Retorna (bool, razon_str)

EXTRACCIÓN:
extraer_componente_por_indice(etiquetas, stats, imagen_rgb, idx):
  mascara = (etiquetas == idx) * 255 uint8
  objeto_color = cv2.bitwise_and(imagen_rgb, cv2.merge([mascara]*3))
  area_px = stats[idx, CC_STAT_AREA]
  Retorna: (mascara, objeto_color, area_px)

extraer_componente_mayor(etiquetas, stats, imagen_rgb):
  Igual pero idx = argmax(stats[1:, CC_STAT_AREA]) + 1
  Retorna: (mascara, objeto_color, idx_principal, area_px)

dibujar_contorno(imagen_rgb, mascara, color=(0,255,0), grosor=2):
  cv2.findContours RETR_EXTERNAL CHAIN_APPROX_SIMPLE
  cv2.drawContours sobre copia de imagen_rgb
  Retorna: imagen RGB con contorno dibujado

Requerimientos app.py sidebar — agregar después de Fase 5:

§6 Fase 6 — Extracción:
st.radio conectividad [4, 8] key="radio_ccl_con" horizontal=True default=8
st.radio selección ["Automático (mayor área)","Manual (por índice)"] 
  key="radio_sel_ccl"
Si Manual: st.number_input índice 1-999 key="num_idx_ccl"
  st.caption "Consulta la tabla CCL para ver los índices"
st.slider grosor contorno 1-5 default=2 key="slider_grosor_contorno"

Requerimientos app.py main() — agregar al final:

BLOQUE FASE 6:
st.markdown "## 🔬 Fase 6 · Análisis de Componentes y Extracción"

1. Convertir img_binarizada a binaria estricta (>0)*255
2. aplicar_ccl con conectividad seleccionada
3. Guardar n_etiquetas, etiquetas, stats, centroides en session_state
4. generar_mapa_color_ccl
5. Detección de bordes simple: cv2.Canny(img_binarizada, 50, 150)

Mostrar tres columnas:
  col1: mapa de color CCL + caption con n_objetos detectados
  col2: bordes Canny + caption
  col3: histograma de img_binarizada key="chart_ccl_hist"

Tabla CCL en st.expander "📋 Estadísticas de componentes":
  DataFrame con columnas: Etiqueta, Área(px²), Ancho, Alto, Elongación,
  ¿Botella? (usando es_probable_botella)
  Ordenado por Área descendente
  st.dataframe sin hide_index

Extracción:
  Si modo automático: extraer_componente_mayor
  Si modo manual: extraer_componente_por_indice con validación de rango
  
  Si idx inválido: st.error y return

Guardar en session_state: img_mascara_final, img_objeto_color

st.markdown "### Resultado final"
Dos columnas:
  col izq: "🔲 Máscara definitiva" → st.image mascara_final channels="GRAY"
  col der: "🍶 Botella extraída" → st.image objeto_color channels="RGB"

Debajo: st.markdown "### 🎯 Contorno sobre imagen original"
  Llamar dibujar_contorno con img_original y mascara_final
  st.image resultado width="stretch" channels="RGB"

Métricas en cuatro columnas:
  Etiqueta CCL, Área px², Bounding box, ¿Es probable botella? (✅/❌)

Comentar todo en español.

## Prompt 5
Última fase del Dashboard de Segmentación de Botellas PET.
Todas las fases anteriores están completas.
La Fase 7 analiza la simetría bilateral de la máscara extraída
para confirmar matemáticamente si el objeto segmentado es una botella.

REGLAS ESTRICTAS:
- width="stretch" en st.image y st.plotly_chart
- key único en todos los elementos interactivos y st.plotly_chart
- st.image NO lleva key

Requerimientos procesamiento.py — funciones nuevas de Fase 7:

calcular_eje_simetria(mascara_binaria):
  Usa cv2.moments(mascara_binaria) para obtener M00, M10, M01
  Centroide: cx = M10/M00, cy = M01/M00
  El eje principal de simetría se estima con los momentos centrales:
  mu20 = M20/M00 - cx²
  mu02 = M02/M00 - cy²  
  mu11 = M11/M00 - cx·cy
  Ángulo: theta = 0.5 · arctan2(2·mu11, mu20-mu02) en grados
  Retorna: (cx, cy, theta)

calcular_indice_simetria(mascara_binaria, eje="vertical"):
  Dada la máscara binaria (0/255):
  1. Si eje="vertical": voltear con cv2.flip(mascara, 1) → horizontal
     Si eje="horizontal": cv2.flip(mascara, 0) → vertical
  2. Interseccion = cv2.bitwise_and(mascara, mascara_volteada)
  3. Union = cv2.bitwise_or(mascara, mascara_volteada)
  4. indice = sum(Interseccion>0) / sum(Union>0)   [IoU de simetría]
  Retorna: float en [0,1]
  Interpretación:
    > 0.85 → simetría alta (muy probable botella cilíndrica)
    0.70-0.85 → simetría media (botella deformada o parcialmente oculta)
    < 0.70 → simetría baja (probablemente no es botella)

calcular_ambos_ejes_simetria(mascara_binaria):
  Calcula índice para eje vertical Y eje horizontal
  Retorna: (simetria_v, simetria_h)
  Las botellas PET tienen simetría vertical alta y horizontal baja

visualizar_simetria(mascara_binaria, cx, cy, theta):
  Sobre fondo negro dibuja:
  1. La máscara en blanco
  2. El centroide como círculo verde radio=6
  3. El eje de simetría estimado como línea roja grosor=2
     que cruza toda la imagen pasando por (cx,cy) con ángulo theta
  Retorna: imagen RGB uint8

clasificar_por_simetria(simetria_v, simetria_h, area_px, elongacion):
  Regla de clasificación combinada:
  Si simetria_v > 0.80 AND area_px > 1000 AND elongacion > 1.3:
    return "✅ Probable Botella PET", "success"
  Elif simetria_v > 0.65:
    return "⚠️ Posible Botella (verificar)", "warning"  
  Else:
    return "❌ No parece una botella", "error"

Requerimientos app.py sidebar — agregar AL FINAL del sidebar:

§7 Fase 7 — Simetría:
st.markdown "#### 🪞 Fase 7 — Análisis de Simetría"
st.caption "Confirma matemáticamente si el objeto es una botella"
st.radio "Eje de análisis principal" ["vertical","horizontal","ambos"]
  key="radio_eje_simetria" default="vertical"
st.slider "Umbral de simetría" 0.50-1.00 step=0.05 default=0.80
  key="slider_umbral_simetria"
  help="Índice IoU mínimo para clasificar como botella"

Requerimientos app.py main() — agregar al final de todo:

BLOQUE FASE 7:
st.markdown "---"
st.markdown "## 🪞 Fase 7 · Análisis de Simetría Bilateral"
st.caption "La simetría bilateral es una propiedad geométrica 
  de los envases cilíndricos que confirma si el objeto es una botella"

Verificar que img_mascara_final esté en session_state, si no: st.warning y return

mascara = st.session_state["img_mascara_final"]

1. calcular_eje_simetria(mascara) → cx, cy, theta
2. calcular_ambos_ejes_simetria(mascara) → simetria_v, simetria_h
3. descriptores del componente principal (desde session_state)
4. clasificar_por_simetria → etiqueta, tipo

Mostrar resultado de clasificación:
  Si tipo="success": st.success con etiqueta
  Si tipo="warning": st.warning con etiqueta
  Si tipo="error": st.error con etiqueta

Dos columnas:
  col izq: visualizar_simetria(mascara, cx, cy, theta)
    st.image width="stretch" channels="RGB"
    caption "Línea roja = eje de simetría estimado · Punto verde = centroide"
  col der: cuatro métricas apiladas
    st.metric "Simetría Vertical" f"{simetria_v:.3f}" 
      delta="Alta ✅" si >umbral_simetria else "Baja ❌"
    st.metric "Simetría Horizontal" f"{simetria_h:.3f}"
    st.metric "Eje estimado" f"{theta:.1f}°"
    st.metric "Centroide" f"({cx:.0f}, {cy:.0f}) px"

Debajo: comparativa de simetría con Plotly bar chart:
  Dos barras: "Eje Vertical" y "Eje Horizontal"
  Línea horizontal en el umbral de simetría configurado
  Color verde si supera el umbral, rojo si no
  key="chart_simetria_barras"

st.markdown "### 📋 Resumen final del análisis"
Tabla o columnas con el veredicto completo:
  Área, Elongación, Simetría V, Simetría H, Eje, Clasificación final

st.success final: "✅ Análisis completo. Pipeline de 7 fases ejecutado."

Comentar todo en español. Dame solo el código nuevo para cada archivo
indicando exactamente dónde insertar cada bloque.
 
## Prompt A
Continuamos el Dashboard de Segmentación de Botellas PET.
Añade SOLO las siguientes funciones nuevas al FINAL de procesamiento.py.
No modifiques nada existente. Añade estos imports al inicio si no están:
  from scipy.ndimage import gaussian_filter1d
  from scipy.signal import find_peaks

═══════════════════════════════════════════════════════════════════
FUNCIÓN 1: analizar_histograma(imagen_rgb, imagen_gris) -> dict
═══════════════════════════════════════════════════════════════════
Analiza ambos histogramas y devuelve un dict con métricas de los dos.
El histograma en grises alimenta la elección del umbral.
El histograma RGB alimenta la clasificación del tipo de escena.

── Bloque A: métricas del histograma en GRISES ──────────────────
  media        = float(np.mean(imagen_gris))
  std          = float(np.std(imagen_gris))
  mediana_px   = float(np.median(imagen_gris))
  min_px       = int(imagen_gris.min())
  max_px       = int(imagen_gris.max())
  rango_din    = max_px - min_px

  asimetria: (media - mediana_px) / std si std > 0 else 0.0
    Positiva → cola hacia valores altos (brillos dominantes)
    Negativa → cola hacia valores bajos (sombras dominantes)

  curtosis: np.mean((img_f - media)^4) / std^4 - 3 si std>0 else 0.0
    donde img_f = imagen_gris.astype(float)

  entropia: calcular sobre hist normalizado de 256 bins:
    hist, _ = np.histogram(imagen_gris.flatten(), 256, (0,256))
    prob = hist / hist.sum()
    entropia = float(-np.sum(prob * np.log2(prob + 1e-12)))
    Rango teórico [0,8]. Alta → imagen compleja con muchos tonos.

  Detección de picos del histograma suavizado:
    hist_suave = gaussian_filter1d(hist.astype(float), sigma=4)
    picos, props = find_peaks(hist_suave,
                              height=hist_suave.max() * 0.05,
                              distance=20)
    n_picos          = len(picos)
    pico_dominante   = int(picos[np.argmax(props["peak_heights"])]) 
                       si n_picos > 0 else int(media)
    segundo_pico     = None (si n_picos < 2) o int del segundo pico
                       más alto (ordenar props["peak_heights"] desc,
                       tomar el índice [1])
    separacion_picos = abs(pico_dominante - segundo_pico) si existe
                       else 0.0

  Distribución por zonas:
    total = imagen_gris.size
    zona_oscura = float((imagen_gris < 85).sum()  / total)
    zona_media  = float(((imagen_gris >= 85) & (imagen_gris <= 170)).sum() / total)
    zona_clara  = float((imagen_gris > 170).sum() / total)

── Bloque B: métricas del histograma RGB ────────────────────────
  # Medias por canal
  media_r = float(np.mean(imagen_rgb[:,:,0]))
  media_g = float(np.mean(imagen_rgb[:,:,1]))
  media_b = float(np.mean(imagen_rgb[:,:,2]))

  # Varianzas por canal (cuál canal lleva más información)
  var_r = float(np.var(imagen_rgb[:,:,0]))
  var_g = float(np.var(imagen_rgb[:,:,1]))
  var_b = float(np.var(imagen_rgb[:,:,2]))
  canal_dominante = ["R","G","B"][int(np.argmax([var_r,var_g,var_b]))]

  # Diferencias entre medias de canales
  diff_gr = media_g - media_r   # positivo → predomina el verde
  diff_gb = media_g - media_b   # positivo → más verde que azul
  diff_rb = media_r - media_b   # positivo → más rojo que azul

  # Saturación cromática media aproximada (sin HSV completo)
  max_c = imagen_rgb.max(axis=2).astype(np.float32)
  min_c = imagen_rgb.min(axis=2).astype(np.float32)
  sat_map = np.where(max_c > 0, (max_c - min_c) / max_c, 0.0)
  saturacion_media = float(sat_map.mean())
  # < 0.10 → escena sin color (gris, blanco, negro)
  # 0.10-0.30 → colores desaturados (agua turbia, plástico blanco)
  # > 0.30 → colores vivos (agua turquesa, botella de color)

  # Clasificación del tipo de fondo hídrico por color dominante:
  if diff_gb > 15 and diff_gr > 10:
      tipo_agua = "verde"        # río con vegetación, algas
  elif diff_gb < -15 and diff_rb < -10:
      tipo_agua = "azul"         # agua clara, piscina, océano
  elif saturacion_media < 0.12:
      tipo_agua = "turbia_cafe"  # agua sucia, río fangoso
  else:
      tipo_agua = "indefinido"   # iluminación mixta o imagen compleja

  # Balance de blancos aproximado: distancia entre canales
  balance_rgb = float(np.std([media_r, media_g, media_b]))
  # < 8  → imagen casi gris (poco color) → botella blanca probable
  # > 20 → imagen con color marcado → botella de color o fondo colorido

── Retornar dict completo con TODAS las claves ──────────────────
  Incluir todos los valores calculados en los dos bloques.
  Nombrar las claves exactamente como se definen arriba.

═══════════════════════════════════════════════════════════════════
FUNCIÓN 2: clasificar_escena(metricas) -> tuple[str, str, str]
═══════════════════════════════════════════════════════════════════
Recibe el dict de analizar_histograma().
Devuelve (tipo_escena: str, descripcion: str, emoji: str).

Evaluar las reglas EN ESTE ORDEN EXACTO (la primera que coincida gana):

REGLA 1 — BAJO CONTRASTE
  Si rango_din < 80 O std < 22:
    tipo  = "bajo_contraste"
    emoji = "⚠️"
    desc  = (f"Contraste muy bajo (σ={metricas['std']:.1f}, "
             f"rango={metricas['rango_din']} niveles). "
             "El histograma está concentrado. "
             "Se necesita expansión de contraste antes de segmentar.")

REGLA 2 — BOTELLA CLARA EN AGUA VERDE
  Si tipo_agua == "verde" Y zona_oscura > 0.35 Y zona_clara > 0.08:
    tipo  = "botella_clara_agua_verde"
    emoji = "🌿"
    desc  = (f"Agua con tonalidad verde (diff_gr={metricas['diff_gr']:.1f}). "
             "Botella clara detectada sobre fondo oscuro verdoso. "
             "Pipeline: Bilateral → Gamma alto → Otsu.")

REGLA 3 — BOTELLA EN AGUA AZUL CLARA
  Si tipo_agua == "azul" Y zona_clara > 0.40:
    tipo  = "botella_agua_azul_clara"
    emoji = "🌊"
    desc  = (f"Agua de tono azul claro (diff_gb={metricas['diff_gb']:.1f}). "
             "Fondo predominantemente claro. "
             "Pipeline: Gaussiano → Ecualización → Kapur invertido.")

REGLA 4 — BOTELLA EN AGUA TURBIA O CAFÉ
  Si tipo_agua == "turbia_cafe":
    tipo  = "botella_agua_turbia"
    emoji = "🟫"
    desc  = ("Agua turbia o contaminada (baja saturación cromática). "
             "Imagen casi acromática. "
             "Pipeline: Mediana → Expansión de contraste → Media.")

REGLA 5 — BOTELLA CLARA SOBRE FONDO OSCURO (genérico)
  Si zona_oscura > 0.45 Y zona_clara > 0.10 Y pico_dominante < 110:
    tipo  = "botella_clara_fondo_oscuro"
    emoji = "🔦"
    desc  = (f"Fondo oscuro dominante ({metricas['zona_oscura']:.0%} "
             "de píxeles bajo 85). Objeto claro destacado. "
             "Pipeline: Mediana → Bilateral → Gamma → Otsu.")

REGLA 6 — BOTELLA OSCURA SOBRE FONDO CLARO (genérico)
  Si zona_clara > 0.45 Y zona_oscura > 0.08 Y pico_dominante > 135:
    tipo  = "botella_oscura_fondo_claro"
    emoji = "☀️"
    desc  = (f"Fondo claro dominante ({metricas['zona_clara']:.0%} "
             "de píxeles sobre 170). Objeto oscuro. "
             "Pipeline: Gaussiano → Ecualización Uniforme → Kapur invertido.")

REGLA 7 — CONTRASTE MODERADO (caso por defecto)
  Para cualquier otro caso:
    tipo  = "contraste_moderado"
    emoji = "📊"
    desc  = (f"Distribución equilibrada (σ={metricas['std']:.1f}, "
             f"separación de picos={metricas['separacion_picos']:.0f}). "
             "Se aplica enhancement previo a la segmentación.")

## Prompt B
Continuamos el Dashboard de Segmentación de Botellas PET.
Añade SOLO estas dos funciones nuevas al FINAL de procesamiento.py.

IMPORTANTE: Los dicts de filtros deben ser compatibles con la función
aplicar_filtro() ya existente. Los dicts de mejoras deben ser
compatibles con aplicar_mejora() que se agregará en el V2.

═══════════════════════════════════════════════════════════════════
FUNCIÓN 1: sugerir_pipeline(tipo_escena, metricas) -> dict
═══════════════════════════════════════════════════════════════════
Devuelve un dict con esta estructura EXACTA (sin alterar los nombres):
{
  "filtros":  [ {tipo+params}, ... ],
  "fft":      {"activo": bool, "tipo": str, "cutoff": float},
  "mejoras":  [ {tipo+params}, ... ],
  "umbral":   {"metodo": str, "invertir": bool, "params": dict},
  "razon":    str,
}

━━━ Caso "botella_clara_agua_verde" ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
filtros:
  [{"tipo":"Mediana","ksize":5},
   {"tipo":"Bilateral","d":9,"sigma_color":65.0,"sigma_space":65.0}]
  Razón filtros: la mediana elimina brillos de hojas/algas sobre el
  agua; el bilateral preserva el borde plástico vs fondo orgánico.

fft: {"activo":True,"tipo":"lowpass","cutoff":0.20}
  Razón FFT: la vegetación y el agua crean textura periódica de alta
  frecuencia; el paso bajas la elimina sin borrar la botella.

mejoras:
  [{"tipo":"Corrección Gamma","gamma":2.0}]
  Razón mejoras: oscurece el fondo verde, separa mejor el pico de
  la botella clara en el histograma.

umbral: {"metodo":"Otsu","invertir":False,"params":{}}
  Razón umbral: dos picos bien separados tras el gamma → Otsu óptimo.

razon: (f"Agua verde detectada (diff_gr={metricas['diff_gr']:.1f}). "
        f"Canal G dominante. σ={metricas['std']:.1f}. "
        "Mediana elimina ruido orgánico, Bilateral preserva borde "
        "plástico, Gamma γ=2.0 separa botella clara del fondo verdoso.")

━━━ Caso "botella_agua_azul_clara" ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
filtros:
  [{"tipo":"Gaussiano","ksize":5,"sigma":1.5},
   {"tipo":"Mediana","ksize":3}]
  Razón filtros: el agua azul clara es uniforme; gaussiano homogeniza
  sin destruir el borde. Mediana limpia reflejos residuales.

fft: {"activo":False,"tipo":"lowpass","cutoff":0.15}
  Razón FFT: fondo uniforme no tiene textura de alta frecuencia.

mejoras:
  [{"tipo":"Ecualización Uniforme"},
   {"tipo":"Corrección Gamma","gamma":0.65}]
  Razón mejoras: HE aplana el histograma sesgado; gamma < 1 aclara
  la imagen para que el pico oscuro de la botella destaque.

umbral: {"metodo":"Kapur","invertir":True,"params":{}}
  Razón umbral: histograma sesgado a la derecha (fondo claro dominante);
  Kapur maneja mejor que Otsu esta distribución asimétrica.
  Invertir porque la botella es el objeto más oscuro.

razon: (f"Agua azul clara detectada (diff_gb={metricas['diff_gb']:.1f}). "
        f"zona_clara={metricas['zona_clara']:.0%}. "
        "Inversión de máscara necesaria: botella más oscura que el agua.")

━━━ Caso "botella_agua_turbia" ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
filtros:
  [{"tipo":"Mediana","ksize":7},
   {"tipo":"Bilateral","d":7,"sigma_color":90.0,"sigma_space":90.0}]
  Razón filtros: agua turbia tiene mucho ruido de sedimentos;
  mediana con ksize grande los elimina; bilateral con sigma alto
  acepta mayor rango de similitud (imagen acromática).

fft: {"activo":True,"tipo":"highpass","cutoff":0.06}
  Razón FFT: en imagen casi gris, el paso altas muy suave resalta
  los pocos bordes presentes (contorno de la botella).

mejoras:
  [{"tipo":"Contracción / Expansión",
    "a_in": max(0, int(metricas["pico_dominante"]) - 35),
    "b_in": min(255, int(metricas["pico_dominante"]) + 35),
    "a_out": 0, "b_out": 255},
   {"tipo":"Ecualización Log. Hiperbólica"}]
  Razón mejoras: expansión alrededor del pico dominante estira el
  histograma comprimido; log hiperbólica expande los detalles sutiles.

umbral:
  invertir_val = metricas["media"] > 127
  {"metodo":"Media","invertir":invertir_val,"params":{}}
  Razón umbral: bajo contraste → media es el umbral más robusto.
  Invertir si la media es alta (fondo claro domina).

razon: (f"Agua turbia detectada (sat={metricas['saturacion_media']:.2f}). "
        "Imagen casi acromática. Expansión de contraste + log hiperbólica "
        "para maximizar la separación residual entre botella y agua.")

━━━ Caso "botella_clara_fondo_oscuro" ━━━━━━━━━━━━━━━━━━━━━━━━━━━
filtros:
  [{"tipo":"Mediana","ksize":5},
   {"tipo":"Bilateral","d":9,"sigma_color":60.0,"sigma_space":60.0}]

fft: {"activo":True,"tipo":"lowpass","cutoff":0.25}

mejoras:
  [{"tipo":"Corrección Gamma","gamma":1.8}]

umbral: {"metodo":"Otsu","invertir":False,"params":{}}

razon: (f"Fondo oscuro genérico (zona_oscura={metricas['zona_oscura']:.0%}). "
        f"pico_dominante={metricas['pico_dominante']}. "
        "Gamma γ=1.8 oscurece el fondo residual. Otsu binariza la zona clara.")

━━━ Caso "botella_oscura_fondo_claro" ━━━━━━━━━━━━━━━━━━━━━━━━━━━
filtros:
  [{"tipo":"Gaussiano","ksize":3,"sigma":0.0},
   {"tipo":"Mediana","ksize":3}]

fft: {"activo":False,"tipo":"lowpass","cutoff":0.15}

mejoras:
  [{"tipo":"Ecualización Uniforme"},
   {"tipo":"Corrección Gamma","gamma":0.7}]

umbral: {"metodo":"Kapur","invertir":True,"params":{}}

razon: (f"Fondo claro genérico (zona_clara={metricas['zona_clara']:.0%}). "
        "Invertir máscara: botella más oscura que el agua.")

━━━ Caso "bajo_contraste" ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
filtros:
  [{"tipo":"Bilateral","d":7,"sigma_color":85.0,"sigma_space":85.0},
   {"tipo":"Mediana","ksize":3}]

fft: {"activo":True,"tipo":"highpass","cutoff":0.04}

mejoras:
  [{"tipo":"Contracción / Expansión",
    "a_in": max(0, int(metricas["min_px"]) + 5),
    "b_in": min(255, int(metricas["max_px"]) - 5),
    "a_out": 0, "b_out": 255},
   {"tipo":"Ecualización Uniforme"}]

umbral: {"metodo":"Media","invertir":False,"params":{}}

razon: (f"Bajo contraste (σ={metricas['std']:.1f}, "
        f"rango={metricas['rango_din']} niveles). "
        "Expansión al rango dinámico real + HE para maximizar separación.")

━━━ Caso "contraste_moderado" (por defecto) ━━━━━━━━━━━━━━━━━━━━━
Construir dinámicamente según las métricas:

filtros:
  Si std < 40:
    [{"tipo":"Bilateral","d":9,"sigma_color":75.0,"sigma_space":75.0},
     {"tipo":"Mediana","ksize":5}]
  Else:
    [{"tipo":"Mediana","ksize":3},
     {"tipo":"Gaussiano","ksize":3,"sigma":0.0}]

fft: {"activo":False,"tipo":"lowpass","cutoff":0.15}

mejoras:
  Si separacion_picos > 45:
    [{"tipo":"Corrección Gamma","gamma":1.4}]
  Else:
    [{"tipo":"Ecualización Uniforme"},
     {"tipo":"Corrección Gamma","gamma":1.3}]

umbral:
  invertir_val = metricas["zona_clara"] > metricas["zona_oscura"]
  Si separacion_picos > 55:
    metodo = "Otsu"
  Else:
    metodo = "Kapur"
  {"metodo":metodo,"invertir":invertir_val,"params":{}}

razon: (f"Contraste moderado (σ={metricas['std']:.1f}, "
        f"sep_picos={metricas['separacion_picos']:.0f}, "
        f"tipo_agua={metricas['tipo_agua']}). "
        f"Pipeline adaptado. Invertir={invertir_val}.")

═══════════════════════════════════════════════════════════════════
FUNCIÓN 2: ejecutar_pipeline_sugerido(imagen_gris, imagen_rgb,
                                      pipeline_dict) -> dict
═══════════════════════════════════════════════════════════════════
Aplica el pipeline_dict completo en orden secuencial.
Necesita imagen_rgb para la FFT y para guardar contexto.

Importar al inicio si no están:
  from procesamiento import (aplicar_filtro, fft_filter,
                              aplicar_mejora, aplicar_umbral)
  (o simplemente llamarlas directamente si están en el mismo archivo)

Pasos en orden:

1. FILTROS: aplicar cada config de pipeline_dict["filtros"] en
   secuencia usando aplicar_filtro(). Guardar imagen tras cada paso.
   imgs_filtros = []
   img_actual = imagen_gris.copy()
   for cfg in pipeline_dict["filtros"]:
       img_actual = aplicar_filtro(img_actual, cfg)
       imgs_filtros.append(img_actual.copy())

2. FFT: si pipeline_dict["fft"]["activo"]:
   img_fft, mascara_fft, espectro_fft = fft_filter(
       img_actual,
       cutoff=pipeline_dict["fft"]["cutoff"],
       tipo=pipeline_dict["fft"]["tipo"],
   )
   img_actual = img_fft
   (guardar img_post_fft, mascara_fft, espectro_fft)
   else: img_post_fft = mascara_fft = espectro_fft = None

3. MEJORAS: aplicar cada config de pipeline_dict["mejoras"].
   imgs_mejoras = []
   for cfg in pipeline_dict["mejoras"]:
       img_actual = aplicar_mejora(img_actual, cfg)
       imgs_mejoras.append(img_actual.copy())
   img_mejorada = img_actual.copy()

4. UMBRALIZACIÓN:
   u = pipeline_dict["umbral"]
   img_bin, umbral_info = aplicar_umbral(
       img_actual,
       metodo=u["metodo"],
       invertir=u["invertir"],
       **u["params"],
   )

5. CIERRE MORFOLÓGICO fijo (siempre, para limpiar máscara):
   kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(7,7))
   img_binarizada = cv2.morphologyEx(img_bin, cv2.MORPH_CLOSE, kernel)

Retornar dict:
{
  "imgs_filtros":    imgs_filtros,      # list[np.ndarray]
  "img_post_fft":    img_post_fft,      # np.ndarray | None
  "mascara_fft":     mascara_fft,       # np.ndarray | None
  "espectro_fft":    espectro_fft,      # np.ndarray | None
  "imgs_mejoras":    imgs_mejoras,      # list[np.ndarray]
  "img_mejorada":    img_mejorada,      # np.ndarray
  "img_binarizada":  img_binarizada,    # np.ndarray
  "umbral_info":     umbral_info,       # dict
}

## Prompt C
Continuamos el Dashboard de Segmentación de Botellas PET.
Modifica SOLO el sidebar de app.py. No toques main().

IMPORTS a agregar al inicio de app.py (junto a los existentes):
  from procesamiento import (
      analizar_histograma,
      clasificar_escena,
      sugerir_pipeline,
      ejecutar_pipeline_sugerido,
  )

INICIALIZACIÓN de session_state (añadir junto a los existentes):
  if "modo_auto" not in st.session_state:
      st.session_state["modo_auto"] = True
  if "pipeline_ejecutado" not in st.session_state:
      st.session_state["pipeline_ejecutado"] = False
  if "historial_mejoras" not in st.session_state:
      st.session_state["historial_mejoras"] = []

MODIFICACIÓN DEL SIDEBAR:
Añadir como la PRIMERA sección, justo después de:
    st.title("🍶 Botellas PET")
    st.caption("Dashboard de Segmentación — V2")
    st.divider()

y ANTES de la sección §1 — Fase 1 — Imagen.

─────────────────────────────────────────────────────────────────
BLOQUE A INSERTAR EN EL SIDEBAR:
─────────────────────────────────────────────────────────────────

    # =========================================================================
    # § 0 — MODO DE OPERACIÓN
    # =========================================================================
    st.markdown("#### ⚙️ Modo de operación")

    modo_op = st.radio(
        label="¿Cómo quieres trabajar?",
        options=["🤖 Automático", "🛠️ Manual"],
        index=0 if st.session_state["modo_auto"] else 1,
        horizontal=True,
        key="radio_modo_operacion",
        help=(
            "Automático: el sistema analiza el histograma RGB y sugiere "
            "el pipeline óptimo. Un botón lo aplica completo.\n"
            "Manual: tú configuras cada fase individualmente."
        ),
    )
    st.session_state["modo_auto"] = (modo_op == "🤖 Automático")

    if st.session_state["modo_auto"]:
        st.info(
            "📊 El sistema analizará el histograma **RGB** para clasificar "
            "la escena (tipo de agua, contraste, distribución de brillo) "
            "y sugerirá automáticamente filtros, FFT, mejoras y método "
            "de umbralización.\n\n"
            "Pulsa **▶ Ejecutar** en el área principal para aplicarlo.",
            icon="🤖",
        )
    else:
        st.caption("🛠️ Configura manualmente cada fase en las secciones de abajo.")

    st.divider()

─────────────────────────────────────────────────────────────────
MODIFICACIÓN DE LAS SECCIONES §1 A §5 EXISTENTES:
─────────────────────────────────────────────────────────────────
En modo automático las secciones manuales siguen visibles pero
muestran un aviso. Añade esto al INICIO de cada sección §3, §3.5,
§4 y §5 (después del st.markdown del título):

    if st.session_state["modo_auto"]:
        st.caption(
            "⚙️ En modo automático esta sección es configurada "
            "por el sistema. Cambia a Manual para ajustar."
        )

El §1 (file_uploader) y §2 (modo visual) NO llevan este aviso
porque siempre son necesarios independientemente del modo.

## Prompt D
Continuamos el Dashboard de Segmentación de Botellas PET.
Modifica SOLO la función main() de app.py.

REGLAS ESTRICTAS:
- width="stretch" en st.image y st.plotly_chart
- key único en TODOS los st.plotly_chart y elementos interactivos
- st.image NO lleva key

UBICACIÓN DEL BLOQUE: insertar en main() DESPUÉS de guardar
st.session_state["img_gris"] = imagen_gris
y ANTES del st.divider() que precede al Paso 1.

─────────────────────────────────────────────────────────────────
BLOQUE COMPLETO A INSERTAR:
─────────────────────────────────────────────────────────────────

    # ─────────────────────────────────────────────────────────────────────────
    # ANÁLISIS DUAL DEL HISTOGRAMA (RGB + Grises)
    # Se ejecuta siempre, en ambos modos (auto y manual).
    # ─────────────────────────────────────────────────────────────────────────
    metricas = analizar_histograma(imagen_rgb, imagen_gris)
    tipo_escena, desc_escena, emoji_escena = clasificar_escena(metricas)
    pipeline_sugerido = sugerir_pipeline(tipo_escena, metricas)

    st.session_state["metricas"]          = metricas
    st.session_state["tipo_escena"]       = tipo_escena
    st.session_state["pipeline_sugerido"] = pipeline_sugerido

    # ── Panel de diagnóstico (siempre visible, expandido en modo auto) ────────
    with st.expander(
        f"{emoji_escena} Diagnóstico automático — "
        f"Escena detectada: **{tipo_escena.replace('_',' ').title()}**",
        expanded=st.session_state["modo_auto"],
    ):
        st.caption(desc_escena)
        st.divider()

        # Métricas del histograma en grises
        st.markdown("**📊 Métricas del histograma en grises**")
        cg1, cg2, cg3, cg4 = st.columns(4)
        with cg1:
            st.metric("Media (μ)",         f"{metricas['media']:.1f}")
            st.metric("Zona oscura",       f"{metricas['zona_oscura']:.0%}")
        with cg2:
            st.metric("Desv. estándar (σ)",f"{metricas['std']:.1f}")
            st.metric("Zona media",        f"{metricas['zona_media']:.0%}")
        with cg3:
            st.metric("Entropía",          f"{metricas['entropia']:.2f} bits")
            st.metric("Zona clara",        f"{metricas['zona_clara']:.0%}")
        with cg4:
            st.metric("Rango dinámico",    f"{metricas['rango_din']} niveles")
            sep = metricas["separacion_picos"]
            st.metric(
                "Separación de picos",
                f"{sep:.0f} niveles",
                delta="✅ Buena" if sep > 60 else "⚠️ Baja",
            )

        st.divider()

        # Métricas del histograma RGB
        st.markdown("**🎨 Métricas del histograma RGB**")
        cr1, cr2, cr3, cr4 = st.columns(4)
        with cr1:
            st.metric("Canal dominante",   metricas["canal_dominante"])
            st.metric("Media R",           f"{metricas['media_r']:.1f}")
        with cr2:
            st.metric("Tipo de agua",      metricas["tipo_agua"])
            st.metric("Media G",           f"{metricas['media_g']:.1f}")
        with cr3:
            st.metric("Saturación media",  f"{metricas['saturacion_media']:.3f}")
            st.metric("Media B",           f"{metricas['media_b']:.1f}")
        with cr4:
            st.metric("Balance RGB (σ)",   f"{metricas['balance_rgb']:.1f}")
            st.metric("diff G-R",          f"{metricas['diff_gr']:+.1f}")

        # Histograma RGB interactivo
        st.divider()
        st.markdown("**Histograma RGB de la imagen cargada**")
        fig_rgb_diag = calcular_histograma(imagen_rgb)
        st.plotly_chart(
            fig_rgb_diag,
            width="stretch",
            key="chart_hist_rgb_diagnostico",
        )

        st.divider()

        # Pipeline sugerido en 4 columnas
        st.markdown("**🤖 Pipeline sugerido para esta escena:**")
        cp1, cp2, cp3, cp4 = st.columns(4)

        with cp1:
            st.markdown("**🧩 Filtros**")
            for i, f in enumerate(pipeline_sugerido["filtros"]):
                params_f = ", ".join(
                    f"{k}={v}" for k,v in f.items() if k != "tipo"
                )
                st.markdown(f"{i+1}. `{f['tipo']}`")
                if params_f:
                    st.caption(params_f)

        with cp2:
            st.markdown("**〰️ FFT**")
            fft_c = pipeline_sugerido["fft"]
            if fft_c["activo"]:
                st.markdown(f"✅ `{fft_c['tipo']}`")
                st.caption(f"cutoff = {fft_c['cutoff']}")
            else:
                st.markdown("⬜ Desactivada")

        with cp3:
            st.markdown("**✨ Mejoras**")
            for i, m in enumerate(pipeline_sugerido["mejoras"]):
                params_m = ", ".join(
                    f"{k}={v}" for k,v in m.items() if k != "tipo"
                )
                st.markdown(f"{i+1}. `{m['tipo']}`")
                if params_m:
                    st.caption(params_m)

        with cp4:
            st.markdown("**🎯 Umbral**")
            u = pipeline_sugerido["umbral"]
            st.markdown(f"`{u['metodo']}`")
            st.caption(
                f"Invertir: {'✅ Sí' if u['invertir'] else '❌ No'}"
            )

        st.caption(f"**Razón:** {pipeline_sugerido['razon']}")

    # ── Bloque de ejecución automática (solo en modo auto) ────────────────────
    if st.session_state["modo_auto"]:
        st.markdown("---")

        col_btn, col_desc = st.columns([1, 3])
        with col_btn:
            ejecutar = st.button(
                "▶ Ejecutar pipeline sugerido",
                key="btn_ejecutar_auto",
                type="primary",
            )
        with col_desc:
            n_f = len(pipeline_sugerido["filtros"])
            n_m = len(pipeline_sugerido["mejoras"])
            fft_activa = pipeline_sugerido["fft"]["activo"]
            st.info(
                f"**{emoji_escena} {tipo_escena.replace('_',' ').title()}** — "
                f"{n_f} filtro(s) · "
                f"FFT {'✅' if fft_activa else '⬜'} · "
                f"{n_m} mejora(s) · "
                f"Umbral: {pipeline_sugerido['umbral']['metodo']}",
                icon="🤖",
            )

        if ejecutar:
            with st.spinner("🤖 Ejecutando pipeline sugerido…"):
                resultado = ejecutar_pipeline_sugerido(
                    imagen_gris, imagen_rgb, pipeline_sugerido
                )

            # Guardar resultados en session_state
            imgs_f = resultado["imgs_filtros"]
            st.session_state["img_filtrada"]   = (
                resultado["img_post_fft"] if resultado["img_post_fft"] is not None
                else imgs_f[-1] if imgs_f else imagen_gris
            )
            st.session_state["img_mejorada"]   = resultado["img_mejorada"]
            st.session_state["img_binarizada"] = resultado["img_binarizada"]
            st.session_state["resultado_auto"] = resultado
            st.session_state["pipeline_ejecutado"] = True
            st.toast("✅ Pipeline completado", icon="🤖")

        # Mostrar resultados solo si ya se ejecutó
        if st.session_state.get("pipeline_ejecutado") \
                and "resultado_auto" in st.session_state:

            res = st.session_state["resultado_auto"]
            st.markdown("## 🤖 Resultado del Pipeline Automático")
            st.success(
                f"Escena **{tipo_escena.replace('_',' ')}** procesada. "
                f"Método de umbralización: {pipeline_sugerido['umbral']['metodo']}. "
                f"Máscara {'invertida' if pipeline_sugerido['umbral']['invertir'] else 'directa'}.",
                icon="✅",
            )

            # 4 columnas: entrada → filtros → mejoras → binarizada
            st.markdown("### Progresión del pipeline")
            rc1, rc2, rc3, rc4 = st.columns(4)

            with rc1:
                st.markdown("**Entrada**")
                st.image(imagen_gris, channels="GRAY",
                         width="stretch", clamp=True)
                st.caption(f"μ={metricas['media']:.1f}  σ={metricas['std']:.1f}")

            with rc2:
                img_post_f = res["imgs_filtros"][-1] if res["imgs_filtros"] \
                             else imagen_gris
                st.markdown("**Tras filtros**")
                st.image(img_post_f, channels="GRAY",
                         width="stretch", clamp=True)
                st.caption(
                    f"μ={np.mean(img_post_f):.1f}  "
                    f"σ={np.std(img_post_f):.1f}"
                )

            with rc3:
                st.markdown("**Tras mejoras**")
                st.image(res["img_mejorada"], channels="GRAY",
                         width="stretch", clamp=True)
                st.caption(
                    f"μ={np.mean(res['img_mejorada']):.1f}  "
                    f"σ={np.std(res['img_mejorada']):.1f}"
                )

            with rc4:
                st.markdown("**Máscara binaria**")
                st.image(res["img_binarizada"], channels="GRAY",
                         width="stretch", clamp=True)
                px_obj = int((res["img_binarizada"] > 0).sum())
                total  = res["img_binarizada"].size
                st.caption(
                    f"Objeto: {px_obj:,} px "
                    f"({100*px_obj/total:.1f}%)"
                )

            # Histograma comparativo: grises originales vs binarizada
            st.markdown("### 📊 Histograma: Grises originales vs Máscara binaria")
            fig_comp = calcular_histograma_comparativo(
                imagen_gris,
                res["img_binarizada"],
            )
            st.plotly_chart(
                fig_comp,
                width="stretch",
                key="chart_hist_auto_comp",
            )
            st.caption(
                "🔴 Grises originales · 🔵 Máscara binarizada. "
                "Un buen resultado muestra el histograma azul con "
                "solo dos barras: una en 0 (fondo) y una en 255 (botella)."
            )

            # Detalle expandible de cada etapa
            with st.expander("🔍 Detalle por etapa", expanded=False):

                if res["imgs_filtros"]:
                    st.markdown("**Filtros aplicados**")
                    cols_f = st.columns(len(res["imgs_filtros"]))
                    for i, (cfg, img_f) in enumerate(
                        zip(pipeline_sugerido["filtros"], res["imgs_filtros"])
                    ):
                        with cols_f[i]:
                            params_s = {k:v for k,v in cfg.items() if k!="tipo"}
                            st.markdown(f"`{cfg['tipo']}`")
                            st.caption(str(params_s) if params_s else "—")
                            st.image(img_f, channels="GRAY",
                                     width="stretch", clamp=True)
                            fig_fi = calcular_histograma(img_f)
                            fig_fi.update_layout(height=180)
                            st.plotly_chart(
                                fig_fi, width="stretch",
                                key=f"chart_det_f_{i}",
                            )

                if res["img_post_fft"] is not None:
                    st.markdown("**FFT aplicada**")
                    cf1, cf2 = st.columns(2)
                    with cf1:
                        st.markdown("Espectro + máscara")
                        esp_vis = crear_espectro_con_mascara(
                            res["espectro_fft"], res["mascara_fft"]
                        )
                        st.image(esp_vis, channels="RGB",
                                 width="stretch", clamp=True)
                    with cf2:
                        st.markdown("Imagen post-FFT")
                        st.image(res["img_post_fft"], channels="GRAY",
                                 width="stretch", clamp=True)

                if res["imgs_mejoras"]:
                    st.markdown("**Mejoras aplicadas**")
                    cols_m = st.columns(len(res["imgs_mejoras"]))
                    for i, (cfg, img_m) in enumerate(
                        zip(pipeline_sugerido["mejoras"], res["imgs_mejoras"])
                    ):
                        with cols_m[i]:
                            params_s = {k:v for k,v in cfg.items() if k!="tipo"}
                            st.markdown(f"`{cfg['tipo']}`")
                            st.caption(str(params_s) if params_s else "—")
                            st.image(img_m, channels="GRAY",
                                     width="stretch", clamp=True)
                            fig_mi = calcular_histograma(img_m)
                            fig_mi.update_layout(height=180)
                            st.plotly_chart(
                                fig_mi, width="stretch",
                                key=f"chart_det_m_{i}",
                            )

            st.info(
                "💡 **¿El resultado no es perfecto?** "
                "Cambia a **Modo Manual** en la barra lateral. "
                "El pipeline sugerido es el punto de partida óptimo "
                "basado en el análisis del histograma RGB, pero cada "
                "imagen es diferente.",
                icon="💡",
            )
            
## Prompt E