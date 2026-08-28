# Gesture Pop

Proyecto Python para entrenar gestos con la webcam usando landmarks de manos/cara y mostrar una imagen cuando el gesto se reconoce de forma estable.

Los datos personales de entrenamiento no se publican: `.gitignore` excluye capturas de camara, vectores CSV, referencias locales, entornos virtuales y modelos generados.

## Interfaz visual Qt

Haz doble clic en `NO_TOCAR/ABRIR_GESTURE_POP.bat` para abrir Gesture Pop Studio. La interfaz usa PySide6/Qt, iconos QtAwesome y transiciones breves; organiza el flujo en cinco vistas:

- `Captura`: camara, landmarks, estabilidad, telemetria y captura guiada.
- `Muestras`: galeria de fotos, referencias y vectores antiguos sin foto.
- `Entrenamiento`: cobertura por clase y creacion del modelo.
- `Reconocimiento`: acciones configuradas y lanzamiento del detector en vivo.
- `Evidencias`: fotos independientes con los vectores dibujados, sin agregarlas al entrenamiento.

La flecha de la esquina superior del panel lateral lo reduce a un riel de iconos y lo vuelve a desplegar hacia la derecha. Tambien puedes alternarlo con `Ctrl+B` o desde `Ver > Panel lateral`.

Desde una sola ventana puedes:

- Agregar imagenes PNG, JPG, JPEG, WEBP, BMP o TIFF.
- Asignar un nombre de gesto a cada imagen.
- Subir fotos referenciales y revisar los vectores antes de aceptarlas.
- Seleccionar visualmente que imagen recibira las muestras.
- Ver la camara, mano, cara y landmarks en vivo.
- Ver una caja por mano con posicion X/Y, zona, giro e inclinacion.
- Activar RTMPose como respaldo cuando MediaPipe pierde una mano.
- Capturar y deshacer muestras.
- Ejecutar una captura guiada automatica por zonas de la camara.
- Administrar muestras individuales, incluidas las antiguas que solo tienen vector.
- Entrenar el modelo y abrir el reconocimiento.
- Abrir la imagen asociada en el visor predeterminado cuando el gesto se confirma.

La columna izquierda contiene la navegacion y el menu de gestos. En `Captura`, el centro muestra la camara y los vectores; la columna derecha confirma el gesto activo, su progreso y la posicion/angulos detectados.

El lanzador principal y VS Code abren `gesture_studio_qt.py`.

### Captura guiada

Selecciona un gesto y pulsa `Iniciar captura guiada`. Elige cuantas muestras quieres crear y manten el mismo gesto mientras mueves la mano entre los objetivos de centro, izquierda, derecha, arriba y abajo. La aplicacion captura automaticamente cuando la mano entra en el cuadro y el vector permanece estable.

### Administrador de muestras

Pulsa `Administrar muestras` para filtrar y revisar muestras con foto, referencias de entrenamiento y vectores antiguos sin imagen. Puedes eliminar una muestra concreta; si era de camara tambien se elimina su JPG, mientras que una referencia externa se conserva como guia y deja de participar en el entrenamiento.

### Telemetria de manos

Cada mano detectada usa un color distinto y muestra:

- `X` / `Y`: centro de la mano como porcentaje del cuadro.
- `A`: giro 2D; `0deg` apunta arriba, `+90deg` a la derecha y `-90deg` a la izquierda.
- `T`: inclinacion estimada en profundidad entre muneca y centro de la palma.
- `zona`: ubicacion aproximada, por ejemplo `arriba-izq` o `medio-centro`.

El seguimiento equilibrado conserva las identidades `M1` y `M2` usando posicion, velocidad, forma y lateralidad de MediaPipe. Si las manos se cruzan, la interfaz muestra `Cruce protegido`; si una desaparece unas decimas de segundo, conserva su ultimo vector mientras intenta recuperarla. Un salto extremo reinicia la ventana de estabilidad para evitar guardar una muestra corrupta.

La opcion `Asistencia RTMPose` agrega un segundo detector de 21 puntos. MediaPipe sigue ejecutandose en cada frame; RTMDet + RTMPose-M trabajan en otro proceso a intervalos y solo completan manos faltantes. Durante un cruce se consulta con mayor frecuencia. La etiqueta sobre cada caja indica `MP` o `RTM`, y el inspector muestra estado, manos y latencia. Si el proceso pesado falla o no esta instalado, el video continua con MediaPipe.

La caja y los angulos son telemetria visual. No cambian el vector de entrenamiento, por lo que las muestras antiguas siguen siendo compatibles. `T` es una estimacion basada en la profundidad relativa de MediaPipe, no una medicion fisica calibrada.

### Imagenes referenciales

Selecciona primero el gesto destino y pulsa `Agregar referencia`. El inspector muestra lado a lado la imagen original y la deteccion de MediaPipe. Revisa manos, cara, caja, landmarks, posicion y angulos antes de elegir:

- `Guardar solo como guia`: conserva imagen y vector de referencia, pero no cambia el entrenamiento.
- `Aceptar y agregar al entrenamiento`: conserva la referencia y agrega su vector al dataset del gesto.
- `Cancelar`: no guarda nada.

El inspector rechaza imagenes sin manos y avisa sobre desenfoque, mala luz, manos pequenas o recortadas. Las referencias aceptadas quedan en `data/references/<gesto>/`; sus metadatos viven en `data/reference_manifest.csv` y sus vectores en `data/reference_vectors.csv`.

Las referencias estan separadas de las imagenes destino y de las fotos de camara:

```text
imagenes/                 imagen mostrada al reconocer
data/references/          referencias externas revisadas
data/captures/            fotos tomadas con la webcam
data/gesture_samples.csv  vectores usados para entrenar
```

## Abrir desde VS Code

1. Abre esta carpeta completa en VS Code.
2. Pulsa `Ctrl+Shift+P`, busca `Python: Select Interpreter` y elige `.venv`.
3. Abre la vista `Run and Debug`.
4. Elige `Gesture Pop Studio (Qt)` y pulsa el boton de play.

Tambien puedes hacer doble clic en:

- `NO_TOCAR/ABRIR_GESTURE_POP.bat`
- `NO_TOCAR/ABRIR_RECONOCIMIENTO.bat`

## Instalacion

Usa Python 3.11 o 3.12:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Para instalar la asistencia pesada, ejecuta `NO_TOCAR/INSTALAR_RTMPOSE.bat`. Instala `rtmlib`/ONNX Runtime y descarga los pesos oficiales RTMDet-nano + RTMPose-M en `models/heavy/`. Los pesos son locales y no se publican en Git.

Los modelos de MediaPipe deben estar exactamente aqui:

```text
models/
  hand_landmarker.task
  face_landmarker.task
```

`hand_landmarker.task` es obligatorio. El modelo de cara es opcional, aunque recomendado para gestos que mezclan mano y expresion/posicion facial.

## Mapeo de imagenes

Las imagenes visibles viven en `imagenes/`. El archivo `gesture_map.json` relaciona cada etiqueta entrenada con su imagen:

```json
{
  "paraguay_existe": "paraguay_existe.png",
  "pure_autism": "pure_autism.png",
  "tuthio": "tuthio.png"
}
```

Puedes cambiar una imagen sin reentrenar si conservas la etiqueta del lado izquierdo. Las imagenes nuevas que no esten en el mapa se agregan automaticamente usando su nombre de archivo como etiqueta.

## Entrenamiento

La forma recomendada es usar Gesture Studio. Pulsa `+ Agregar`, elige una imagen, escribe el nombre del gesto y seleccionala en la columna izquierda. Cuando la barra de estabilidad este completa, pulsa `Capturar gesto`. La foto guardada incluye los puntos detectados para que puedas revisarla.

El entrenador clasico por teclado sigue disponible ejecutando `train_gestures.py`.

Ejecuta `train_gestures.py`. El flujo recomendado es:

1. Selecciona una etiqueta con `1`, `2`, `3` y hasta `9`.
2. Haz el gesto y espera a que aparezca `LISTO PARA CAPTURAR`.
3. Pulsa `c` una vez.
4. Repite cambiando ligeramente distancia, angulo y luz.
5. Llega idealmente a 20 muestras por gesto, con cantidades similares entre clases.
6. Pulsa `s` para entrenar y guardar el modelo.

Controles:

- `1` a `9`: cambia la etiqueta activa.
- `c`: guarda vector y foto de revision.
- `u`: elimina la ultima muestra de la etiqueta activa.
- `s`: entrena, valida y guarda el modelo.
- `v`: muestra u oculta landmarks/vectores.
- `q`: sale.

Las fotos de revision quedan separadas por etiqueta en `data/captures/`. Los vectores siguen en `data/gesture_samples.csv`; tus muestras anteriores se conservan. Al capturar se abre durante unos segundos una vista de la ultima foto para detectar una postura mal tomada y deshacerla con `u`.

Al pulsar `s`, el entrenador exige al menos 3 muestras por clase. Desde 4 por clase calcula ademas una precision estimada mediante validacion cruzada. Esa cifra orienta, pero la prueba real siempre es usar la camara con luz y posiciones nuevas.

## Reconocimiento en vivo

Ejecuta `gesture_launcher.py`. Ya no dispara por un solo frame:

1. Cada frame produce probabilidades para todos los gestos.
2. Se rechaza una prediccion si tiene poca confianza o esta demasiado cerca del segundo lugar.
3. Se juntan varios frames en una ventana temporal.
4. La imagen aparece solo cuando suficientes frames coinciden.
5. Debes soltar/cambiar el gesto antes de que el mismo gesto pueda dispararse de nuevo.

La interfaz muestra las tres probabilidades principales, el progreso de estabilidad y si el sistema esta armado. Usa `v` para alternar landmarks y `q` para salir.

## Ajustes

Los parametros viven en `app_config.json`:

- `target_samples_per_gesture`: objetivo visual de muestras por clase.
- `confidence_threshold`: confianza minima del mejor gesto.
- `confidence_margin`: separacion minima frente al segundo gesto.
- `prediction_window`: cantidad de frames recientes considerados.
- `stability_frames`: votos necesarios para confirmar.
- `release_frames`: frames distintos/sin gesto para rearmar.
- `overlay_seconds`: tiempo visible de la imagen.
- `camera_index`: cambia a `1` si tienes otra webcam.

Si reconoce poco, baja `confidence_threshold` de `0.68` a `0.60`. Si lanza imagenes incorrectas, subelo o aumenta `confidence_margin`/`stability_frames`.

## Como se forman los vectores

MediaPipe extrae 21 puntos por mano y puntos clave de cara. El proyecto centra y escala cada grupo de puntos, agrega distancias entre dedos y relaciones mano-cara, y normaliza todo antes del clasificador KNN. Cuando aparecen dos manos por primera vez se asignan de izquierda a derecha; despues el rastreador conserva esas identidades aunque se crucen para que las dos mitades del vector no se intercambien.
