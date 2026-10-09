# Gesture Pop

Gesture Pop convierte gestos de la webcam en acciones: entrenas una señal con tu mano y, cuando el programa la reconoce, abre la imagen que le asignaste. La idea es que puedas preparar y probar todo desde una sola ventana, sin pelearte con comandos para el uso diario.

El procesamiento ocurre en tu equipo. La aplicacion no transmite el video ni las muestras a un servidor. Las capturas, referencias, vectores y modelos entrenados se guardan localmente y estan excluidos de Git.

## Que incluye

- Una interfaz para capturar, revisar y entrenar gestos.
- Deteccion de manos con RTMDet y RTMPose, con seguimiento rapido de MediaPipe entre detecciones.
- Captura guiada, referencias de imagen y evidencias de los landmarks.
- Reconocimiento con estabilidad temporal para evitar que una imagen se abra por un solo cuadro dudoso.
- Un lanzador clasico por teclado para entrenar o reconocer sin la interfaz Qt.

## Requisitos

- Windows 10/11 y Python 3.12.
- Una webcam. Si Teams, Zoom u otra aplicacion la esta usando, cierrala antes de iniciar Gesture Pop.
- Los modelos de MediaPipe y RTMPose instalados localmente; se explica abajo.

## Instalacion

Abre PowerShell en la carpeta del proyecto y prepara el entorno:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Descarga los modelos de MediaPipe y colocalos con estos nombres:

```text
models/hand_landmarker.task   requerido
models/face_landmarker.task   opcional
```

Para instalar RTMDet y RTMPose, ejecuta `NO_TOCAR/INSTALAR_RTMPOSE.bat`. El instalador prepara las dependencias y guarda los modelos en `models/heavy/`. Esta descarga puede tardar la primera vez.

Cuando termine, abre el programa con doble clic en `NO_TOCAR/ABRIR_GESTURE_POP.bat`. Tambien puedes abrir `gesture_studio_qt.py` desde VS Code usando el interprete `.venv`.

## Guia rapida

1. En `Captura`, pulsa `Agregar` y elige la imagen que quieres asociar al gesto. Tambien puedes dejar imagenes en `imagenes/`; el nombre del archivo se usa como etiqueta si no hay otra asignacion.
2. Selecciona el gesto en la lista. Coloca la mano completa dentro de la camara y espera a que la lectura se estabilice.
3. Pulsa `Capturar gesto`. Repite unas 20 veces por gesto; varia un poco la distancia, el angulo y la posicion para que el modelo no dependa de una sola pose.
4. En `Muestras`, revisa las fotos y elimina las que salieron mal. `Captura guiada` ayuda a variar la posicion de la mano.
5. En `Entrenamiento`, comprueba que cada gesto tenga muestras y entrena el modelo. Como minimo se requieren 3 por gesto; cantidades parecidas y unas 20 suelen dar una prueba mas util.
6. En `Reconocimiento`, inicia la deteccion y manten el gesto estable. Si el programa lo confirma, abre la imagen asociada.

La opcion `Evidencia 5s` toma una foto con los puntos dibujados, pero no la agrega al entrenamiento. Las referencias subidas tambien se pueden inspeccionar antes de decidir si sirven solo como guia o si deben aportar una muestra.

## Como se siguen las manos

RTMDet y RTMPose vuelven a localizar las manos de forma periodica y aportan puntos de referencia nuevos. MediaPipe actualiza los landmarks entre esas detecciones para que el movimiento no espere a que termine cada inferencia pesada. La camara solicita 960 x 540 y la vista previa se limita a 24 actualizaciones por segundo; los calculos de calidad se refrescan con menos frecuencia para no competir con el seguimiento.

La frecuencia de RTMPose sube temporalmente si falta una mano, hay un cruce o el seguimiento necesita recuperarse. El procesamiento pesado corre separado del video, asi que una inferencia lenta no deberia congelar la ventana.

## Donde se guardan tus datos

```text
data/captures/            fotos de entrenamiento
data/gesture_samples.csv  vectores usados por el entrenador
data/references/          imagenes de referencia
data/evidencias_vectores/ fotos de evidencia, fuera del entrenamiento
models/                   pesos locales y modelo entrenado
gesture_settings.json     ajustes por gesto
```

Estas carpetas y archivos estan ignorados por Git. No agregues manualmente tus capturas, referencias ni pesos al repositorio. Las imagenes que ya forman parte de `imagenes/` son recursos del proyecto; las imagenes nuevas en esa carpeta quedan ignoradas por defecto.

## Si algo no funciona

- **No aparece video:** cierra Teams o el navegador que use la webcam, reinicia la aplicacion y prueba otra camara desde el selector.
- **No hay puntos de mano:** confirma que `models/hand_landmarker.task` exista y mejora la luz; procura que la mano no quede cortada.
- **RTMPose no esta listo:** vuelve a ejecutar `NO_TOCAR/INSTALAR_RTMPOSE.bat` y revisa que haya modelos `.onnx` dentro de `models/heavy/`. MediaPipe puede mantener el seguimiento mientras solucionas la instalacion.
- **Se equivoca de gesto:** captura mas ejemplos variados y equilibrados. En Reconocimiento, sube la seguridad minima para evitar disparos dudosos; bajarla lo hace responder con menos exigencia.
- **Se siente lento:** cierra otras aplicaciones que usen la camara o consuman CPU. RTMPose es el detector de mayor costo y, por ahora, se ejecuta con ONNX Runtime en CPU.

## Ejecucion clasica

La interfaz Qt es la opcion recomendada. Para trabajar desde terminal tambien puedes usar:

```powershell
python train_gestures.py
python gesture_launcher.py
```

El entrenador usa `1` a `9` para elegir gesto, `c` para capturar, `u` para deshacer, `s` para entrenar, `v` para alternar los puntos y `q` para salir.

## Pruebas

Con las dependencias instaladas, ejecuta:

```powershell
python -m unittest discover -s tests
```

Las pruebas no necesitan abrir la webcam ni descargar los modelos pesados.

## Mejoras pendientes

- Medir FPS y tiempo por etapa en equipos distintos, no solo estimar el cuello de botella.
- Probar aceleracion de ONNX Runtime cuando el equipo y sus controladores la permitan.
- Seguir afinando los cruces y las oclusiones con ejemplos de captura variados.
- Comparar los resultados por gesto con muestras nuevas, no solo con las usadas al entrenar.

## Licencia

Este repositorio aun no incluye una licencia. Antes de reutilizar o redistribuir el codigo, falta elegir y agregar una licencia al proyecto.
