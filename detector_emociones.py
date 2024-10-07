import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Cargar el modelo entrenado
model = load_model('models/best.keras')

# Definir las etiquetas de las emociones
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# Función para preprocesar la imagen de entrada
def preprocess_input(image):
    image = image.astype('float32')
    image = image / 255.0
    image = image - 0.5
    image = image * 2.0
    return image

# Capturar video desde la cámara (o se puede poner la ruta de un video)
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    raise RuntimeError("No se pudo abrir la cámara")

while True:
    # Leer un cuadro del video
    ret, frame = cap.read()
    if not ret:
        break

    # Convertir a escala de grises
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Redimensionar a 48x48 para el modelo
    face_resized = cv2.resize(gray_frame, (48, 48))
    face_resized = np.expand_dims(face_resized, axis=-1)  # Añadir canal de profundidad
    face_resized = np.expand_dims(face_resized, axis=0)  # Añadir dimensión del lote
    face_resized = preprocess_input(face_resized)  # Preprocesar imagen

    # Realizar la predicción
    prediction = model.predict(face_resized)
    emotion_class = np.argmax(prediction)  # Obtener la clase con mayor probabilidad
    emotion_label = emotion_labels[emotion_class]

    # Mostrar la emoción en la imagen
    cv2.putText(frame, emotion_label, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    # Mostrar el cuadro en una ventana
    cv2.imshow('Reconocimiento de Emociones', frame)

    # Salir si se presiona la tecla 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar la cámara y cerrar las ventanas
cap.release()
cv2.destroyAllWindows()