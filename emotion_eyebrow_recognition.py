import cv2
import numpy as np
from tensorflow.keras.models import load_model
from scipy.spatial import distance as dist
from imutils import face_utils
import dlib
from tensorflow.keras.preprocessing.image import img_to_array
import pandas as pd
import time
import os

def cargar_modelo(path='models/best.keras'):
    # Cargar el modelo entrenado para la detección de emociones
    model = load_model(path)

    # Definir las etiquetas de las emociones
    emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

    # Inicializar el detector de rostros de dlib
    face_detector = dlib.get_frontal_face_detector()

    # Inicializar el predictor de puntos faciales de dlib
    predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

    return model, emotion_labels, face_detector, predictor

# Función para preprocesar la imagen de entrada
def preprocess_input(image):
    image = image.astype('float32')
    image = image / 255.0
    image = image - 0.5
    image = image * 2.0
    return image

# Función para calcular la distancia entre las cejas
def eye_brow_distance(leye, reye):
    return dist.euclidean(leye, reye)

# Función para encontrar la emoción de un rostro detectado
def emotion_finder(model, emotion_labels, face_image):
    if face_image.size == 0:
        return None, None
    face_image = cv2.resize(face_image, (48, 48))
    face_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
    face_image = img_to_array(face_image)
    face_image = np.expand_dims(face_image, axis=0)
    face_image = preprocess_input(face_image)
    emotion_prediction = model.predict(face_image)
    return emotion_prediction, emotion_labels[int(np.argmax(emotion_prediction))]

# Función para normalizar los valores y calcular el nivel de estrés
def normalize_values(points, disp):
    normalized_value = abs(disp - np.min(points)) / abs(np.max(points) - np.min(points))
    stress_value = np.exp(-(normalized_value))
    if stress_value >= 0.75:
        return stress_value, "High Stress"
    else:
        return stress_value, "Low Stress"
    

def analisis_video(path, model, emotion_labels, face_detector, predictor):
    # Capturar video desde un archivo
    video_capture = cv2.VideoCapture(path + '.mp4')

    # Verificar si el video se abrió correctamente
    if not video_capture.isOpened():
        print("Error: No se puede abrir el archivo de video.")
        exit()

    # Inicializar variables para el análisis por segundo
    start_time = time.time()
    frame_count = 0
    emotion_sums = np.zeros(len(emotion_labels))
    data = []
    current_second = 0
    points = []

    while True:
        # Leer el cuadro actual del video
        ret, frame = video_capture.read()
        if not ret:
            break

        # Convertir el cuadro a escala de grises
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detectar rostros en la imagen
        faces = face_detector(gray, 0)

        for face in faces:
            # Obtener los puntos faciales
            shape = predictor(gray, face)
            shape = face_utils.shape_to_np(shape)

            # Coordenadas de los ojos
            left_eye = shape[36]
            right_eye = shape[45]

            # Calcular la distancia entre las cejas
            eyebrow_dist = eye_brow_distance(left_eye, right_eye)
            points.append(int(eyebrow_dist))

            # Dibujar el rectángulo del rostro
            (x, y, w, h) = face_utils.rect_to_bb(face)
            if w > 0 and h > 0:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

                # Encontrar la emoción del rostro
                emotion_prediction, emotion = emotion_finder(model, emotion_labels, frame[y:y + h, x:x + w])

                if emotion_prediction is not None:
                    # Acumular las probabilidades de cada emoción
                    emotion_sums += emotion_prediction[0]
                    frame_count += 1

                # Mostrar la emoción y la distancia entre cejas
                cv2.putText(frame, f'Emotion: {emotion}', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
                cv2.putText(frame, f'Eyebrow Dist: {int(eyebrow_dist)}', (x, y + h + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

                # Calcular y mostrar el nivel de estrés
                stress_value, stress_label = normalize_values(points, eyebrow_dist)
                cv2.putText(frame, f'Stress Level: {stress_label}', (x, y + h + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # Mostrar el cuadro con las anotaciones
        cv2.imshow('Emotion and Eyebrow Detection', frame)

        # Cada segundo, calcular el promedio de las probabilidades de las emociones
        elapsed_time = int(time.time() - start_time)
        if elapsed_time > current_second:
            if frame_count > 0:
                avg_emotions = emotion_sums / frame_count
                stress_value, stress_label = normalize_values(points, eyebrow_dist)
                data.append([current_second] + avg_emotions.tolist() + [stress_label] + [stress_value])
            # Reiniciar las variables
            current_second = elapsed_time
            frame_count = 0
            emotion_sums = np.zeros(len(emotion_labels))

        # Salir si se presiona la tecla 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Liberar los recursos 
    video_capture.release()
    cv2.destroyAllWindows()

    return data


def realizar_csv(path, data, emotion_labels):
    print(path)
    # Generar un nombre de archivo secuencial
    file_name = f'emotion_analysis_{path}.csv'

    # Guardar los datos en un archivo Excel
    df = pd.DataFrame(data, columns=['Time (s)'] + emotion_labels + ['Stress Level'] + ['Stress Value'])
    df.to_csv(file_name, index=False)
    
    # Crear una carpeta para almacenar los archivos .csv si no existe
    output_folder = 'csv_outputs'
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Guardar los datos en un archivo .csv dentro de la carpeta
    file_name = os.path.join(output_folder, f'emotion_analysis_{path}.csv')
    df.to_csv(file_name, index=False)



