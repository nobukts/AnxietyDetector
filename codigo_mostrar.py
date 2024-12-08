import emotion_eyebrow_recognition as eer

#Cargar modelo
model, emotion_labels, face_detector, predictor = eer.cargar_modelo()

# El vídeo debe de estar en la carpeta raiz
path = input("Ingrese el nombre del archivo: ")

# Analizar el video
data = eer.analisis_video(path , model, emotion_labels, face_detector, predictor)

# Crear el archivo CSV, aparecera en la carpeta raiz
eer.realizar_csv(path, data, emotion_labels)