import os
import emotion_eyebrow_recognition as eer

# Revisa el contenido de la carpeta videos
def listar_videos(carpeta_videos):
    extensiones_validas = ('.mp4')
    videos = [f for f in os.listdir(carpeta_videos) if f.endswith(extensiones_validas)]
    return videos

#Cargar modelo
model, emotion_labels, face_detector, predictor = eer.cargar_modelo()

# Listar videos
videos = listar_videos("videos")

# Analizar videos
for video in videos:
    video_name = os.path.join("videos", video)
    video_name = video_name.replace('.mp4', '')
    print(f"Analizando video: {video_name}")
    data = eer.analisis_video(video_name, model, emotion_labels, face_detector, predictor)
    eer.realizar_csv(video.replace('.mp4', ''), data, emotion_labels)