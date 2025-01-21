import os
import logging
from flask import Flask, request, render_template, jsonify
from fer import FER
from PIL import Image
import numpy as np
from supabase import create_client, Client
from datetime import datetime
import cv2

SUPABASE_URL = "https://vlwgtkkthokbudbobjxi.supabase.co"
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InZsd2d0a2t0aG9rYnVkYm9ianhpIiwicm9sZSI6ImFub24iLCJpYXQiOjE3MzQzODg3NDQsImV4cCI6MjA0OTk2NDc0NH0.fNSZ4aGd8nsy4r1V7Qh1XWEMIgdkyETCZKUxFC_5BAo"

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

app = Flask(__name__)
detector = FER()

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger()

def save_to_database(file_name, emotion, score):
    """Função para salvar os dados no Supabase"""
    try:
        data_to_insert = {
            "data_hora": datetime.now().isoformat(),
            "arquivo": file_name,
            "emocao": emotion,
            "porcentagem": score
        }
        response = supabase.table("analises_emocao").insert(data_to_insert).execute()
        if response.error is not None:
            logger.error(f"Erro ao salvar no banco de dados: {response.error}")
            return False

        logger.info(f"Dados salvos com sucesso: Arquivo={file_name}, Emoção={emotion}, Porcentagem={score}")
        return True
    except Exception as e:
        logger.info(f"Dados salvos com sucesso: Arquivo={file_name}, Emoção={emotion}, Porcentagem={score}")
        return False

def is_face_present(img_array):
    """Função para verificar se há rostos na imagem usando OpenCV"""
    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY) 
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
    return len(faces) > 0 
@app.route('/')
def index():
    return render_template('upload.html')

@app.route('/analise_emocional', methods=['POST'])
def analisar_emocao():
    if 'image' not in request.files:
        logger.warning("Nenhuma imagem foi enviada.")
        return jsonify({'error': 'Nenhuma imagem enviada'}), 400 

    file = request.files['image']
    file_name = file.filename
    img = Image.open(file.stream).convert('RGB')
    img_array = np.array(img)

    if not is_face_present(img_array):
        logger.warning("Nenhum rosto detectado na imagem.")
        return jsonify({'error': 'Nenhum rosto detectado'}), 400 

    emotion, score = detector.top_emotion(img_array)
    confidence_threshold = 0.5

    if emotion is None or score < confidence_threshold:
        logger.warning("Nenhuma emoção detectada com confiança suficiente.")
        return jsonify({'error': 'Nenhuma emoção detectada com confiança suficiente'}), 400 

    if  save_to_database(file_name, emotion, round(score * 100, 1)):
        return jsonify({'error': 'Erro ao salvar os dados no banco de dados'}), 500 

    return jsonify({
        'emocao': emotion,
        'porcentagem': f"{round(score * 100, 1)}%",
        'arquivo': file_name
    }), 200

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0')
            
