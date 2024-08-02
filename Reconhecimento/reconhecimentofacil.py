import cv2
import mediapipe as mp
import sqlite3
import numpy as np
import os
import warnings

# Suprimir avisos
warnings.filterwarnings('ignore', category=UserWarning, message='SymbolDatabase.GetPrototype() is deprecated. Please')

# Diretório para salvar as imagens
image_dir = 'saved_faces'
os.makedirs(image_dir, exist_ok=True)

# Configuração do banco de dados
conn = sqlite3.connect('faces.db')
c = conn.cursor()
c.execute('''CREATE TABLE IF NOT EXISTS faces
             (name TEXT, image_path TEXT)''')
conn.commit()

# Função para salvar imagem facial no banco de dados
def save_face(name, image_path):
    c.execute("INSERT INTO faces (name, image_path) VALUES (?, ?)", (name, image_path))
    conn.commit()

# Função para carregar faces do banco de dados
def load_faces():
    c.execute("SELECT name, image_path FROM faces")
    all_faces = c.fetchall()
    known_face_images = []
    known_face_names = []
    for face in all_faces:
        known_face_names.append(face[0])
        face_image = cv2.imread(face[1])
        known_face_images.append(face_image)
    return known_face_names, known_face_images

# Função para encontrar a correspondência de rosto
def find_matching_face(known_face_images, face_image):
    for i, known_image in enumerate(known_face_images):
        result = cv2.matchTemplate(face_image, known_image, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, _ = cv2.minMaxLoc(result)
        if max_val > 0.7:  # Ajuste o limiar conforme necessário
            return i
    return None

# Iniciar webcam
webcam = cv2.VideoCapture(0)

# Inicializar MediaPipe Face Detection
reconhecimento_rosto = mp.solutions.face_detection
desenho = mp.solutions.drawing_utils
reconhecedor_rosto = reconhecimento_rosto.FaceDetection()

while webcam.isOpened():
    validacao, frame = webcam.read()
    if not validacao:
        break
    imagem = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    lista_rostos = reconhecedor_rosto.process(imagem)

    known_face_names, known_face_images = load_faces()

    if lista_rostos.detections:
        for rosto in lista_rostos.detections:
            # Extraindo a região da face da imagem
            bboxC = rosto.location_data.relative_bounding_box
            ih, iw, _ = frame.shape
            (x, y, w, h) = (int(bboxC.xmin * iw), int(bboxC.ymin * ih), int(bboxC.width * iw), int(bboxC.height * ih))
            face_image = frame[y:y+h, x:x+w]

            try:
                if face_image.size > 0:
                    match_index = find_matching_face(known_face_images, face_image)
                    if match_index is not None:
                        name = known_face_names[match_index]
                        # Desenhar retângulo verde ao redor da face reconhecida e adicionar o nome
                        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                        cv2.putText(frame, name, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                    else:
                        # Desenhar retângulo vermelho ao redor da face desconhecida
                        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
                        cv2.putText(frame, "Desconhecido", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
            except Exception as e:
                print(f"Erro durante o reconhecimento facial: {e}")

    # Exibir o frame com as faces reconhecidas
    cv2.imshow("Rostos na sua webcam", frame)

    # Aguardar tecla 's' para salvar nova face
    if cv2.waitKey(5) & 0xFF == ord('s'):
        if lista_rostos.detections:
            for rosto in lista_rostos.detections:
                bboxC = rosto.location_data.relative_bounding_box
                ih, iw, _ = frame.shape
                (x, y, w, h) = (int(bboxC.xmin * iw), int(bboxC.ymin * ih), int(bboxC.width * iw), int(bboxC.height * ih))
                face_image = frame[y:y+h, x:x+w]
                if face_image.size > 0:
                    name = input("Digite seu nome: ")
                    image_path = os.path.join(image_dir, f"{name}.jpg")
                    cv2.imwrite(image_path, face_image)
                    save_face(name, image_path)
                    print(f"Face de {name} salva com sucesso em {image_path}.")

    # Aguardar tecla 'q' para sair
    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

webcam.release()
cv2.destroyAllWindows()
conn.close()
