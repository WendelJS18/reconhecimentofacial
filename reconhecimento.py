import cv2
import mediapipe as mp
import numpy as np
import os

# Diretório onde os rostos registrados serão armazenados
face_db_dir = "face_db"
if not os.path.exists(face_db_dir):
    os.makedirs(face_db_dir)

# Inicializando o módulo de detecção de rostos do MediaPipe
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

# Função para calcular histograma de uma imagem
def calculate_histogram(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    hist = cv2.calcHist([image], [0], None, [256], [0, 256])
    cv2.normalize(hist, hist)
    return hist

# Função para registrar um rosto
def register_face(name):
    cap = cv2.VideoCapture(0)
    face_detection = mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5)
    print(f"Registrando rosto para {name}. Pressione 'q' para capturar a imagem.")

    while True:
        ret, frame = cap.read()
        if not ret:
            continue
        
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_detection.process(frame_rgb)
        
        if results.detections:
            for detection in results.detections:
                mp_drawing.draw_detection(frame, detection)
        
        cv2.imshow('Registro de Rosto', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            if results.detections:
                for i, detection in enumerate(results.detections):
                    bboxC = detection.location_data.relative_bounding_box
                    ih, iw, _ = frame.shape
                    (x, y, w, h) = (int(bboxC.xmin * iw), int(bboxC.ymin * ih),
                                    int(bboxC.width * iw), int(bboxC.height * ih))
                    
                    face_img = frame[y:y+h, x:x+w]
                    face_img = cv2.resize(face_img, (100, 100))
                    np.save(os.path.join(face_db_dir, f"{name}_{i}.npy"), face_img)
                    print(f"Rosto {i} registrado para {name}.")
            else:
                print("Nenhum rosto encontrado. Tente novamente.")
            break

    cap.release()
    cv2.destroyAllWindows()
    face_detection.close()

# Função para carregar os rostos registrados
def load_registered_faces():
    face_images = []
    face_names = []

    for file in os.listdir(face_db_dir):
        if file.endswith(".npy"):
            name = os.path.splitext(file)[0]
            face_img = np.load(os.path.join(face_db_dir, file))
            face_images.append(face_img)
            face_names.append(name)

    return face_images, face_names

# Função para reconhecer rostos em tempo real
def recognize_faces():
    cap = cv2.VideoCapture(0)
    face_detection = mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5)

    # Carregar os rostos registrados
    known_face_images, known_face_names = load_registered_faces()
    known_face_histograms = [calculate_histogram(face_img) for face_img in known_face_images]

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_detection.process(frame_rgb)

        if results.detections:
            for detection in results.detections:
                bboxC = detection.location_data.relative_bounding_box
                ih, iw, _ = frame.shape
                (x, y, w, h) = (int(bboxC.xmin * iw), int(bboxC.ymin * ih),
                                int(bboxC.width * iw), int(bboxC.height * ih))
                
                face_img = frame[y:y+h, x:x+w]
                
                # Verifique se a região do rosto é válida antes de redimensionar
                if face_img.size == 0:
                    continue
                
                face_img = cv2.resize(face_img, (100, 100))
                face_hist = calculate_histogram(face_img)
                name = "Desconhecido"

                # Comparar a imagem do rosto capturado com os rostos registrados
                for i, known_face_hist in enumerate(known_face_histograms):
                    score = cv2.compareHist(face_hist, known_face_hist, cv2.HISTCMP_CORREL)
                    if score > 0.7:  # Threshold para reconhecimento
                        name = known_face_names[i].split('_')[0]  # Remover índice do nome
                        break

                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
                cv2.rectangle(frame, (x, y-20), (x+w, y), (0, 0, 255), cv2.FILLED)
                font = cv2.FONT_HERSHEY_DUPLEX
                cv2.putText(frame, name, (x + 6, y - 6), font, 0.5, (255, 255, 255), 1)

        cv2.imshow('Reconhecimento Facial', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    face_detection.close()

# Menu principal
def main():
    while True:
        print("1. Registrar Rosto")
        print("2. Reconhecer Rostos")
        print("3. Sair")
        choice = input("Escolha uma opção: ")

        if choice == '1':
            name = input("Digite o nome para registro: ")
            register_face(name)
        elif choice == '2':
            recognize_faces()
        elif choice == '3':
            break
        else:
            print("Opção inválida. Tente novamente.")

if __name__ == "__main__":
    main()
