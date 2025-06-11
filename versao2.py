
from facenet_pytorch import InceptionResnetV1, MTCNN
import torch
from PIL import Image
import numpy as np
import os
from google.colab import drive
import matplotlib.pyplot as plt

drive.mount('/content/drive')


#DATASET_PATH = '/content/drive/MyDrive/'  # pasta com subpastas por pessoa
#NEW_PERSON_IMAGE_PATH = '/content/drive/MyDrive/.../marcelinho_no_db.jpg'
#MASK_IMAGE_PATH = '/content/drive/MyDrive/.../marcelinho_na_inferencia.jpg'

#Configurar dispositivo, detector e modelo
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Usando dispositivo: {device}")

mtcnn = MTCNN(image_size=160, margin=0, device=device)
facenet_model = InceptionResnetV1(pretrained='vggface2').eval().to(device)

# Função para extrair embedding normalizado de uma imagem
def extract_embedding(image_path):
    img = Image.open(image_path).convert('RGB')
    face = mtcnn(img)
    if face is None:
        print(f'[WARN] Nenhum rosto detectado na imagem: {image_path}')
        return None
    with torch.no_grad():
        embedding = facenet_model(face.unsqueeze(0).to(device))
    embedding = embedding[0].cpu().numpy()
    embedding /= np.linalg.norm(embedding)  # Normalizar vetor
    return embedding

# Criar banco de dados de embeddings para o dataset inteiro (múltiplas imagens por pessoa)
def build_database(dataset_path):
    database = {}
    for person_name in os.listdir(dataset_path):
        person_folder = os.path.join(dataset_path, person_name)
        if not os.path.isdir(person_folder):
            continue
        embeddings = []
        for img_name in os.listdir(person_folder):
            img_path = os.path.join(person_folder, img_name)
            emb = extract_embedding(img_path)
            if emb is not None:
                embeddings.append(emb)
        if embeddings:
            # Média dos embeddings para a pessoa
            mean_embedding = np.mean(embeddings, axis=0)
            mean_embedding /= np.linalg.norm(mean_embedding)
            database[person_name] = mean_embedding
            print(f"[INFO] Embeddings computados para {person_name} com {len(embeddings)} imagens.")
        else:
            print(f"[WARN] Nenhuma imagem válida para {person_name}")
    return database

#Adicionar nova pessoa ao banco
def add_person_to_database(database, person_name, image_path):
    emb = extract_embedding(image_path)
    if emb is not None:
        database[person_name] = emb
        print(f"[INFO] Pessoa '{person_name}' adicionada ao banco de dados.")
    else:
        print(f"[ERRO] Não foi possível extrair embedding da imagem {image_path}.")

# Função para reconhecer pessoa via embedding
def recognize_face(database, image_path, threshold=1.0):
    emb = extract_embedding(image_path)
    if emb is None:
        print("[ERRO] Nenhum rosto detectado para reconhecimento.")
        return None, None

    min_dist = float('inf')
    identity = None
    for person_name, db_emb in database.items():
        dist = np.linalg.norm(emb - db_emb)
        if dist < min_dist:
            min_dist = dist
            identity = person_name

    # Mostrar imagem
    img = Image.open(image_path).convert('RGB')
    plt.imshow(img)
    plt.axis('off')
    plt.title(f"Reconhecimento Facial - Score (distância): {min_dist:.4f}")
    plt.show()

    if min_dist > threshold:
        print(f"[RESULTADO] Não reconhecido com confiança (distância {min_dist:.4f} > threshold {threshold})")
        return None, min_dist
    else:
        print(f"[RESULTADO] Reconhecido como: {identity} (distância {min_dist:.4f})")
        return identity, min_dist

# --- Executar pipeline ---

print("\n[PASSO 1] Construindo banco de dados com dataset...\n")
database = build_database(DATASET_PATH)

print("\n[PASSO 2] Adicionando nova pessoa ao banco de dados...\n")
add_person_to_database(database, "marcelinho", NEW_PERSON_IMAGE_PATH)

print("\n[PASSO 3] Reconhecendo pessoa na imagem com máscara...\n")
recognize_face(database, MASK_IMAGE_PATH)
