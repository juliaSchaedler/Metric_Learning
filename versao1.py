import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize
import pickle
from google.colab import drive
from PIL import Image
import random
from tqdm import tqdm
import sys
from tensorflow.keras.applications import EfficientNetB0

# Configurar GPU se disponível
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("GPU configurada com sucesso")
    except RuntimeError as e:
        print("Erro ao configurar GPU:", e)
else:
    print("Usando CPU")

# Configurações do modelo (otimizadas para economia de memória)
IMG_SIZE = 224
BATCH_SIZE = 32
EMBEDDING_DIM = 512
EPOCHS = 50
LEARNING_RATE = 0.0001
MARGIN = 1.0  # Para triplet loss
MIN_IMAGES_PER_CLASS = 3

#Pré-processamento
def get_augmentation_pipeline():
    """Pipeline de data augmentation usando Albumentations"""
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.Rotate(limit=15, p=0.5),
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
        A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=10, p=0.3),
        A.GaussNoise(var_limit=(10, 50), p=0.2),
        A.OneOf([
            A.MotionBlur(blur_limit=3, p=0.5),
            A.GaussianBlur(blur_limit=3, p=0.5),
        ], p=0.2),
        A.CoarseDropout(max_holes=8, max_height=16, max_width=16, p=0.3),
    ])

def load_and_preprocess_image(image_path, target_size=(IMG_SIZE, IMG_SIZE), augment=False):
    """Carrega e preprocessa uma imagem com data augmentation opcional"""
    try:
        image = cv2.imread(image_path)
        if image is None:
            return None

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, target_size)

        # Aplicar data augmentation se solicitado
        if augment:
            aug_pipeline = get_augmentation_pipeline()
            image = aug_pipeline(image=image)['image']

        # Normalização melhorada (ImageNet stats)
        image = image.astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        image = (image - mean) / std

        return image
    except Exception as e:
        print(f"Erro ao carregar imagem {image_path}: {e}")
        return None

def scan_dataset_improved(data_dir, min_images=MIN_IMAGES_PER_CLASS):
    print(" Escaneando dataset...")

    classes = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
    class_files = {}

    for class_name in tqdm(classes, desc="Processando classes"):
        class_path = os.path.join(data_dir, class_name)
        image_files = []

        for img_file in os.listdir(class_path):
            if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                img_path = os.path.join(class_path, img_file)
                # Verificar se a imagem pode ser carregada
                if load_and_preprocess_image(img_path) is not None:
                    image_files.append(img_path)

        # Filtrar classes com número mínimo de imagens
        if len(image_files) >= min_images:
            class_files[class_name] = image_files
            print(f"{class_name}: {len(image_files)} imagens")
        else:
            print(f" {class_name}: apenas {len(image_files)} imagens (mínimo: {min_images})")

    print(f" Total de classes válidas: {len(class_files)}")
    return class_files

try:
    import albumentations as A
except ImportError:
    print("Albumentations não encontrado. Instale com: !pip install albumentations")
    # Definir um A.Compose vazio para evitar erro, ou sair
    class EmptyCompose:
        def __call__(self, image):
            return {'image': image}
    A = EmptyCompose()


class TripletGenerator(keras.utils.Sequence):
    """Gerador de triplets para Triplet Loss"""

    def __init__(self, class_files, batch_size=32, triplets_per_class=20):
        self.class_files = class_files
        self.batch_size = batch_size
        self.class_names = list(class_files.keys())
        self.triplets_per_class = triplets_per_class

        # Pré-computar triplets para eficiência
        self.triplets = self._generate_triplets()
        print(f"🔄 Gerador criado com {len(self.triplets)} triplets")

    def _generate_triplets(self):
        """Gera triplets (anchor, positive, negative)"""
        triplets = []

        # Garantir que há pelo menos 2 classes para formar negativos válidos
        if len(self.class_names) < 2:
             print("Aviso: Necessárias pelo menos 2 classes para gerar triplets negativos.")
             return triplets # Retorna lista vazia se não houver classes suficientes

        for class_name in self.class_names:
            files = self.class_files[class_name]

            # Gerar triplets para esta classe
            for _ in range(self.triplets_per_class):
                if len(files) >= 2:
                    try:
                        # Anchor e Positive da mesma classe
                        anchor_idx, positive_idx = random.sample(range(len(files)), 2)

                        # Negative de classe diferente
                        available_negative_classes = [c for c in self.class_names if c != class_name and len(self.class_files[c]) > 0]
                        if not available_negative_classes:
                            continue 
                        negative_class = random.choice(available_negative_classes)
                        negative_idx = random.randint(0, len(self.class_files[negative_class]) - 1)

                        triplets.append({
                            'anchor': (class_name, anchor_idx),
                            'positive': (class_name, positive_idx),
                            'negative': (negative_class, negative_idx)
                        })
                    except ValueError as e:
                        # Catch random.sample error if len(files) becomes less than 2 unexpectedly
                         print(f" Aviso: Erro ao selecionar amostras para classe {class_name}: {e}. Pulando classe.")
                         continue


        random.shuffle(triplets)
        return triplets

    def __len__(self):
        # Evitar divisão por zero se não houver triplets
        if len(self.triplets) == 0:
            return 0
        return int(np.ceil(len(self.triplets) / self.batch_size))

    def __getitem__(self, idx):
        batch_triplets = self.triplets[idx * self.batch_size:(idx + 1) * self.batch_size]

        anchors = []
        positives = []
        negatives = []

        for triplet in batch_triplets:
            try:
                # Carregar anchor
                anchor_class, anchor_idx = triplet['anchor']
                anchor_path = self.class_files[anchor_class][anchor_idx]
              
                anchor_img = load_and_preprocess_image(anchor_path, augment=True)

                # Carregar positive
                positive_class, positive_idx = triplet['positive']
                positive_path = self.class_files[positive_class][positive_idx]
                
                positive_img = load_and_preprocess_image(positive_path, augment=True)

                # Carregar negative
                negative_class, negative_idx = triplet['negative']
                negative_path = self.class_files[negative_class][negative_idx]
                 
                negative_img = load_and_preprocess_image(negative_path, augment=True)

                if all(img is not None for img in [anchor_img, positive_img, negative_img]):
                    anchors.append(anchor_img)
                    positives.append(positive_img)
                    negatives.append(negative_img)
            except Exception as e:
                 print(f"❌ Erro ao carregar triplet: {triplet}. Erro: {e}. Pulando triplet.")
                 continue

        anchors_arr = np.array(anchors)
        positives_arr = np.array(positives)
        negatives_arr = np.array(negatives)

        dummy_targets = np.zeros((len(anchors_arr), 1), dtype=np.float32)
      
        return [anchors_arr, positives_arr, negatives_arr], dummy_targets


    def on_epoch_end(self):
        """Regenera triplets a cada época"""
        self.triplets = self._generate_triplets()
        print(f"🔄 Gerador: Tripletos regenerados. Total: {len(self.triplets)}")


    @property
    def output_signature(self):
        return (
            [ 
                tf.TensorSpec(shape=(None, IMG_SIZE, IMG_SIZE, 3), dtype=tf.float32),
                tf.TensorSpec(shape=(None, IMG_SIZE, IMG_SIZE, 3), dtype=tf.float32), 
                tf.TensorSpec(shape=(None, IMG_SIZE, IMG_SIZE, 3), dtype=tf.float32)  
            ],
            tf.TensorSpec(shape=(None, 1), dtype=tf.float32) 
        )

def create_embedding_network():
    """Cria rede para extração de embeddings"""

    # Usar EfficientNetB0 como backbone (mais eficiente que ResNet50)
    base_model = EfficientNetB0(
        weights='imagenet',
        include_top=False,
        input_shape=(IMG_SIZE, IMG_SIZE, 3)
    )

    # Descongelar algumas camadas do final para fine-tuning
    for layer in base_model.layers[:-20]:
        layer.trainable = False

    # Camadas de embedding
    model = keras.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(1024, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        layers.Dense(512, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.2),
        layers.Dense(EMBEDDING_DIM),
        layers.Lambda(lambda x: tf.nn.l2_normalize(x, axis=1))
    ])

    return model

def triplet_loss(y_true, y_pred, margin=MARGIN):
    """Triplet Loss melhorado"""
    anchor, positive, negative = y_pred[:, 0], y_pred[:, 1], y_pred[:, 2]

    # Calcular distâncias
    pos_dist = tf.reduce_sum(tf.square(anchor - positive), axis=1)
    neg_dist = tf.reduce_sum(tf.square(anchor - negative), axis=1)

    # Triplet loss com margem
    basic_loss = pos_dist - neg_dist + margin
    loss = tf.reduce_mean(tf.maximum(basic_loss, 0.0))

    return loss

def create_triplet_model():
    """Cria modelo para triplet learning"""

    embedding_network = create_embedding_network()

    # Três entradas para o triplet
    anchor_input = layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3), name='anchor')
    positive_input = layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3), name='positive')
    negative_input = layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3), name='negative')

    # Processar com a mesma rede
    anchor_embedding = embedding_network(anchor_input)
    positive_embedding = embedding_network(positive_input)
    negative_embedding = embedding_network(negative_input)

    # Concatenar embeddings para o loss
    output = layers.concatenate([anchor_embedding, positive_embedding, negative_embedding])

    model = keras.Model(
        inputs=[anchor_input, positive_input, negative_input],
        outputs=output
    )

    return model, embedding_network


class_files = scan_dataset_improved(DATASET_PATH)

if not class_files:
    print("❌ Nenhum dado válido encontrado no dataset com o mínimo de imagens por classe.")
else:

    train_generator = TripletGenerator(
        class_files,
        batch_size=BATCH_SIZE,
        triplets_per_class=15 
    )

 
    if len(train_generator) == 0:
        print("O gerador não conseguiu criar nenhum triplet válido com os dados disponíveis.")
    else:

        print("Criando modelo triplet...")
        triplet_model, embedding_network = create_triplet_model()

        triplet_model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
            loss=triplet_loss
        )

        print("📋 Arquitetura do modelo:")
        embedding_network.summary()

        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor='loss',
                patience=10,
                restore_best_weights=True,
                verbose=1
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='loss',
                factor=0.7,
                patience=5,
                min_lr=1e-7,
                verbose=1
            ),
            keras.callbacks.ModelCheckpoint(
                '/content/best_triplet_model.h5',
                monitor='loss',
                save_best_only=True,
                verbose=1
            ),
            keras.callbacks.LambdaCallback(
                on_epoch_end=lambda epoch, logs: print(f" Época {epoch+1}: Loss = {logs.get('loss', 'N/A'):.4f}")
            )
        ]

        print(" Iniciando treinamento...")
        history = triplet_model.fit(
            train_generator,
            epochs=EPOCHS,
            callbacks=callbacks,
            verbose=1
            # workers=2,
            # use_multiprocessing=False
        )
        print("Treinamento concluído!")

#Visualização do treinamento

plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Função de Loss')
plt.xlabel('Época')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Acurácia')
plt.xlabel('Época')
plt.ylabel('Acurácia')
plt.legend()

plt.tight_layout()
plt.show()

#Criar banco de dados
def extract_embeddings_batch(file_paths, model, batch_size=16):
    """Extrai embeddings em lotes"""
    embeddings = []

    for i in tqdm(range(0, len(file_paths), batch_size), desc="Extraindo embeddings"):
        batch_paths = file_paths[i:i+batch_size]
        batch_images = []

        for path in batch_paths:
            img = load_and_preprocess_image(path, augment=False)
            if img is not None:
                batch_images.append(img)

        if batch_images:
            batch_array = np.array(batch_images)
            batch_embeddings = model.predict(batch_array, verbose=0)
            embeddings.extend(batch_embeddings)

    return np.array(embeddings)

print("🏗️  Criando banco de dados de embeddings...")

# Criar banco de dados
celebrity_database = {}
all_embeddings = []
all_names = []

for class_name, file_paths in tqdm(class_files.items(), desc="Processando celebridades"):
    # Usar todas as imagens disponíveis
    embeddings = extract_embeddings_batch(file_paths, embedding_network, batch_size=8)

    if len(embeddings) > 0:
        # Calcular embedding médio
        mean_embedding = np.mean(embeddings, axis=0)

        celebrity_database[class_name] = {
            'embedding': mean_embedding,
            'all_embeddings': embeddings,
            'num_images': len(embeddings),
            'file_paths': file_paths
        }

        all_embeddings.append(mean_embedding)
        all_names.append(class_name)

all_embeddings = np.array(all_embeddings)

print(f"✅ Banco de dados criado com {len(celebrity_database)} celebridades")


# Adição da Nova Pessoa

print("Adicionando nova pessoa ao banco de dados")

new_person_name = "nova_pessoa"
new_person_files = []

if os.path.exists(NEW_PERSON_PATH):
    for img_file in os.listdir(NEW_PERSON_PATH):
        if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
            img_path = os.path.join(NEW_PERSON_PATH, img_file)
            new_person_files.append(img_path)

    if new_person_files:
        new_embeddings = extract_embeddings_batch(new_person_files, embedding_network)

        if len(new_embeddings) > 0:
            mean_new_embedding = np.mean(new_embeddings, axis=0)

            celebrity_database[new_person_name] = {
                'embedding': mean_new_embedding,
                'all_embeddings': new_embeddings,
                'num_images': len(new_embeddings),
                'file_paths': new_person_files
            }

            all_embeddings = np.vstack([all_embeddings, mean_new_embedding])
            all_names.append(new_person_name)

            print(f" Nova pessoa adicionada com {len(new_embeddings)} imagens")


# Função de Reconhecimento Melhorada

def recognize_face_improved(image_path, database, embeddings_array, names_array, threshold=0.4):
    """Reconhecimento facial melhorado com múltiplas métricas"""

    query_image = load_and_preprocess_image(image_path, augment=False)
    if query_image is None:
        return None, 0, None, None

    # Extrair embedding
    query_embedding = embedding_network.predict(np.expand_dims(query_image, axis=0), verbose=0)[0]

    # Calcular múltiplas métricas
    cosine_similarities = cosine_similarity([query_embedding], embeddings_array)[0]
    euclidean_distances = euclidean_distances([query_embedding], embeddings_array)[0]

    # Converter distâncias euclidianas para similaridades
    euclidean_similarities = 1 / (1 + euclidean_distances)

    # Combinar métricas (média ponderada)
    combined_scores = 0.7 * cosine_similarities + 0.3 * euclidean_similarities

    # Encontrar melhor match
    best_match_idx = np.argmax(combined_scores)
    best_score = combined_scores[best_match_idx]
    best_name = names_array[best_match_idx]

    # Análise de confiança
    confidence_metrics = {
        'cosine_similarity': cosine_similarities[best_match_idx],
        'euclidean_similarity': euclidean_similarities[best_match_idx],
        'combined_score': best_score,
        'rank_position': np.where(np.argsort(combined_scores)[::-1] == best_match_idx)[0][0] + 1
    }

    if best_score >= threshold:
        return best_name, best_score, query_image, confidence_metrics
    else:
        return "Desconhecido", best_score, query_image, confidence_metrics


# Reconhecimento da Pessoa com Máscara

print("\n" + "="*60)
print("🔍 RECONHECIMENTO FACIAL COM MÁSCARA")
print("="*60)

if os.path.exists(MASK_IMAGE_PATH):
    # Realizar reconhecimento
    recognized_person, similarity_score, query_img, metrics = recognize_face_improved(
        MASK_IMAGE_PATH,
        celebrity_database,
        all_embeddings,
        all_names,
        threshold=0.35
    )

    # Visualização dos resultados
    plt.figure(figsize=(15, 8))

    # Imagem de consulta
    plt.subplot(2, 3, 1)
    plt.imshow((query_img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])).clip(0, 1))
    plt.title('🔍 Imagem para Reconhecimento\n(Pessoa com Máscara)', fontweight='bold')
    plt.axis('off')

    # Resultado do reconhecimento
    plt.subplot(2, 3, 2)
    if recognized_person != "Desconhecido" and recognized_person in celebrity_database:
        # Mostrar imagem da pessoa reconhecida
        sample_path = celebrity_database[recognized_person]['file_paths'][0]
        sample_img = load_and_preprocess_image(sample_path, augment=False)
        if sample_img is not None:
            plt.imshow((sample_img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])).clip(0, 1))
            plt.title(f' Identificado: {recognized_person}', fontweight='bold', color='green')
        else:
            plt.text(0.5, 0.5, f' {recognized_person}', ha='center', va='center',
                    fontsize=14, transform=plt.gca().transAxes)
            plt.title('Pessoa Identificada', fontweight='bold', color='green')
    else:
        plt.text(0.5, 0.5, ' Pessoa\nDesconhecida', ha='center', va='center',
                fontsize=14, transform=plt.gca().transAxes, color='red')
        plt.title('Não Identificado', fontweight='bold', color='red')
    plt.axis('off')

    # Gráfico de métricas
    plt.subplot(2, 3, 3)
    if metrics:
        metric_names = ['Cosine\nSimilarity', 'Euclidean\nSimilarity', 'Combined\nScore']
        metric_values = [metrics['cosine_similarity'], metrics['euclidean_similarity'], metrics['combined_score']]
        colors = ['skyblue', 'lightcoral', 'lightgreen']

        bars = plt.bar(metric_names, metric_values, color=colors)
        plt.title('📊 Métricas de Similaridade', fontweight='bold')
        plt.ylabel('Score')
        plt.ylim(0, 1)

        # Adicionar valores nos gráficos
        for bar, value in zip(bars, metric_values):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{value:.3f}', ha='center', va='bottom', fontweight='bold')

    # Top 5 candidatos
    query_embedding = embedding_network.predict(np.expand_dims(query_img, axis=0), verbose=0)[0]
    cosine_sims = cosine_similarity([query_embedding], all_embeddings)[0]
    euclidean_dists = euclidean_distances([query_embedding], all_embeddings)[0]
    euclidean_sims = 1 / (1 + euclidean_dists)
    combined_scores = 0.7 * cosine_sims + 0.3 * euclidean_sims

    top_indices = np.argsort(combined_scores)[::-1][:5]

    plt.subplot(2, 1, 2)
    top_names = [all_names[i] for i in top_indices]
    top_scores = [combined_scores[i] for i in top_indices]
    colors = ['gold', 'silver', '#CD7F32', 'lightblue', 'lightgray']

    bars = plt.barh(range(len(top_names)), top_scores, color=colors)
    plt.yticks(range(len(top_names)), [f"{i+1}. {name}" for i, name in enumerate(top_names)])
    plt.xlabel('Score de Similaridade')
    plt.title(' Top 5 Candidatos Mais Similares', fontweight='bold')
    plt.xlim(0, 1)

    # Adicionar valores nas barras
    for i, (bar, score) in enumerate(zip(bars, top_scores)):
        plt.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
                f'{score:.4f}', va='center', fontweight='bold')

    plt.tight_layout()
    plt.show()


    print("\n RELATÓRIO DETALHADO DO RECONHECIMENTO")
    print("="*50)
    print(f" Pessoa identificada: {recognized_person}")
    print(f" Score combinado: {similarity_score:.4f}")
    print(f" Similaridade cosseno: {metrics['cosine_similarity']:.4f}")
    print(f" Similaridade euclidiana: {metrics['euclidean_similarity']:.4f}")
    print(f" Posição no ranking: {metrics['rank_position']}")

    if similarity_score >= 0.35:
        print(f" Status: RECONHECIMENTO BEM-SUCEDIDO")
        print(f" Confiança: {similarity_score*100:.2f}%")
    else:
        print(f" Status: PESSOA NÃO RECONHECIDA")
        print(f" Score abaixo do threshold (35%)")

    print("\n ranking 5 primeiros:")
    print("-" * 45)
    for i, idx in enumerate(top_indices, 1):
        name = all_names[idx]
        score = combined_scores[idx]
        print(f"{i:2d}. {name:20s} | {score:.4f} ({score*100:.2f}%)")

    print("="*50)

else:
    print(" Imagem com máscara não encontrada!")

  
#Outras Análises
print("\n=== ESTATÍSTICAS DO MODELO ===")
print(f"Número total de celebridades no banco: {len(celebrity_database)}")
print(f"Dimensão dos vetores descritores: {EMBEDDING_DIM}")
print(f"Arquitetura base: ResNet50 + Camadas Dense")
print(f"Função de loss: Binary Crossentropy (Siamese Network)")
print(f"Métrica de similaridade: Similaridade do Cosseno")

# Avaliar performance do modelo nos dados de validação
print("\n=== AVALIAÇÃO NO CONJUNTO DE VALIDAÇÃO ===")
val_predictions = siamese_model.predict(X_val_split, verbose=0)
val_predictions_binary = (val_predictions > 0.5).astype(int).flatten()

from sklearn.metrics import classification_report, confusion_matrix

print("\nRelatório de Classificação:")
print(classification_report(y_val_split, val_predictions_binary))

print("\nMatriz de Confusão:")
cm = confusion_matrix(y_val_split, val_predictions_binary)
print(cm)



print("\n💾 Salvando modelo e resultados...")
# Salvar modelos
triplet_model.save('/content/triplet_model.h5')
embedding_network.save('/content/embedding_network.h5')

# Salvar banco de dados
with open('/content/celebrity_database_improved.pkl', 'wb') as f:
    pickle.dump(celebrity_database, f)

np.save('/content/celebrity_embeddings.npy', all_embeddings)
with open('/content/celebrity_names.pkl', 'wb') as f:
    pickle.dump(all_names, f)

# Salvar resultados finais
final_results = {
    'recognized_person': recognized_person,
    'similarity_score': similarity_score,
    'confidence_metrics': metrics,
    'model_architecture': 'EfficientNetB0 + Triplet Loss',
    'total_celebrities': len(celebrity_database),
    'embedding_dimension': EMBEDDING_DIM,
    'training_epochs': len(history.history['loss']),
    'final_loss': min(history.history['loss'])
}

with open('/content/final_results.pkl', 'wb') as f:
    pickle.dump(final_results, f)

print(f"\n RESULTADO FINAL: {recognized_person} (Score: {similarity_score:.4f})")
