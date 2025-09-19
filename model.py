import tensorflow as tf
from tensorflow import keras
from keras import layers
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.model_selection import train_test_split
import json
from PIL import Image, ImageDraw, ImageFont
import random
import string


def setup_gpu():
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            tf.keras.mixed_precision.set_global_policy('mixed_float16')
        except RuntimeError as e:
            print(e)
    return len(gpus) > 0

if tf.test.is_gpu_available():
    print("GPU is available.")
setup_gpu()

class HandwritingDataGenerator:
    def __init__(self, output_dir = 'training_data'):
        self.output_dir = output_dir
        self.vocab = string.ascii_letters + string.digits + " .,/$()-"
        self.max_len = 50
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(f"{output_dir}/images", exist_ok=True)

    def generate_synthetic_handwriting(self, text, variations=5):
        images = []
        for i in range(variations):
            img = Image.new('RGB', (400,60), 'white')
            draw = ImageDraw.Draw(img)
            x_offset = random.randint(5, 20)
            y_offset = random.randint(10,20)
            for j, char in enumerate(text):
                char_x = x_offset + j * random.randint(12, 18)
                char_y = y_offset + random.randint(-3, 3)
                font_size = random.randint(16, 24)
                try:
                    font = ImageFont.truetype("arial.ttf", font_size)
                except:
                    font = ImageFont.load_default()
                draw.text((char_x, char_y), char, fill='black', font=font)

            img_array = np.array(img)
            img_array = cv2.GaussianBlur(img_array, (1, 1), 0)
            noise = np.random.normal(0, 5, img_array.shape)
            img_array = np.clip(img_array + noise, 0, 255).astype(np.uint8)
            images.append(img_array)
        return images
    
    def create_training_dataset(self, num_samples=5000):
        invoice_terms = [
            "pumped septic tank", "locate headyard", "second compartment",
            "cleaned baffle", "inspect system", "pump waste", "clean filter",
            "repair pipe", "install valve", "check pressure", "maintenance service",
            "emergency call", "weekend service", "after hours", "travel time",
            "parts and labor", "service call", "diagnostic fee", "permit fee"
        ]
        names = ["John Smith", "Mary Johnson", "Bob Wilson", "Lisa Davis", "Mike Brown"]
        addresses = ["123 Main St", "456 Oak Ave", "789 Pine Rd", "321 Elm Dr"]
        numbers = [str(random.randint(100, 999)) for _ in range(100)]
        amounts = [f"${random.randint(50, 500)}.{random.randint(10, 99)}" for _ in range(100)]
        all_texts = invoice_terms + names + addresses + numbers + amounts
        dataset = []
        labels = []
        print(f"Generating {num_samples} training samples...")

        for i, text in enumerate(random.choices(all_texts, k=num_samples)):
            images = self.generate_synthetic_handwriting(text, variations=1)
            for img in images:
                gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
                resized = cv2.resize(gray, (200, 32))
                normalized = resized.astype(np.float32) / 255.0
                dataset.append(normalized)
                labels.append(text)
            if i % 500 == 0:
                print(f"Generated {i} samples...")
        return np.array(dataset), labels

class HandwritingOCRModel:
    def __init__(self, vocab, max_length=50):
        self.vocab = vocab
        self.char_to_idx = {char: idx for idx, char in enumerate(vocab)}
        self.idx_to_char = {idx: char for idx, char in enumerate(vocab)}
        self.max_length = max_length
        self.model = None
    
    def encode_labels(self, labels):
        sequence_length = 25 
        encoded = []
        for label in labels:
            sequence = [self.char_to_idx.get(char, 0) for char in label[:sequence_length]]
            sequence += [0] * (sequence_length - len(sequence))
            encoded.append(sequence)
        return np.array(encoded)
    
    def decode_prediction(self, prediction):
        chars = []
        for idx in prediction:
            if idx > 0 and idx < len(self.vocab):
                chars.append(self.idx_to_char[idx])
        return ''.join(chars).strip()

    def build_model(self):
        input_img = layers.Input(shape=(32, 200, 1), name='image')
        x = layers.Conv2D(32, (3,3), activation='relu', padding='same')(input_img)
        x = layers.MaxPooling2D((2,2))(x)
        x = layers.Conv2D(64, (3,3), activation='relu', padding='same')(x)
        x = layers.MaxPooling2D((2,2))(x)
        x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
        x = layers.MaxPooling2D((2,2))(x)

        new_shape = ((200 // 8), (32 // 8) * 128)
        x = layers.Reshape(target_shape=new_shape)(x)
        x = layers.Dense(64, activation='relu')(x)
        x = layers.Bidirectional(layers.LSTM(128, return_sequences=True))(x)
        x = layers.Bidirectional(layers.LSTM(64, return_sequences=True))(x)
        output = layers.Dense(len(self.vocab), activation='softmax', name='output')(x)
        self.model = keras.Model(inputs=input_img, outputs=output)
        return self.model
    
    def compile_model(self, learning_rate=0.001):
        if tf.test.is_gpu_available():
            optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
            optimizer = keras.mixed_precision.LossScaleOptimizer(optimizer)
        else:
            optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
        self.model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    def train(self, x_train, y_train, x_val, y_val, epochs=50):
        callbacks = [
            keras.callbacks.EarlyStopping(patience=15, restore_best_weights=True, monitor='val_loss'),
            keras.callbacks.ReduceLROnPlateau(patience=8, factor=0.5, min_lr=1e-7, monitor='val_loss'),
            keras.callbacks.ModelCheckpoint('best_handwriting_model.h5', save_best_only=True, monitor='val_loss'),
        ]
        
        x_train = x_train.reshape(-1, 32, 200, 1)
        x_val = x_val.reshape(-1, 32, 200, 1)
        
        batch_size = 64 if tf.test.is_gpu_available() else 16
        
        train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
        train_dataset = train_dataset.shuffle(1000).batch(batch_size).prefetch(tf.data.AUTOTUNE)
        
        val_dataset = tf.data.Dataset.from_tensor_slices((x_val, y_val))
        val_dataset = val_dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
        
        with tf.device('/GPU:0' if tf.test.is_gpu_available() else '/CPU:0'):
            history = self.model.fit(
                train_dataset,
                validation_data=val_dataset,
                epochs=epochs,
                callbacks=callbacks,
                verbose=1
            )
        return history
    
    def predict_text(self, image):
        if len(image.shape) == 2:
            image = image.reshape(1, 32, 200, 1)
        elif len(image.shape) == 3:
            image = image.reshape(1, 32, 200, 1)
        prediction = self.model.predict(image, verbose=0)

        predicted_chars = np.argmax(prediction[0], axis=-1)
        return self.decode_prediction(predicted_chars)
    
    def save_model(self, filepath):
        self.model.save(filepath)
        with open(filepath.replace('.h5', '_vocab.json'), 'w') as f:
            json.dump({
                'vocab': self.vocab,
                'char_to_idx': self.char_to_idx,
                'idx_to_char': self.idx_to_char,
                'max_length': self.max_length
            }, f)
    
    def load_model(self, filepath):
        self.model = keras.models.load_model(filepath)
        with open(filepath.replace('.h5', '_vocab.json'), 'r') as f:
            vocab_data = json.load(f)
            self.vocab = vocab_data['vocab']
            self.char_to_idx = vocab_data['char_to_idx']
            self.idx_to_char = {int(k): v for k, v in vocab_data['idx_to_char'].items()}
            self.max_length = vocab_data['max_length']


def train_custom_ocr_model():
    print('starting training process...')
    print("=" * 60)

    data_gen = HandwritingDataGenerator()
    x, y = data_gen.create_training_dataset(num_samples=2000)
    x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.2, random_state=42)
    print(f"Training samples: {len(x_train)}, Validation samples: {len(x_val)}")
    vocab = string.ascii_letters + string.digits + " .,/$()-"
    ocr_model = HandwritingOCRModel(vocab=vocab)
    model = ocr_model.build_model()
    ocr_model.compile_model()
    print("model architecture:")
    model.summary()

    y_train_encoded = ocr_model.encode_labels(y_train)
    y_val_encoded = ocr_model.encode_labels(y_val)

    print("starting training")
    history = ocr_model.train(x_train, y_train_encoded, x_val, y_val_encoded, epochs=30)

    ocr_model.save_model('handwriting_ocr_model.h5')
    return ocr_model, history

def test_custom_model():
    ocr_model = HandwritingOCRModel(string.ascii_letters + string.digits + " .,/$()-")
    ocr_model.load_model('handwriting_ocr_model.h5')
    form_img = cv2.imread('form_img.jpeg')
    if form_img is not None:
        x1, y1, x2, y2 = (71, 338, 635, 370)
        region = form_img[y1:y2, x1:x2]
        gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, (200, 32))
        normalized = resized.astype(np.float32) / 255.0

        predicted_text = ocr_model.predict_text(normalized)
        print(f"Predicted text: {predicted_text}")
        cv2.imwrite('test_region.png', region)
        cv2.imwrite('test_processed.png', (normalized * 255).astype(np.uint8))

if __name__ == "__main__":
    if os.path.exists('handwriting_ocr_model.h5'):
        test_custom_model()
    else:
        ocr_model, history = train_custom_ocr_model()
        test_custom_model()