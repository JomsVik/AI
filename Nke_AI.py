import numpy as np
import matplotlib.pyplot as plt
import cv2
import utils
from collections import Counter
from typing import List
import os

# ============ НАСТРОЙКИ ============
NUM_OF_IMAGES = 4

# ============ УЛУЧШЕННАЯ ОБРАБОТКА КАПЧ ============

def remove_lines_and_noise(img):
    """Удаление линий и шума с капчи"""
    
    # Конвертируем в оттенки серого
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Инвертируем цвета (чтобы цифры были белыми на черном фоне)
    gray = cv2.bitwise_not(gray)
    
    # Медианный фильтр для удаления шума
    denoised = cv2.medianBlur(gray, 3)
    
    # Используем пороговую обработку для выделения цифр
    _, binary = cv2.threshold(denoised, 150, 255, cv2.THRESH_BINARY)
    
    # Удаляем маленькие объекты (шум)
    kernel = np.ones((2, 2), np.uint8)
    cleaned = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel)
    
    return cleaned


def split_into_digits(img):
    """Разделение капчи на отдельные цифры"""
    h, w = img.shape
    digit_width = w // NUM_OF_IMAGES
    digits = []
    
    for i in range(NUM_OF_IMAGES):
        # Вырезаем цифру
        digit = img[:, i * digit_width:(i + 1) * digit_width]
        
        # Находим границы цифры
        contours, _ = cv2.findContours(digit, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            # Находим bounding box цифры
            x, y, dw, dh = cv2.boundingRect(max(contours, key=cv2.contourArea))
            
            # Добавляем отступы
            padding = 4
            x1 = max(0, x - padding)
            y1 = max(0, y - padding)
            x2 = min(digit.shape[1], x + dw + padding)
            y2 = min(digit.shape[0], y + dh + padding)
            
            # Вырезаем и центрируем
            digit = digit[y1:y2, x1:x2]
        
        # Изменяем размер до 28x28
        if digit.size > 0:
            digit = cv2.resize(digit, (28, 28))
        else:
            digit = np.zeros((28, 28))
        
        # Нормализуем
        digit = digit.astype(np.float32) / 255.0
        
        digits.append(digit)
    
    return digits


def preprocess_captcha_complete(img):
    """Полная предобработка капчи"""
    # Удаляем линии и шум
    cleaned = remove_lines_and_noise(img)
    # Разделяем на цифры
    digits = split_into_digits(cleaned)
    return digits


# ============ НЕЙРОСЕТЬ ============

class NeuralNetwork:
    def __init__(self, input_size=784, hidden_size=128, output_size=10):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        
        # Инициализация весов
        np.random.seed(42)
        self.weights_input_to_hidden = np.random.uniform(-0.3, 0.3, (hidden_size, input_size))
        self.weights_hidden_to_output = np.random.uniform(-0.3, 0.3, (output_size, hidden_size))
        self.bias_input_to_hidden = np.zeros((hidden_size, 1))
        self.bias_hidden_to_output = np.zeros((output_size, 1))
    
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))
    
    def forward(self, X):
        X = np.reshape(X, (-1, 1))
        hidden_raw = self.bias_input_to_hidden + self.weights_input_to_hidden @ X
        hidden = self.sigmoid(hidden_raw)
        output_raw = self.bias_hidden_to_output + self.weights_hidden_to_output @ hidden
        output = self.sigmoid(output_raw)
        return output
    
    def predict(self, X):
        output = self.forward(X)
        return np.argmax(output)
    
    def predict_with_confidence(self, X):
        output = self.forward(X)
        predicted = np.argmax(output)
        confidence = np.max(output) * 100
        return predicted, confidence
    
    def train(self, images, labels, epochs=10, learning_rate=0.01):
        n_samples = len(images)
        
        for epoch in range(epochs):
            e_loss = 0
            e_correct = 0
            
            for i in range(n_samples):
                image = images[i].flatten().reshape(-1, 1)
                label = labels[i].reshape(-1, 1)
                
                # Forward
                hidden_raw = self.bias_input_to_hidden + self.weights_input_to_hidden @ image
                hidden = self.sigmoid(hidden_raw)
                output_raw = self.bias_hidden_to_output + self.weights_hidden_to_output @ hidden
                output = self.sigmoid(output_raw)
                
                # Loss
                e_loss += 1 / len(output) * np.sum((output - label) ** 2, axis=0)
                e_correct += int(np.argmax(output) == np.argmax(label))
                
                # Backward
                delta_output = output - label
                self.weights_hidden_to_output += -learning_rate * delta_output @ hidden.T
                self.bias_hidden_to_output += -learning_rate * delta_output
                
                delta_hidden = self.weights_hidden_to_output.T @ delta_output * (hidden * (1 - hidden))
                self.weights_input_to_hidden += -learning_rate * delta_hidden @ image.T
                self.bias_input_to_hidden += -learning_rate * delta_hidden
            
            loss_percent = (e_loss[0] / n_samples) * 100
            accuracy = (e_correct / n_samples) * 100
            print(f"Эпоха {epoch + 1}/{epochs} | Ошибка: {loss_percent:.2f}% | Точность: {accuracy:.2f}%")
        
        return self


# ============ ФУНКЦИИ РАСПОЗНАВАНИЯ ============

def solve_captcha(nn, img_path):
    """Распознавание капчи"""
    img = cv2.imread(img_path)
    if img is None:
        return "ОШИБКА", []
    
    # Предобработка
    digits = preprocess_captcha_complete(img)
    
    # Распознавание каждой цифры
    result = ""
    confidences = []
    
    for digit in digits:
        pred, conf = nn.predict_with_confidence(digit.flatten())
        result += str(pred)
        confidences.append(conf)
    
    return result, confidences, digits


def show_captcha_result(nn, img_path):
    """Отображение результата распознавания капчи"""
    
    img = cv2.imread(img_path)
    if img is None:
        print("Ошибка загрузки")
        return
    
    result, confidences, digits = solve_captcha(nn, img_path)
    
    # Создаем окно
    plt.figure(figsize=(14, 8))
    plt.suptitle(f"РАСПОЗНАВАНИЕ КАПЧИ", fontsize=18, fontweight='bold')
    
    # Оригинальная капча
    plt.subplot(2, 3, 1)
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title("Оригинальная капча", fontsize=12)
    plt.axis('off')
    
    # Обработанная капча
    cleaned = remove_lines_and_noise(img)
    plt.subplot(2, 3, 2)
    plt.imshow(cleaned, cmap='gray')
    plt.title("После обработки", fontsize=12)
    plt.axis('off')
    
    # Результат
    plt.subplot(2, 3, 3)
    plt.axis('off')
    avg_conf = np.mean(confidences)
    
    result_text = f"""
╔═════════════════════╗
║     РЕЗУЛЬТАТ       ║
╠═════════════════════╣
║                     ║
║     {result}           
║                     ║
║  Уверенность:       
║     {avg_conf:.1f}%      
║                     ║
╚═════════════════════╝
"""
    plt.text(0.5, 0.5, result_text, fontsize=14, ha='center', va='center',
             fontfamily='monospace', fontweight='bold')
    
    # Отдельные цифры
    for i, (digit, conf) in enumerate(zip(digits, confidences)):
        plt.subplot(2, 4, 5 + i)
        plt.imshow(digit, cmap='gray')
        
        if conf >= 70:
            color = 'green'
        elif conf >= 50:
            color = 'orange'
        else:
            color = 'red'
        
        plt.title(f"Цифра {i+1}\n{result[i]} ({conf:.0f}%)", color=color, fontsize=10)
        plt.axis('off')
    
    plt.tight_layout()
    plt.show()
    
    return result, confidences


def interactive_captcha_viewer(nn, folder="test_images"):
    """Интерактивный просмотр капч"""
    
    if not os.path.exists(folder):
        print(f"Папка {folder} не найдена")
        return
    
    files = [f for f in os.listdir(folder) if f.endswith(('.png', '.jpg', '.jpeg'))]
    if not files:
        print("Нет изображений")
        return
    
    files.sort()
    current = 0
    total = len(files)
    
    def show_current():
        img_path = os.path.join(folder, files[current])
        img = cv2.imread(img_path)
        
        if img is None:
            return
        
        result, confidences, digits = solve_captcha(nn, img_path)
        avg_conf = np.mean(confidences)
        
        plt.clf()
        plt.suptitle(f"РАСПОЗНАВАНИЕ КАПЧИ - {files[current]}", fontsize=16, fontweight='bold')
        
        # Оригинал
        plt.subplot(2, 3, 1)
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        plt.title(f"Капча {current + 1} из {total}", fontsize=12)
        plt.axis('off')
        
        # Обработанная
        cleaned = remove_lines_and_noise(img)
        plt.subplot(2, 3, 2)
        plt.imshow(cleaned, cmap='gray')
        plt.title("Обработанное изображение", fontsize=12)
        plt.axis('off')
        
        # Результат
        plt.subplot(2, 3, 3)
        plt.axis('off')
        
        result_text = f"""
╔═════════════════════╗
║    РЕЗУЛЬТАТ        ║
╠═════════════════════╣
║                     ║
║      {result}         
║                     ║
║  Уверенность:       
║     {avg_conf:.1f}%      
║                     ║
╚═════════════════════╝
"""
        plt.text(0.5, 0.5, result_text, fontsize=14, ha='center', va='center',
                fontfamily='monospace', fontweight='bold')
        
        # Цифры
        for i, (digit, conf) in enumerate(zip(digits, confidences)):
            plt.subplot(2, 4, 5 + i)
            plt.imshow(digit, cmap='gray')
            
            if conf >= 70:
                color = 'green'
            elif conf >= 50:
                color = 'orange'
            else:
                color = 'red'
            
            plt.title(f"{result[i]} ({conf:.0f}%)", color=color, fontsize=14, fontweight='bold')
            plt.axis('off')
        
        # Управление
        plt.figtext(0.5, 0.02, '◀  Левая стрелка  |  Правая стрелка  ▶  |  ESC - выход',
                   ha='center', fontsize=12, style='italic')
        
        plt.draw()
    
    def on_key(event):
        nonlocal current
        if event.key in ['right', '→']:
            current = (current + 1) % total
            show_current()
        elif event.key in ['left', '←']:
            current = (current - 1) % total
            show_current()
        elif event.key == 'escape':
            plt.close()
    
    plt.figure(figsize=(14, 8))
    plt.gcf().canvas.mpl_connect('key_press_event', on_key)
    show_current()
    plt.show()


# ============ ОБУЧЕНИЕ И ТЕСТИРОВАНИЕ ============

def augment_data(images, labels):
    """Аугментация данных для улучшения обучения"""
    augmented_images = []
    augmented_labels = []
    
    for img, label in zip(images, labels):
        # Оригинал
        augmented_images.append(img)
        augmented_labels.append(label)
        
        # Поворот на +5 градусов
        from scipy.ndimage import rotate
        rotated = rotate(img.reshape(28, 28), 5, reshape=False)
        rotated = rotated.flatten()
        augmented_images.append(rotated)
        augmented_labels.append(label)
        
        # Поворот на -5 градусов
        rotated = rotate(img.reshape(28, 28), -5, reshape=False)
        rotated = rotated.flatten()
        augmented_images.append(rotated)
        augmented_labels.append(label)
        
        # Сдвиг
        from scipy.ndimage import shift
        shifted = shift(img.reshape(28, 28), [1, 1], mode='constant', cval=0)
        shifted = shifted.flatten()
        augmented_images.append(shifted)
        augmented_labels.append(label)
    
    return np.array(augmented_images), np.array(augmented_labels)


def main():
    print("\n" + "="*60)
    print("🤖 ЗАГРУЗКА И ОБУЧЕНИЕ НЕЙРОСЕТИ")
    print("="*60)
    
    # Загрузка данных
    print("\nЗагрузка данных MNIST...")
    images, labels = utils.load_dataset()
    
    # Аугментация
    print("Аугментация данных...")
    images_aug, labels_aug = augment_data(images[:10000], labels[:10000])
    print(f"Создано {len(images_aug)} обучающих примеров")
    
    # Обучение
    print("\nОбучение нейросети...")
    nn = NeuralNetwork(input_size=784, hidden_size=128, output_size=10)
    nn.train(images_aug, labels_aug, epochs=10, learning_rate=0.01)
    
    # Меню
    while True:
        print("\n" + "="*50)
        print("ГЛАВНОЕ МЕНЮ")
        print("="*50)
        print("1 - Показать результат капчи")
        print("2 - Интерактивный просмотр капч")
        print("3 - Выйти")
        print("="*50)
        
        choice = input("Ваш выбор: ")
        
        if choice == '1':
            path = input("Путь к капче: ")
            if os.path.exists(path):
                show_captcha_result(nn, path)
            else:
                print("Файл не найден!")
        
        elif choice == '2':
            interactive_captcha_viewer(nn, "test_images")
        
        elif choice == '3':
            print("До свидания!")
            break


if __name__ == "__main__":
    main()