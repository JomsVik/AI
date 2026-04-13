import numpy as np
import matplotlib.pyplot as plt
import cv2
import utils
from collections import Counter
from typing import List
import os

# ============ НАСТРОЙКИ ============
NUM_OF_IMAGES = 4
KERNEL_SIZE = 4
OFFSET = 3

# ============ ФУНКЦИИ ДЛЯ ОБРАБОТКИ КАПЧ ============

def remove_line1(img: np.ndarray) -> np.ndarray:
    new_img = img.copy()
    for row in range(0, new_img.shape[0] - KERNEL_SIZE, OFFSET):
        for col in range(0, new_img.shape[1] - KERNEL_SIZE, OFFSET):
            conv = new_img[row:row + KERNEL_SIZE, col:col + KERNEL_SIZE]
            colors = np.zeros((KERNEL_SIZE, KERNEL_SIZE))
            for i in range(KERNEL_SIZE):
                for j in range(KERNEL_SIZE):
                    colors[i, j] = int(conv[i, j, 0]) + int(conv[i, j, 1]) + int(conv[i, j, 2])
            colors = list(Counter(colors.reshape(KERNEL_SIZE ** 2)))
            if len(colors) == 2 and (colors[0] == 765 or colors[1] == 765):
                new_img[row:row + KERNEL_SIZE, col:col + KERNEL_SIZE] = [255, 255, 255]
    return new_img


def remove_line2(number_image: np.ndarray) -> np.ndarray:
    new_img = cv2.cvtColor(number_image, cv2.COLOR_BGR2HSV)
    colors = np.zeros((new_img.shape[0], new_img.shape[1]), dtype=int)
    for row in range(new_img.shape[0]):
        for col in range(new_img.shape[1]):
            colors[row, col] = int(new_img[row, col, 0]) + int(new_img[row, col, 1]) + int(new_img[row, col, 2])
    colors = Counter(colors.reshape(colors.shape[0] * colors.shape[1]))
    colors.pop(255)
    if len(colors) > 0:
        colors.pop(max(colors, key=colors.get))
    for row in range(new_img.shape[0]):
        for col in range(new_img.shape[1]):
            for color in colors:
                if int(new_img[row, col, 0]) + int(new_img[row, col, 1]) + int(new_img[row, col, 2]) == color:
                    new_img[row, col] = [0, 0, 255]
    return cv2.cvtColor(new_img, cv2.COLOR_HSV2BGR)


def to_binary(img: np.ndarray) -> np.ndarray:
    if len(img.shape) == 3:
        new_img = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)
        for row in range(img.shape[0]):
            for col in range(img.shape[1]):
                if (img[row, col] == [255, 255, 255]).all():
                    new_img[row, col] = 0
                else:
                    new_img[row, col] = 1
        return new_img
    else:
        return (img > 127).astype(np.uint8)


def get_numbers(img: np.ndarray) -> List[np.ndarray]:
    w = img.shape[1] // NUM_OF_IMAGES
    numbers = []
    for i in range(NUM_OF_IMAGES):
        numbers.append(img[:, i * w: i * w + w])
    return numbers


def preprocess_captcha(img: np.ndarray, debug: bool = False) -> List[np.ndarray]:
    img = remove_line1(img)
    numbers = get_numbers(img)
    processed_numbers = []
    
    for i, number in enumerate(numbers):
        _, number = cv2.threshold(number, 127, 255, cv2.THRESH_BINARY)
        number = remove_line2(number)
        number = to_binary(number)
        number = cv2.resize(number.astype(np.float32), (28, 28))
        processed_numbers.append(number)
    
    return processed_numbers


# ============ НЕЙРОСЕТЬ ============

class NeuralNetwork:
    def __init__(self, input_size=784, hidden_size=20, output_size=10):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.weights_input_to_hidden = np.random.uniform(-0.5, 0.5, (hidden_size, input_size))
        self.weights_hidden_to_output = np.random.uniform(-0.5, 0.5, (output_size, hidden_size))
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
        """Возвращает предсказание и уверенность"""
        output = self.forward(X)
        predicted = np.argmax(output)
        confidence = np.max(output) * 100
        return predicted, confidence
    
    def train(self, images, labels, epochs=3, learning_rate=0.01):
        n_samples = len(images)
        for epoch in range(epochs):
            e_loss = 0
            e_correct = 0
            for i in range(n_samples):
                image = images[i].flatten().reshape(-1, 1)
                label = labels[i].reshape(-1, 1)
                
                hidden_raw = self.bias_input_to_hidden + self.weights_input_to_hidden @ image
                hidden = self.sigmoid(hidden_raw)
                output_raw = self.bias_hidden_to_output + self.weights_hidden_to_output @ hidden
                output = self.sigmoid(output_raw)
                
                e_loss += 1 / len(output) * np.sum((output - label) ** 2, axis=0)
                e_correct += int(np.argmax(output) == np.argmax(label))
                
                delta_output = output - label
                self.weights_hidden_to_output += -learning_rate * delta_output @ hidden.T
                self.bias_hidden_to_output += -learning_rate * delta_output
                
                delta_hidden = self.weights_hidden_to_output.T @ delta_output * (hidden * (1 - hidden))
                self.weights_input_to_hidden += -learning_rate * delta_hidden @ image.T
                self.bias_input_to_hidden += -learning_rate * delta_hidden
            
            loss_percent = (e_loss[0] / n_samples) * 100
            accuracy = (e_correct / n_samples) * 100
            print(f"Эпоха {epoch + 1}/{epochs} | Ошибка: {loss_percent:.3f}% | Точность: {accuracy:.2f}%")


# ============ ФУНКЦИИ ДЛЯ РАСПОЗНАВАНИЯ ============

def solve_captcha(nn: NeuralNetwork, captcha_path: str, visualize: bool = False) -> str:
    """Распознает капчу"""
    if isinstance(captcha_path, str):
        img = cv2.imread(captcha_path)
    else:
        img = captcha_path
    
    if img is None:
        return "Ошибка загрузки"
    
    numbers = preprocess_captcha(img)
    
    if visualize:
        fig, axes = plt.subplots(2, 4, figsize=(12, 6))
        fig.suptitle("Распознавание капчи", fontsize=14)
    
    answer = ""
    for i, number in enumerate(numbers):
        digit, conf = nn.predict_with_confidence(number.flatten())
        answer += str(digit)
        
        if visualize:
            axes[0, i].imshow(number, cmap='gray')
            axes[0, i].set_title(f"Цифра {i+1}")
            axes[0, i].axis('off')
            axes[1, i].text(0.5, 0.5, f"{digit}\n({conf:.0f}%)", 
                           ha='center', va='center', fontsize=14)
            axes[1, i].axis('off')
    
    if visualize:
        plt.tight_layout()
        plt.show()
    
    return answer


def diagnose_captcha(nn: NeuralNetwork, captcha_path: str):
    """Диагностика распознавания капчи"""
    img = cv2.imread(captcha_path)
    if img is None:
        print("Ошибка загрузки")
        return
    
    print(f"\n=== ДИАГНОСТИКА: {os.path.basename(captcha_path)} ===\n")
    
    numbers = preprocess_captcha(img)
    result = ""
    
    for i, number in enumerate(numbers):
        predicted, confidence = nn.predict_with_confidence(number.flatten())
        result += str(predicted)
        
        if confidence < 50:
            status = "⚠️ НИЗКАЯ"
        elif confidence < 80:
            status = "⚠️ СРЕДНЯЯ"
        else:
            status = "✓ ХОРОШАЯ"
        
        print(f"Цифра {i+1}: {predicted} (уверенность: {confidence:.1f}%) {status}")
        
        plt.figure(figsize=(3, 3))
        plt.imshow(number, cmap='gray')
        plt.title(f"Цифра {i+1} -> {predicted} ({confidence:.0f}%)")
        plt.axis('off')
        plt.show()
    
    print(f"\nРЕЗУЛЬТАТ: {result}")


def batch_test_captcha(nn: NeuralNetwork, captcha_folder="test_images"):
    """Массовое тестирование"""
    if not os.path.exists(captcha_folder):
        print(f"Папка {captcha_folder} не найдена")
        return
    
    captcha_files = [f for f in os.listdir(captcha_folder) 
                     if f.endswith(('.png', '.jpg', '.jpeg'))]
    
    if not captcha_files:
        print("Нет изображений")
        return
    
    print("\n" + "="*50)
    print("ТЕСТИРОВАНИЕ КАПЧ")
    print("="*50)
    
    for img_name in captcha_files:
        img_path = os.path.join(captcha_folder, img_name)
        result = solve_captcha(nn, img_path, visualize=False)
        print(f"{img_name}: {result}")


# ============ ИНТЕРАКТИВНЫЙ ПРОСМОТР ============

def interactive_digits_viewer(nn: NeuralNetwork, images, labels):
    """Просмотр цифр MNIST"""
    current_index = 0
    total = len(images)
    
    fig, ax = plt.subplots(figsize=(8, 8))
    plt.subplots_adjust(bottom=0.15)
    
    def update():
        ax.clear()
        img = images[current_index]
        true_label = np.argmax(labels[current_index])
        predicted, conf = nn.predict_with_confidence(img.flatten())
        
        color = 'green' if predicted == true_label else 'red'
        status = "✓" if predicted == true_label else "✗"
        
        ax.imshow(img.reshape(28, 28), cmap="Greys")
        ax.set_title(f"{current_index+1}/{total} | Нейросеть: {predicted} | Ответ: {true_label} | {status} | {conf:.0f}%", 
                    color=color, fontsize=12)
        ax.axis('off')
        fig.canvas.draw()
    
    def on_key(event):
        nonlocal current_index
        if event.key in ['right', '→']:
            current_index = (current_index + 1) % total
            update()
        elif event.key in ['left', '←']:
            current_index = (current_index - 1) % total
            update()
        elif event.key == 'escape':
            plt.close()
    
    fig.canvas.mpl_connect('key_press_event', on_key)
    update()
    plt.show()


def interactive_captcha_viewer(nn: NeuralNetwork, folder="test_images"):
    """Просмотр капч"""
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
    
    fig, ax = plt.subplots(figsize=(12, 8))
    plt.subplots_adjust(bottom=0.15)
    
    def update():
        ax.clear()
        img_path = os.path.join(folder, files[current])
        img = cv2.imread(img_path)
        
        if img is None:
            ax.text(0.5, 0.5, "Ошибка", ha='center', va='center')
            return
        
        numbers = preprocess_captcha(img)
        result = ""
        confs = []
        
        for num in numbers:
            pred, conf = nn.predict_with_confidence(num.flatten())
            result += str(pred)
            confs.append(conf)
        
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        ax.imshow(img_rgb)
        ax.set_title(f"{current+1}/{total} | Результат: {result} | Уверенность: {np.mean(confs):.0f}%", fontsize=12)
        ax.axis('off')
        
        fig.text(0.5, 0.02, '← → - листать | Esc - выход', ha='center', fontsize=10)
        fig.canvas.draw()
    
    def on_key(event):
        nonlocal current
        if event.key in ['right', '→']:
            current = (current + 1) % total
            update()
        elif event.key in ['left', '←']:
            current = (current - 1) % total
            update()
        elif event.key == 'escape':
            plt.close()
    
    fig.canvas.mpl_connect('key_press_event', on_key)
    update()
    plt.show()


# ============ ОСНОВНАЯ ПРОГРАММА ============

def main():
    print("\n" + "="*50)
    print("ЗАГРУЗКА И ОБУЧЕНИЕ НЕЙРОСЕТИ")
    print("="*50)
    
    images, labels = utils.load_dataset()
    
    nn = NeuralNetwork(input_size=784, hidden_size=20, output_size=10)
    nn.train(images, labels, epochs=3, learning_rate=0.01)
    
    while True:
        print("\n" + "="*50)
        print("ГЛАВНОЕ МЕНЮ")
        print("="*50)
        print("1 - Просмотр цифр MNIST (листать стрелками)")
        print("2 - Просмотр капч из test_images (листать стрелками)")
        print("3 - Быстрый тест всех капч (консоль)")
        print("4 - Диагностика одной капчи")
        print("5 - Выйти")
        print("="*50)
        
        choice = input("Ваш выбор: ")
        
        if choice == '1':
            interactive_digits_viewer(nn, images, labels)
        
        elif choice == '2':
            interactive_captcha_viewer(nn, "test_images")
        
        elif choice == '3':
            batch_test_captcha(nn, "test_images")
        
        elif choice == '4':
            path = input("Путь к капче: ")
            if os.path.exists(path):
                diagnose_captcha(nn, path)
            else:
                print("Файл не найден!")
        
        elif choice == '5':
            print("До свидания!")
            break
        
        else:
            print("Неверный выбор!")


if __name__ == "__main__":
    main()