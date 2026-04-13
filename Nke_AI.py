import numpy as np
import matplotlib.pyplot as plt
import utils

# Загрузка набора данных
print("Загрузка данных...")
images, labels = utils.load_dataset()
print(f"Загружено {len(images)} изображений")

# Инициализация весов и смещений случайными значениями
weights_input_to_hidden = np.random.uniform(-0.5, 0.5, (20, 784))  # веса от входного к скрытому слою
weights_hidden_to_output = np.random.uniform(-0.5, 0.5, (10, 20))  # веса от скрытого к выходному слою
bias_input_to_hidden = np.zeros((20, 1))   # смещения для скрытого слоя
bias_hidden_to_output = np.zeros((10, 1))  # смещения для выходного слоя

# Параметры обучения
epochs = 3          # количество эпох
e_loss = 0          # накопленная ошибка за эпоху
e_correct = 0       # количество правильных предсказаний за эпоху
learning_rate = 0.01  # скорость обучения

print("Начало обучения...")
print("-" * 50)

# Основной цикл обучения по эпохам
for epoch in range(epochs):
    print(f"Эпоха №{epoch + 1}")
    
    # Обучение на каждом примере
    for image, label in zip(images, labels):
        # Преобразование входных данных в вектор-столбец
        image = np.reshape(image, (-1, 1))
        label = np.reshape(label, (-1, 1))
        
        # === Прямое распространение (Forward propagation) ===
        
        # Скрытый слой
        hidden_raw = bias_input_to_hidden + weights_input_to_hidden @ image  # линейная комбинация
        hidden = 1 / (1 + np.exp(-hidden_raw))  # сигмоидная функция активации
        
        # Выходной слой
        output_raw = bias_hidden_to_output + weights_hidden_to_output @ hidden  # линейная комбинация
        output = 1 / (1 + np.exp(-output_raw))  # сигмоидная функция активации
        
        # === Расчет ошибки ===
        # Накопление среднеквадратичной ошибки для эпохи
        e_loss += 1 / len(output) * np.sum((output - label) ** 2, axis=0)
        # Подсчет правильных предсказаний (сравнение индексов максимальных значений)
        e_correct += int(np.argmax(output) == np.argmax(label))
        
        # === Обратное распространение (Backpropagation) ===
        
        # Выходной слой: градиент ошибки
        delta_output = output - label
        # Обновление весов и смещений выходного слоя
        weights_hidden_to_output += -learning_rate * delta_output @ np.transpose(hidden)
        bias_hidden_to_output += -learning_rate * delta_output
        
        # Скрытый слой: распространение ошибки назад
        delta_hidden = np.transpose(weights_hidden_to_output) @ delta_output * (hidden * (1 - hidden))
        # Обновление весов и смещений скрытого слоя
        weights_input_to_hidden += -learning_rate * delta_hidden @ np.transpose(image)
        bias_input_to_hidden += -learning_rate * delta_hidden
        
        # Конец обработки одного примера
    
    # Вывод статистики после завершения эпохи
    print(f"  Ошибка: {round((e_loss[0] / images.shape[0]) * 100, 3)}%")
    print(f"  Точность: {round((e_correct / images.shape[0]) * 100, 3)}%")
    print("-" * 50)
    
    # Сброс накопителей для следующей эпохи
    e_loss = 0
    e_correct = 0

print("Обучение завершено!")
print("\n" + "=" * 50)
print("ИНТЕРАКТИВНЫЙ РЕЖИМ ПРОСМОТРА")
print("=" * 50)
print("Управление:")
print("  → или Right - следующее изображение")
print("  ← или Left - предыдущее изображение")
print("  Esc - закрыть окно")
print("=" * 50)

# Функция предсказания для одного изображения
def predict(image):
    """
    Предсказывает цифру на изображении
    
    Параметры:
    image - входное изображение (784 пикселя)
    
    Возвращает:
    predicted_digit - предсказанная цифра (0-9)
    output - выходные значения нейросети
    """
    image = np.reshape(image, (-1, 1))
    hidden_raw = bias_input_to_hidden + weights_input_to_hidden @ image
    hidden = 1 / (1 + np.exp(-hidden_raw))
    output_raw = bias_hidden_to_output + weights_hidden_to_output @ hidden
    output = 1 / (1 + np.exp(-output_raw))
    return output.argmax(), output

# Интерактивный просмотр изображений
def interactive_view(images_list, start_index=0):
    """
    Создает интерактивное окно для просмотра изображений
    
    Параметры:
    images_list - список изображений для просмотра
    start_index - начальный индекс изображения
    """
    current_index = start_index
    
    # Создание фигуры для отображения
    fig, ax = plt.subplots(figsize=(8, 8))
    plt.subplots_adjust(bottom=0.15)
    
    # Функция обновления отображения
    def update_display():
        """Обновляет текущее изображение и предсказание"""
        ax.clear()
        test_image = images_list[current_index]
        predicted_digit, output = predict(test_image)
        
        # Получение вероятностей для каждого класса
        confidence = np.max(output) * 100  # уверенность в процентах
        
        # Отображение изображения
        ax.imshow(test_image.reshape(28, 28), cmap="Greys")
        
        # Создание заголовка с информацией
        title = (f"Изображение {current_index + 1}/{len(images_list)}\n"
                f"Нейросеть предполагает: {predicted_digit}\n"
                f"Уверенность: {confidence:.1f}%")
        
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.axis('off')
        
        # Добавление текста с подсказками по управлению внизу
        fig.text(0.5, 0.02, '← Предыдущее | Следующее → | Esc - выход', 
                ha='center', fontsize=10, style='italic')
        
        fig.canvas.draw()
    
    # Обработчик нажатий клавиш
    def on_key(event):
        nonlocal current_index
        if event.key == 'right' or event.key == '→':
            current_index = (current_index + 1) % len(images_list)
            update_display()
        elif event.key == 'left' or event.key == '←':
            current_index = (current_index - 1) % len(images_list)
            update_display()
        elif event.key == 'escape':
            plt.close()
            print("\nПрограмма завершена. Спасибо за использование!")
    
    # Подключение обработчика событий клавиатуры
    fig.canvas.mpl_connect('key_press_event', on_key)
    
    # Отображение первого изображения
    update_display()
    
    # Показ окна
    plt.show()

# Запуск интерактивного просмотра
try:
    interactive_view(images)
except KeyboardInterrupt:
    print("\nПрограмма прервана пользователем")
except Exception as e:
    print(f"\nПроизошла ошибка: {e}")