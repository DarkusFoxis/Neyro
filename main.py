import tensorflow as tf
import numpy as np
import tflearn
import nltk
import json

nltk.download('punkt')

with open('data.json', encoding='utf-8') as file:
    data = json.load(file)

# Загрузка и создание предобученных векторов слов
embeddings_index = {}
embedding_dim = 100  # Размерность вектора слов
with open('glove.6B.100d.txt', encoding='utf-8') as f:
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs

words = []
for dialogue in data['dialogues']:
    print(dialogue)  # Убедиться, что данные в порядке
    # Проверить тип dialogue
    print(type(dialogue))
    for sentence in dialogue['dialogue']:
        sentence_words = nltk.word_tokenize(sentence)
        words.extend(sentence_words)
words = [word for word in words if word.isalnum()]
words = [word.lower() for word in words]

word_freq = nltk.FreqDist(words)

common_words = word_freq.most_common(1000)
word_list = [word[0] for word in common_words]
word_dict = {word: index for index, word in enumerate(word_list)}

max_sequence_length = 48
max_output_length = 201
training_data = []

for dialogues in data['dialogues']:
    for i in range(len(dialogues["dialogue"]) - 1):
        input_sentence = dialogues['dialogue'][i]
        output_sentence = dialogues['dialogue'][i + 1]

        input_words = nltk.word_tokenize(input_sentence)
        output_words = nltk.word_tokenize(output_sentence)

        input_words = [word for word in input_words if word.isalnum()]
        output_words = [word for word in output_words if word.isalnum()]

        input_vector = [0] * max_sequence_length
        for i, word in enumerate(input_words):
            if i >= max_sequence_length:
                break
            if word in word_dict:
                index = word_dict[word]
                input_vector[i] = index

        output_vector = [0] * max_output_length
        for i, word in enumerate(output_words):
            if i >= max_output_length:
                break
            if word in word_dict:
                index = word_dict[word]
                output_vector[i] = index

        training_data.append([input_vector, output_vector])

print(training_data)

# Построение GloVe векторов и матрицы весов для embedding слоя
embedding_matrix = np.zeros((len(word_list), embedding_dim), dtype=np.float32)  # Инициализация матрицы с нужным типом данных
for word, i in word_dict.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector.astype(np.float32)

tensor_embedding = tf.Variable(initial_value=embedding_matrix, dtype=tf.float32, trainable=False)

max_sequence_length = max(len(nltk.word_tokenize(sentence)) for dialogue in data['dialogues'] for sentence in dialogue['dialogue'])

input_dim = len(embeddings_index)  # Размер словаря, input_dim = len(word_list)
output_dim = 100  # Предполагаемый размер эмбеддингов (может отличаться в зависимости от GloVe векторов)
word_to_check = 'example_word'
if word_to_check in embeddings_index:
    output_dim = len(embeddings_index[word_to_check])
    # далее используете output_dim
else:
    print(f"Слово '{word_to_check}' отсутствует в embeddings_index")
trainable = False  # Предполагаем, что эмбеддинги не обучаемы
weights_init = tf.constant(embedding_matrix, dtype=tf.float32)

# Создаем input_data для модели
net = tflearn.input_data(shape=[None, max_sequence_length])

# Создаем слой Embedding
net = tflearn.embedding(net, input_dim=input_dim, output_dim=output_dim, weights_init=weights_init, trainable=trainable)

# Добавим LSTM слои для обработки текста
net = tflearn.lstm(net, 64, return_seq=True)
net = tflearn.dropout(net, 0.5)
net = tflearn.lstm(net, 64)
net = tflearn.dropout(net, 0.5)

# Добавим полностью связанный слой для вывода
net = tflearn.fully_connected(net, len(word_list), activation='softmax')

# Укажем функцию потерь и метод оптимизации
net = tflearn.regression(net, optimizer='adam', loss='categorical_crossentropy', learning_rate=0.001)

#Объявляем, и обучаем модель...
model = tflearn.DNN(net)
model.fit([data[0] for data in training_data], [data[1] for data in training_data], n_epoch=5000, batch_size=4, show_metric=True)

#Запрос при вопросе...
def get_response(question):
    question_words = nltk.word_tokenize(question)
    question_words = [word for word in question_words if word.isalnum()]

    max_input_length = 48  # Максимальная длина входной последовательности

    question_vector = [0] * max_input_length
    for i, word in enumerate(question_words):
        if i >= max_input_length:
            break
        if word in word_dict:
            index = word_dict[word]
            question_vector[i] = index

    prediction = model.predict([question_vector])[0]
    responce_vertor = np.zeros(len(word_list))
    responce_vertor[np.argmax(prediction)] = 1

    responce_words = []
    for index, value in enumerate(responce_vertor):
        if value == 1:
            responce_words.append(word_list[index])

    responce = ' '.join(responce_words)
    return responce

while True:
    question = input("Вы: ")
    response = get_response(question)
    print("AI: " + response)