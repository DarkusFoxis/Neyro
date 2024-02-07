import tensorflow as tf
import numpy as np
import tflearn
import nltk
import json

nltk.download('punkt')

with open('data_kyaru.json', encoding='utf-8') as file:
    data = json.load(file)

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

training_data = []
for dialogues in data['dialogues']:
    for i in range(len(dialogues["dialogue"]) - 1):
        input_sentence = dialogues['dialogue'][i]
        output_sentence = dialogues['dialogue'][i + 1]

        input_words = nltk.word_tokenize(input_sentence)
        output_words = nltk.word_tokenize(output_sentence)

        input_words = [word for word in input_words if word.isalnum()]
        output_words = [word for word in output_words if word.isalnum()]

        input_vector = [0] * len(word_list)
        for word in input_words:
            if word in word_dict:
                index = word_dict[word]
                input_vector[index] = 1

        output_vector = [0] * len(word_list)
        for word in output_words:
            if word in word_dict:
                index = word_dict[word]
                output_vector[index] = 1

        training_data.append([input_vector, output_vector])

#Постройка модели нейронки...
net = tflearn.input_data(shape=[None, len(word_list)])
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, len(word_list), activation='softmax')
net = tflearn.regression(net)

#Объявляем модель...
model = tflearn.DNN(net)

#Обучение модели...
model.fit([data[0] for data in training_data], [data[1] for data in training_data], n_epoch=20000, batch_size=2, show_metric=True)

#Запрос при вопросе...
def get_response(question):
    question_words = nltk.word_tokenize(question)
    question_words = [word for word in question_words if word.isalnum()]

    question_vector = [0] * len(word_list)
    for word in question_words:
        if word in word_dict:
            index = word_dict[word]
            question_vector[index] = 1

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