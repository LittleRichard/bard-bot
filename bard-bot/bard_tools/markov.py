from keras.utils import np_utils
from keras.preprocessing.sequence import pad_sequences
import numpy as np
from random import randint
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, BatchNormalization, Embedding
from tensorflow.keras.constraints import max_norm
from tensorflow.keras.callbacks import (EarlyStopping as EarlyStopping,
                                        LambdaCallback)

from bard_tools.text_utils import get_text_stats


class Dataset:
    def __init__(self, word_to_indices, vocab_size):
        self.word_to_indices = word_to_indices
        self.vocab_size = vocab_size

        self.text_sequences = list()
        self.X_train = list()
        self.y_train = list()
        self.X_val = list()
        self.y_val = list()

    def make_dataset(self, words, input_sequence_length=10,
                     output_sequence_length=1):
        sequence_length = input_sequence_length + output_sequence_length
        tokens = self.__create_tokens(words, sequence_length)
        self.__create_training_and_validation_set(tokens)
        return self

    def __create_tokens(self, words, sequence_length):
        encoded_words = [self.word_to_indices[word] for word in words]
        tokens = list()
        for i in range(sequence_length, len(words)):
            line = ' '.join(words[i - sequence_length:i])
            self.text_sequences.append(line)
            tokens.append(encoded_words[i - sequence_length:i])
        return tokens

    def __create_training_and_validation_set(self, tokens):
        data = np.asarray(tokens)
        X, y = data[:, :-1], data[:, -1]
        y = np_utils.to_categorical(y, num_classes=self.vocab_size)
        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(X,
                                                                              y,
                                                                              test_size=0.2,
                                                                              shuffle=True)

    def get_random_sequence(self):
        return self.text_sequences[randint(0, len(self.text_sequences))]


# TODO: make config params out of the model inputs
class Model:
    def __init__(self, vocab_size, input_sequence_length):
        self.vocab_size = vocab_size
        self.input_sequence_length = input_sequence_length

        self.model = Sequential()
        self.__build_model()
        self.__compile_model()
        self.model.summary()

    def __build_model(self):
        self.model.add(Embedding(self.vocab_size, 50, input_length=self.input_sequence_length))
        self.model.add(LSTM(100, return_sequences=True, recurrent_initializer='glorot_uniform', kernel_constraint=max_norm(3)))
        self.model.add(BatchNormalization())
        self.model.add(LSTM(100, recurrent_initializer='glorot_uniform', kernel_constraint=max_norm(3)))
        self.model.add(BatchNormalization())
        self.model.add(Dense(self.vocab_size, activation='softmax'))

    def __compile_model(self):
        self.model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    def fit_model(self, x_train, y_train, validation_data, epochs, batch_size, callbacks):
       return self.model.fit(
           x_train,
           y_train,
           validation_data=validation_data,
           epochs=epochs,
           batch_size=batch_size,
           verbose=2,
           callbacks=callbacks
       )


def generate_text_from_model(model, seed, words_amount, word_to_indices,
                             input_sequence_length, indices_to_word):
    result = list()
    input_text = seed
    for _ in range(words_amount):
        encoded_text = [word_to_indices[word] for word in
                        input_text.split()]
        encoded_text = pad_sequences([encoded_text],
                                     maxlen=input_sequence_length,
                                     truncating='pre')
        predictions = model.predict_classes(encoded_text, verbose=0)
        predicted_word = indices_to_word[predictions[0]]
        input_text += ' ' + predicted_word
        result.append(predicted_word)
    result = ' '.join(result).replace(" ,", ",").replace(" .", ".\n")
    return result


def noop(*arg, **kwargs):
    pass


def markov_get_dataset_and_model(text,
                                 show_training_stage_test=False,
                                 send_output=noop):
    """ A bunch of Markov madness copied from the internet, organized here
    with minimal understanding of how it works.
    """

    unique_characters, words, vocab = get_text_stats(text)
    send_output('Unique characters ({}): {}'.format(
        len(unique_characters), unique_characters))
    send_output('Total words: {}'.format(len(words)))
    send_output('Unique words: {}'.format(len(vocab)))

    send_output('Building indices')
    word_to_indices = dict((w, i) for i, w in enumerate(vocab))
    indices_to_word = dict((i, w) for i, w in enumerate(vocab))

    # config?
    input_sequence_length = 10
    output_sequence_length = 1

    send_output('Creating dataset')
    dataset = Dataset(
        word_to_indices, len(vocab)
    ).make_dataset(
        words,
        input_sequence_length,
        output_sequence_length,
    )

    send_output('Total Sequences: {}'.format(len(dataset.text_sequences)))

    model = Model(len(vocab), input_sequence_length)

    early_stopping = EarlyStopping(
        monitor='val_loss',
        mode='min',
        verbose=2,
        patience=50
    )

    batch_size = 64
    epochs = 100

    callbacks = [early_stopping]
    if show_training_stage_test:
        seed_for_epochs = dataset.get_random_sequence()
        send_output("Checking stages with seed:", seed_for_epochs)

        def _on_epoch_end(epoch, _):
            send_output('-- Start generated text --\n')
            send_output(
                generate_text_from_model(
                    model.model,
                    seed_for_epochs,
                    50,  # words_amount
                    word_to_indices,
                    input_sequence_length,
                    indices_to_word
                )
            )
            send_output('\n-- End generated text --\n')

        callbacks.append(LambdaCallback(on_epoch_end=_on_epoch_end))

    send_output('Fitting model...')
    model.fit_model(
        dataset.X_train,
        dataset.y_train,
        validation_data=(dataset.X_val, dataset.y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks
    )

    return model, dataset
