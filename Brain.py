import random
import numpy as np
# import tensorflow as tf
from tensorflow import keras


class Brain:
    def __init__(self, person):
        model = keras.Sequential()
        model.add(keras.layers.Dense(8, input_shape=(8,)))  # input layer (1)
        model.add(keras.layers.Activation('tanh'))
        model.add(keras.layers.Dense(6, input_shape=(8,)))  # hidden layer (2)
        model.add(keras.layers.Activation('tanh'))
        model.add(keras.layers.Dense(4, input_shape=(6,)))  # hidden layer (2)
        model.add(keras.layers.Activation('tanh'))
        model.add(keras.layers.Dense(2, input_shape=(4,)))  # output layer (3)
        model.add(keras.layers.Activation('softmax'))
        model.compile(optimizer='adam',
                      loss='mse',
                      metrics=['accuracy'])
        self.model = model
        self.person = person

    def decision_making(self, gender, age, strength, pregnancy, biowatch, readyness, previous_choice):
        if random.randint(1, 10) > 3:
            # this list is for convinience, to know what does each index of option means.
            # choices = ["connection", "strength"]

            # flatten histories & prepare input
            neural_input = [gender, age, strength, pregnancy, biowatch, readyness]

            previous_choices = [0 for _ in range(2)]
            if previous_choice:
                previous_choices[previous_choice-1] = 1
            neural_input.extend(previous_choices)

            # handle input
            neural_input = np.asarray([neural_input])
            neural_input = np.atleast_2d(neural_input)

            # needs to preprocess the data to squash it between 0 and 1. ?

            # output
            output_prob: np.ndarray = self.model.predict(neural_input, batch_size=1, verbose=0)[0]

            return np.argmax(output_prob)
        else:
            return random.randint(0, 1)

    def gene_inhritance(self):
        father_weights = self.person.father.brain.model.get_weights()
        mother_weights = self.person.mother.brain.model.get_weights()

        # inheritance of parent's weights
        new_weights = father_weights
        for i in range(len(new_weights)):
            if random.uniform(0, 1) > 0.5:
                new_weights[i] = mother_weights[i]

        # mutation and personal uniqueness
        for i in new_weights:
            if random.uniform(0, 1) > 0.8:
                i += random.uniform(-0.5, 0.5)

        return new_weights

    def evolvement(self):
        new_weights: list[np.ndarray] = self.model.get_weights()

        for matrix in new_weights[::2]:
            matrix += np.array(
                [[(random.uniform(-0.01, 0.01) if random.random() < 0.2 else 0) for _ in range(matrix.shape[1])]
                for _ in range(matrix.shape[0])]
            )

        self.model.set_weights(new_weights)
