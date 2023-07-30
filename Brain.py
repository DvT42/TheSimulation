import random
import numpy as np
# import tensorflow as tf
from tensorflow import keras


class Brain:
    def __init__(self, person):
        model = keras.Sequential()
        model.add(keras.layers.Dense(12, input_shape=(12,)))  # input layer (1)
        model.add(keras.layers.Activation('tanh'))
        model.add(keras.layers.Dense(6, input_shape=(12,)))  # hidden layer (2)
        model.add(keras.layers.Activation('tanh'))
        model.add(keras.layers.Dense(4, input_shape=(6,)))  # hidden layer (3)
        model.add(keras.layers.Activation('tanh'))
        model.add(keras.layers.Dense(2, input_shape=(4,)))  # output layer (4)
        model.add(keras.layers.Activation('softmax'))
        model.compile(optimizer='adam',
                      loss='mse',
                      metrics=['accuracy'])
        self.model = model
        self.person = person

    def decision_making(
            self, gender, age, strength, pregnancy, biowatch, readiness, previous_choice, father_choice,
            mother_choice
    ):
        if random.randint(1, 10) > 3:
            # this list is for convinience, to know what does each index of option means.
            # choices = ["connection", "strength"]

            # flatten histories & prepare input
            neural_input = [gender, age, strength, pregnancy, biowatch, readiness]

            neural_input.extend(get_choice_nodes(previous_choice))
            neural_input.extend(get_choice_nodes(father_choice))
            neural_input.extend(get_choice_nodes(mother_choice))

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


def get_choice_nodes(choice):
    """Returns a list of zeroes with 1 at the index of the choice - 1, or no 1 at all if 0 is supplied"""

    nodes = [0 for _ in range(2)]
    if choice:
        nodes[choice - 1] = 1
    return nodes
