import random
import numpy as np
# import tensorflow as tf
from tensorflow import keras


class Brain:
    def __init__(self, person):
        model = keras.Sequential()
        model.add(keras.layers.Dense(3607, input_shape=(3607,)))  # input layer (1)
        model.add(keras.layers.Activation('sigmoid'))
        model.add(keras.layers.Dense(60, input_shape=(3607,)))  # hidden layer (2)
        model.add(keras.layers.Activation('relu'))
        model.add(keras.layers.Dense(1, input_shape=(60,)))  # output layer (3)
        model.add(keras.layers.Activation('sigmoid'))
        model.compile(optimizer='adam',
                      loss='mse',
                      metrics=['accuracy'])
        self.model = model
        self.person = person

    def decision_making(self, gender, age, strength, pregnancy, biowatch, readyness, father_his, mother_his, his,
                        previous_choice):
        if random.uniform(0, 1) > 0.3:
            # this list is for convinience, to know what does each index of option means.
            # choices = ["connection", "strength"]

            # flatten histories & prepare input
            neural_input = [gender, age, strength, pregnancy, biowatch, readyness, previous_choice]
            for i in father_his:
                neural_input.append(i)
            for i in mother_his:
                neural_input.append(i)
            for i in his:
                neural_input.append(i)

            # handle input
            neural_input = np.asarray([neural_input])
            neural_input = np.atleast_2d(neural_input)

            # needs to preprocess the data to squash it between 0 and 1. ?

            # output
            output_prob = self.model.predict(neural_input, 1, verbose=0)[0]

            if output_prob[0] <= .5:
                return 0
            else:
                return 1
        else:
            if random.uniform(0, 1) > 0.5:
                return 0
            else:
                return 1

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
        new_weights = self.model.get_weights()
        for i in new_weights:
            i += random.uniform(-0.05, 0.05)
        self.model.set_weights(new_weights)