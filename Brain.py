import random
import numpy as np
from tensorflow import keras
from abc import ABC, abstractmethod


class Brain:
    def __init__(self, person):
        self.person = person
        self.brainparts = {"PFC": PrefrontalCortex(person)}

    def inherit(self, person):
        for brainpart in self.brainparts.values():
            brainpart.model.set_weights(brainpart.gene_inheritance(person))

    def decision_making(self):
        if self.person.gender == 0:
            preg = self.person.pregnancy
            biowatch = self.person.biowatch
        else:
            preg = 0
            biowatch = 0
        if self.person.isManual:
            return self.brainparts.get("PFC").decision_making(
                self.person.gender, self.person.age(), self.person.strength, preg, biowatch, self.person.readiness,
                self.person.history[self.person.age() - 1], 0, 0
            )
        else:
            return self.brainparts.get("PFC").decision_making(
                self.person.gender, self.person.age(), self.person.strength, preg, biowatch, self.person.readiness,
                self.person.history[self.person.age() - 1], self.person.father.history[self.person.age()],
                self.person.mother.history[self.person.age()]
            )

    def evolve(self):
        for brainpart in self.brainparts.values():
            brainpart.evolvement(brainpart.model)


class BrainPart(ABC):
    @abstractmethod
    def gene_inheritance(self, person):
        father_weights = person.father.brain.brainparts.get("PFC").model.get_weights()
        mother_weights = person.mother.brain.brainparts.get("PFC").model.get_weights()

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

    @abstractmethod
    def evolvement(self, model):
        new_weights: list[np.ndarray] = model.get_weights()

        for matrix in new_weights[::2]:
            matrix += np.array(
                [[(random.uniform(-0.01, 0.01) if random.random() < 0.2 else 0) for _ in range(matrix.shape[1])]
                for _ in range(matrix.shape[0])]
            )

        model.set_weights(new_weights)


class PrefrontalCortex(BrainPart):
    """
    That is the part in the brain that is responsible for making decisions.
    """

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
        self.person = person
        self.model = model

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

    def gene_inheritance(self, person):
        return super().gene_inheritance(person)

    def evolvement(self, model):
        super().evolvement(self.model)


def get_choice_nodes(choice):
    """Returns a list of zeroes with 1 at the index of the choice - 1, or no 1 at all if 0 is supplied"""

    nodes = [0 for _ in range(2)]
    if choice:
        nodes[choice - 1] = 1
    return nodes
