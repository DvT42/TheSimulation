import random
import numpy as np
from tensorflow import keras


class Brain:
    def __init__(self, person):
        self.person = person
        self.brainparts = {"PFC": PrefrontalCortex(self.person),
                           "AMG": Amygdala(self.person),
                           "HPC": Hippocampus(self.person)}

    def inherit(self, person):
        for key, brainpart in self.brainparts.values():
            brainpart.model.set_weights(brainpart.gene_inheritance(person, key))

    def evolve(self):
        for brainpart in self.brainparts.values():
            brainpart.evolvement()

    def call_decision_making(self):
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

    def get_first_impression(self, other):
        impression = self.brainparts.get("AMG").first_impression(other)

        self.brainparts.get("HPC").first_impressions[f"{other.id}"] = impression
        self.brainparts.get("HPC").attitiudes[f"{other.id}"] = impression

        return impression


class BrainPart:
    def __init__(self):
        self.model = None

    def gene_inheritance(self, person, part: str):
        if self.model:
            father_weights = person.father.brain.brainparts.get(part).model.get_weights()
            mother_weights = person.mother.brain.brainparts.get(part).model.get_weights()

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
        if self.model:
            new_weights: list[np.ndarray] = self.model.get_weights()

            for matrix in new_weights[::2]:
                matrix += np.array(
                    [[(random.uniform(-0.01, 0.01) if random.random() < 0.2 else 0) for _ in range(matrix.shape[1])]
                    for _ in range(matrix.shape[0])]
                )

            self.model.set_weights(new_weights)


class PrefrontalCortex(BrainPart):
    """
    That is the part in the brain that is responsible for making decisions.
    """
    def __init__(self, person):
        super().__init__()
        # TODO: consider minimizing this model.
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


class Amygdala(BrainPart):
    """
    The part of the brain responsible for first impression.
    """
    def __init__(self, person):
        super().__init__()
        self.person = person

        model = keras.Sequential()
        model.add(keras.layers.Dense(16, input_shape=(16,)))  # input layer (1)
        model.add(keras.layers.Activation('tanh'))
        model.add(keras.layers.Dense(4, input_shape=(16,)))  # hidden layer (2)
        model.add(keras.layers.Activation('tanh'))
        model.add(keras.layers.Dense(1, input_shape=(4,)))  # output layer (3)
        model.add(keras.layers.Activation('sigmoid'))
        model.compile(optimizer='adam',
                      loss='mse',
                      metrics=['accuracy'])
        self.model = model

    def prepare_neural_input(self, other):
        # prepare the person's attributes.
        my_gender = self.person.gender
        my_age = self.person.age()
        my_strength = self.person.strength
        if my_gender == 0:
            my_pregnancy = self.person.pregnancy
            my_biowatch = self.person.biowatch
        else:
            my_pregnancy = 0
            my_biowatch = 0
        my_readiness = self.person.readiness

        # prepare the other person's attributes.
        other_gender = other.gender
        other_age = other.age()
        other_strength = other.strength
        if other_gender == 0:
            other_pregnancy = other.pregnancy
            other_biowatch = other.biowatch
        else:
            other_pregnancy = 0
            other_biowatch = 0
        other_readiness = other.readiness

        # prepare the difference between two people
        gender_dif = abs(my_gender - other_gender)
        age_dif = abs(my_age - other_age)
        strength_dif = abs(my_strength - other_strength)
        readiness_dif = abs(my_readiness - other_readiness)

        # organize all the information.
        neural_input = np.asarray([
            my_gender, my_age, my_strength, my_pregnancy, my_biowatch, my_readiness,
            other_gender, other_age, other_strength, other_pregnancy, other_biowatch, other_readiness,
            gender_dif, age_dif, strength_dif, readiness_dif
        ])
        neural_input = np.atleast_2d(neural_input)

        return neural_input

    def first_impression(self, other):
        output: np.ndarray = self.model.predict(self.prepare_neural_input(other), batch_size=1, verbose=0)[0]
        return output


class Hippocampus(BrainPart):
    """
    The part of the brain responsible for memory.
    """
    # TODO: transfer "person.history" into the Hippocampus.
    def __init__(self, person):
        super().__init__()
        self.person = person

        self.first_impressions = {}
        self.attitiudes = {}


def get_choice_nodes(choice):
    """Returns a list of zeroes with 1 at the index of the choice - 1, or no 1 at all if 0 is supplied"""

    nodes = [0 for _ in range(2)]
    if choice:
        nodes[choice - 1] = 1
    return nodes
