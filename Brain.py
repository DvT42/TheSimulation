import random
import numpy as np
from Neural_Network import NeuralNetwork


class Brain:
    def __init__(self, person):
        self.person = person
        self.brainparts = {"PFC": PrefrontalCortex(self.person),
                           "AMG": Amygdala(self.person),
                           "HPC": Hippocampus(self.person)}

    def inherit(self, person):
        for key, brainpart in self.brainparts.items():
            if brainpart.model:
                new_weights, new_biases = brainpart.gene_inheritance(person, key)
                brainpart.model.set_weights(new_weights)
                brainpart.model.set_biases(new_biases)

    def evolve(self):  # Evolvement currently on hold
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
                self.get_action_from_history(self.person.age() - 1), 0, 0
            )
        else:
            return self.brainparts.get("PFC").decision_making(
                self.person.gender, self.person.age(), self.person.strength, preg, biowatch, self.person.readiness,
                self.get_action_from_history(self.person.age() - 1),
                self.person.father.brain.get_action_from_history(self.person.age()),
                self.person.mother.brain.get_action_from_history(self.person.age())
            )

    def get_first_impression(self, other):
        impression = self.brainparts.get("AMG").first_impression(other)

        self.brainparts.get("HPC").first_impressions[other] = impression
        self.brainparts.get("HPC").attitiudes[other] = impression
        self.update_positives()

        return impression

    def get_action_from_history(self, index):
        if index < len(self.brainparts.get("HPC").history):
            return self.brainparts.get("HPC").history[index]
        else:
            return 0

    def get_history(self):
        return self.brainparts.get("HPC").history

    def set_history(self, index: int, value):
        if self.person.age() < Hippocampus.SHOULD_PROBABLY_BE_DEAD:
            self.brainparts.get("HPC").history[index] = value
        else:
            self.brainparts.get("HPC").history = np.append(self.brainparts.get("HPC").history, value)

    def get_attitudes(self, other=None):
        if other:
            return self.brainparts.get("HPC").attitiudes[other]
        return self.brainparts.get("HPC").attitiudes

    def get_positives(self):
        return self.brainparts.get("HPC").positives

    def improve_attitude(self, other, value=0):
        if other in self.brainparts.get("HPC").attitiudes:
            if value == 0:
                self.brainparts.get("HPC").attitiudes[other] += Hippocampus.ATTITUDE_IMPROVEMENT_BONUS
            else:
                self.brainparts.get("HPC").attitiudes[other] += value

    def update_positives(self):
        self.brainparts.get("HPC").update_positives()


class BrainPart:

    INHERITENCE_RATIO = 0.5
    MUTATION_RATIO = 0.2
    MUTATION_NORMALIZATION_RATIO = 0.01

    def __init__(self):
        self.model = None

    def gene_inheritance(self, person, part: str):
        if self.model:
            father_weights = person.father.brain.brainparts.get(part).model.get_weights()
            mother_weights = person.mother.brain.brainparts.get(part).model.get_weights()
            father_biases = person.father.brain.brainparts.get(part).model.get_biases()
            mother_biases = person.mother.brain.brainparts.get(part).model.get_biases()

            # inheritance of parent's weights & biases
            new_weights = father_weights
            new_biases = father_biases
            for lnum, layer_weights in enumerate(new_weights):
                for index, value in np.ndenumerate(layer_weights):
                    if random.uniform(0, 1) < BrainPart.INHERITENCE_RATIO:
                        new_weights[lnum][index] = mother_weights[lnum][index]
            for lnum, layer_biases in enumerate(new_biases):
                for index, value in np.ndenumerate(layer_biases):
                    if random.uniform(0, 1) < BrainPart.INHERITENCE_RATIO:
                        new_weights[lnum][index] = mother_biases[lnum][index]

            for layer_weights in new_weights:
                layer_weights += BrainPart.MUTATION_NORMALIZATION_RATIO * np.random.randn(*np.shape(layer_weights))
            for layer_biases in new_biases:
                layer_biases += BrainPart.MUTATION_NORMALIZATION_RATIO * np.random.randn(*np.shape(layer_biases))

            return new_weights, new_biases

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
        model = NeuralNetwork()
        model.add_layer(12, input_num=12, activation='relu')  # inp layer (1)
        model.add_layer(6, input_num=12, activation='relu')  # hidden layer (2)
        model.add_layer(4, input_num=6, activation='relu')  # hidden layer (3)
        model.add_layer(2, input_num=4, activation='softmax')  # output layer (4)

        self.person = person
        self.model = model

    def decision_making(
            self, gender, age, strength, pregnancy, biowatch, readiness, previous_choice, father_choice,
            mother_choice
    ):
        if random.randint(1, 10) > 3:
            # this list is for convinience, to know what does each index of option means.
            # choices = ["social connection", "strength"]

            # flatten histories & prepare input
            neural_input = [gender, age, strength, pregnancy, biowatch, readiness]

            neural_input.extend(get_choice_nodes(previous_choice))
            neural_input.extend(get_choice_nodes(father_choice))
            neural_input.extend(get_choice_nodes(mother_choice))

            # handle input
            neural_input = np.asarray([neural_input])

            # need to preprocess the data to squash it between 0 and 1. ?

            # output
            output_prob: np.ndarray = self.model.feed(neural_input)

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

        model = NeuralNetwork()
        model.add_layer(16, input_num=16, activation='relu')  # inp layer (1)
        model.add_layer(4, input_num=16, activation='relu')  # hidden layer (2)
        model.add_layer(1, input_num=4, activation='sigmoid')  # output layer (3)
        self.model = model

    def prepare_neural_input(self, other):
        # prepare the person's attributes.
        my_gender = int(self.person.gender)
        my_age = int(self.person.age())
        my_strength = self.person.strength
        if my_gender == 0:
            my_pregnancy = self.person.pregnancy
            my_biowatch = self.person.biowatch
        else:
            my_pregnancy = 0
            my_biowatch = 0
        my_readiness = self.person.readiness

        # prepare the other person's attributes.
        other_gender = int(other.gender)
        other_age = int(other.age())
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

        return neural_input

    def first_impression(self, other):
        neural_input = self.prepare_neural_input(other)
        output_prob: np.ndarray = self.model.feed(neural_input)

        return output_prob


class Hippocampus(BrainPart):
    """
    The part of the brain responsible for memory.
    """
    SHOULD_PROBABLY_BE_DEAD = 120 * 12
    ATTITUDE_IMPROVEMENT_BONUS = 0.05

    def __init__(self, person):
        super().__init__()
        self.person = person

        self.history = np.zeros(120 * 12, dtype=int)

        self.first_impressions = {}
        self.attitiudes = {}
        self.positives = []

    def update_positives(self):
        self.positives = []
        for person, value in self.attitiudes.items():
            if value > 0:
                self.positives.append(person)


def get_choice_nodes(choice):
    """Returns a list of zeroes with 1 at the index of the choice - 1, or no 1 at all if 0 is supplied"""

    nodes = [0 for _ in range(2)]
    if choice:
        nodes[choice - 1] = 1
    return nodes
