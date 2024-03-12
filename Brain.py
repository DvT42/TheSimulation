import numpy as np
from numpy import random

from Neural_Network import NeuralNetwork


class Collective:
    # Collective Constants:
    BASIC_POPULATION = 1000

    # brainpart constants:
    INHERITANCE_RATIO = 0.5
    MUTATION_RATIO = 0.2
    MUTATION_NORMALIZATION_RATIO = 0.3

    # PFC constants
    CHOICE_RANDOMIZER = 0.1
    CHOICE_NUM = 2

    # HPC constants
    SHOULD_PROBABLY_BE_DEAD = 120 * 12
    ATTITUDE_IMPROVEMENT_BONUS = 0.05
    SELF_ATTITUDE_IMPROVEMENT_BONUS = 0.1

    def __init__(self):
        self.population_size = 0
        self.world_attitudes = np.zeros((Collective.BASIC_POPULATION, Collective.BASIC_POPULATION), dtype=float)
        self.aranged_indexes = np.arange(Collective.BASIC_POPULATION)
        self.historical_population = []

    def add_person(self, person):
        self.historical_population.append(person)

        if self.population_size >= Collective.BASIC_POPULATION:
            new_world_attitudes = np.zeros((self.population_size + 1, self.population_size + 1))
            new_world_attitudes[:self.population_size, :self.population_size] = self.world_attitudes
            self.world_attitudes = new_world_attitudes

            self.aranged_indexes = np.append(self.aranged_indexes, self.population_size)

        self.population_size += 1


class Brain:
    def __init__(self, person=None, f=None, m=None, collective=None, models=None):
        self.person = person
        self.collective = collective

        if person and collective:
            self.id = person.id
            self.brainparts = {"PFC": PrefrontalCortex(),
                               "AMG": Amygdala(),
                               "HPC": Hippocampus()}

            for part, bp in self.brainparts.items():
                if bp.model:
                    new_weights, new_biases = bp.gene_inheritance(self.person.isManual, f, m, part)
                    bp.model.set_weights(new_weights)
                    bp.model.set_biases(new_biases)

        elif type(models) is np.ndarray and models.any():
            self.id = None
            self.brainparts = {"PFC": PrefrontalCortex(models[0]),
                               "AMG": Amygdala(models[1]),
                               "HPC": Hippocampus()}

    def evolve(self):  # Evolvement currently on hold
        for brainpart in self.brainparts.values():
            brainpart.evolvement()

    def call_decision_making(self):
        if self.person.gender == 0:
            preg = self.person.pregnancy
            youngness = self.person.youngness
        else:
            preg = 0
            youngness = 0
        if self.person.isManual:
            father_choice = 0
            mother_choice = 0
        else:
            father_choice = Brain.get_action_from_history(self.person.age(), self.person.father_history)
            mother_choice = Brain.get_action_from_history(self.person.age(), self.person.mother_history)
        if self.person.partner:
            partner_choice = Brain.get_action_from_history(self.person.age(), self.person.partner.brain.get_history())
        else:
            partner_choice = 0

        return self.brainparts.get("PFC").decision_making(
            self.person.gender, self.person.age(), self.person.strength, preg, youngness, self.person.readiness,
            self.person.child_num, Brain.get_action_from_history(self.person.age() - 1, self.get_history()),
            father_choice, mother_choice, partner_choice
        )

    def get_first_impression(self, other):
        impression = self.brainparts.get("AMG").first_impression(self.person, other)

        self.brainparts.get("HPC").first_impressions[other] = impression
        self.collective.world_attitudes[self.id, other.id] = impression

        return impression

    @staticmethod
    def get_action_from_history(index, history):
        if index < len(history):
            return history[index]
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
            return self.collective.world_attitudes[self.id, other.id]
        return self.collective.world_attitudes[self.id, :self.collective.population_size]

    def improve_my_attitudes(self, multiplier=1):
        self.collective.world_attitudes[self.id, :self.collective.population_size] += (
                Collective.SELF_ATTITUDE_IMPROVEMENT_BONUS * multiplier)

    def improve_attitudes_toward_me(self, region):
        arr = np.copy(self.collective.world_attitudes[:self.collective.population_size, self.person.id])
        arr = np.where((arr > 0) &
            np.isin(self.collective.aranged_indexes[:self.collective.population_size], region.pop_id),
            arr + Hippocampus.ATTITUDE_IMPROVEMENT_BONUS, arr)

        self.collective.world_attitudes[:self.collective.population_size, self.person.id] = arr

    def update_location_history(self):
        self.brainparts.get("HPC").location_history.append((self.person.location, self.person.age()))

    def get_location_history(self):
        return self.brainparts.get("HPC").location_history

    def is_friendly(self):
        return np.max(self.collective.world_attitudes[self.id]) > 0

    def transfer_brain(self, new_person):
        self.person = new_person
        self.collective = new_person.collective
        self.id = new_person.id
        self.brainparts["HPC"] = Hippocampus()

    def get_models(self):
        return self.brainparts.get("PFC").model, self.brainparts.get("AMG").model


class BrainPart:
    INHERITENCE_RATIO = Collective.INHERITANCE_RATIO
    MUTATION_RATIO = Collective.MUTATION_RATIO
    MUTATION_NORMALIZATION_RATIO = Collective.MUTATION_NORMALIZATION_RATIO

    def __init__(self):
        self.model = None

    def gene_inheritance(self, isManual, father, mother, part: str):
        if not isManual:
            father_weights = father.brain.brainparts.get(part).model.get_weights()
            mother_weights = mother.brain.brainparts.get(part).model.get_weights()
            father_biases = father.brain.brainparts.get(part).model.get_biases()
            mother_biases = mother.brain.brainparts.get(part).model.get_biases()

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
                        new_biases[lnum][index] = mother_biases[lnum][index]

            for layer_weights in new_weights:
                layer_weights += BrainPart.MUTATION_NORMALIZATION_RATIO * np.random.randn(*np.shape(layer_weights))
            for layer_biases in new_biases:
                layer_biases += BrainPart.MUTATION_NORMALIZATION_RATIO * np.random.randn(*np.shape(layer_biases))

            return new_weights, new_biases
        else:
            return self.model.get_weights(), self.model.get_biases()

    def evolvement(self):
        if self.model:
            new_weights = np.asarray(self.model.get_weights())

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
    CHOICE_RANDOMIZER = Collective.CHOICE_RANDOMIZER
    CHOICE_NUM = Collective.CHOICE_NUM

    def __init__(self, model=None):
        super().__init__()
        if model:
            self.model = model

        else:
            input_num = 7 + PrefrontalCortex.CHOICE_NUM * 4

            model = NeuralNetwork()
            model.add_layer(input_num, input_num=input_num, activation='relu')  # input layer (1)
            model.add_layer(12, input_num=input_num, activation='relu')  # hidden layer (2)
            model.add_layer(6, input_num=12, activation='relu')  # hidden layer (3)
            model.add_layer(PrefrontalCortex.CHOICE_NUM, input_num=6, activation='softmax')  # output layer (4)

            self.model = model

    def decision_making(
            self, gender, age, strength, pregnancy, biowatch, readiness, child_num, previous_choice, father_choice,
            mother_choice, partner_choice
    ):
        if random.random() > PrefrontalCortex.CHOICE_RANDOMIZER:
            # this list is for convenience, to know what does each index of option means.
            # choices = ["social connection", "strength", "location"]

            # flatten histories & prepare input
            neural_input = [gender, age, strength, pregnancy, biowatch, readiness, child_num]

            neural_input.extend(get_choice_nodes(previous_choice))
            neural_input.extend(get_choice_nodes(father_choice))
            neural_input.extend(get_choice_nodes(mother_choice))
            neural_input.extend(get_choice_nodes(partner_choice))

            # handle input
            neural_input = np.asarray([neural_input])

            # need to preprocess the data to squash it between 0 and 1. ?

            # output
            output_prob: np.ndarray = self.model.feed(neural_input)

            return np.argmax(output_prob)
        else:
            r = random.random()
            # Calculate the slice index
            choice_index = int(np.floor(r * PrefrontalCortex.CHOICE_NUM))

            # Adjust the slice index if the random number falls exactly on a slice boundary
            if r == choice_index / PrefrontalCortex.CHOICE_NUM:
                choice_index -= 1
            return choice_index


class Amygdala(BrainPart):
    """
    The part of the brain responsible for first impression.
    """

    def __init__(self, model=None):
        super().__init__()

        if model:
            self.model = model

        else:
            model = NeuralNetwork()
            model.add_layer(16, input_num=16, activation='relu')  # inp layer (1)
            model.add_layer(4, input_num=16, activation='relu')  # hidden layer (2)
            model.add_layer(1, input_num=4, activation='sigmoid')  # output layer (3)
            self.model = model

    @staticmethod
    def prepare_neural_input(person, other):
        # prepare the person's attributes.
        my_gender = int(person.gender)
        my_age = int(person.age())
        my_strength = person.strength
        if my_gender == 0:
            my_pregnancy = person.pregnancy
            my_biowatch = person.youngness
        else:
            my_pregnancy = 0
            my_biowatch = 0
        my_readiness = person.readiness

        # prepare the other person's attributes.
        other_gender = int(other.gender)
        other_age = int(other.age())
        other_strength = other.strength
        if other_gender == 0:
            other_pregnancy = other.pregnancy
            other_biowatch = other.youngness
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

    def first_impression(self, person, other):
        neural_input = self.prepare_neural_input(person, other)
        output_prob: np.ndarray = self.model.feed(neural_input)

        return output_prob[0][0]


class Hippocampus(BrainPart):
    """
    The part of the brain responsible for memory.
    """
    SHOULD_PROBABLY_BE_DEAD = Collective.SHOULD_PROBABLY_BE_DEAD
    ATTITUDE_IMPROVEMENT_BONUS = Collective.ATTITUDE_IMPROVEMENT_BONUS

    def __init__(self):
        super().__init__()

        self.history = np.zeros(120 * 12, dtype=int)
        self.location_history = []

        self.dead_impressions = {}
        self.first_impressions = {}


def get_choice_nodes(choice, options=PrefrontalCortex.CHOICE_NUM):
    """Returns a list of zeroes with 1 at the index of the choice - 1, or no 1 at all if 0 is supplied"""

    nodes = [0 for _ in range(options)]
    if choice:
        nodes[choice - 1] = 1
    return nodes
