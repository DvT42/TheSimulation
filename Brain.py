import random
from numpy import random as nprnd
from Neural_Network import *
from threading import Lock


class Collective:
    # Collective Constants:
    BASIC_POPULATION = 10000

    # brainpart constants:
    INHERITANCE_RATIO = 0.5
    MUTATION_NORMALIZATION_RATIO = 0.3

    # PFC constants
    CHOICE_RANDOMIZER = 0.083
    CHOICE_NUM = 10

    # HPC constants
    SHOULD_PROBABLY_BE_DEAD = 120 * 12
    ATTITUDE_IMPROVEMENT_BONUS = 0.05
    SELF_ATTITUDE_IMPROVEMENT_BONUS = 0.1

    def __init__(self):
        self.population_size = 0
        self._dead = 0
        self.world_attitudes = np.zeros((Collective.BASIC_POPULATION, Collective.BASIC_POPULATION), dtype=float)
        self.arranged_indexes = np.arange(Collective.BASIC_POPULATION)
        self.historical_population = []
        self._lock = Lock()

    def add_person(self, person):
        with self._lock:
            self.historical_population.append(person)

            if self.population_size >= Collective.BASIC_POPULATION:
                new_world_attitudes = np.zeros((self.population_size + 1, self.population_size + 1))
                new_world_attitudes[:self.population_size, :self.population_size] = self.world_attitudes
                self.world_attitudes = new_world_attitudes

                self.arranged_indexes = np.append(self.arranged_indexes, self.population_size)

            self.population_size += 1

    @property
    def dead(self):
        return self._dead

    def remove_person(self):
        with self._lock:
            self._dead += 1


class Brain:

    def __init__(self, person=None, f=None, m=None, models=None, mutate=True):
        self.person = person

        if person:
            self.id = person.id
            self.collective = self.person.collective
            self.brainparts = {"PFC": PrefrontalCortex(),
                               "AMG": Amygdala(),
                               "HPC": Hippocampus()}

            for part, bp in self.brainparts.items():
                if bp.model:
                    new_weights, new_biases = bp.gene_inheritance(father=f if not person.isManual else None,
                                                                  mother=m if not person.isManual else None,
                                                                  part=part)
                    bp.model.set_weights(new_weights)
                    bp.model.set_biases(new_biases)

        elif models is not None:
            self.id = None
            self.brainparts = {"PFC": PrefrontalCortex(models[0], mutate=mutate),
                               "AMG": Amygdala(models[1], mutate=mutate),
                               "HPC": Hippocampus()}

    # def evolve(self):  # Evolvement currently on hold
    #     for brainpart in self.brainparts.values():
    #         brainpart.evolvement()

    def copy(self):
        return Brain(models=self.get_models(), mutate=False)

    def call_decision_making(self, region):
        if self.person.gender == 0:
            preg = self.person.pregnancy
            youngness = self.person.youth
        else:
            preg = 0
            youngness = 0
        if self.person.isManual:
            father_choice = 0
            mother_choice = 0
        else:
            father_choice = Brain.get_action_from_history(self.person.age, self.person.father_history)
            mother_choice = Brain.get_action_from_history(self.person.age, self.person.mother_history)
        partner = self.person.partner

        if partner:
            partner_choice = Brain.get_action_from_history(self.person.age, partner.brain.get_history())
            partner_loc = partner.location
        else:
            partner_choice = 0
            partner_loc = np.array([0, 0])

        loc = region.location
        regional_biomes = region.surr_biomes
        regional_pop = region.surr_pop()
        regional_resources = np.zeros(9)

        return self.brainparts.get("PFC").decision_making(self.person.gender, self.person.age, self.person.strength,
                                                          preg, youngness, self.person.readiness, self.person.child_num,
                                                          father_choice, mother_choice, partner_choice, loc,
                                                          partner_loc, regional_biomes, regional_pop,
                                                          regional_resources)

    def get_first_impression(self, other):
        impression = self.brainparts.get("AMG").first_impression(
            np.concatenate((Amygdala.get_relevant_info(self.person), Amygdala.get_relevant_info(other)), axis=1)
        ).item()

        self.collective.world_attitudes[self.id, other.id] = impression
        self.brainparts.get("HPC").already_met.append(other.id)

        return impression

    def raw_get_first_impression(self, neural_input):
        impression = self.brainparts.get("AMG").first_impression(neural_input).item()

        return impression

    def get_mass_first_impressions(self, ids, info_batch):
        ids = np.array(ids)
        destined_indexes = ids.astype(int)
        impressions = self.brainparts.get("AMG").first_impression(info_batch).flatten()
        self.brainparts.get("HPC").already_met.extend(ids)

        self.collective.world_attitudes[self.id, destined_indexes] = impressions

    @staticmethod
    def get_action_from_history(index, history):
        if index < len(history):
            return history[index]
        else:
            return 0

    def get_history(self, simplified=True):
        if simplified:
            his = self.brainparts.get("HPC").history
            return np.where(his > 2, 3, his)
        return self.brainparts.get("HPC").history

    def set_history(self, index: int, value):
        if self.person.age < Collective.SHOULD_PROBABLY_BE_DEAD:
            self.brainparts.get("HPC").history[index] = value
        else:
            self.brainparts.get("HPC").history = np.append(self.brainparts.get("HPC").history, value)

    def get_attitudes(self, other=None):
        if other:
            return self.collective.world_attitudes[self.id, other.id]
        return self.collective.world_attitudes[self.id, :self.collective.population_size]

    def improve_my_attitudes(self, multiplier=1):
        self.collective.world_attitudes[self.id, self.brainparts.get("HPC").already_met] += (
                Collective.SELF_ATTITUDE_IMPROVEMENT_BONUS * multiplier)

    def improve_attitudes_toward_me(self, region):
        pop_size = self.collective.population_size
        arr = np.copy(self.collective.world_attitudes[:pop_size, self.person.id])
        arr = np.where((arr > 0) &
                       np.isin(self.collective.arranged_indexes[:pop_size], region.pop_id),
                       arr + Collective.ATTITUDE_IMPROVEMENT_BONUS, arr)

        self.collective.world_attitudes[:pop_size, self.person.id] = arr

    def update_location_history(self, loc=None, age=None):
        if type(loc) is not np.ndarray or not loc.any():
            loc = self.person.location
        if age is None:
            age = self.person.age
        self.brainparts.get("HPC").location_history.append(str((loc, age)))

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

    @staticmethod
    def assemble_brains(neural_list):
        brain_couples = []
        for models_couple in neural_list:
            brain_couples.append((Brain(models=models_couple[0]),
                                  Brain(models=models_couple[1])))
        return brain_couples

    @staticmethod
    def assemble_separated_brains(neural_list):
        brains = []
        for model in neural_list:
            brains.append(Brain(models=model, mutate=False))
        return brains


class BrainPart:
    def gene_inheritance(self, part: str, father, mother):
        if father and mother:
            father_weights = father.brain.brainparts.get(part).model.get_weights()
            mother_weights = mother.brain.brainparts.get(part).model.get_weights()
            father_biases = father.brain.brainparts.get(part).model.get_biases()
            mother_biases = mother.brain.brainparts.get(part).model.get_biases()

            # inheritance of parent's weights & biases
            new_weights = father_weights.copy()
            new_biases = father_biases.copy()
            for lnum, layer_weights in enumerate(new_weights):
                for index, value in np.ndenumerate(layer_weights):
                    if nprnd.uniform(0, 1) < Collective.INHERITANCE_RATIO:
                        new_weights[lnum][index] = mother_weights[lnum][index]
            for lnum, layer_biases in enumerate(new_biases):
                for index, value in np.ndenumerate(layer_biases):
                    if nprnd.uniform(0, 1) < Collective.INHERITANCE_RATIO:
                        new_biases[lnum][index] = mother_biases[lnum][index]

            new_weights, new_biases = BrainPart.mutate(new_weights, new_biases)

            return new_weights, new_biases
        else:
            return self.model.get_weights(), self.model.get_biases()

    @staticmethod
    def mutate(weights, biases):
        mutated_weights = []
        mutated_biases = []
        for layer_weights in weights:
            mutated_weights.append(
                layer_weights + Collective.MUTATION_NORMALIZATION_RATIO * nprnd.randn(*np.shape(layer_weights)))
        # should be gotten rid of?
        for layer_biases in biases:
            mutated_biases.append(
                layer_biases + Collective.MUTATION_NORMALIZATION_RATIO * nprnd.randn(*np.shape(layer_biases)))
        return mutated_weights, mutated_biases


class PrefrontalCortex(BrainPart):
    """
    That is the part in the brain that is responsible for making decisions.
    """
    def __init__(self, model=None, mutate=True):
        if model:
            self.model = model.copy()
            if mutate:
                new_weights, new_biases = BrainPart.mutate(self.model.get_weights(), self.model.get_biases())
                self.model.set_weights(new_weights)
                self.model.set_biases(new_biases)

        else:
            input_num = 38 + Collective.CHOICE_NUM * 3

            model = NeuralNetwork()
            model.add_layer(input_num, input_num=input_num, activation='relu')  # input layer (1)
            model.add_layer(39, input_num=input_num, activation='relu')  # hidden layer (2)
            model.add_layer(Collective.CHOICE_NUM, input_num=39, activation='softmax')  # output layer (4)

            self.model = model

    def decision_making(
            self, gender, age, strength, pregnancy, youth, readiness, child_num, father_choice,
            mother_choice, partner_choice, location, partner_location, regional_biomes, regional_pop, regional_resources
    ):
        if nprnd.random() > Collective.CHOICE_RANDOMIZER:
            # this list is for convenience, to know what does each index of option means.
            # choices = ["social connection", "strength", "location"[8]]

            neural_input = np.array([gender, age, strength, pregnancy, youth, readiness, child_num,
                                     *PrefrontalCortex.get_choice_nodes(father_choice),
                                     *PrefrontalCortex.get_choice_nodes(mother_choice),
                                     *PrefrontalCortex.get_choice_nodes(partner_choice),
                                     *location,
                                     *partner_location,
                                     *regional_biomes.flatten(),
                                     *regional_pop,
                                     *regional_resources])

            # need to preprocess the data to squash it between 0 and 1. ?

            # output
            output_prob: np.ndarray = self.model.feed(neural_input)

            return np.argmax(output_prob)
        else:
            choice_index = random.randint(0, Collective.CHOICE_NUM - 8 - 1)
            # You can never relocate randomly because of its unproportional magnitude.

            return choice_index

    @staticmethod
    def get_choice_nodes(choice, options=Collective.CHOICE_NUM):
        """Returns a list of zeroes with 1 at the index of the choice - 1, or no 1 at all if 0 is supplied"""

        nodes = np.zeros(options, dtype=int)
        if choice:
            nodes[choice - 1] = 1
        return nodes


class Amygdala(BrainPart):
    """
    The part of the brain responsible for first impression.
    """

    def __init__(self, model=None, mutate=True):
        if model:
            self.model = model.copy()
            if mutate:
                new_weights, new_biases = BrainPart.mutate(self.model.get_weights(), self.model.get_biases())
                self.model.set_weights(new_weights)
                self.model.set_biases(new_biases)

        else:
            model = NeuralNetwork()
            model.add_layer(14, input_num=14, activation='relu')  # inp layer (1)
            model.add_layer(7, input_num=14, activation='relu')  # hidden layer (2)
            model.add_layer(1, input_num=7, activation='sigmoid')  # output layer (3)
            self.model = model

    @staticmethod
    def get_relevant_info(person):
        gender = int(person.gender)
        age = int(person.age)
        strength = person.strength

        if gender == 0:
            pregnancy = person.pregnancy
            youth = person.youth

        else:
            pregnancy = 0
            youth = 0

        my_readiness = person.readiness
        my_child_num = person.child_num

        return np.asarray([gender, age, strength, pregnancy, youth, my_readiness, my_child_num])

    def first_impression(self, neural_input):
        output_prob: np.ndarray = self.model.feed(neural_input)

        return output_prob


class Hippocampus(BrainPart):
    """
    The part of the brain responsible for memory.
    """
    def __init__(self):
        self.model = None
        self.history = np.zeros(Collective.SHOULD_PROBABLY_BE_DEAD * 12, dtype=int)
        self.location_history = []

        self.already_met = []

