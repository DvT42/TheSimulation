import random
from numpy import random as nprnd

from Neural_Network import NeuralNetwork
from Collective import *


class Brain:
    """
        Represents the cognitive system of an individual, encompassing decision-making, social evaluation, and memory.

        This class encapsulates three distinct brain regions: Prefrontal Cortex (PFC), Amygdala (AMG), and Hippocampus (HPC), each with its specialized functions.

        Attributes:
            person: The Person object to which this brain belongs.
            collective: The Collective object to which the person belongs.
            id: The unique identifier of the person.
            brainparts: A dictionary containing the three brain regions:
                "PFC": PrefrontalCortex object responsible for decision-making.
                "AMG": Amygdala object responsible for initial impressions of others.
                "HPC": Hippocampus object responsible for storing experiences.
    """
    def __init__(self, person=None, f=None, m=None, models=None, mutate=True):
        """
            Initializes the Brain object and its components.

            Args:
                person (Person, optional): The Person object to which this brain belongs. Defaults to None.
                f (Person, optional): The father of the person. Defaults to None.
                m (Person, optional): The mother of the person. Defaults to None.
                models (tuple, optional): A tuple of neural networks for the PFC and AMG. Defaults to None.
                mutate (bool, optional): Whether to mutate the neural networks. Defaults to True.
            """
        self.person = person

        if person:
            self.id = person.id
            self.collective = self.person.collective
            self.brainparts = {"PFC": PrefrontalCortex(),
                               "AMG": Amygdala(),
                               "HPC": Hippocampus()}

            for part, bp in self.brainparts.items():
                if bp.model:
                    new_weights, new_biases = bp.gene_inheritance(gender=person.gender,
                                                                  father=f if not person.isManual else None,
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
        """
            Creates a copy of the Brain object with identical brain parts and neural networks.

            Returns:
                Brain: The newly created copy of the Brain object.
            """
        return Brain(models=self.get_models(), mutate=False)

    def call_decision_making(self, region):
        """
            Invokes the decision-making process using the PFC neural network.

            Args:
                region (Region.Region): The region where the person resides.

            Returns:
                str: The chosen action or decision.
        """
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
        regional_resources = region.surr_resources()

        return self.brainparts.get("PFC").decision_making(self.person.gender, self.person.age, self.person.strength,
                                                          preg, youngness, self.person.readiness, self.person.child_num,
                                                          father_choice, mother_choice, partner_choice, loc,
                                                          partner_loc, regional_biomes, regional_pop,
                                                          regional_resources)

    def get_first_impression(self, other):
        """
            Evaluates an initial impression of another person using the AMG neural network.

            Args:
                other (Person.Person): The person to evaluate.

            Returns:
                float: The initial impression score, ranging from 0 (negative) to 1 (positive).
            """
        impression = self.brainparts.get("AMG").first_impression(
            np.concatenate((Amygdala.get_relevant_info(self.person), Amygdala.get_relevant_info(other)), axis=1)
        ).item()

        self.collective.world_attitudes[self.id, other.id] = impression
        self.brainparts.get("HPC").already_met.append(other.id)

        return impression

    def raw_get_first_impression(self, neural_input):
        """
            Evaluates an initial impression score directly from neural network input.

            Args:
                neural_input (np.ndarray): The raw neural network input vector.

            Returns:
                float: The initial impression score, ranging from 0 (negative) to 1 (positive).
            """
        impression = self.brainparts.get("AMG").first_impression(neural_input).item()

        return impression

    def get_mass_first_impressions(self, ids, info_batch):
        """
            Processes a batch of data for multiple individuals to generate initial impressions.

            This function adapts the 'mass_encounter' method of the 'region' module to handle
            and return results for a large group of individuals. It inputs data in a batch
            format into the AMG (Assumed Model Generator) and retrieves the corresponding
            impressions.

            Args:
                ids (list): A list of identifiers for the individuals whose data is being processed.
                info_batch (ndarray): A collection of data for each individual, formatted as a batch.
                                   Each index in the array corresponds to an individual's data.

            Returns:
                list: A list of initial impressions for each individual based on the input data.
            """
        ids = np.array(ids)
        # destined_indexes = ids.astype(int)
        impressions = self.brainparts.get("AMG").first_impression(info_batch).flatten()
        # self.brainparts.get("HPC").already_met.extend(ids)

        self.collective.world_attitudes[self.id, ids] = impressions

    @staticmethod
    def get_action_from_history(index, history):
        """
            Retrieves the value at the specified index from the given history.

            This function is static to allow it to be used on histories of individuals other than
            the one containing the brain, such as the parents of that individual.

            Args:
                index (int): The index in the history to search for.
                history (list): The history to search in.

            Returns:
                The value at the specified index in the history.
            """
        if index < len(history):
            return history[index]
        else:
            return 0

    def get_history(self, simplified=True):
        """
            Retrieves the history stored within the HPC.

            Args:
                simplified (bool, optional): If set to True, returns the history with all movement choices represented as "3".
                Defaults to True.

            Returns:
                The history stored within the HPC.
            """
        if simplified:
            his = self.brainparts.get("HPC").history
            return np.where(his > 2, 3, his)
        return self.brainparts.get("HPC").history

    def set_history(self, index: int, value):
        """
            Sets a given value at the specified index in the history stored within the HPC.

            Args:
                index (int): The index in the history to set the value at.
                value: The value to set.
            """
        if self.person.age < Collective.SHOULD_PROBABLY_BE_DEAD:
            self.brainparts.get("HPC").history[index] = value
        else:
            self.brainparts.get("HPC").history = np.append(self.brainparts.get("HPC").history, value)

    def get_attitudes(self, other=None):
        """
            Retrieves the attitude of the individual containing the brain towards another individual.

            If no other individual is specified, returns all attitudes of the individual towards others.

            Args:
                other (Individual, optional): The other individual to get the attitude towards. Defaults to None.

            Returns:
                The attitude towards the specified individual, or an array of attitudes towards all other individuals.
            """
        if other:
            return self.collective.world_attitudes[self.id, other.id]
        return self.collective.world_attitudes[self.id, :self.collective.population_size]

    def improve_my_attitudes(self, multiplier=1):
        """
            Improves the individual's attitudes towards others by the collective's attitude improvement bonus constant.

            Args:
                multiplier (float, optional): A factor to adjust the bonus by. Defaults to 1.
            """
        self.collective.world_attitudes[self.id, self.brainparts.get("HPC").already_met] += (
                Collective.SELF_ATTITUDE_IMPROVEMENT_BONUS * multiplier)

    def improve_attitudes_toward_me(self, region):
        """
            Improves the attitudes of others in the specified region towards the individual containing the brain by the collective's attitude improvement bonus constant.

            Args:
                region (Region.Region): The region in which the individual is located.
            """
        pop_size = self.collective.population_size
        arr = np.copy(self.collective.world_attitudes[:pop_size, self.person.id])
        arr = np.where((arr > 0) &
                       np.isin(self.collective.arranged_indexes[:pop_size], region.pop_id),
                       arr + Collective.ATTITUDE_IMPROVEMENT_BONUS, arr)

        self.collective.world_attitudes[:pop_size, self.person.id] = arr

    def update_location_history(self, loc=None, age=None):
        """
            Adds a new location to the individual's location history. If no specific location or age is provided, adds the current location and age.

            Args:
                loc (str, optional): The location to add to the history. Defaults to None.
                age (int, optional): The age at which the individual reached the location. Defaults to None.
            """
        if type(loc) is not np.ndarray or not loc.any():
            loc = self.person.location
        if age is None:
            age = self.person.age
        self.brainparts.get("HPC").location_history.append(str((loc, age)))

    def get_location_history(self):
        """
            Retrieves the location history of the individual.

            Returns:
                The list of location entries in the history.
            """
        return self.brainparts.get("HPC").location_history

    def is_friendly(self):
        """
            Checks whether the individual has any positive attitudes towards others.

            Returns:
                True if the individual has at least one positive attitude, False otherwise.
            """
        return np.max(self.collective.world_attitudes[self.id]) > 0

    def transfer_brain(self, new_person):
        """
            Transfers the brain to another individual, connecting the brain to the new individual's data and collective.

            Note: This function has only been used for transferring brains created without an individual.

            Args:
                new_person (Individual): The individual to transfer the brain to.
            """
        self.person = new_person
        self.collective = new_person.collective
        self.id = new_person.id
        self.brainparts["HPC"] = Hippocampus()

    def get_models(self):
        """
            Returns a tuple of the brain's neural networks, with each network associated with a brain region.

            Returns:
                Tuple of (brain_region, neural_network) pairs.
            """
        return self.brainparts.get("PFC").model, self.brainparts.get("AMG").model

    @staticmethod
    def assemble_brains(neural_list):
        """
            Assembles pairs of brains from a list of neural network models.

            Args:
                neural_list (List[Tuple[Tuple[NeuralNetwork, NeuralNetwork]]]): A list of tuples of brain models, where each brain model is a tuple of (visual_network, motor_network).

            Returns:
                List[Brain]: A list of assembled brains.
            """
        brain_couples = []
        for models_couple in neural_list:
            brain_couples.append((Brain(models=models_couple[0]),
                                  Brain(models=models_couple[1])))
        return brain_couples

    @staticmethod
    def assemble_separated_brains(neural_list):
        """
            Assembles individual brains from a list of neural network models.

            Similar to assemble_brains, but returns a list of individual brains instead of pairs.

            Args:
                neural_list (List[Tuple[NeuralNetwork]]): A list of tuples of brain models, where each brain model is a tuple of (visual_network, motor_network).

            Returns:
                List[Brain]: A list of assembled brains.
            """
        brains = []
        for model in neural_list:
            brains.append(Brain(models=model, mutate=False))
        return brains


class BrainPart:
    """
        Base class representing a component of the brain.

        This class serves as a template from which all brain part classes in the `brain.brainparts` dictionary inherit.
        It provides essential functionality and structure for defining and managing various brain components.
    """
    def gene_inheritance(self, gender, part: str, father, mother):
        """
            Simulates the inheritance of genetic information from parents to create a child's brain part model.

            Args:
                father (Person.Person): The father's brain.
                mother (Person.Person): The mother's brain.
                part (str): The key (string) corresponding to the brain part in the `brainparts` dictionary within the `brain` class.

            Returns:
                new_weights: The child's new brain-part model's weights.
                new_biases: The child's new brain-part model's biases.
        """
        if father and mother:
            same_gender = father if gender == 1 else mother
            other_gender = mother if gender == 0 else father

            same_gender_weights = same_gender.brain.brainparts.get(part).model.get_weights()
            other_gender_weights = other_gender.brain.brainparts.get(part).model.get_weights()
            same_gender_biases = same_gender.brain.brainparts.get(part).model.get_biases()
            other_gender_biases = other_gender.brain.brainparts.get(part).model.get_biases()

            # inheritance of parent's weights & biases
            new_weights = same_gender_weights.copy()
            new_biases = same_gender_biases.copy()
            for lnum, layer_weights in enumerate(new_weights):
                for index in range(len(layer_weights.T)):
                    if nprnd.uniform(0, 1) < Collective.INHERITANCE_RATIO:
                        new_weights[lnum][:, index] = other_gender_weights[lnum][:, index]
                        new_biases[lnum][0, index] = other_gender_biases[lnum][0, index]

            new_weights, new_biases = BrainPart.mutate(new_weights, new_biases)

            return new_weights, new_biases
        else:
            return self.model.get_weights(), self.model.get_biases()

    @staticmethod
    def mutate(weights, biases):
        """
            Applies random Gaussian mutations to weights and biases.

            Args:
                weights (ndarray): The weights to be mutated.
                biases (ndarray): The biases to be mutated.

            Returns:
                Tuple[ndarray, ndarray]: The mutated weights and biases.
        """
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
        """
            Constructor for the BrainPart class.

            Args:
                model (NeuralNetwork, optional): An existing neural network model to be used. If None, a new model will be created.
                mutate (bool, optional): Whether to apply mutations to the model. Defaults to True.
            """
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
        """
            Performs decision-making using the brain part's neural network model.

            This function takes various input parameters representing the individual's characteristics, environment, and partner's choices, and feeds them into the brain part's neural network model. The model then processes the input data and generates an output representing the individual's decision.

            Args:
                gender (str): The individual's gender ("male" or "female").
                age (int): The individual's age.
                strength (float): The individual's strength value.
                pregnancy (int): The individual's pregnancy status (0 for non-pregnant, 1-9 for pregnancy progression).
                youth (int): The individual's reproductive potential (0 for non-pregnant, 1-9 for increasing potential).
                readiness (int): The individual's biological readiness for reproduction.
                child_num (int): The number of children the individual has already raised.
                father_choice (int): The choice made by the individual's father at his current age (0 if not applicable).
                mother_choice (int): The choice made by the individual's mother at her current age (0 if not applicable).
                partner_choice (int): The choice made by the individual's partner last month (0 if no partner).
                location (str): The individual's current location.
                partner_location (str): The individual's partner's current location.
                regional_biomes (ndarray): A 2D array representing the biomes in the individual's region (3x3).
                regional_pop (ndarray): A 2D array representing the population density in the individual's region (3x3).
                regional_resources (ndarray): A 2D array representing the resource availability in the individual's region (3x3).

            Returns:
                int: The individual's decision based on the neural network's output.
            """
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
        """
        Generates an array representing a specific choice out of a set of options.

        This helper function creates a zero-filled array of the size of the number of options, with a single '1' at the index corresponding to the selected choice. This format is useful for training and simplifying the neural network's learning process.

        Args:
            choice (int): The selected choice index (0-based).
            options (int, optional): The total number of available choices. Defaults to the value of `Collective.CHOICE_NUM`.

        Returns:
            ndarray: An array of size `options` representing the selected choice.
        """
        nodes = np.zeros(options, dtype=int)
        if choice:
            nodes[choice - 1] = 1
        return nodes


class Amygdala(BrainPart):
    """
    The part of the brain responsible for first impression.
    """

    def __init__(self, model=None, mutate=True):
        """
            Constructor for the BrainPart class.

            Args:
                model (NeuralNetwork, optional): An existing neural network model to be used. If None, a new model will be created.
                mutate (bool, optional): Whether to apply mutations to the model. Defaults to True.
            """
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
        """
            Gathers relevant information about a person to form an initial impression.

            This static method collects various details about the specified individual, potentially including their characteristics, environment, and past choices, to gain a more comprehensive understanding of their situation and potential actions.

            Args:
                person (Person.Person): The Person object representing the individual for whom information is being gathered.

            Returns:
                ndarray: An array containing the collected relevant information about the person.
            """
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
        """
            Generates a first impression of a person based on the provided neural network input.

            This method takes the processed neural network input, potentially representing various characteristics and environmental factors, and utilizes it to form an initial assessment of the individual. The output of this function could be used to guide further interactions or decision-making processes.

            Args:
                neural_input (ndarray): The processed neural network input data.

            Returns:
                The first impression score, representing the initial assessment of the individual.
        """
        output_prob: np.ndarray = self.model.feed(neural_input)

        return output_prob


class Hippocampus(BrainPart):
    """
    Represents the hippocampus, a brain part responsible for memory formation and storage.

    The Hippocampus object maintains information about an individual's past actions, encounters, and locations, serving as a repository of their experiences. This information can be used to influence decision-making, interactions, and behavior modeling within a simulation framework.

    Attributes:
        history (np.ndarray): A list of actions performed by the individual in the past, ordered by their age at the time of action. Each element in the list represents one month.
        location_history (list): A list of tuples containing the locations visited by the individual and their age at the time of each visit.
        already_met (list): A list of IDs of individuals that the current individual has encountered in the past.
    """

    def __init__(self):
        self.model = None
        self.history = np.zeros(Collective.SHOULD_PROBABLY_BE_DEAD * 12, dtype=int)
        self.location_history = []

        self.already_met = []
