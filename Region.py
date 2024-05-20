from Person import *


class Region:
    """
        Represents a region on the map with a population of individuals (or who once had a population).

        :cvar RESOURCE_LEGEND:
        :cvar RISK_LEGEND:

        Attributes:
            location (Tuple[int, int]): The coordinates (x, y) of the region's location.

            biome (int): An integer representing the biome type of the region.

            surr_biomes (NumPy array): A 3x3 NumPy array containing the biomes of the surrounding regions.

            _neighbors (NumPy array): A 3x3 NumPy array containing the neighboring regions of this region.

            _dead (List): An updated list of individuals who died this month and need to be removed from the region's population.

            action_pool (List): An updated list of the distribution of choices among the region's population, used for debugging purposes only.

            _newborns (List): An updated list of children born this month who need to be added to the region's population.

            _social_connectors (List): An updated list of individuals from this region who chose to be social this month.

            _relocating_people (List): An updated list of individuals who chose to leave this region this month and need to be removed from the region's population.

            _newcomers (List): An updated list of individuals who chose to enter this region this month and need to be added to the region's population.

            _lock (Lock): A `Lock` object preventing concurrent modification of values within the region.

            _population (List): An updated list of all living individuals in this region.

            pop_id (List): An updated list containing the unique IDs of all individuals in the `_population` list. Saves the need to recreate it repeatedly.
        """
    RESOURCE_LEGEND = {  # The numbers were taken from Gemini's estimations.
        0: 0,
        1: 10,
        2: 30,
        3: 300,
        4: 20,
        5: 500,
        6: 1000,
        7: 200,
        8: 300,
        9: 10,
        10: 700
    }
    RISK_LEGEND = {  # The numbers are by my own estimations. changeable, obviously.
        0: 2,
        1: 1,
        2: 0.5,
        3: 0.4,
        4: 0.7,
        5: 0.4,
        6: 0.8,
        7: 0.3,
        8: 0.4,
        9: 1.2,
        10: 0.4
    }

    def __init__(self, location, surrounding_biomes, neighbors=np.empty((3, 3), dtype=object), population=None):
        """
            Initializes the `Region` instance with the specified location, surrounding biomes, and optional population.

            Args:
                location (Tuple[int, int]): The coordinates (x, y) of the region's location.

                surrounding_biomes (NumPy array): A 3x3 NumPy array representing the biomes of the surrounding regions.

                neighbors (NumPy array, optional): A 3x3 NumPy array of `Region` objects representing the neighboring regions. Defaults to an empty array.
                neighbors (NumPy array, optional): A 3x3 NumPy array of `Region` objects representing the neighboring regions. Defaults to an empty array.

                population (List[Person], optional): An initial list of `Person` objects representing the existing population in the region. Defaults to `None`.
        """

        self.location = np.array(location)
        self.biome = surrounding_biomes[1, 1]
        self.resources = Region.RESOURCE_LEGEND[self.biome]
        self.risk = Region.RISK_LEGEND[self.biome]
        self.surr_biomes = surrounding_biomes
        self._neighbors = neighbors
        self._neighbors[1, 1] = self

        self._dead = []
        self.action_pool = [0, 0, 0]
        self._newborns = []
        self._social_connectors = []
        self._relocating_people = []
        self._newcomers = []
        self._lock = Lock()

        if population:
            self._Population = population
        else:
            self._Population = []
        self.pop_id = [p.id for p in self._Population]

    @property
    def neighbors(self):
        return self._neighbors

    @neighbors.setter
    def neighbors(self, new_neighbors):
        """
            Updates the `_neighbors` array with a new neighbors array. This function is not yet used anywhere.

            Args:
                new_neighbors: an array that will replace the former neighbors array completely
        """
        with self._lock:
            self._neighbors = new_neighbors

    def neighbors_update(self, pos, value):
        """
            Updates the `_neighbors` array with the specified `value` at the given `pos` position.

            Args:
                pos (Tuple[int, int]): The (row, column) coordinates in the `_neighbors` array where the `value` should be placed.

                value (Region): The `Region` object to be placed at the specified position.
        """
        with self._lock:
            self._neighbors[pos] = value

    @property
    def newborns(self):
        return self._newborns

    @newborns.setter
    def newborns(self, new_newborn):
        """
            Sets the `_newborns` list with the specified `new_newborn`.

            This method is used to update the list of newborns for the current region.

            Args:
                new_newborn (Person): The `Person` object representing the newborn to be added to the list.
        """
        with self._lock:
            self._newborns.append(new_newborn)

    @property
    def dead(self):
        return self._dead

    @dead.setter
    def dead(self, new_dead):
        """
            Sets the `_dead` list with the specified `new_dead`.

            This method is used to update the list of individuals who died in the current region during the current simulation step.

            Args:
                new_dead (Person): The `Person` object representing the deceased individual to be added to the list.
            """
        with self._lock:
            self._dead.append(new_dead)

    @property
    def social_connectors(self):
        return self._social_connectors

    @social_connectors.setter
    def social_connectors(self, new_social_connector):
        """
            Sets the `_social_connectors` list with the specified `new_social_connector`.

            This method is used to update the list of individuals who chose to be social in the current region during the current simulation step.

            Args:
                new_social_connector (Person): The `Person` object representing the individual who chose to be social to be added to the list.
            """
        with self._lock:
            self._social_connectors.append(new_social_connector)

    @property
    def relocating_people(self):
        return self._relocating_people

    @relocating_people.setter
    def relocating_people(self, new_relocating_person):
        """
            Sets the `_relocating_people` list with the specified `new_relocating_person`.

            This method is used to update the list of individuals who chose to leave the current region during the current simulation step.

            Args:
                new_relocating_person (Person): The `Person` object representing the individual who chose to leave to be added to the list.
            """
        with self._lock:
            self._relocating_people.append(new_relocating_person)

    @property
    def newcomers(self):
        return self._newcomers

    @newcomers.setter
    def newcomers(self, new_newcomer):
        """
            Sets the `_newcomers` list with the specified `new_newcomer`.

            This method is used to update the list of individuals who arrived in the current region during the current simulation step.

            Args:
                new_newcomer (Person): The `Person` object representing the individual who arrived to be added to the list.
            """
        with self._lock:
            self._newborns.append(new_newcomer)

    def clear(self):
        """
            Clears the monthly update lists of the region (dead, action_pool, newborns, social_connectors, and newcomers).

            This method is called at the beginning of each month to reset the lists that track individuals who died, were added to the action pool, were born, chose to be social, or arrived in the region during the current month.
        """
        self._dead = []
        self.action_pool = [0, 0, 0]
        self._newborns = []
        self._social_connectors = []
        self._relocating_people = []
        self._newcomers = []

    @property
    def Population(self):
        with self._lock:
            return self._Population

    def add_person(self, person):
        """
            Adds the specified `person` to the region's population and assigns a unique population ID.

            This method is responsible for incorporating a new individual into the region's population and assigning them a unique identifier within the region's context.

            Args:
                person (Person): The `Person` object representing the individual to be added to the region.
            """
        with self._lock:
            if person.id not in self.pop_id:
                self._Population.append(person)
                self.pop_id.append(person.id)
                self.newcomers.append(person)

    def remove_person(self, person):
        """
            Removes the specified `person` from the region's population.

            This method handles the removal of an individual from the region's population, either due to death or relocation to another region.

            Args:
                person (Person): The `Person` object representing the individual to be removed from the region.
            """
        with self._lock:
            while person.id in self.pop_id:
                self._Population.remove(person)
                self.pop_id.remove(person.id)

    # noinspection PyTypeChecker
    def introduce_newcomers(self):
        """
        Introduces the newcomers to the existing residents of the region and creates initial impressions.

        This method facilitates interactions between newcomers and existing residents, allowing them to form initial impressions of each other.
        """
        for n in self.newcomers:
            self.mass_encounter(person=n)

    def mass_encounter(self, person=None, pop_lst=None, ids=None):
        """
            Efficiently generates encounters between a large number of individuals.

            This method utilizes numpy arrays and vectorized operations to minimize runtime overhead when performing mass encounters.

            Args:
                pop_lst (list of Person.Person): The list of individuals involved in the encounters. If a list of `Person` objects is provided, it will be used directly.
                ids (list of int): The list of individual IDs involved in the encounters.
                person (Person.Person): If specified, the method will generate encounters between this person and all individuals in `pop_lst` or all individuals in the region if `pop_lst` is not provided.
            """
        pop_lst = pop_lst if pop_lst else self.Population
        ids = ids if ids else self.pop_id
        info_batch = np.empty(shape=(len(pop_lst), 7))

        for i, p in enumerate(pop_lst):
            info_batch[i] = Amygdala.get_relevant_info(person=p)

        if person:
            person: Person
            p_info = Amygdala.get_relevant_info(person=person)
            never_met = [(ids[i], p, info_batch[i]) for i, p in enumerate(pop_lst) if
                         ids[i] not in person.brain.brainparts.get("HPC").already_met]
            info_batch = np.array([i[2] for i in never_met])
            never_met = np.array([i[:2] for i in never_met], dtype=object)
            if never_met.any():
                with self._lock:
                    new_ids = never_met[:, 0].astype(int)
                    tiled = np.tile(p_info, (len(new_ids), 1))
                    person.brain.get_mass_first_impressions(new_ids, np.concatenate((info_batch, tiled), axis=1))

                    reflective = np.concatenate((tiled, info_batch), axis=1)
                    impressions = np.empty((len(new_ids)))
                    for i, p in enumerate(never_met[:, 1]):
                        impressions[i] = p.brain.raw_get_first_impression(reflective[i])
                        p.brain.brainparts.get("HPC").already_met.append(person.id)
                    person.collective.world_attitudes[new_ids, person.id] = impressions
        else:
            for i, p in enumerate(pop_lst):
                p_info = info_batch[i]
                tiled = np.tile(p_info, (len(pop_lst), 1))
                p.brain.get_mass_first_impressions(ids, np.concatenate((info_batch, tiled), axis=1))

    def falsify_action_flags(self):
        """
            Sets all action flags of individuals in the region's population to False.

            This method is used to reset the action flags of all individuals, indicating that they have not yet taken any actions during the current simulation step.

            It iterates through the region's population and sets the `action_flag` attribute of each `Person` object to `False`.
            """
        for p in self.Population:
            p.action_flag = False

    def pop(self):
        """
            Returns the size of the region's population.

            This method provides a convenient way to access the current number of individuals residing in the region.
            """
        return len(self.Population)

    def resource_distribution(self):
        """
            Calculates and returns the average resource availability per person in the region.

            This method provides a measure of the resource abundance within the region, considering both the total available resources (`self.resources`) and the current population size (`self.pop()`).

            Returns:
                float: The average resource availability per person, rounded to two decimal places.
        """
        return round(min(self.resources / self.pop(), 2), 2)

    def surr_pop(self):
        """
        Gathers population information from surrounding regions and returns it as a 3x3 array.

        This method collects population data from the eight neighboring regions (assuming a 3x3 grid structure) and returns it as a structured array for easy access and analysis.

        Returns:
            numpy.ndarray: A 3x3 array containing the population sizes of the surrounding regions.
        """
        lst = np.zeros(9).reshape((3, 3))
        for i, j in zip(*np.where(self.neighbors)):
            lst[i, j] = self.neighbors[i, j].pop()
        return lst.flatten()

    def surr_resources(self):
        """
            Gathers resource information from surrounding regions and returns it as a flattened array.

            This method collects resource data from the eight neighboring regions (assuming a 3x3 grid structure) and returns a flat array containing resource codes based on a pre-defined legend.

            Returns:
                numpy.ndarray: A flattened 1D array containing resource codes of the surrounding regions.
        """
        lst = np.zeros(9).reshape((3, 3))
        for i in range(len(self.neighbors)):
            for j in range(len(self.neighbors[i])):
                lst[i, j] = Region.RESOURCE_LEGEND[self.surr_biomes[i, j]]
        return lst.flatten()

    def __iter__(self):
        """
            Magic method that allows iteration over the Region object.

            This method enables the Region class to be used in 'for' loops, allowing you to iterate over the individuals in the region's population.
        """
        return (p for p in self.Population)

    def __repr__(self):
        """
            Magic method that defines the string representation of the Region object.

            This method provides a detailed description of the Region object when it is converted to a string, including its position, population size, and resources.
            """
        return f'({self.location}, {self.biome})'

    def display(self):
        """
            Displays a comprehensive overview of the Region's characteristics and state.

            This method provides a detailed presentation of the region's attributes, including its position, population, resources, biome, and surrounding regions.
        """
        txt = f'location: {self.location}:\n----------\n'

        for p in self:
            txt += f"{p.display()}\n\n"

        txt += f'Current population: {len(self.Population)}'

        return txt
