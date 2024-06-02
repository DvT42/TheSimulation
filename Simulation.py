import concurrent.futures

from Region import *


class Simulation:
    """
      A class that manages the overall simulation process and handles macro-level interactions.

      This class is responsible for:

      * Initializing and managing the simulation environment
      * Maintaining the state of the population and their interactions
      * Simulating the passage of time and its effects
      * Providing access to simulation data and statistics

      Constants:
          IMMUNITY_TIME (int): The initial immunity period after simulation start (in months) during which the mortality function is not applied.
          SELECTED_COUPLES (int): The number of selected couples to include in the simulation (may be irrelevant depending on simulation settings).
          INITIAL_COUPLES (int): The initial number of couples to populate the simulation (must be greater than SELECTED_COUPLES if both are used).
          STARTING_LOCATION (tuple): The starting location on the map for all individuals.
          INITIAL_STRENGTH (int): The initial strength value given to each individual.
          NEW_REGION_LOCK (Lock): A lock to protect the creation of new regions, preventing concurrent region creation at the same location.
          ACTION_POOL_LOCK (Lock): A lock to protect the modification of each region's action pool, a tool for visualizing action distribution within a region.

      Attributes:
          collective (Collective): A unique Collective object for each simulation.
          map (Map): The Map object that manages distance calculations and the simulation grid.
          regions (2D array): A 2D array of Region objects indexed by their location, or None if no region exists at that location.
          region_iterator (list): An updated list of all regions with any population. This allows efficient iteration over the populated regions without traversing the entire array each time.
          new_regions (list): A list that is reset each month and holds the regions that should be added to the region iterator in the current month. This ensures the iterator is not updated during runtime.
          Time (int): The elapsed time since simulation start in simulated months.
    """
    VISUAL = False
    IMMUNITY_TIME = 0 * 12
    SELECTED_COUPLES = 10
    INITIAL_COUPLES = 1000
    STARTING_LOCATION = (770, 330)
    INITIAL_STRENGTH = 10
    NEW_REGION_LOCK = Lock()
    ACTION_POOL_LOCK = Lock()

    def __init__(self, sim_map, imported=None, separated_imported=None, all_couples=False):
        """
        The object handling macro operations and containing the whole simulation.

        :param sim_map: A map with size (1600, 800) pixels. This map will determine the terrain on which the simulation
        will run.
        :param imported: a list containing couples of disassembled brains. imported.shape=(num, 2, 2)
        :param all_couples: boolean parameter whether Simulation.SELECTED_COUPLES should be disregarded.
        """
        self.collective = Collective()
        Person.Person_reset()

        self.map = sim_map

        self.regions = np.empty((800, 1600), dtype=Region)  # an array containing all regions indexed by location.
        initial_region_index = tuple(np.flip(Simulation.STARTING_LOCATION, 0))

        self.regions[initial_region_index] = (  # First region to contain all initial Person-s.
            Region(location=Simulation.STARTING_LOCATION,
                   surrounding_biomes=self.map.get_surroundings(self.map.biome_map, Simulation.STARTING_LOCATION)))

        # a list containing all existing regions' indexes, to make the iteration easier.
        self.region_iterator = [initial_region_index]

        if all_couples and imported:
            actual_brains = Brain.assemble_brains(imported)

            for brain_couple in actual_brains:
                for _ in range(Simulation.INITIAL_COUPLES // len(actual_brains)):
                    # initiate the male from the couple.
                    m = Person(
                        properties=[Gender.Male, Simulation.INITIAL_STRENGTH, np.array(Simulation.STARTING_LOCATION)],
                        collective=self.collective,
                        brain=brain_couple[0])
                    m.brain.transfer_brain(m)

                    # initiate the female from the couple.
                    f = Person(
                        properties=[Gender.Female, Simulation.INITIAL_STRENGTH, np.array(Simulation.STARTING_LOCATION)],
                        collective=self.collective,
                        brain=brain_couple[1])
                    f.brain.transfer_brain(f)

                    self.regions[initial_region_index].add_person(m)
                    self.regions[initial_region_index].add_person(f)

            # the remaining people will be initiated from the top brains.
            for i in range(Simulation.INITIAL_COUPLES % len(actual_brains)):
                # initiate the male from the couple.
                m = Person(
                    properties=[Gender.Male, Simulation.INITIAL_STRENGTH, np.array(Simulation.STARTING_LOCATION)],
                    collective=self.collective,
                    brain=actual_brains[i][0])
                m.brain.transfer_brain(m)

                # initiate the female from the couple.
                f = Person(
                    properties=[Gender.Female, Simulation.INITIAL_STRENGTH, np.array(Simulation.STARTING_LOCATION)],
                    collective=self.collective,
                    brain=actual_brains[i][1])
                f.brain.transfer_brain(f)

                self.regions[initial_region_index].add_person(m)
                self.regions[initial_region_index].add_person(f)

        elif Simulation.SELECTED_COUPLES and type(imported) is np.ndarray and imported.any():
            actual_brains = Brain.assemble_brains(imported[:Simulation.SELECTED_COUPLES])

            for brain_couple in actual_brains:
                for _ in range(Simulation.INITIAL_COUPLES // Simulation.SELECTED_COUPLES):
                    # initiate the male from the couple.
                    m = Person(
                        properties=[Gender.Male, Simulation.INITIAL_STRENGTH, np.array(Simulation.STARTING_LOCATION)],
                        collective=self.collective,
                        brain=brain_couple[0])
                    m.brain.transfer_brain(m)

                    # initiate the female from the couple.
                    f = Person(
                        properties=[Gender.Female, Simulation.INITIAL_STRENGTH, np.array(Simulation.STARTING_LOCATION)],
                        collective=self.collective,
                        brain=brain_couple[1])
                    f.brain.transfer_brain(f)

                    self.regions[initial_region_index].add_person(m)
                    self.regions[initial_region_index].add_person(f)

            for brain_couple in actual_brains[:Simulation.INITIAL_COUPLES % Simulation.SELECTED_COUPLES]:
                # initiate the male from the couple.
                m = Person(
                    properties=[Gender.Male, Simulation.INITIAL_STRENGTH, np.array(Simulation.STARTING_LOCATION)],
                    collective=self.collective,
                    brain=brain_couple[0])
                m.brain.transfer_brain(m)

                # initiate the female from the couple.
                f = Person(
                    properties=[Gender.Female, Simulation.INITIAL_STRENGTH, np.array(Simulation.STARTING_LOCATION)],
                    collective=self.collective,
                    brain=brain_couple[1])
                f.brain.transfer_brain(f)

                self.regions[initial_region_index].add_person(m)
                self.regions[initial_region_index].add_person(f)

        elif separated_imported:
            actual_male_brains = Brain.assemble_separated_brains(separated_imported[0])
            actual_female_brains = Brain.assemble_separated_brains(separated_imported[1])

            for brain in actual_male_brains:
                brain: Brain
                for _ in range(Simulation.INITIAL_COUPLES // len(actual_male_brains)):
                    # initiate the male from the couple.
                    m = Person(
                        properties=[Gender.Male, Simulation.INITIAL_STRENGTH, np.array(Simulation.STARTING_LOCATION)],
                        collective=self.collective,
                        brain=brain)
                    m.brain.transfer_brain(m)
                    m.brain.update_location_history()

                    self.regions[initial_region_index].add_person(m)

            for brain in actual_male_brains[
                         :Simulation.INITIAL_COUPLES - len(self.regions[initial_region_index].Population)]:
                # initiate the male from the couple.
                m = Person(
                    properties=[Gender.Male, Simulation.INITIAL_STRENGTH, np.array(Simulation.STARTING_LOCATION)],
                    collective=self.collective,
                    brain=brain)
                m.brain.transfer_brain(m)
                m.brain.update_location_history()
                self.regions[initial_region_index].add_person(m)

            for brain in actual_female_brains:
                brain: Brain
                for _ in range(Simulation.INITIAL_COUPLES // len(actual_female_brains)):
                    # initiate the female from the couple.
                    f = Person(
                        properties=[Gender.Female, Simulation.INITIAL_STRENGTH, np.array(Simulation.STARTING_LOCATION)],
                        collective=self.collective,
                        brain=brain)
                    f.brain.transfer_brain(f)
                    f.brain.update_location_history()

                    self.regions[initial_region_index].add_person(f)

            for brain in actual_female_brains[
                         :Simulation.INITIAL_COUPLES * 2 - len(self.regions[initial_region_index].Population)]:
                # initiate the female from the couple.
                f = Person(
                    properties=[Gender.Female, Simulation.INITIAL_STRENGTH, np.array(Simulation.STARTING_LOCATION)],
                    collective=self.collective,
                    brain=brain)
                f.brain.transfer_brain(f)
                f.brain.update_location_history()

                self.regions[initial_region_index].add_person(f)

        else:
            for i in range(Simulation.INITIAL_COUPLES):  # people initiated with random brains.
                self.regions[initial_region_index].add_person(Person(
                    properties=[Gender.Male, Simulation.INITIAL_STRENGTH, np.array(Simulation.STARTING_LOCATION)],
                    collective=self.collective))
                self.regions[initial_region_index].add_person(Person(
                    properties=[Gender.Female, Simulation.INITIAL_STRENGTH, np.array(Simulation.STARTING_LOCATION)],
                    collective=self.collective))

        for p in self.regions[initial_region_index].Population:
            self.collective.add_person(p)

        # get all the first impressions of the initial people.
        self.regions[initial_region_index].mass_encounter()

        self.new_regions = []

        self.Time = 0

    # @jit(target_backend='cuda')
    def month_advancement(self, executors: concurrent.futures.ThreadPoolExecutor):
        """
        This function constitutes all operations needed to be preformed each month.
        :parameter executors: parameter used to distribute work across regions.
        """
        self.Time += 1
        self.new_regions = []

        for i, j in self.region_iterator:
            self.regions[i, j].falsify_action_flags()

        # noinspection PyTypeChecker
        futures_region = [executors.submit(self.handle_region, self.regions[i, j]) for i, j in
                          self.region_iterator]
        concurrent.futures.wait(futures_region)
        # for i, j in self.region_iterator:
        #     self.handle_region(self.regions[i, j])

        # futures_region = [executors.submit(self.regions[i, j].introduce_newcomers) for i, j in
        #                   self.region_iterator]
        # concurrent.futures.wait(futures_region)
        for i, j in self.region_iterator:
            self.regions[i, j].introduce_newcomers()

        self.region_iterator.extend(self.new_regions)
        empty_regions = []
        for i, j in self.region_iterator:
            if len(self.regions[i, j].Population) == 0:
                empty_regions.append((i, j))

        for i, j in empty_regions:
            self.region_iterator.remove((i, j))
            # self.regions[i, j] = None
            # neighbors = self.map.get_surroundings(self.regions, (j, i), dtype=Region)
            # self.update_neighbors(neighbors)

        pop = 0
        for i, j in self.region_iterator:
            pop += len(self.regions[i, j].Population)

        if self.pop_num() != pop:
            print(pop)

        if Simulation.VISUAL:
            self.map.update_map(self.get_locations(), self.pop_density())

    def handle_region(self, reg: Region):
        """
            Handles region-level operations during a simulation month.

            Args:
                reg: The Region object to be processed

            This function is called exclusively within the month advancement process. It performs all necessary region-level actions for a single simulation month.
            """
        try:
            reg.clear()

            # person_exec = concurrent.futures.ThreadPoolExecutor()
            # futures_person = [person_exec.submit(self.handle_person, person, reg) for person in reg.Population]
            # concurrent.futures.wait(futures_person)
            # person_exec.shutdown()
            for p in reg.Population:
                self.handle_person(p, reg)

            for newborn in reg.newborns:
                self.collective.add_person(newborn)
                reg.add_person(newborn)

            for social_connector in reg.social_connectors:
                social_connector: Person
                if social_connector.brain.is_friendly():  # whether he likes any other people.
                    social_connector.brain.improve_attitudes_toward_me(region=reg)
                    social_connector.brain.improve_my_attitudes(multiplier=0.5)

                    if social_connector.partner:
                        if social_connector.partner in reg.social_connectors:
                            social_connector.prepare_next_generation(social_connector.partner)
                        elif self.map.distance(social_connector.location,
                                               social_connector.partner.location) > Person.GIVE_UP_DISTANCE:
                            social_connector.partner = None
                    else:
                        social_connector.partner = social_connector.partner_selection(region=reg)
                        if social_connector.partner:
                            social_connector.partner.partner = social_connector
                else:
                    social_connector.brain.improve_my_attitudes()

            for d in reg.dead:
                self.kill_person(d, reg)
            for rlp in reg.relocating_people:
                reg.remove_person(rlp)

        except Exception as ex:
            print('handle_region - Exception caught:', ex)

    def handle_person(self, p: Person, reg: Region):
        """
            Handles individual-level operations during a simulation month.

            Args:
                p: The Person object to be processed
                reg: The Region object in which the person resides

            This function is called exclusively within the `handle_region()` function. It performs all necessary individual-level actions for a single simulation month.
            """
        if not p.action_flag:
            # Person.ages[:Person.runningID] += 1 // might decide to transfer it to this format later
            p.age += 1

            #  - handle pregnancy
            if p.gender == Gender.Female:
                if p.father_of_child:
                    if p.pregnancy == Person.PREGNANCY_LENGTH:
                        # print("P2: ", reg.location, p.id)
                        newborn = p.birth()
                        reg.newborns = newborn
                    else:
                        p.pregnancy += 1
                elif p.age > p.readiness and p.youth > 0:
                    p.youth -= 1

            p.aging()  # handles growing up and old people's aging process.

            if p.natural_death_chance(reg) and self.Time > Simulation.IMMUNITY_TIME:
                reg.dead = p
                return None

            action = p.action(region=reg)
            if action == 0:  # social connection.
                reg.social_connectors = p
                with self.ACTION_POOL_LOCK:
                    reg.action_pool[0] += 1
            elif action == 1:  # strength.
                with self.ACTION_POOL_LOCK:
                    reg.action_pool[1] += 1
            elif action in np.arange(2, 10):  # relocation.
                with self.ACTION_POOL_LOCK:
                    reg.action_pool[2] += 1

                if p.location[1] >= 800 or p.location[1] < 0:  # if exited the boundaries of the map.
                    reg.dead = p
                    return None

                reg.relocating_people = p  # to not mess up the indexing in the original region.
                location = tuple(np.flip(p.location))
                new_reg = self.regions[location]
                if new_reg and new_reg.Population:  # if tried to relocate into an occupied region.
                    new_reg.add_person(p)
                else:
                    with (self.NEW_REGION_LOCK):
                        if self.new_regions.count(location) == 0 and \
                                self.region_iterator.count(location) == 0:
                            self.new_regions.append(location)
                        new_reg = self.regions[location]
                        if new_reg:  # if tried to relocate into an occupied region.
                            new_reg.add_person(p)
                        else:
                            neighbors = self.map.get_surroundings(self.regions, p.location, dtype=Region)
                            new_reg = Region(location=p.location,
                                             surrounding_biomes=self.map.get_surroundings(self.map.biome_map,
                                                                                          p.location),
                                             neighbors=neighbors)
                            new_reg.add_person(p)
                            self.update_neighbors(neighbors)  # informs the neighbors that a person joined new_reg.
                        self.regions[location] = new_reg

            p.action_flag = True  # so that it won't get iterated upon again if relocated.

    def is_eradicated(self):
        """
            Checks if the entire population has been eradicated.

            Returns:
                True if the population is extinct, False otherwise
            """
        return not self.region_iterator

    @staticmethod
    def update_neighbors(area):
        """
           Updates neighboring regions with information about a newly added region.

           Args:
               area: A 3x3 2D array representing the surrounding area to be updated. The new region is located at the center.
           """
        center = area[tuple(np.array(area.shape) // 2)]
        for i, j in zip(*np.where(area)):
            relative_position = np.array(area.shape) // 2 - [i, j]
            area[i, j].neighbors_update(tuple([1, 1] + relative_position), center)

    def kill_person(self, p, reg):
        """
            Handles the removal of a deceased person from the simulation.

            Args:
                p: The Person object representing the deceased individual
                reg: The Region object from which the person should be removed
            """
        self.collective.remove_person()
        reg.remove_person(p)
        p.isAlive = False
        if p.partner:
            p.partner.partner = None

    def get_historical_figure(self, pid):
        """
            Retrieves historical data about a specific person from the simulation.

            Args:
                pid (int): The unique identifier of the person to retrieve data for

            Returns:
                Person: An object representing the person with the specified ID, or None if no such person exists
                np.ndarray: The person's history.

            Raises:
                KeyError: If the provided person ID is not found in the simulation's historical records
            """
        try:
            hf = self.collective.historical_population[pid]
            history = hf.brain.get_history()[1:hf.age + 1]
            return hf, np.reshape(np.append(history, np.zeros((12 - len(history) % 12) % 12)), (-1, 12))
        except KeyError:
            print("ID not in the simulation")

    def get_attitudes(self, pid):
        """
            Retrieves the attitudes of a specific person from the simulation.

            Args:
                pid (int): The unique identifier of the person to retrieve attitudes for

            Returns:
                dict: A dictionary containing the person's attitudes towards various aspects of the simulation, or None if no such person exists

            Raises:
                KeyError: If the provided person ID is not found in the simulation's current population
            """
        try:
            return self.collective.historical_population[pid].collective.world_attitudes[pid]
        except KeyError:
            print("ID not in the simulation")

    def evaluate(self, by_alive=False):
        """
            Evaluates the simulation and returns a summary of data for each individual.

            Args:
                by_alive (bool, optional): If True, filters the results to include only living individuals (default: False)

            Returns:
                tuple[list]: A tuple of lists, each containing summary data for a single individual.
        """
        if by_alive:
            return ([p.brain.get_models()
                     for p in self.collective.historical_population if p.isAlive and p.gender == 1],
                    [p.brain.get_models()
                     for p in self.collective.historical_population if p.isAlive and p.gender == 0])

        else:
            return ([person.brain.get_models() for person in self.collective.historical_population],
                    [p.gender for p in self.collective.historical_population],
                    [person.child_num for person in self.collective.historical_population],
                    [p.age for p in self.collective.historical_population])

    @staticmethod
    def find_best_minds(evaluated_list, children_bearers='best'):
        """
           Identifies the best individuals from a list of evaluated individuals and returns their data.

           Args:
               evaluated_list (tuple[list]): A 2D array containing individual evaluation data, typically obtained from the `evaluate()` function.
              children_bearers (str, optional): Determines the selection of individuals for procreation:
                   'all': Includes all individuals
                   'best': Selects the top-performing individuals
                   'enough': Selects enough individuals to fill the next simulation (default: 'best')

           Returns:
               tuple[list]: A tuple of lists containing data for the selected best individuals.
           """
        neural_list, genders, children, ages = evaluated_list
        best_minds = []
        his = np.swapaxes(np.array([np.arange(len(neural_list)), genders, children, ages]), 0, 1)
        sorted_idx = np.lexsort((his[:, 3], his[:, 2], his[:, 1]))
        sorted_his = np.array([his[i] for i in sorted_idx])

        gender_idx = np.argmax(sorted_his[:, 1])
        male_lst, female_lst = sorted_his[gender_idx:], sorted_his[:gender_idx]

        if children_bearers == 'all':
            m_children_antidx = np.argmin(np.flip(male_lst[:, 2]))
            f_children_antidx = np.argmin(np.flip(female_lst[:, 2]))

            for i in range(min(m_children_antidx, f_children_antidx)):
                best_minds.append((neural_list[male_lst[-i - 1][0]], neural_list[female_lst[-i - 1][0]]))

            return (best_minds,
                    male_lst[-len(best_minds):],
                    female_lst[-len(best_minds):])

        elif children_bearers == 'enough':
            m_children_antidx = np.argmin(np.flip(male_lst[:, 2]))
            f_children_antidx = np.argmin(np.flip(female_lst[:, 2]))

            for i in range(min(m_children_antidx, f_children_antidx, Simulation.INITIAL_COUPLES)):
                best_minds.append((neural_list[male_lst[-i - 1][0]], neural_list[female_lst[-i - 1][0]]))

            return (best_minds,
                    male_lst[-len(best_minds):],
                    female_lst[-len(best_minds):])

        for i in range(Simulation.SELECTED_COUPLES):
            best_minds.append((neural_list[male_lst[-i - 1][0]], neural_list[female_lst[-i - 1][0]]))

        return (best_minds,
                male_lst[-Simulation.SELECTED_COUPLES:],
                female_lst[-Simulation.SELECTED_COUPLES:])

    @staticmethod
    def prepare_best_for_reprocess(best_minds, male_lst, female_lst):
        """
           Prepares the best individuals for reprocessing in the simulation.

           Args:
               best_minds (List[tuple[tuple[Neural_Network]]]): A list containing the minds of the best individuals.
               male_lst (List): A list of data for all male individuals.
               female_lst (List): A list of data for all female individuals.

           Returns:
               reprocessed_best_minds(List[tuple[Neural_Network]]): A list containing the minds of the best individuals, reorganized so that the first half is for the males and the second the females.
               unified_lst(list): a list containing male_lst and female_lst along the same axis.
           """
        reprocessed_best_minds = []
        temp = []
        for i in best_minds:
            reprocessed_best_minds.append(i[0])
            temp.append(i[1])
        reprocessed_best_minds.extend(temp)

        unified_lst = np.append(male_lst, female_lst, axis=0)
        return reprocessed_best_minds, unified_lst

    def pop_num(self):
        """
            Calculates the current population size, counting only living individuals.

            Returns:
                int: The number of living individuals in the simulation.
            """
        return self.collective.population_size - self.collective.dead

    def divide_by_generation(self):
        """
            Calculates the number of individuals in each generation based on the historical population data.

            Returns:
                List[int]: A list where each index represents a generation and the value at that index represents the number of individuals in that generation.
        """
        gens = []
        for p in self.collective.historical_population:
            if p.generation >= len(gens):
                for i in range(p.generation - len(gens) + 1):
                    gens.append(0)
            gens[p.generation] += 1
        return gens

    def pop_density(self):
        density = []
        for ij in self.region_iterator:
            density.append(len(self.regions[ij].Population))
        return density

    def get_number_of_children(self):
        """
            Calculates the number of children based on the current population size and initial number of couples.

            Returns:
                int: The number of children born in the simulation.
            """
        return self.collective.population_size - Simulation.INITIAL_COUPLES * 2

    def get_locations(self):
        return np.flip(np.asarray(self.region_iterator), axis=1)

    def __repr__(self):
        """
        Represents the Simulation object as a string for debugging and printing purposes.

        Returns:
            str: A string representation of the Simulation object, presenting its current Time.
        """
        txt = f"Year: {self.Time // 12}"
        return txt

    def display(self):
        """
            Displays a summary of the current state of the simulation, including population details, regions, and the simulation year.
        """
        txt = f"Year: {self.Time // 12}\n\n"

        for i, j in self.region_iterator:
            txt += self.regions[i, j].display()
            txt += '\n----------\n\n\n'

        if self.is_eradicated():
            txt = "SPECIES GONE"

        print(txt)
        print(self.pop_num())
