from Region import *


class Simulation:
    IMMUNITY_TIME = 0 * 12
    SELECTED_COUPLES = 10
    INITIAL_COUPLES = 1000
    STARTING_LOCATION = (850, 400)
    INITIAL_STRENGTH = 10
    NEW_REGION_LOCK = Lock()
    ACTION_POOL_LOCK = Lock()

    def __init__(self, sim_map, imported=None, separated_imported=None, all_couples=False):
        """
        The object handling macro operations and containing the whole simulation.

        :param sim_map: A map with size (1600, 800) pixels. This map will determine the terrain on which the simulation
        will run.
        :param imported: a list containing couples of disassembled brains. imported.shape=(num, 2, 2)
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
                f = Person(properties=[Gender.Female, Simulation.INITIAL_STRENGTH, np.array(Simulation.STARTING_LOCATION)],
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

            for brain in actual_male_brains[:Simulation.INITIAL_COUPLES - len(self.regions[initial_region_index].Population)]:
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

            for brain in actual_female_brains[:Simulation.INITIAL_COUPLES * 2 - len(self.regions[initial_region_index].Population)]:
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

    def handle_region(self, reg: Region):
        try:
            reg.clear()
            # person_exec = concurrent.futures.ThreadPoolExecutor()
            # futures_person = [person_exec.submit(self.handle_person, person, reg) for person in reg.Population]
            # concurrent.futures.wait(futures_person)
            # person_exec.shutdown()
            for p in reg.Population:
                self.handle_person(p, reg)

            # print("3: ", reg.location)
            for newborn in reg.newborns:
                # print("4a: ", reg.location)
                self.collective.add_person(newborn)
                reg.add_person(newborn)
                # print("4b: ", reg.location)

            for social_connector in reg.social_connectors:
                # print("5a: ", reg.location)
                social_connector: Person
                if social_connector.brain.is_friendly():  # whether he likes any other people.
                    social_connector.brain.improve_attitudes_toward_me(region=reg)
                    social_connector.brain.improve_my_attitudes(multiplier=0.5)
                    # print("5b: ", reg.location)

                    if social_connector.partner:
                        if social_connector.partner in reg.social_connectors:
                            social_connector.prepare_next_generation(social_connector.partner)
                            # print("5c: ", reg.location)
                        elif self.map.distance(social_connector.location,
                                               social_connector.partner.location) > Person.GIVE_UP_DISTANCE:
                            social_connector.partner = None
                    else:
                        # print("5d: ", reg.location)
                        social_connector.partner = social_connector.partner_selection(region=reg)
                        # print("5e: ", reg.location)
                        if social_connector.partner:
                            social_connector.partner.partner = social_connector
                            # print("5f: ", reg.location)
                else:
                    # print("5g: ", reg.location)
                    social_connector.brain.improve_my_attitudes()

            # print("6: ", reg.location)
            for d in reg.dead:
                # print("7a: ", reg.location)
                self.kill_person(d, reg)
                # print("7b: ", reg.location)
            for rlp in reg.relocating_people:
                # print("8a: ", reg.location)
                reg.remove_person(rlp)
                # print("8b: ", reg.location)

        except Exception as ex:
            print('handle_region - Exception caught:', ex)

    def handle_person(self, p: Person, reg: Region):
        # print("P1: ", reg.location, p.id)
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

            # print("P3: ", reg.location, p.id)
            p.aging()  # handles growing up and old people's aging process.

            # print("P4: ", reg.location, p.id)
            if p.natural_death_chance() and self.Time > Simulation.IMMUNITY_TIME:
                reg.dead = p
                # print("P5: ", reg.location, p.id)
                return None

            # print("P6: ", reg.location, p.id)
            action = p.action(region=reg)
            # print("P7: ", reg.location, p.id)
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

                # print("P8: ", reg.location, p.id)
                if p.location[1] >= 800 or p.location[1] < 0:  # if exited the boundaries of the map.
                    reg.dead = p
                    # print("P9: ", reg.location, p.id)
                    return None

                reg.relocating_people = p  # to not mess up the indexing in the original region.
                location = tuple(np.flip(p.location))
                new_reg = self.regions[location]
                # print("P10: ", reg.location, p.id)
                if new_reg and new_reg.Population:  # if tried to relocate into an occupied region.
                    new_reg.add_person(p)
                    # print("P11a: ", reg.location, p.id)
                else:
                    with (self.NEW_REGION_LOCK):
                        if self.new_regions.count(location) == 0 and \
                           self.region_iterator.count(location) == 0:
                            self.new_regions.append(location)
                        new_reg = self.regions[location]
                        # print("P10: ", reg.location, p.id)
                        if new_reg:  # if tried to relocate into an occupied region.
                            new_reg.add_person(p)
                        else:
                            # print("P11b: ", reg.location, p.id)
                            neighbors = self.map.get_surroundings(self.regions, p.location, dtype=Region)
                            # print("P11c: ", reg.location, p.id)
                            new_reg = Region(location=p.location,
                                             surrounding_biomes=self.map.get_surroundings(self.map.biome_map, p.location),
                                             neighbors=neighbors)
                            # print("P11d: ", reg.location, p.id)
                            new_reg.add_person(p)
                            # print("P11e: ", reg.location, p.id)
                            self.update_neighbors(neighbors)  # informs the neighbors that a person joined new_reg.
                            # print("P11f: ", reg.location, p.id)
                        self.regions[location] = new_reg

            p.action_flag = True  # so that it won't get iterated upon again if relocated.
            # print("P12: ", reg.location, p.id)

    def is_eradicated(self):
        return not self.region_iterator

    @staticmethod
    def update_neighbors(area):
        center = area[tuple(np.array(area.shape) // 2)]
        for i, j in zip(*np.where(area)):
            relative_position = np.array(area.shape) // 2 - [i, j]
            area[i, j].neighbors_update(tuple([1, 1] + relative_position), center)

    def kill_person(self, p, reg):
        self.collective.remove_person()
        reg.remove_person(p)
        p.isAlive = False
        if p.partner:
            p.partner.partner = None

    def get_historical_figure(self, pid):
        hf = self.collective.historical_population[pid]
        history = hf.brain.get_history()[1:hf.age + 1]
        return hf, np.reshape(np.append(history, np.zeros((12 - len(history) % 12) % 12)), (-1, 12))

    def get_attitudes(self, pid):
        return self.collective.historical_population[pid].collective.world_attitudes[pid]

    def evaluate(self, by_alive=False):
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
        reprocessed_best_minds = []
        temp = []
        for i in best_minds:
            reprocessed_best_minds.append(i[0])
            temp.append(i[1])
        reprocessed_best_minds.extend(temp)

        unified_lst = np.append(male_lst, female_lst, axis=0)
        return reprocessed_best_minds, unified_lst

    def kill_all_travelers(self):
        for i, j in self.region_iterator:
            if (i, j) != (400, 850):
                self.regions[i, j]._Population = None
                self.regions[i, j] = None
                neighbors = self.map.get_surroundings(self.regions, (j, i), dtype=Region)
                self.update_neighbors(neighbors)
        self.region_iterator = [(400, 850)]

    def pop_num(self):
        return self.collective.population_size - self.collective.dead

    def __repr__(self):
        txt = f"Year: {self.Time // 12}"
        return txt

    def display(self):
        txt = f"Year: {self.Time // 12}\n\n"

        for i, j in self.region_iterator:

            txt += self.regions[i, j].display()
            txt += '\n----------\n\n\n'

        if self.is_eradicated():
            txt = "SPECIES GONE"

        print(txt)
        print(self.pop_num())
