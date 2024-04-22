from multiprocessing import Queue
from Region import *


class Simulation:
    IMMUNITY_TIME = 0 * 12
    SELECTED_COUPLES = 10
    INITIAL_COUPLES = 1000
    STARTING_LOCATION = (850, 400)
    INITIAL_STRENGTH = 10

    def __init__(self, sim_map, imported=None, separated_imported=None):
        """
        The object handling macro operations and containing the whole simulation.

        :param sim_map: A map with size (1600, 800) pixels. This map will determine the terrain on which the simulation
        will run.
        :param imported: an array containing couples of disassembled brains. imported.shape=(num, 2, 2)
        """
        self.collective = Collective()
        Person.Person_reset(Simulation.INITIAL_COUPLES)

        self.map = sim_map
        self.regions = np.empty((800, 1600), dtype=Region)  # an array containing all regions indexed by location.
        initial_region_index = tuple(np.flip(Simulation.STARTING_LOCATION, 0))

        self.regions[initial_region_index] = (  # First region to contain all initial Person-s.
            Region(location=Simulation.STARTING_LOCATION,
                   surrounding_biomes=self.map.get_surroundings(self.map.biome_map, Simulation.STARTING_LOCATION),
                   biome=self.map.get_biome(Simulation.STARTING_LOCATION)))

        # a list containing all existing regions' indexes, to make the iteration easier.
        self.region_iterator = [initial_region_index]

        if Simulation.SELECTED_COUPLES and type(imported) is np.ndarray and imported.any():  # if importation is needed.
            actual_brains = Simulation.assemble_brains(imported[:Simulation.SELECTED_COUPLES])

            for brain_couple in actual_brains:
                for _ in range(Simulation.INITIAL_COUPLES // Simulation.SELECTED_COUPLES):
                    # initiate the male from the couple.
                    m = Person(father=[Gender.Male, Simulation.INITIAL_STRENGTH, np.array(Simulation.STARTING_LOCATION)], collective=self.collective)
                    m.brain = brain_couple[0]
                    brain_couple[0].transfer_brain(m)

                    # initiate the female from the couple.
                    f = Person(father=[Gender.Female, Simulation.INITIAL_STRENGTH, np.array(Simulation.STARTING_LOCATION)], collective=self.collective)
                    f.brain = brain_couple[1]
                    brain_couple[1].transfer_brain(f)

                    self.regions[initial_region_index].add_person(m)
                    self.regions[initial_region_index].add_person(f)

            # the remaining people will be initiated with random brains.
            for c in range(Simulation.INITIAL_COUPLES % Simulation.SELECTED_COUPLES):
                self.regions[initial_region_index].add_person(Person(
                    father=[Gender.Male, Simulation.INITIAL_STRENGTH, np.array(Simulation.STARTING_LOCATION)], collective=self.collective))
                self.regions[initial_region_index].add_person(Person(
                    father=[Gender.Female, Simulation.INITIAL_STRENGTH, np.array(Simulation.STARTING_LOCATION)], collective=self.collective))

        elif type(separated_imported) is np.ndarray and separated_imported.any():
            actual_male_brains = Simulation.assemble_separated_brains(separated_imported[0])
            actual_female_brains = Simulation.assemble_separated_brains(separated_imported[1])

            for brain in actual_male_brains:
                brain: Brain
                for _ in range(Simulation.INITIAL_COUPLES // len(actual_male_brains)):
                    # initiate the male from the couple.
                    m = Person(
                        father=[Gender.Male, Simulation.INITIAL_STRENGTH, np.array(Simulation.STARTING_LOCATION)],
                        collective=self.collective)
                    m.brain = brain
                    brain.transfer_brain(m)

                    self.regions[initial_region_index].add_person(m)

            for brain in actual_female_brains:
                brain: Brain
                for _ in range(Simulation.INITIAL_COUPLES // len(actual_female_brains)):
                    # initiate the female from the couple.
                    f = Person(
                        father=[Gender.Female, Simulation.INITIAL_STRENGTH, np.array(Simulation.STARTING_LOCATION)],
                        collective=self.collective)
                    f.brain = brain
                    brain.transfer_brain(f)

                    self.regions[initial_region_index].add_person(f)

            for c in range(Simulation.INITIAL_COUPLES - len(self.regions[initial_region_index].Population)):
                self.regions[initial_region_index].add_person(Person(
                    father=[Gender.Male, Simulation.INITIAL_STRENGTH, np.array(Simulation.STARTING_LOCATION)], collective=self.collective))
                self.regions[initial_region_index].add_person(Person(
                    father=[Gender.Female, Simulation.INITIAL_STRENGTH, np.array(Simulation.STARTING_LOCATION)], collective=self.collective))

        else:
            for i in range(Simulation.INITIAL_COUPLES):  # people initiated with random brains.
                self.regions[initial_region_index].add_person(Person(
                    father=[Gender.Male, Simulation.INITIAL_STRENGTH, np.array(Simulation.STARTING_LOCATION)], collective=self.collective))
                self.regions[initial_region_index].add_person(Person(
                    father=[Gender.Female, Simulation.INITIAL_STRENGTH, np.array(Simulation.STARTING_LOCATION)], collective=self.collective))

        for p in self.regions[initial_region_index].Population:
            self.collective.add_person(p)

        # get all the first impressions of the initial people.
        self.regions[initial_region_index].mass_encounter()

        self.new_regions = []

        self.Time = 0

    # @jit(target_backend='cuda')
    def month_advancement(self, executor: concurrent.futures.Executor):
        """
        This function constitutes all operations needed to be preformed each month.
        """
        self.Time += 1
        self.new_regions = []

        for i, j in self.region_iterator:
            self.regions[i, j].falsify_action_flags()

        # results_region = Queue()
        futures_region = [executor.submit(self.handle_region, self.regions[i, j], executor) for i, j in
                          self.region_iterator]
        # for future in concurrent.futures.as_completed(futures_region):
        #     results_region.put(future.result())
        concurrent.futures.wait(futures_region)

        results_region = Queue()
        futures_region = [executor.submit(self.regions[i, j].introduce_newcomers, executor) for i, j in
                          self.region_iterator]
        for future in concurrent.futures.as_completed(futures_region):
            results_region.put(future.result())
        concurrent.futures.wait(futures_region)

        empty_regions = []
        for i, j in self.region_iterator:
            reg = self.regions[i, j]
            if not reg.Population:
                empty_regions.append((i, j))

        for i, j in empty_regions:
            self.regions[i, j] = None
            self.region_iterator.remove((i, j))
            neighbors = self.map.get_surroundings(self.regions, (j, i), dtype=Region)
            self.update_neighbors(neighbors)

        self.region_iterator.extend(self.new_regions)

    def handle_region(self, reg: Region, executor):
        reg.dead = []
        reg.action_pool = [0, 0, 0]
        reg.newborns = []
        reg.social_connectors = []
        reg.relocating_people = []
        reg.newcomers = []

        # results = Queue()
        futures_person = [executor.submit(self.handle_person, person, reg) for person in reg.Population]
        # for future in concurrent.futures.as_completed(futures_person):
        #     results.put(future.result())
        concurrent.futures.wait(futures_person)

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
            Simulation.kill_person(d, reg)
        for rlp in reg.relocating_people:
            reg.remove_person(rlp)

    def handle_person(self, p: Person, reg: Region):
        if not p.action_flag:
            # Person.ages[:Person.runningID] += 1 // might decide to transfer it to this format later
            Person.ages[p.id] += 1

            #  - handle pregnancy
            if p.gender == Gender.Female:
                if p.father_of_child:
                    if p.pregnancy == Person.PREGNANCY_LENGTH:
                        newborn = p.birth()
                        reg.newborns.append(newborn)
                    else:
                        p.pregnancy += 1
                elif p.age() > p.readiness and p.youngness > 0:
                    p.youngness -= 1

            p.aging()  # handles old people's aging process.

            # handle growing up
            if p.age() < 15 * 12:
                p.strength += 0.25

            if p.natural_death_chance() and self.Time > Simulation.IMMUNITY_TIME:
                reg.dead.append(p)
                return None

            action = p.action(region=reg)
            if action == 0:  # social connection.
                reg.social_connectors.append(p)
                reg.action_pool[0] += 1
            elif action == 1:  # strength.
                reg.action_pool[1] += 1
            elif action in np.arange(2, 10):  # relocation.
                reg.action_pool[2] += 1

                if p.location[1] >= 800 or p.location[1] < 0:  # if exited the boundaries of the map.
                    reg.dead.append(p)
                    return None

                new_reg = self.regions[tuple(np.flip(p.location))]
                reg.relocating_people.append(p)  # to not mess up the indexing in the original region.

                if new_reg:  # if tried to relocate into an occupied region.
                    new_reg.add_person(p)

                else:
                    neighbors = self.map.get_surroundings(self.regions, p.location, dtype=Region)
                    new_reg = Region(location=p.location,
                                     surrounding_biomes=self.map.get_surroundings(self.map.biome_map, p.location),
                                     biome=self.map.get_biome(p.location),
                                     neighbors=neighbors)
                    new_reg.add_person(p)
                    self.regions[tuple(np.flip(p.location))] = new_reg
                    self.update_neighbors(neighbors)  # informs the neighbors that a person joined new_reg.
                    self.new_regions.append(tuple(np.flip(p.location)))

            p.action_flag = True  # so that it won't get iterated upon again if relocated.

    def is_eradicated(self):
        return not np.any(self.regions)

    @staticmethod
    def update_neighbors(area):
        center = area[tuple(np.array(area.shape) // 2)]
        for i, j in zip(*np.where(area)):
            relative_position = np.array(area.shape) // 2 - [i, j]
            area[i, j].neighbors[tuple([1, 1] + relative_position)] = center

    @staticmethod
    def kill_person(p, reg):
        reg.remove_person(p)
        p.isAlive = False
        if p.partner:
            p.partner.partner = None

    def get_historical_figure(self, id):
        hf = self.collective.historical_population[id]
        return hf, hf.brain.get_history()[:hf.age() + 1]

    def get_attitudes(self, id):
        return self.collective.historical_population[id].collective.world_attitudes[id]

    def evaluate(self, by_alive=False):
        if by_alive:
            return ([p.brain.get_models() for p in self.collective.historical_population if p.isAlive and p.gender == 1],
                    [p.brain.get_models() for p in self.collective.historical_population if p.isAlive and p.gender == 0])

        else:
            return ([person.brain.get_models() for person in self.collective.historical_population],
                    [p.gender for p in self.collective.historical_population],
                    [person.child_num for person in self.collective.historical_population],
                    Person.ages[:len(self.collective.historical_population)])

    @staticmethod
    def find_best_minds(evaluated_list, date=False):
        neural_list, genders, children, ages = evaluated_list
        best_minds = []
        his = np.swapaxes(np.array([np.arange(len(neural_list)), genders, children, ages]), 0, 1)
        if date:
            sorted_idx = np.lexsort((his[:, 4], his[:, 3], his[:, 2], his[:, 1]))
            sorted_his = np.array([his[i] for i in sorted_idx])
        else:
            sorted_idx = np.lexsort((his[:, 3], his[:, 2], his[:, 1]))
            sorted_his = np.array([his[i] for i in sorted_idx])

        gender_idx = np.argmax(sorted_his[:, 1])
        male_lst, female_lst = sorted_his[gender_idx:], sorted_his[:gender_idx]

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

    @staticmethod
    def assemble_brains(neural_list):
        brain_couples = []
        for models_couple in neural_list:
            brain_couples.append((Brain(models=models_couple[0]), Brain(models=models_couple[1])))
        return brain_couples

    @staticmethod
    def assemble_separated_brains(neural_list):
        brains = []
        for models in neural_list:
            brains.append((Brain(models=models[0]), Brain(models=models[1])))
        return brains

    @staticmethod
    def disassemble_brains(brain_couples):
        neural_list = []
        for brain_couple in brain_couples:
            neural_list.append((brain_couple[0].get_models(), brain_couple[1].get_models()))
        return neural_list

    def __repr__(self):
        txt = f"Year: {self.Time // 12}"
        return txt

    def display(self):
        txt = f"Year: {self.Time // 12}\n\n"
        pop_num = 0

        for i, j in self.region_iterator:
            pop_num += len(self.regions[i, j].Population)

            txt += self.regions[i, j].display()
            txt += '\n----------\n\n\n'

        if self.is_eradicated():
            txt = "SPECIES GONE"

        print(txt)
        print(pop_num)
