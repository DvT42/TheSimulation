from Person import *
import concurrent.futures


class Region:
    def __init__(self, location, surrounding_biomes, neighbors=np.empty((3, 3), dtype=object), population=None):
        self.location = np.array(location)
        self.biome = surrounding_biomes[1, 1]
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
        with self._lock:
            self._neighbors = new_neighbors

    def neighbors_update(self, pos, value):
        with self._lock:
            self._neighbors[pos] = value

    @property
    def newborns(self):
        return self._newborns

    @newborns.setter
    def newborns(self, new_newborn):
        with self._lock:
            self._newborns.append(new_newborn)

    @property
    def dead(self):
        return self._dead

    @dead.setter
    def dead(self, new_dead):
        with self._lock:
            self._dead.append(new_dead)

    @property
    def social_connectors(self):
        return self._social_connectors

    @social_connectors.setter
    def social_connectors(self, new_social_connector):
        with self._lock:
            self._social_connectors.append(new_social_connector)

    @property
    def relocating_people(self):
        return self._relocating_people

    @relocating_people.setter
    def relocating_people(self, new_relocating_person):
        with self._lock:
            self._relocating_people.append(new_relocating_person)

    @property
    def newcomers(self):
        return self._newcomers

    @newcomers.setter
    def newcomers(self, new_newcomer):
        with self._lock:
            self._newborns.append(new_newcomer)

    def clear(self):
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
        with self._lock:
            if person.id not in self.pop_id:
                self._Population.append(person)
                self.pop_id.append(person.id)
                self.newcomers.append(person)

    def remove_person(self, person):
        with self._lock:
            if person.id in self.pop_id:
                self._Population.remove(person)
                self.pop_id.remove(person.id)

    # noinspection PyTypeChecker
    def introduce_newcomers(self):
        newcomers_ids = [n.id for n in self.newcomers]
        newcomers_exec = concurrent.futures.ThreadPoolExecutor()
        futures_region = [newcomers_exec.submit(self.mass_encounter,
                                                (self.newcomers + self.Population, newcomers_ids + self.pop_id, n)
                                                ) for n in self.newcomers]
        concurrent.futures.wait(futures_region)

        newcomers_exec.shutdown()
        for n in self.newcomers:
            self.mass_encounter(n)
            self.mass_encounter(self.newcomers, newcomers_ids, n)

    def mass_encounter(self, pop_lst=None, ids=None, person=None):
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
        for p in self.Population:
            p.action_flag = False

    def surr_pop(self):
            lst = np.zeros(9).reshape((3, 3))
            for i, j in zip(*np.where(self.neighbors)):
                lst[i, j] = len(self.neighbors[i, j].Population)
            return lst.flatten()

    def __iter__(self):
        return (p for p in self.Population)

    def __repr__(self):
        return f'({self.location}, {self.biome})'

    def display(self):
        txt = f'location: {self.location}:\n----------\n'

        for p in self:
            txt += f"{p.display()}\n\n"

        txt += f'Current population: {len(self.Population)}'

        return txt
