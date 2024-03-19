import numpy as np


class Region:
    def __init__(self, location, surrounding_biomes, biome, neighbors=np.empty((3, 3), dtype=object), population=None):
        self.location = np.array(location)
        self.biome = biome
        self.surr_biomes = surrounding_biomes
        self.neighbors = neighbors
        self.neighbors[1, 1] = self

        if population:
            self.Population = population
        else:
            self.Population = []
        self.pop_id = [p.id for p in self.Population]

    def add_person(self, person):
        self.Population.append(person)
        self.pop_id.append(person.id)

    def remove_person(self, person):
        self.Population.remove(person)
        self.pop_id.remove(person.id)

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
