import numpy as np


class Region:
    def __init__(self, location, surroundings, biome, population=None):
        self.location = np.array(location)
        self.biome = biome
        self.surr_biomes, self.surr_pop = surroundings

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
