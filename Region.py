class Region:
    def __init__(self, location, biome, population=None):
        self.location = location
        self.biome = biome
        if population:
            self.Population = population
        else:
            self.Population = []

    def __iter__(self):
        return (p for p in self.Population)

    def display(self):
        txt = f'location: {self.location}:\n----------\n'

        for p in self:
            txt += f"{p.display()}\n\n"

        txt += f'Current population: {len(self.Population)}'

        return txt
