from Brain import Collective
from Person import Person, Gender


class Simulation:
    MARRIAGE_AGE = 12 * 12
    SHOULD_PROBABLY_BE_DEAD = 120 * 12
    IMMUNITY_TIME = 10 * 12

    def __init__(self):
        self.collective = Collective()

        self.Adam = Person(father=[Gender.Male, 100], collective=self.collective)
        self.Eve = Person(father=[Gender.Female, 100], collective=self.collective)
        self.Population: list[Person] = [self.Adam, self.Eve]
        self.collective.add_person(self.Adam)
        self.collective.add_person(self.Eve)
        self.Adam.brain.get_first_impression(self.Eve)
        self.Eve.brain.get_first_impression(self.Adam)

        self.History = [self.Adam, self.Eve]
        self.Time = 0
        self.Pregnant_Women = []

    # @jit(target_backend='cuda')
    def month_advancement(self):
        self.Time += 1
        newborns = []

        # Person.ages[:Person.runningID] += 1 // might decide to transfer it to this format later

        for idx, p in enumerate(self):
            p: Person
            Person.ages[p.id] += 1

            #  - handle pregnancy
            if p.gender == Gender.Female:
                if p.father_of_child is not None:
                    if p.pregnancy == 9:
                        newborn = p.birth()
                        newborns.append(newborn)
                    else:
                        p.pregnancy += 1
                elif p.age() > p.readiness and p.youngness > 0:
                    p.youngness -= 1

            p.aging()  # handles old people's aging process.

            # handle growing up
            if p.year() < 15:
                p.strength += 0.25

            if p.natural_death_chance() and self.Time > Simulation.IMMUNITY_TIME:
                self.Population.remove(p)
                p.isAlive = False
                if p.partner:
                    p.partner.partner = None
                continue
            p.action()

        for social_connector in Person.social_connectors:
            social_connector: Person
            if social_connector.brain.is_friendly():
                social_connector.brain.improve_attitudes_toward_me()

                if social_connector.partner:
                    if social_connector.partner in Person.social_connectors:
                        social_connector.prepare_next_generation(social_connector.partner)
                else:
                    social_connector.partner = social_connector.partner_selection()
                    if social_connector.partner:
                        social_connector.partner.partner = social_connector

            else:
                social_connector.brain.improve_my_attitudes()
        Person.social_connectors = []

        for newborn in newborns:
            self.collective.add_person(newborn)

            for other in self:
                newborn.brain.get_first_impression(other)
                other.brain.get_first_impression(newborn)

            self.Population.append(newborn)
            self.History.append(newborn)

    def is_eradicated(self):
        return not self.Population

    def get_historical_figure(self, id):
        return self.History[id], self.History[id].brain.get_history()[:self.History[id].age()]

    def get_attitudes(self, id):
        return self.History[id].collective.world_attitudes[id]

    def __iter__(self):
        return (p for p in self.Population)

    def __repr__(self):
        txt = f"Year: {self.Time // 12};"
        for p in self:
            txt += f" {p}"
        return txt

    def display(self):
        txt = f"Year: {self.Time // 12}\n\n"
        for p in self:
            txt += f"{p.display()}\n\n"

        txt += f'Current world population: {len(self.Population)}'

        if self.is_eradicated():
            txt = "SPECIES GONE"

        print(txt)