from Brain import *
from Person import *


class Simulation:
    MARRIAGE_AGE = 12 * 12
    SHOULD_PROBABLY_BE_DEAD = 120 * 12
    IMMUNITY_TIME = 10 * 12
    INITIAL_COUPLES = 2

    def __init__(self, imported=None):
        self.collective = Collective()
        Person.Person_reset(Simulation.INITIAL_COUPLES)

        self.Population: list[Person] = []
        self.History: list[Person] = []

        if imported:
            for brain_couple in imported:
                m = Person(father=[Gender.Male, 100], collective=self.collective)
                m.brain = brain_couple[0]
                brain_couple[0].transfer_brain(m)

                f = Person(father=[Gender.Female, 100], collective=self.collective)
                f.brain = brain_couple[1]
                brain_couple[1].transfer_brain(f)

                self.Population.extend([m, f])

        else:
            for i in range(Simulation.INITIAL_COUPLES):
                self.Population.extend([Person(father=[Gender.Male, 100], collective=self.collective),
                                        Person(father=[Gender.Female, 100], collective=self.collective)])

        for p in self.Population:
            self.collective.add_person(p)
            self.History.append(p)

        for i in self.Population:
            for j in self.Population:
                if j is not i:
                    i.brain.get_first_impression(j)

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

    def evaluate(self):
        return ([person.child_num for person in self.History], Person.ages[:len(self.History)],
                                                              [p.gender for p in self.History])

    def find_best_minds(self):
        children, ages, genders = self.evaluate()
        best_minds = []
        his = np.swapaxes(np.array([np.arange(len(self.History)), genders, children, ages]), 0, 1)
        sorted_idx = np.lexsort((his[:, 3], his[:, 2], his[:, 1]))
        sorted_his = np.array([his[i] for i in sorted_idx])

        gender_idx = np.argmax(sorted_his[:, 1])
        male_lst, female_lst = sorted_his[gender_idx:], sorted_his[:gender_idx]

        for i in range(self.INITIAL_COUPLES):
            best_minds.append((self.History[male_lst[-i][0]].brain, self.History[female_lst[-i][0]].brain))

        return best_minds

    @staticmethod
    def assemble_brains(neural_list):
        brain_couples = []
        for models_couple in neural_list:
            brain_couples.append((Brain(models=models_couple[0]), Brain(models=models_couple[1])))
        return brain_couples

    @staticmethod
    def disassemble_brains(brain_couples):
        neural_list = []
        for brain_couple in brain_couples:
            neural_list.append((brain_couple[0].get_models(), brain_couple[1].get_models()))
        return neural_list

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
