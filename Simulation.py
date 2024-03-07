from Brain import *
from Person import *
from Region import *


class Simulation:
    MARRIAGE_AGE = 12 * 12
    SHOULD_PROBABLY_BE_DEAD = 120 * 12
    IMMUNITY_TIME = 10 * 12
    INITIAL_COUPLES = 6

    def __init__(self, imported=None):
        self.collective = Collective()
        Person.Person_reset(Simulation.INITIAL_COUPLES)

        self.regions = [Region(location=[0, 0], biome=1)]

        if type(imported) is np.ndarray and imported.any():
            actual_brains = Simulation.assemble_brains(imported[:Simulation.INITIAL_COUPLES])
            for brain_couple in actual_brains:
                m = Person(father=[Gender.Male, 100, [0, 0]], collective=self.collective)
                m.brain = brain_couple[0]
                brain_couple[0].transfer_brain(m)

                f = Person(father=[Gender.Female, 100, [0, 0]], collective=self.collective)
                f.brain = brain_couple[1]
                brain_couple[1].transfer_brain(f)

                self.regions[0].add_person(m)
                self.regions[0].add_person(f)

        else:
            for i in range(Simulation.INITIAL_COUPLES):
                self.regions[0].add_person(Person(father=[Gender.Male, 100, [0, 0]], collective=self.collective))
                self.regions[0].add_person(Person(father=[Gender.Female, 100, [0, 0]], collective=self.collective))

        for p in self.regions[0].Population:
            self.collective.add_person(p)

        for i in self.regions[0].Population:
            for j in self.regions[0].Population:
                if j is not i:
                    i.brain.get_first_impression(j)

        self.Time = 0
        self.Pregnant_Women = []

    # @jit(target_backend='cuda')
    def month_advancement(self):
        self.Time += 1

        # Person.ages[:Person.runningID] += 1 // might decide to transfer it to this format later
        for reg in self.regions:
            newborns = []
            social_connectors = []

            for idx, p in enumerate(reg):
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
                    reg.remove_person(p)
                    p.isAlive = False
                    if p.partner:
                        p.partner.partner = None
                    continue

                action = p.action()
                if action == 0:
                    social_connectors.append(p)

            for newborn in newborns:
                self.collective.add_person(newborn)

                for other in reg:
                    newborn.brain.get_first_impression(other)
                    other.brain.get_first_impression(newborn)

                reg.add_person(newborn)

            if not reg.Population:
                self.regions.remove(reg)

            for social_connector in social_connectors:
                social_connector: Person
                if social_connector.brain.is_friendly():
                    social_connector.brain.improve_attitudes_toward_me(region=reg)
                    social_connector.brain.improve_my_attitudes(multiplier=0.5)

                    if social_connector.partner:
                        if social_connector.partner in social_connectors:
                            social_connector.prepare_next_generation(social_connector.partner)
                    else:
                        social_connector.partner = social_connector.partner_selection()
                        if social_connector.partner:
                            social_connector.partner.partner = social_connector

                else:
                    social_connector.brain.improve_my_attitudes()

    def is_eradicated(self):
        return not self.regions

    def get_historical_figure(self, id):
        hf = self.collective.historical_population[id]
        return hf, hf.brain.get_history()[:hf.age()]

    def get_attitudes(self, id):
        return self.collective.historical_population[id].collective.world_attitudes[id]

    def evaluate(self):
        return ([person.brain.get_models() for person in self.collective.historical_population],
                [p.gender for p in self.collective.historical_population],
                [person.child_num for person in self.collective.historical_population],
                Person.ages[:len(self.collective.historical_population)])

    @staticmethod
    def find_best_minds(evaluated_list):
        neural_list, genders, children, ages = evaluated_list
        best_minds = []
        his = np.swapaxes(np.array([np.arange(len(neural_list)), genders, children, ages]), 0, 1)
        sorted_idx = np.lexsort((his[:, 3], his[:, 2], his[:, 1]))
        sorted_his = np.array([his[i] for i in sorted_idx])

        gender_idx = np.argmax(sorted_his[:, 1])
        male_lst, female_lst = sorted_his[gender_idx:], sorted_his[:gender_idx]

        for i in range(Simulation.INITIAL_COUPLES):
            best_minds.append((neural_list[male_lst[-i - 1][0]], neural_list[female_lst[-i - 1][0]]))

        return (best_minds,
                male_lst[-Simulation.INITIAL_COUPLES:],
                female_lst[-Simulation.INITIAL_COUPLES:])

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

        for reg in self.regions:
            txt += reg.display()
            txt += '\n----------\n'

        if self.is_eradicated():
            txt = "SPECIES GONE"

        print(txt)
