import math
import random
import enum
import numpy as np
from Brain import Brain


class Person:
    MAX_POPULATION = 10000000
    AGING_STARTING_AGE = 40 * 12
    DRASTIC_AGING_AGE = 75 * 12
    DIFF_AGE = 15 * 12
    DEATH_NORMALIZER = 0.3

    runningID = 0
    ages = np.zeros(MAX_POPULATION, dtype=int)
    ages[0], ages[1] = 20 * 12, 20 * 12

    def __init__(self, collective, father, mother=None):
        # ID assignment
        self.isAlive = True
        self.isManual = type(father) is list
        self.id = Person.runningID
        Person.runningID += 1

        self.father: Person = father
        self.mother: Person = mother
        self.collective = collective

        # Attributes that don't depend on the parents
        self.gender = random.choice(list(Gender))
        if Person.MAX_POPULATION < Person.runningID:
            print("Reached Max Person.runningID")

        # list to save former actions in
        self.readiness = 12 * 12 + int(random.normalvariate(0, 12))

        self.brain = Brain(self, collective)

        self.partner = None
        self.child_num = 0

        # Attributes that depend on the parents:
        # - standard creation
        if type(father) is Person:
            self.fatherID = father.id
            self.motherID = mother.id

            # strength - represents physical ability and health. {mean strength of parents}/2 + {normal distribution}/2
            self.starting_strength = (
                    (father.starting_strength + mother.starting_strength) // 4 + int(random.normalvariate(50, 10)))
            self.strength = self.starting_strength // 10

            # get born in the mother's place
            self.location = mother.location

        # - Manual creation
        else:
            # [0]
            self.gender = father[0]

            # [1]
            self.strength = father[1]
            self.starting_strength = self.strength

            # [2]
            self.location = father[2]

        self.brain.update_location_history()  # insert the birthplace into location history.

        # procreation availability.
        if self.gender == Gender.Female:
            self.pregnancy = 0
            self.father_of_child = None
            self.youngness = 30 * 12 + int(random.normalvariate(0, 24))
            if self.isManual:
                self.youngness -= 8 * 12

    def prepare_next_generation(self, other):
        other: Person
        if self.gender == Gender.Female:
            if self.pregnancy == 0 and self.father_of_child is None and self.age() >= self.readiness and self.youngness > 0:
                self.father_of_child = other
                self.partner = other
                other.partner = self
        else:
            if other.pregnancy == 0 and other.father_of_child is None and other.age() >= other.readiness and other.youngness > 0:
                other.father_of_child = self
                self.partner = other
                other.partner = self

    def birth(self):
        f: Person = self.father_of_child
        self.father_of_child = None

        self.pregnancy = 0
        self.child_num += 1
        f.child_num += 1

        return Person(collective=self.collective, father=f, mother=self)

    def natural_death_chance(self):
        death_chance = Person.DEATH_NORMALIZER * 0.06 * math.exp(-0.02 * self.strength)
        random_number = random.random()
        return random_number < death_chance

    # noinspection PyTypeChecker
    def action(self):
        decision = self.brain.call_decision_making()

        if decision == 0:
            # should improve attitudes/merge
            self.brain.set_history(self.age(), 1)
        if decision == 1:
            self.strength += 0.5
            self.brain.set_history(self.age(), 2)
        # if decision == 2:
        #     # (function that checks the best place within range)
        #
        #     self.location[random.randint(0, 1)] += int(Person.RANGE * random.uniform(-1, 1))
        #     self.brain.update_location_history()
        #     self.brain.set_history(self.age(), 3)
        return decision

    def aging(self):
        if self.age() > Person.AGING_STARTING_AGE:
            self.strength -= 1
            if self.age() > Person.DRASTIC_AGING_AGE:
                self.strength -= 1

    def is_possible_partner(self, other):
        if other.isAlive:
            if other.age() > other.readiness:
                if self.gender != other.gender and not other.partner and abs(
                        self.age() - other.age()) < Person.DIFF_AGE:
                    if self.brain.get_attitudes(other) > 0.7 and other.brain.get_attitudes(self) > 0.7:
                        return True
        return False

    def partner_selection(self):
        if self.age() > self.readiness:
            arr: np.ndarray = np.copy(self.collective.world_attitudes[self.id, :self.collective.population_size])
            while arr.any():
                other = self.collective.historical_population[np.argmax(arr)]
                if self.is_possible_partner(other):
                    return other
                arr[np.argmax(arr)] = 0
            return None

    @staticmethod
    def Person_reset(initial_couples):
        Person.runningID = 0
        Person.ages = np.zeros(Person.MAX_POPULATION, dtype=int)
        Person.ages[:initial_couples * 2] = 20 * 12

    # Override of the conversion to string
    def __repr__(self):
        # basic information
        txt = f"{self.id}: " \
              f"gender: {self.gender.name}, age: {self.year()}, children: {self.child_num} " \
              f"str: {self.strength}"

        # pregnancy data
        if self.gender == Gender.Female and self.pregnancy != 0:
            txt += f", pregnancy: {self.pregnancy}, mate: {self.father_of_child.id}"

        return txt + ";"

    def display(self):
        # basic information
        if not self.isManual:
            txt = f"{self.id}: \n" \
                  f"parents: [{self.father.id}, {self.mother.id}] \n"
        else:
            txt = f"{self.id}: \n"

        txt += f"gender: {self.gender.name}, age: {self.year()}\n" \
               f"strength: {self.strength}\n" \
               f"last action: {self.brain.get_action_from_history(self.age())}"

        # pregnancy data
        if self.gender == Gender.Female:
            if self.pregnancy != 0:
                txt += f"\n pregnancy: {self.pregnancy}, mate: {self.father_of_child.id}"
            else:
                txt += f"\n timer: {self.youngness}"

        return txt

    # noinspection PyTypeChecker
    def age(self):
        return Person.ages[self.id]

    def year(self):
        return self.age() // 12


class Gender(enum.IntEnum):
    Female = 0
    Male = 1
