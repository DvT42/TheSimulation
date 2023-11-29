import math
import random
import enum
import numpy as np
from Brain import Brain


class Person:
    MAX_POPULATION = 10000000
    AGING_STARTING_AGE = 40 * 12
    DRASTIC_AGING_AGE = 75 * 12
    DEATH_NORMALIZER = 0.5

    runningID = 0
    ages = np.zeros(MAX_POPULATION, dtype=int)
    ages[0], ages[1] = 20 * 12, 20 * 12

    # list for social connectors
    social_connectors = []

    def __init__(self, father, mother=None):
        # ID assignment
        self.isAlive = True
        self.id = Person.runningID
        Person.runningID += 1

        self.father: Person = father
        self.mother: Person = mother

        # Attributes that don't depend on the parents
        self.gender = random.choice(list(Gender))
        if Person.MAX_POPULATION < Person.runningID:
            print("Reached Max Person.runningID")

        # list to save former actions in
        self.readiness = 12 * 12 + int(random.normalvariate(0, 12))

        self.brain = Brain(self)

        # Attributes that depend on the parents:
        # - standard creation
        if type(father) is Person:
            self.isManual = False
            self.fatherID = father.id
            self.motherID = mother.id

            # strength - represents physical ability and health. {mean strength of parents}/2 + {normal distribution}/2
            self.starting_strength = (
                    (father.starting_strength + mother.starting_strength) // 4 + int(random.normalvariate(50, 10)))
            self.strength = self.starting_strength // 10

            # inherit parents' brain
            self.brain.inherit(self)

        # - Manual creation
        else:
            self.isManual = True

            # [0]
            self.gender = father[0]

            # [1]
            self.strength = father[1]
            self.starting_strength = self.strength

        # procreation availability.
        if self.gender == Gender.Female:
            self.pregnancy = 0
            self.father_of_child = None
            self.biowatch = 30*12 + int(random.normalvariate(0, 24))
            if self.isManual:
                self.biowatch -= 8*12

    def merge(self, mate):
        mate: Person
        if self.gender == Gender.Female:
            if self.pregnancy == 0 and self.father_of_child is None and self.age() >= self.readiness and self.biowatch > 0:
                self.father_of_child = mate
        else:
            if mate.pregnancy == 0 and mate.father_of_child is None and mate.age() >= mate.readiness and mate.biowatch > 0:
                mate.father_of_child = self

    def birth(self):
        f = self.father_of_child
        self.father_of_child = None
        self.pregnancy = 0
        return Person(f, self)

    def did_die(self):
        death_chance = Person.DEATH_NORMALIZER * 0.06 * math.exp(-0.02 * self.strength)
        random_number = random.random()
        return random_number < death_chance

    def action(self):
        dec = self.brain.call_decision_making()

        if dec == 0:
            # should improve attitudes/merge
            Person.social_connectors.append(self)
            self.brain.set_history(self.age(), 1)
        if dec == 1:
            self.strength += 0.5
            self.brain.set_history(self.age(), 2)

    def aging(self):
        if self.age() > Person.AGING_STARTING_AGE:
            self.strength -= 1
            if self.age() > Person.DRASTIC_AGING_AGE:
                self.strength -= 1

    # Override of the conversion to string
    def __repr__(self):
        # basic information
        txt = f"{self.id}: " \
              f"gender: {self.gender.name}, age: {self.year()}, " \
              f"str: {self.strength}"

        # pregnancy data
        if self.gender == Gender.Female and self.pregnancy != 0:
            txt += f", pregnancy: {self.pregnancy}, mate: {self.father_of_child.id}"

        return txt + ";"

    def display(self):
        # basic information
        if not self.isManual:
            txt = f"{self.id}: \n" \
                  f"parents: [{self.father.id}, {self.mother.id}] \n" \
                  f"gender: {self.gender.name}, age: {self.year()} \n" \
                  f"strength: {self.strength} \n" \
                  f"last action: {self.brain.get_action_from_history(self.age())}"
        else:
            txt = f"{self.id}: \n" \
                  f"gender: {self.gender.name}, age: {self.year()} \n" \
                  f"strength: {self.strength}\n" \
                  f"last action: {self.brain.get_action_from_history(self.age())}"

        # pregnancy data
        if self.gender == Gender.Female:
            if self.pregnancy != 0:
                txt += f"\n pregnancy: {self.pregnancy}, mate: {self.father_of_child.id}"
            else:
                txt += f"\n timer: {self.biowatch}"

        return txt

    def age(self) -> int:
        return Person.ages[self.id]

    def year(self):
        return self.age() // 12


class Gender(enum.IntEnum):
    Female = 0
    Male = 1
