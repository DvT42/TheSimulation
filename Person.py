import math
import random
import enum
import numpy as np
from Brain import Brain


class Person:
    runningID = 0
    MAX_POPULATION = 10000000
    ages = np.empty(MAX_POPULATION, "uint16")
    ages[0], ages[1] = 20 * 12, 20 * 12

    # list for merging people
    merging = np.array([], dtype=object)

    def __init__(self, father, mother=None):
        # ID assignment
        self.id = Person.runningID
        Person.runningID += 1

        self.father = father
        self.mother = mother

        # Attributes that don't depend on the parents
        self.gender = random.choice(list(Gender))
        if Person.MAX_POPULATION < Person.runningID:
            print("Reached Max Person.runningID")

        # list to save former actions in
        self.history = np.zeros(120*12, dtype=int)
        self.readiness = 12 * 12 + int(random.normalvariate(0, 12))

        self.brain = Brain(self)

        # Attributes that depend on the parents:
        # - standard creation
        if type(father) is Person:
            self.isManual = False
            self.fatherID = father.id
            self.motherID = mother.id

            Person.ages[self.id] = 0

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
        if self.gender == Gender.Female:
            if self.pregnancy == 0 and self.age() >= self.readiness and self.biowatch > 0:
                self.father_of_child = mate
        elif mate.pregnancy == 0:
            mate.father_of_child = self

    def birth(self):
        f = self.father_of_child
        self.father_of_child = None
        self.pregnancy = 0
        return Person(f, self)

    # TODO: try to find ways for death to be less emennent.
    def death(self):
        death_chance = 0.06 * math.exp(-0.02 * self.strength)
        random_number = random.random()
        return random_number < death_chance

    def action(self):
        dec = self.brain.call_decision_making()

        if dec == 0:
            # should improve attitudes/merge
            if self.age() > self.readiness:
                Person.merging = np.append(Person.merging, np.array([self], dtype=object))
            return 1
        if dec == 1:
            self.strength += 0.5
            return 2

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
                  f"last action: {self.history[self.age()]}"
        else:
            txt = f"{self.id}: \n" \
                  f"gender: {self.gender.name}, age: {self.year()} \n" \
                  f"strength: {self.strength}\n" \
                  f"last action: {self.history[self.age()]}"

        # pregnancy data
        if self.gender == Gender.Female:
            if self.pregnancy != 0:
                txt += f"\n pregnancy: {self.pregnancy}, mate: {self.father_of_child.id}"
            else:
                txt += f"\n timer: {self.biowatch}"

        return txt

    @staticmethod
    def newMonth():
        Person.ages += 1

    def age(self):
        return Person.ages[self.id]

    def year(self):
        return self.age() // 12


class Gender(enum.IntEnum):
    Female = 0
    Male = 1
