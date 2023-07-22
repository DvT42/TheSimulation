import random
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
        self.gender = random.randint(0, 1)
        if Person.MAX_POPULATION < Person.runningID:
            print("Reached Max Person.runningID")

        # list to save former actions in
        self.history = np.zeros(100*12)

        # Attributes that depend on the parents:
        # - standard creation
        if type(father) is Person:
            self.isManual = False
            self.fatherID = father.id
            self.motherID = mother.id

            self.isMan = self.gender == 1
            self.isWoman = not self.isMan
            Person.ages[self.id] = 0

            # strength - represents physical ability and health. {mean strength of parents}/2 + {normal distribution}/2
            self.strength = ((father.strength + mother.strength) // 4 + int(random.normalvariate(50, 10))) // 10

            # inherit parents' brain
            self.brain = Brain(self)
            self.brain.model.set_weights(self.brain.gene_inhritance())

        # - isManual creation
        else:
            self.isManual = True

            # create a brain
            self.brain = Brain(self)

            # [0]
            self.gender = father[0]
            self.isMan = self.gender == 1
            self.isWoman = not self.isMan

            # [1]
            self.strength = father[1]

        # procreation availability.
        if self.isWoman:
            self.pregnancy = 0
            self.father_of_child = None

    def merge(self, mate):
        # empregnantating ('cause I just made that word up)
        if self.isWoman:
            if self.pregnancy == 0:
                self.father_of_child = mate
        elif mate.pregnancy == 0:
            mate.father_of_child = self

    def birth(self):
        f = self.father_of_child
        self.father_of_child = None
        self.pregnancy = 0
        return Person(f, self)

    def action(self):
        if self.isWoman:
            preg = self.pregnancy
        else:
            preg = 0
        if self.isManual:
            dec = self.brain.decision_making(self.gender, self.age(), self.strength, preg, np.zeros(100*12),
                                       np.zeros(100*12), self.history, self.history[self.age() - 1])
        else:
            dec = self.brain.decision_making(self.gender, self.age(), self.strength, preg, self.father.history,
                                   self.mother.history, self.history, self.history[self.age() - 1])

        if dec == 0:
            # should improve attitudes/merge
            Person.merging = np.append(Person.merging, np.array([self], dtype=object))
            return 1
        if dec == 1:
            self.strength += 1
            return 2

    # Override of the conversion to string
    def __repr__(self):
        # basic information
        txt = f"{self.id}: " \
              f"gender: {self.gender}, age: {self.year()}" \
              f"str: {self.strength}"

        # pregnancy data
        if self.isWoman and self.pregnancy != 0:
            txt += f", pregnancy: {self.pregnancy}, mate: {self.father_of_child.id}"

        return txt + ";"

    def display(self):
        # basic information
        if not self.isManual:
            txt = f"{self.id}: \n" \
                  f"parents: [{self.father.id}, {self.mother.id}] \n" \
                  f"gender: {self.gender}, age: {self.year()} \n" \
                  f"strength: {self.strength}"
        else:
            txt = f"{self.id}: \n" \
                  f"gender: {self.gender}, age: {self.year()} \n" \
                  f"strength: {self.strength}"

        # pregnancy data
        if self.isWoman and self.pregnancy != 0:
            txt += f"\n pregnancy: {self.pregnancy}, mate: {self.father_of_child.id}"

        return txt

    @staticmethod
    def newMonth():
        Person.ages += 1

    def age(self):
        return Person.ages[self.id]

    def year(self):
        return self.age() // 12
