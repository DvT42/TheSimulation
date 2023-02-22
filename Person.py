import random
import numpy as np


class Person:
    runningID = 0
    MAX_POPULATION = 10000000
    ages = np.empty(MAX_POPULATION, "uint16")
    ages[0], ages[1] = 20 * 12, 20 * 12

    def __init__(self, father, mother=None):
        # ID assignment

        self.id = Person.runningID
        Person.runningID += 1

        # Attributes that don't depend on the parents
        self.gender = random.randint(0, 1)
        if Person.MAX_POPULATION < Person.runningID:
            print("Reached Max Person.runningID")

        # Attributes that depend on the parents:
        # - standard creation
        if type(father) is Person:
            self.fatherID = father.id
            self.motherID = mother.id

            self.isMan = self.gender == 1
            self.isWoman = not self.isMan
            Person.ages[self.id] = 0

            # strength - represents physical ability and health. {mean strength of parents}/2 + {normal distribution}/2
            self.strength = ((father.strength + mother.strength) // 4 + int(random.normalvariate(50, 10))) // 10
            pass

        # - manual creation
        else:
            # [0]
            self.gender = father[0]
            self.isMan = self.gender == 1
            self.isWoman = not self.isMan

            # [1]
            self.strength = father[1]
            pass

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

    # Override of the conversion to string
    def __repr__(self):
        # basic information
        txt = f"{self.id}: " \
              f"gender: {self.gender}, age: {self.age() // 12}" \
              f"str: {self.strength}"

        # pregnancy data
        if self.isWoman and self.pregnancy != 0:
            txt += f", pregnancy: {self.pregnancy}, mate: {self.father_of_child.id}"

        return txt + ";"

    def display(self):
        # basic information
        txt = f"{self.id}: \n" \
              f"gender: {self.gender}, age: {self.age() // 12} \n" \
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
