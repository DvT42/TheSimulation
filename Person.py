import random
import numpy as np


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
        self.history = np.array([[0, 0]])

        # Attributes that depend on the parents:
        # - standard creation
        if type(father) is Person:
            self.manual = False
            self.fatherID = father.id
            self.motherID = mother.id

            self.isMan = self.gender == 1
            self.isWoman = not self.isMan
            Person.ages[self.id] = 0

            # strength - represents physical ability and health. {mean strength of parents}/2 + {normal distribution}/2
            self.strength = ((father.strength + mother.strength) // 4 + int(random.normalvariate(50, 10))) // 10

        # - manual creation
        else:
            self.manual = True
            for i in range(19):
                self.history = np.append(self.history, [[0, 0]], axis=0)

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

    def decision_making(self):
        # this list is for convinience, to know what does each index of option means.
        choices = ["emotional connection", "strength"]

        # parental influence depends on the former choices and attitude of the parents toward each option.
        if self.father is Person:
            parental_influence = [(self.father.history[self.year()][0] + self.mother.history[self.year()][0]) / 2,
                                  (self.father.history[self.year()][1] + self.mother.history[self.year()][1]) / 2]
        else:
            parental_influence = [0, 0]

        # history influence won't be available for toddlers.
        if self.year() > 1:  # and not self.manual:
            history_influence = self.history[-2]
            if self.year() < len(self.history):
                history_influence[0] += self.history[-1][0] * (12 / self.age() % 12)
                history_influence[1] += self.history[-1][1] * (12 / self.age() % 12)
        else:
            history_influence = [0, 0]

        # calculating attitue toward each action for present time:
        if not self.manual:
            if self.year() >= len(self.history):  # this line checks if the person hadn't acted this year yet
                # until the age of 16, parents have greater impact on their child's actions than other people.
                if self.year() < 16:
                    # there are 3 things affecting choice-making: 1. parent's decisions, 2. former choices of the
                    # person, and personal change and taste (represented by a random number, not normal)
                    self.history = np.append(self.history, [[((parental_influence[0] * (16 - self.year()) / 16) +
                                                            (history_influence[0] +
                                                             random.randint(1, 25) / 100) *
                                                            (1 - (16 - self.year()) / 16)) / 12,

                                                             ((parental_influence[1] * (16 - self.year()) / 16) +
                                                             (history_influence[1] +
                                                              random.randint(1, 25) / 100) *
                                                             (1 - (16 - self.year()) / 16)) / 12]],
                                             axis=0)
                else:
                    self.history = np.append(self.history, [[(history_influence[0] + random.randint(1, 25) / 100) / 12,
                                                             (history_influence[1] + random.randint(1, 25) / 100) / 12]],
                                             axis=0)
            else:
                if self.year() < 16:
                    self.history[self.year()][0] += ((parental_influence[0] * (16 - self.year()) / 16) +
                                                     (history_influence[0] +
                                                      random.randint(1, 25) / 100) * (1 - (16 - self.year()) / 16)) / 12
                    self.history[self.year()][1] += ((parental_influence[1] * (16 - self.year()) / 16) +
                                                     (history_influence[1] +
                                                      random.randint(1, 25) / 100) * (1 - (16 - self.year()) / 16)) / 12
                else:
                    self.history[self.year()][0] += (history_influence[0] + random.randint(1, 25) / 100) / 12

                    self.history[self.year()][1] += (history_influence[1] + random.randint(1, 25) / 100) / 12

            return np.where(self.history[self.year()] == np.max(self.history[self.year()]))

        # manual persons currently are uncapable of acting like a human being, and they make decisions randomly.
        else:
            if self.year() >= len(self.history):
                self.history = np.append(self.history, [[0, 0]], axis=0)
            return [[random.randint(0, 1)]]

    def action(self):
        dec = self.decision_making()[0][0]
        if dec == 0:
            # should improve attitudes/merge
            Person.merging = np.append(Person.merging, np.array([self], dtype=object))
        if dec == 1:
            self.strength += 1

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
