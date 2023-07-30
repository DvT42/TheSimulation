import numpy as np
import tqdm

from Person import Person, Gender
import tensorflow as tf
from datetime import datetime


class Simulation:
    MARRIAGE_AGE = 12 * 12
    DIFF_AGE = 15 * 12

    def __init__(self):
        self.Adam = Person([Gender.Male, 100])
        self.Eve = Person([Gender.Female, 100])
        self.Population = np.empty(Person.MAX_POPULATION, dtype=object)
        self.Population[0], self.Population[1] = self.Adam, self.Eve
        self.Time = 0
        self.Pregnant_Women = []

    def month_avancement(self):
        self.Time += 1
        # p => any person.
        Person.newMonth()  # that means that dead people will also age(). Take that into consideration.
        newborns = []
        for i, p in enumerate(self):
            p: Person

            # handle self advancement.
            #  - handle pregnancy
            if p.gender == 0:
                if p.father_of_child is not None:
                    if p.pregnancy == 9:
                        newborn = p.birth()
                        newborns.append(newborn)
                    else:
                        p.pregnancy += 1
                elif p.age() > p.readiness and p.biowatch > 0:
                    p.biowatch -= 1

            # handle advencemnt
            if p.year() < 15:
                p.strength += 0.25

            # taking actions
            action = p.action()
            p.brain.evolvement()
            p.history[p.age()] = action

            # handle interactions between people.
            # for o in self.Population[i+1::]:  # for not intracting with yourself.
            #     pass

        # handle people who want to merge
        for i, p in enumerate(Person.merging):
            for o in Person.merging[i+1::]:
                if o.gender != p.gender and abs(p.age() - o.age()) < Simulation.DIFF_AGE:
                    p.merge(o)
        Person.merging = []

        # self advancement
        for newborn in newborns:
            self.Population[newborn.id] = newborn

    def __iter__(self):
        return (person for person in self.Population[:Person.runningID] if person)

    def __repr__(self):
        txt = f"Year: {self.Time // 12};"
        for p in self:
            txt += f" {p}"
        return txt

    def display(self):
        txt = f"Year: {self.Time // 12}\n\n"
        for p in self:
            txt += f"{p.display()}\n\n"
        print(txt)


class ProgressBar(tqdm.tqdm):
    def __init__(self, iterations):
        super().__init__(range(iterations), ncols=80, ascii="░▒▓█", unit="m", colour="blue")


# running code
TS = Simulation()
while True:
    command = input("please input command: ")
    start = datetime.now()
    if command[0] == "s" or command[0] == "S":
        for j in ProgressBar(int(command[1::])):
            TS.month_avancement()
            tf.keras.backend.clear_session()
        TS.display()
        print(f"{(datetime.now() - start).total_seconds():.02f}s")
    elif command[0] == "y" or command[0] == "Y":
        for j in ProgressBar(int(command[1::]) * 12):
            TS.month_avancement()
            tf.keras.backend.clear_session()
        TS.display()
        print(f"{(datetime.now() - start).total_seconds():.02f}s")
    elif command[0] == "x" or command[0] == "X":
        break

    else:
        TS.month_avancement()
        TS.display()
        print(f"{(datetime.now() - start).total_seconds():.02f}s")
        tf.keras.backend.clear_session()
