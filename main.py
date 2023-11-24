# import numpy as np
import tqdm
from Person import Person, Gender
from datetime import datetime


class Simulation:
    MARRIAGE_AGE = 12 * 12
    DIFF_AGE = 15 * 12
    SHOULD_PROBABLY_BE_DEAD = 120 * 12
    IMMUNITY_TIME = 10 * 12

    def __init__(self):
        self.Adam = Person([Gender.Male, 100])
        self.Eve = Person([Gender.Female, 100])
        self.Adam.brain.get_first_impression(self.Eve)
        self.Eve.brain.get_first_impression(self.Adam)

        self.Population: list[Person] = [self.Adam, self.Eve]
        self.Time = 0
        self.Pregnant_Women = []

    def month_avancement(self):
        self.Time += 1
        newborns = []

        Person.ages[:Person.runningID] += 1

        for idx, person in enumerate(self):
            person: Person

            # handle self advancement.
            #  - handle pregnancy
            if person.gender == Gender.Female:
                if person.father_of_child is not None:
                    if person.pregnancy == 9:
                        newborn = person.birth()
                        newborns.append(newborn)
                    else:
                        person.pregnancy += 1
                elif person.age() > person.readiness and person.biowatch > 0:
                    person.biowatch -= 1

            # TODO: fix absurd strength levels. older people must crumble.
            # handle advencemnt
            if person.year() < 15:
                person.strength += 0.25

            # taking actions
            if person.death() and self.Time > Simulation.IMMUNITY_TIME:
                self.Population.remove(person)
                continue
            person.action()

        for person in Person.social_connectors:
            if person.brain.get_positives():
                for other in person.brain.get_positives():
                    other: Person
                    other.brain.improve_attitude(person)
                    other.brain.update_positives()

                    if person.brain.get_attitudes(other) > 0.7 and other.brain.get_attitudes(person) > 0.7:
                        if person.age() > person.readiness and other.age() > other.readiness:
                            if person.gender != other.gender and abs(person.age() - other.age()) < Simulation.DIFF_AGE:
                                person.merge(other)
            else:
                attitudes = person.brain.get_attitudes()
                person.brain.improve_attitude(max(attitudes, key=attitudes.get), 0.5)
        Person.social_connectors = []

        # self advancement
        for newborn in newborns:
            for other in self:
                newborn.brain.get_first_impression(other)
                other.brain.get_first_impression(newborn)

            self.Population.append(newborn)

    def is_eradicated(self):
        return not self.Population

    def __iter__(self):
        return (person for person in self.Population)

    def __repr__(self):
        txt = f"Year: {self.Time // 12};"
        for p in self:
            txt += f" {p}"
        return txt

    def display(self):
        txt = f"Year: {self.Time // 12}\n\n"
        for p in self:
            txt += f"{p.display()}\n\n"

        if self.is_eradicated():
            txt = "SPECIES GONE"

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
            if TS.is_eradicated():
                break

        TS.display()
        print(f"{(datetime.now() - start).total_seconds():.02f}s")
        if TS.is_eradicated():
            break

    elif command[0] == "y" or command[0] == "Y":
        for j in ProgressBar(int(command[1::]) * 12):
            TS.month_avancement()
            if TS.is_eradicated():
                break

        TS.display()
        print(f"{(datetime.now() - start).total_seconds():.02f}s")
        if TS.is_eradicated():
            break

    elif command[0] == "x" or command[0] == "X":
        break

    else:
        TS.month_avancement()
        TS.display()
        print(f"{(datetime.now() - start).total_seconds():.02f}s")
        if TS.is_eradicated():
            break
