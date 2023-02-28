import numpy as np

from Person import Person


class Simulation:
    MARRIAGE_AGE_YEARS = 12
    DIFF_AGE_YEARS = 15

    def __init__(self):
        self.Adam = Person([1, 100, 100])
        self.Eve = Person([0, 100, 100])
        self.Population = np.array([self.Adam, self.Eve])  # [self.Adam, self.Eve]
        self.Time = 0
        self.Pregnant_Women = []
        self.deadpeople = 0

    def month_avancement(self):
        self.Time += 1
        # p => any person.
        Person.newMonth()  # that means that dead people will also age(). Take that into consideration.
        newborns = np.array([])
        p: Person
        for i in range(len(self.Population) - 1, -1, -1):  # for not provoking outofindex. look in chatgpt.
            p = self.Population[i]
            # handle self avancement.
            #  - handle natural death
            if p.isDeadNaturally():
                self.deadpeople += 1
                self.Population = np.delete(self.Population, i)
                continue

            #  - handle pregnancy
            if p.gender == 0 and p.father_of_child is not None:
                if p.pregnancy == 9:
                    newborn = p.birth()
                    newborns = np.append(newborns, np.array([newborn]))
                else:
                    p.pregnancy += 1

            # handle advencemnt
            if p.year() < 15:
                p.strength += 0.5
            p.action()

            # handle interactions between people.
            for o in self.Population[i+1::]:  # for not intracting with yourself.
                pass

        # handle people who want to merge
        for i, p in enumerate(Person.merging):
            for o in Person.merging[i+1::]:
                if o.gender != p.gender and abs(p.year() - o.year()) < Simulation.DIFF_AGE_YEARS and \
                        p.year() > Simulation.MARRIAGE_AGE_YEARS and o.year() > Simulation.MARRIAGE_AGE_YEARS:
                    p.merge(o)
        Person.merging = []

        # self advancement
        self.Population = np.concatenate((self.Population, newborns))

    def __repr__(self):
        txt = f"Year: {self.Time // 12};"
        for p in self.Population:
            txt += f" {p}"
        return txt

    def display(self):
        txt = f"Year: {self.Time // 12}\n\n"
        for p in self.Population:
            txt += f"{p.display()}\n\n"
        txt += f'Dead: {self.deadpeople}'
        print(txt)


# running code
TS = Simulation()
while True:
    command = input("please input command: ")

    if command[0] == "s" or command[0] == "S":
        for j in range(int(command[1::])):
            TS.month_avancement()
        TS.display()

    elif command[0] == "y" or command[0] == "Y":
        for j in range(int(command[1::]) * 12):
            TS.month_avancement()
        TS.display()

    elif command[0] == "x" or command[0] == "X":
        break

    else:
        TS.month_avancement()
        TS.display()
