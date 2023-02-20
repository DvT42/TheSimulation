import numpy as np

from Person import Person


class Simulation:
    MARRIAGE_AGE = 12 * 12
    DIFF_AGE = 15 * 12

    def __init__(self):
        self.Adam = Person([1])
        self.Eve = Person([0])
        self.Population = np.array([self.Adam, self.Eve], dtype=object)  # [self.Adam, self.Eve]
        self.Time = 0
        self.Pregnant_Women = []

    def month_avancement(self):
        self.Time += 1
        # p => any person.
        Person.newMonth()  # that means that dead people will also age(). Take that into consideration.
        newborns = np.array([], dtype=object)
        for i, p in enumerate(self.Population):
            # handle self avancement.
            if p.gender == 0 and p.father_of_child is not None:
                if p.pregnancy == 9:
                    newborn = p.birth()
                    newborns = np.append(newborns, np.array([newborn], dtype=object))
                else:
                    p.pregnancy += 1

            # handle interactions between people.
            for o in self.Population[i+1::]:  # for not intracting with yourself.

                if o.gender != p.gender and abs(p.age() - o.age()) < Simulation.DIFF_AGE and \
                        p.age() > Simulation.MARRIAGE_AGE and o.age() > Simulation.MARRIAGE_AGE:
                    p.merge(o)

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
            txt += f"{p}\n\n"
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
