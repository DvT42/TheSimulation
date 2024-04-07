import Neural_Network
import numpy as np
import random


class a:
    runningID = 0

    def __init__(self, father=None, mother=None):
        self.id = a.runningID
        a.runningID += 1

        self.brain = Neural_Network.NeuralNetwork()
        self.brain.add_layer(2, 2, activation='relu')
        self.brain.add_layer(7, 2, activation='relu')
        self.brain.add_layer(3, 7, activation='softmax')

        self.history = []
        self.counter = 6
        self.child_num = 0

        if father and mother:
            self.gene_inheritance(father, mother)

    def gene_inheritance(self, father, mother):
        father_weights = father.brain.get_weights()
        mother_weights = mother.brain.get_weights()

        # inheritance of parent's weights & biases
        new_weights = father_weights
        for lnum, layer_weights in enumerate(new_weights):
            for index, value in np.ndenumerate(layer_weights):
                if random.uniform(0, 1) < 0.5:
                    new_weights[lnum][index] = mother_weights[lnum][index]

        for layer_weights in new_weights:
            layer_weights += 0.3 * np.random.randn(*np.shape(layer_weights))

        self.brain.set_weights(new_weights)

    def action(self, time):
        self.counter -= 1
        if self.counter > 0:
            dec = np.argmax(self.brain.feed([self.counter, time]))

            if dec == 0:
                self.history.append(0)
                return 0
            elif dec == 1:
                self.history.append(1)
                return 1
            elif dec > 1:
                self.history.append(2)
                return 2
        else:
            return 2

    def __repr__(self):
        return f'{self.id}'

    def display(self):
        txt = f"{self.id}: \n"
        txt += f"{self.history}"

        return txt


class Sim:
    def __init__(self):
        a.runningID = 0
        self.pop = []
        self.right = 1
        self.timer = 3
        self.whitelst = []
        for i in range(1000):
            self.pop.append(a())
        self.Time = 0

    def month(self):
        for b in self.pop:
            act = b.action(self.Time)
            if act == 2:
                self.pop.remove(b)
                continue
            elif act == self.right:
                self.whitelst.append(b)

        self.timer -= 1
        if self.timer == 0:
            self.right = (self.right + 1) % 2
            self.timer = 2

        for w in self.whitelst:
            for j in self.whitelst:
                if j is not w:
                    self.pop.append(a(w, j))
                    w.child_num += 1
                    j.child_num += 1
                    self.whitelst.remove(w)
                    self.whitelst.remove(j)
                    break
        self.whitelst = []
        self.Time += 1

    def evaluate(self):
        return [[person.id for person in self.pop], [person.child_num for person in self.pop]]

    @staticmethod
    def find_best_minds(evaluated_list):
        ids, children = evaluated_list
        his = np.swapaxes(np.array([ids, children]), 0, 1)
        sorted_idx = np.lexsort([his[:, 1]])
        sorted_his = np.array([his[ix] for ix in sorted_idx])

        return sorted_his[-10:]

    def display(self):
        txt = f"month: {self.Time}\n\n"

        for c in self.pop:
            txt += c.display()
            txt += '\n----------\n'
        print(txt)


if __name__ == '__main__':
    sim = Sim()
    while True:
        prompt = input()
        if prompt.lower() == 'b':
            best = sim.find_best_minds(sim.evaluate())
            for i in range(len(best)):
                print(f'{best[-i - 1][0]}')

        elif prompt.lower() == 'i':
            print(f'{sim.pop[int(input())]}')

        else:
            for i in range(int(prompt)):
                sim.month()
                if not sim.pop:
                    break
            sim.display()
        if not sim.pop:
            sim = Sim()
