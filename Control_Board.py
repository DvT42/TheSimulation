import pickle

import numpy as np
import tqdm
from Simulation import Simulation


class ControlBoard:
    SAVED_BRAINS_PATH = 'data/saved brains.pkl'

    def __init__(self):
        self.best_minds_lst = []
        self.quality_lst = np.empty((0, 3), dtype=int)

    def process_command(self, sim: Simulation, com: str):
        if com.lower() == 'load':
            del sim
            with open(ControlBoard.SAVED_BRAINS_PATH, 'rb') as f:
                models = pickle.load(file=f)
                brains = Simulation.assemble_brains(models)
                sim = Simulation(imported=brains)
                f.close()

        elif com.lower() == 'save':
            with open(ControlBoard.SAVED_BRAINS_PATH, 'wb') as f:
                minds_to_pickle, _, _ = sim.find_best_minds(
                    [self.best_minds_lst, self.quality_lst[:, 0], self.quality_lst[:, 1], self.quality_lst[:, 2]])
                pickle.dump(minds_to_pickle, f)
                f.close()

        elif com.lower() == 'savec':
            with open(ControlBoard.SAVED_BRAINS_PATH, 'wb') as f:
                minds_to_pickle, _, _ = sim.find_best_minds(sim.evaluate())
                pickle.dump(minds_to_pickle, f)
                f.close()

        elif com.lower() == 'best':
            _, best_male, best_female = sim.find_best_minds(sim.evaluate())
            for i in range(len(best_male)):
                print(f'male: {best_male[-i -1][0]}, female: {best_female[-i -1][0]}')

        elif com[0].lower() == 'i':
            if com[1].lower() == 'a':
                print(ControlBoard.info_search(sim, int(com[2:]), is_att=True))
            else:
                print(ControlBoard.info_search(sim, int(com[1:])))

        elif com[0].lower() == "s":
            sim = self.exact_skip(sim, int(com[1:]))
        elif com[0].lower() == "y":
            sim = self.annual_skip(sim, int(com[1:]))

        elif com[0].lower() == "x":
            sim = None

        else:
            sim = self.exact_skip(sim, 1)

        return sim

    @staticmethod
    def info_search(sim: Simulation, id: int, is_att=False):
        if is_att:
            return sim.get_attitudes(id)
        person, history = sim.get_historical_figure(id)
        return (f'\n{person}'
                f'\n{history}')

    def exact_skip(self, sim: Simulation, num: int, failsafe: bool = True, recursion_flag=False):
        dest_time = sim.Time + num
        for _ in ProgressBar(num):
            sim.month_advancement()
            if sim.is_eradicated():
                break
        if sim.Time < dest_time:
            if failsafe:
                best_minds, male_lst, female_lst = sim.find_best_minds(sim.evaluate())
                processed_best_minds, unified_lst = Simulation.prepare_best_for_reprocess(best_minds, male_lst[:, 1:], female_lst[:, 1:])
                self.best_minds_lst.extend(processed_best_minds)
                self.quality_lst = np.append(self.quality_lst, unified_lst, axis=0)

                best_minds, male_lst, female_lst = sim.find_best_minds(
                    [self.best_minds_lst, self.quality_lst[:, 0], self.quality_lst[:, 1], self.quality_lst[:, 2]])
                processed_best_minds, unified_lst = Simulation.prepare_best_for_reprocess(
                    best_minds, male_lst[:, 1:], female_lst[:, 1:])
                self.best_minds_lst = processed_best_minds
                self.quality_lst = unified_lst

                new_sim = Simulation(imported=Simulation.find_best_minds(
                    [self.best_minds_lst, self.quality_lst[:, 0], self.quality_lst[:, 1], self.quality_lst[:, 2]]
                )[0])

                del sim
                sim = new_sim
                sim = self.exact_skip(sim, num=dest_time, recursion_flag=True)
            else:
                sim.display()
                return None
        if not recursion_flag:
            sim.display()
        return sim

    def annual_skip(self, sim: Simulation, num: int, failsafe: bool = True):
        return self.exact_skip(sim, num=num * 12, failsafe=failsafe)


class ProgressBar(tqdm.tqdm):
    def __init__(self, iterations):
        super().__init__(range(iterations), ncols=80, ascii="░▒▓█", unit="m", colour="blue")
