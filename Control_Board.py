import pickle
import numpy as np
import tqdm
from Simulation import Simulation


class ControlBoard:
    SAVED_BRAINS_PATH = 'data/saved brains.pkl'

    @staticmethod
    def process_command(sim: Simulation, com: str):
        if com.lower() == 'load':
            del sim
            with open(ControlBoard.SAVED_BRAINS_PATH, 'rb') as f:
                models = pickle.load(file=f)
                sim = Simulation(imported=models)
                f.close()

        elif com.lower() == 'save':
            with open(ControlBoard.SAVED_BRAINS_PATH, 'wb') as f:
                minds_to_pickle, _, _ = sim.find_best_minds(sim.evaluate())
                pickle.dump(minds_to_pickle, f)
                f.close()

        elif com.lower() == 'best':
            _, best_male, best_female = sim.find_best_minds(sim.evaluate())
            for i in range(len(best_male)):
                print(f'male: {best_male[-i - 1][0]}, female: {best_female[-i - 1][0]}')

        elif com[0].lower() == 'i':
            if com[1].lower() == 'a':
                print(ControlBoard.info_search(sim, int(com[2:]), is_att=True))
            else:
                print(ControlBoard.info_search(sim, int(com[1:])))

        elif com[0].lower() == "s":
            sim = ControlBoard.exact_skip(sim, int(com[1:]))
        elif com[0].lower() == "y":
            sim = ControlBoard.annual_skip(sim, int(com[1:]))

        elif com[0].lower() == "x":
            sim = None

        else:
            sim = ControlBoard.exact_skip(sim, 1)

        return sim

    @staticmethod
    def info_search(sim: Simulation, id: int, is_att=False):
        if is_att:
            return sim.get_attitudes(id)
        person, history = sim.get_historical_figure(id)
        return (f'\n{person}'
                f'\n{history}')

    @staticmethod
    def exact_skip(sim: Simulation, num: int, best_minds_lst=None, quality_lst=np.empty((0, 3), dtype=int),
                   failsafe: bool = True, recursion_flag=False):
        if best_minds_lst is None:
            best_minds_lst = []

        dest_time = sim.Time + num
        for _ in ProgressBar(num):
            sim.month_advancement()
            if sim.is_eradicated():
                break

        if sim.Time < dest_time:
            if failsafe:
                best_minds, male_lst, female_lst = sim.find_best_minds(sim.evaluate())
                processed_best_minds, unified_lst = Simulation.prepare_best_for_reprocess(best_minds, male_lst[:, 1:],
                                                                                          female_lst[:, 1:])
                best_minds_lst.extend(processed_best_minds)
                quality_lst = np.append(quality_lst, unified_lst, axis=0)

                best_minds, male_lst, female_lst = sim.find_best_minds(
                    [best_minds_lst, quality_lst[:, 0], quality_lst[:, 1], quality_lst[:, 2]])
                processed_best_minds, unified_lst = Simulation.prepare_best_for_reprocess(
                    best_minds, male_lst[:, 1:], female_lst[:, 1:])
                best_minds_lst = processed_best_minds
                quality_lst = unified_lst

                new_sim = Simulation(imported=
                np.reshape(
                    np.append(
                        best_minds_lst[len(best_minds_lst) // 2:], best_minds_lst[:len(best_minds_lst) // 2], axis=1,
                    ), (len(best_minds_lst) // 2, 2, 2)))

                del sim
                sim = new_sim
                sim = ControlBoard.exact_skip(sim, num=dest_time, best_minds_lst=best_minds_lst,
                                              quality_lst=quality_lst, recursion_flag=True)
            else:
                sim.display()
                return None
        if not recursion_flag:
            sim.display()
        return sim

    @staticmethod
    def annual_skip(sim: Simulation, num: int, failsafe: bool = True):
        return ControlBoard.exact_skip(sim, num=num * 12, failsafe=failsafe)


class ProgressBar(tqdm.tqdm):
    def __init__(self, iterations):
        super().__init__(range(iterations), ncols=80, ascii="░▒▓█", unit="m", colour="blue")
