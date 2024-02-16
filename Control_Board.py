import pickle

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
                brains = Simulation.assemble_brains(models)
                sim = Simulation(imported=brains)
                f.close()

        elif com.lower() == 'save':
            with open(ControlBoard.SAVED_BRAINS_PATH, 'wb') as f:
                minds_to_pickle = sim.disassemble_brains(sim.find_best_minds())
                pickle.dump(minds_to_pickle, f)
                f.close()

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
    def exact_skip(sim: Simulation, num: int, failsafe: bool = True, recursion_flag=False):
        dest_time = sim.Time + num
        for _ in ProgressBar(num):
            sim.month_advancement()
            if sim.is_eradicated():
                break
        if sim.Time < dest_time:
            if failsafe:
                new_sim = Simulation(imported=sim.find_best_minds())
                del sim
                sim = new_sim
                sim = ControlBoard.exact_skip(sim, num=dest_time, recursion_flag=True)
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
