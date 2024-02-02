import tqdm
from Simulation import Simulation


class ControlBoard:
    @staticmethod
    def process_command(sim: Simulation, com: str):
        if com[0].lower() == 'i':
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
    def exact_skip(sim: Simulation, num: int, failsafe: bool = True):
        for i in ProgressBar(num):
            sim.month_advancement()
            if sim.is_eradicated():
                if failsafe:
                    time = sim.Time
                    sim = Simulation()
                    ControlBoard.exact_skip(sim, num=time)

                sim.display()
                return None

        sim.display()
        return sim

    @staticmethod
    def annual_skip(sim: Simulation, num: int, failsafe: bool = True):
        return ControlBoard.exact_skip(sim, num=num * 12, failsafe=failsafe)


class ProgressBar(tqdm.tqdm):
    def __init__(self, iterations):
        super().__init__(range(iterations), ncols=80, ascii="░▒▓█", unit="m", colour="blue")