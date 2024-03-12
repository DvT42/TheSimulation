from Simulation import Simulation
from Control_Board import ControlBoard
from datetime import datetime
from map.Map import Map

if __name__ == "__main__":

    # running code
    sim_map = Map()
    TS = Simulation(sim_map=sim_map)
    while True:
        command = input("please input command: ")

        start = datetime.now()
        TS = ControlBoard.process_command(TS, command)
        print(f"{(datetime.now() - start).total_seconds():.02f}s")

        if not TS:
            break
