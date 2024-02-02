from Simulation import Simulation
from Control_Board import ControlBoard
from datetime import datetime


if __name__ == "__main__":

    # running code
    TS = Simulation()
    while True:
        command = input("please input command: ")

        start = datetime.now()
        TS = ControlBoard.process_command(TS, command)
        print(f"{(datetime.now() - start).total_seconds():.02f}s")

        if not TS:
            break
