from Control_Board import ControlBoard
from datetime import datetime
from map.Map import Map
from multiprocessing import Process


def runOnce(commands, done):
    sim_map = Map()
    TS = None
    for command in commands:
        start = datetime.now()
        TS = ControlBoard.process_command(
            sim=TS,
            sim_map=sim_map,
            com=command if not done else input("please input command: "))
        print(f"{(datetime.now() - start).total_seconds():.02f}s")


if __name__ == "__main__":
    # running code
    TS = None
    processes = ([['load alive', 'y150', 'save children']] +
                 [['load children', 'y150', 'save children']] * 20)
    done = False
    while True:
        for pro in processes:
            p = Process(target=runOnce, args=(pro, done))
            p.start()
            p.join()  # this blocks until the process terminates
            # runOnce(pro, done)
        done = True
        if not TS:
            break
