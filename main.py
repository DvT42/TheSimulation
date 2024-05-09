from Control_Board import ControlBoard
from datetime import datetime
from map.Map import Map
from multiprocessing import Process


def runOnce(commands, done, is_new_leap):
    sim_map = Map()
    TS = None
    ControlBoard.IS_NEW_LEAP = is_new_leap
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
    processes = [(
            [['y35', 'save both']] +
            [['load both', 'y35', 'save both']] * 20
    ), (
            [['load both', 'y70', 'save both']] * 20
    ), (
            [['load both', 'y100', 'save both']] * 20
    ), (
            [['load both', 'y130', 'save both']] * 20)
    ]
    done = False
    while True:
        for process in processes:
            isNewLeap = True
            for pro in process:
                p = Process(target=runOnce, args=(pro, done, isNewLeap))
                p.start()
                p.join()  # this blocks until the process terminates
                # runOnce(pro, done)
                isNewLeap = False
        done = True
        if not TS:
            break
