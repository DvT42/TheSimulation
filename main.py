"""
    The main.py file handles the input from the user and passes the commands to the `ControlBoard` for processing.

    This code is responsible for the following:

    * Receiving input from the user
    * Parsing the input into commands
    * Passing the commands to the `ControlBoard` for processing
    * Handling errors

    Constants & important variables:
        done: If set to False, the program will perform the commands inside *processes* before giving the user the control on the simulation.
        processes: A list containing tuples of commands.
"""

from datetime import datetime
from multiprocessing import Process

from Control_Board import *
from map.Map import Map


def runOnce(ts, sim_map, command, is_new_leap):
    """
    This function is used for passing a single command into the ControlBoard. Useful especially for memory and predetermined commands.
    :param ts: the Simulation object on which the commands will be performed.
    :param sim_map: the Map object which will be used in the simulation. if ts is passed, the Simulation's map must correspond with this parameter.
    :param command: the requested command to be passed.
    :param is_new_leap: A boolean. if True, the simulation will separate this run's result from the former. Used for certain sets of predestined commands, such as consecutive increasing number of years in the training process.
    :return: The modified Simulation object that was passed, or an alternative one provided by ControlBoard.
    """
    TS = ts
    ControlBoard.IS_NEW_LEAP = is_new_leap
    start = datetime.now()
    TS = ControlBoard.process_command(
        sim=TS,
        sim_map=sim_map,
        com=command)
    print(f"{(datetime.now() - start).total_seconds():.02f}s")
    return TS


if __name__ == "__main__":
    # User-set variables
    done = True

    TS = None
    processes = [(
            [['load both', 'y30', 'save both']] +
            [['load both', 'y150', 'save both']] * 4
    ), (
            [['load both', 'y170', 'save both']] * 4
    ), (
            [['load both', 'y200', 'save both']] * 4
    ), (
            [['load both', 'y230', 'save both']] * 4)
    ]
    sim_map = Map()

    while True:
        if done:
            start = datetime.now()
            command = input("please input command: ")
            TS = ControlBoard.process_command(
                sim=TS,
                sim_map=sim_map,
                com=command)
            print(f"{(datetime.now() - start).total_seconds():.02f}s")
        else:
            for process in processes:
                isNewLeap = True
                for pro in process:
                    p = Process(target=runOnce, args=(TS, sim_map, pro, isNewLeap))
                    p.start()
                    p.join()  # this blocks until the process terminates
                    isNewLeap = False
            done = True
        if not TS:
            break
