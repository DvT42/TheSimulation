import pickle
import os

import numpy as np
import tqdm
from Simulation import *


class ControlBoard:
    BASE_PATH = os.path.dirname(os.path.abspath("__file__"))
    SAVED_BRAINS_PATH = os.path.join(BASE_PATH, 'data', 'saved brains.pkl')
    MALE_BRAINS_PATH = os.path.join(BASE_PATH, 'data', 'saved male brains.pkl')
    FEMALE_BRAINS_PATH = os.path.join(BASE_PATH, 'data', 'saved female brains.pkl')
    SAVED_BRAINS_INFO_PATH = os.path.join(BASE_PATH, 'data', 'saved brains info.pkl')

    @staticmethod
    def process_command(sim: Simulation, com: str, sim_map=None):
        """
        The main function for operating with the simulation.

        :param sim_map: The intended map with which the simulation will work. For some commands this will not be needed.
        :param sim: The current Simulation object.
        :param com: Any command among: {'load', 'save', 'best', 'i'+id, 'ia'+id, 'il'+id, 's'+num, 'y'+num, 'x'}. If a
                    different command is supplied. the simulation will advance in 1 month.
        :return: The given Simulation, modified according to the command. If unsuccessful at executing the command the
                 returned Simulation will be the first that succeeded. If none succeeded, or if the command ordered to
                 terminate the program, None is returned.
        """
        if com.lower() == 'load':
            with open(ControlBoard.SAVED_BRAINS_PATH, 'rb') as f:
                models = pickle.load(file=f)
                new_sim = Simulation(sim_map=sim_map, imported=models)
            sim = new_sim

        elif com.lower() == 'load alive':
            with open(ControlBoard.MALE_BRAINS_PATH, 'rb') as f:
                male_models = pickle.load(file=f)
            with open(ControlBoard.FEMALE_BRAINS_PATH, 'rb') as f:
                female_models = pickle.load(file=f)
            new_sim = Simulation(sim_map=sim_map, separated_imported=(male_models, female_models))
            sim = new_sim

        elif com.lower() == 'load children':
            with open(ControlBoard.SAVED_BRAINS_PATH, 'rb') as f:
                models = pickle.load(file=f)
                with open(ControlBoard.SAVED_BRAINS_INFO_PATH, 'rb') as f2:
                    new_models, _, _ = Simulation.find_best_minds((models, *pickle.load(file=f2)))
                new_sim = Simulation(sim_map=sim_map, imported=new_models, all_couples=True)
            sim = new_sim

        elif com.lower() == 'save':
            with open(ControlBoard.SAVED_BRAINS_PATH, 'wb') as f:
                minds_to_pickle, _, _ = sim.find_best_minds(sim.evaluate())
                pickle.dump(minds_to_pickle, f)

        elif com.lower() == 'save alive':
            male_minds_to_pickle, female_minds_to_pickle = sim.evaluate(by_alive=True)
            with open(ControlBoard.MALE_BRAINS_PATH, 'wb') as f:
                pickle.dump(male_minds_to_pickle, f)
            with open(ControlBoard.FEMALE_BRAINS_PATH, 'wb') as f:
                pickle.dump(female_minds_to_pickle, f)

        elif com.lower() == 'save children':
            with open(ControlBoard.SAVED_BRAINS_PATH, 'wb') as f:
                minds_to_pickle, male_lst, female_lst = sim.find_best_minds(sim.evaluate(), take_all=True)
                pickle.dump(minds_to_pickle, f)
            with open(ControlBoard.SAVED_BRAINS_INFO_PATH, 'wb') as f2:
                pickle.dump(np.append(male_lst, female_lst, axis=0), f2)

        elif com.lower() == 'best':
            _, best_male, best_female = sim.find_best_minds(sim.evaluate())
            for i in range(len(best_male)):
                print(f'male: {best_male[-i - 1][0]}, female: {best_female[-i - 1][0]}')

        elif com[0].lower() == 'i':
            if com[1].lower() == 'a':
                print(ControlBoard.info_search(sim, int(com[2:]), is_att=True))
            elif com[1].lower() == 'l':
                print(ControlBoard.info_search(sim, int(com[2:]), is_loc=True))
            else:
                print(ControlBoard.info_search(sim, int(com[1:])))

        elif com[0].lower() == "x":
            sim = None

        else:
            if sim is None:
                sim = Simulation(sim_map)
            if com[0].lower() == "s":
                sim = ControlBoard.exact_skip(sim, int(com[1:]), regressive='partial')
            elif com[0].lower() == "y":
                sim = ControlBoard.annual_skip(sim, int(com[1:]), regressive='partial')
            else:
                sim = ControlBoard.exact_skip(sim, 1)

        return sim

    @staticmethod
    def info_search(sim: Simulation, id: int, is_att=False, is_loc=False):
        if is_att:
            return sim.get_attitudes(id)
        elif is_loc:
            return sim.get_historical_figure(id)[0].brain.get_location_history()
        person, history = sim.get_historical_figure(id)
        return (f'\n{person}'
                f'\n{history}')

    @staticmethod
    def exact_skip(sim: Simulation, num: int, failsafe: bool = True, regressive='False'):
        """
        The function that handles the advancement of a Simulation over time.

        :param sim: the starting Simulation object.
        :param num: the number of months requested. The Simulation will advance by this number.
        :param failsafe: If true, the function will try to produce new Simulation until one succeeds to match the
        destined Time (sim.Time + num).
        :param regressive: {'True', 'False', 'partial'}. determines whether information will be saved globally across
                           simulations. This directly affects the algorithm by which the Persons will learn.
        :return: The given Simulation, advanced by the requested num of months. If the Simulation got eradicated the
                  returned Simulation will be the first that succeeded. Or, if failsafe was set to False, None
                  is returned.
        """
        dest_time = sim.Time + num
        best_minds_lst = []

        # representing the best male Person across all Simulations. Relevant only if regressive=partial.
        best = [0, 0, 0]

        # representing the best Person-s across all Simulations. Relevant only if regressive=True.
        quality_lst = np.empty((0, 3), dtype=int)
        executors = concurrent.futures.ThreadPoolExecutor()

        while True:
            pbar = ProgressBar(dest_time - sim.Time)
            for _ in pbar:
                pbar.set_description(f'{sim.pop_num()}')
                sim.month_advancement(executors)
                if sim.is_eradicated():
                    break

            if sim.Time < dest_time:
                if failsafe:
                    # number of children born. used to access a Simulation's success
                    print(len(sim.collective.historical_population) - Simulation.INITIAL_COUPLES * 2)

                    best_minds, male_lst, female_lst = sim.find_best_minds(sim.evaluate())
                    processed_best_minds, unified_lst = Simulation.prepare_best_for_reprocess(best_minds,
                                                                                              male_lst[:, 1:],
                                                                                              female_lst[:, 1:])

                    if regressive == 'True':
                        # This learning algorithm always chooses the best across all Simulations. It takes the best from
                        # the last simulation and compares it to the former best
                        best_minds_lst.extend(processed_best_minds)
                        quality_lst = np.append(quality_lst, unified_lst, axis=0)

                        best_minds, male_lst, female_lst = sim.find_best_minds(
                            [best_minds_lst, quality_lst[:, 0], quality_lst[:, 1], quality_lst[:, 2]])
                        processed_best_minds, unified_lst = Simulation.prepare_best_for_reprocess(
                            best_minds, male_lst[:, 1:], female_lst[:, 1:])

                    elif regressive == 'partial':
                        # This learning algorithm replaces all the males with the absolute best across all Simulations
                        # if it weren't better than its former.
                        if (unified_lst[len(processed_best_minds) // 2 - 1][1] > best[1] or
                                (unified_lst[len(processed_best_minds) // 2 - 1][1] == best[1] and
                                 unified_lst[len(processed_best_minds) // 2 - 1][2] > best[2])):
                            best[0] = processed_best_minds[len(processed_best_minds) // 2 - 1]
                            best[1] = unified_lst[len(processed_best_minds) // 2 - 1][1]
                            best[2] = unified_lst[len(processed_best_minds) // 2 - 1][2]
                        else:
                            processed_best_minds = np.array(processed_best_minds)
                            processed_best_minds[len(processed_best_minds) // 2 - 1:] = best[0]

                    best_minds_lst = processed_best_minds
                    quality_lst = unified_lst

                    new_sim = Simulation(sim_map=sim.map, imported=np.reshape(
                        np.append(
                            best_minds_lst[:len(best_minds_lst) // 2],
                            best_minds_lst[len(best_minds_lst) // 2:],
                            axis=1),
                        (len(best_minds_lst) // 2, 3, 2)))

                    del sim
                    sim = new_sim
                else:
                    sim.display()
                    return sim
            else:
                # sim.display()
                return sim

    @staticmethod
    def annual_skip(sim: Simulation, num: int, failsafe: bool = False, regressive: str = 'False'):
        return ControlBoard.exact_skip(sim, num=num * 12, failsafe=failsafe, regressive=regressive)


class ProgressBar(tqdm.tqdm):
    def __init__(self, iterations):
        super().__init__(range(iterations), ncols=80, ascii="░▒▓█", unit="m", colour="blue")
