import os
import pickle
import tqdm

import map.Map
from Simulation import *


class ControlBoard:
    """
     A static class that manages the processing of user commands and mediates between the user and the simulation.

     This class provides functionalities for:

     * Processing user commands related to simulation control, brain data management, and learning processes.
     * Mediating between the user and the simulation by loading/saving brain data, running simulations, and providing results.

     Attributes:
         BASE_PATH (str): The path to the file from which the program is launched.
         SAVED_BRAINS_PATH (str): The path to the file of saved brains in pairs.
         MALE_BRAINS_PATH (str): The path to the file of male brains (which are saved separately).
         FEMALE_BRAINS_PATH (str): The path to the file of female brains (which are saved separately).
         SAVED_BRAINS_INFO_PATH (str): The path to the file of information about the saved brains in pairs.
         REGRESSIVE (str): A constant affecting the learning method from simulation to simulation. Can be 'full', 'partial', or 'none'.
    """

    BASE_PATH = os.path.dirname(os.path.abspath("__file__"))
    SAVED_BRAINS_PATH = os.path.join(BASE_PATH, 'data', 'saved brains.pkl')
    MALE_BRAINS_PATH = os.path.join(BASE_PATH, 'data', 'saved male brains.pkl')
    FEMALE_BRAINS_PATH = os.path.join(BASE_PATH, 'data', 'saved female brains.pkl')
    SAVED_BRAINS_INFO_PATH = os.path.join(BASE_PATH, 'data', 'saved brains info.pkl')
    REGRESSIVE = 'partial'
    IS_NEW_LEAP = True

    @staticmethod
    def process_command(sim: Simulation, com: str, sim_map: map.Map.Map=None):
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
                models = np.array(pickle.load(file=f))
                models = np.append(models[:, 0], models[:, 1], axis=0)
                with open(ControlBoard.SAVED_BRAINS_INFO_PATH, 'rb') as f2:
                    new_models, _, _ = Simulation.find_best_minds(
                        (models, *(pickle.load(file=f2))[:, 1:].T), children_bearers='enough')
                new_sim = Simulation(sim_map=sim_map, imported=new_models, all_couples=True)
            sim = new_sim

        elif com.lower() == 'load both':
            with open(ControlBoard.MALE_BRAINS_PATH, 'rb') as f:
                male_models = pickle.load(file=f)
            with open(ControlBoard.FEMALE_BRAINS_PATH, 'rb') as f:
                female_models = pickle.load(file=f)
            with open(ControlBoard.SAVED_BRAINS_PATH, 'rb') as f:
                data = pickle.load(file=f)
                if data:
                    models = np.array(data)
                    minds_to_pickle = models[:, 0]
                    female_lst = models[:, 1]
                    if len(male_models) > len(minds_to_pickle):
                        minds_to_pickle = np.tile(minds_to_pickle, (len(male_models) // len(minds_to_pickle), 1))
                    if len(female_models) > len(female_lst):
                        female_lst = np.tile(female_lst, (len(female_models) // len(female_lst), 1))
                    male_models = np.append(male_models, minds_to_pickle, axis=0) \
                        if len(male_models) > 0 else minds_to_pickle
                    female_models = np.append(female_models, female_lst, axis=0) \
                        if len(female_models) > 0 else female_lst
            new_sim = Simulation(sim_map=sim_map, separated_imported=(male_models, female_models))
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
                minds_to_pickle, male_lst, female_lst = sim.find_best_minds(sim.evaluate(), children_bearers='all')
                pickle.dump(minds_to_pickle, f)
            with open(ControlBoard.SAVED_BRAINS_INFO_PATH, 'wb') as f2:
                pickle.dump(np.append(male_lst, female_lst, axis=0), f2)

        elif com.lower() == 'save both':
            male_minds_to_pickle, female_minds_to_pickle = sim.evaluate(by_alive=True)
            minds_to_pickle, male_lst, female_lst = sim.find_best_minds(sim.evaluate(), children_bearers='all')

            with open(ControlBoard.MALE_BRAINS_PATH, 'rb') as f:
                male_models = pickle.load(file=f)
            with open(ControlBoard.FEMALE_BRAINS_PATH, 'rb') as f:
                female_models = pickle.load(file=f)
            with open(ControlBoard.SAVED_BRAINS_PATH, 'rb') as f:
                data = pickle.load(file=f)
            if ControlBoard.IS_NEW_LEAP or len(male_minds_to_pickle) >= len(male_models):
                with open(ControlBoard.MALE_BRAINS_PATH, 'wb') as f:
                    pickle.dump(male_minds_to_pickle, f)
                    print('saving male_minds_to_pickle: ', len(male_minds_to_pickle))
            if ControlBoard.IS_NEW_LEAP or len(female_minds_to_pickle) >= len(female_models):
                with open(ControlBoard.FEMALE_BRAINS_PATH, 'wb') as f:
                    pickle.dump(female_minds_to_pickle, f)
                    print('saving female_minds_to_pickle: ', len(female_minds_to_pickle))

            if ControlBoard.IS_NEW_LEAP or len(minds_to_pickle) >= len(data):
                with open(ControlBoard.SAVED_BRAINS_PATH, 'wb') as f:
                    pickle.dump(minds_to_pickle, f)
                    print('saving minds_to_pickle: ', len(minds_to_pickle))
            ControlBoard.IS_NEW_LEAP = False

        elif com.lower() == 'best':
            _, best_male, best_female = sim.find_best_minds(sim.evaluate())
            for i in range(len(best_male)):
                print(f'male: {best_male[-i - 1][0]}, female: {best_female[-i - 1][0]}')

        elif com[0].lower() == 'i':
            if com[1].lower() == 'a':
                print(ControlBoard.info_search(sim, int(com[2:]), info='attitudes'))
            elif com[1].lower() == 'l':
                print(ControlBoard.info_search(sim, int(com[2:]), info='locations'))
            else:
                print(ControlBoard.info_search(sim, int(com[1:])))

        elif com.lower() == 'display':
            sim.display()

        elif com.lower() == 'visualize':
            sim_map.plot_map(locations=sim.get_locations(), pops=sim.pop_density())
            return sim

        elif com[0].lower() == "x":
            sim = None

        else:
            if sim is None:
                sim = Simulation(sim_map)
            if com[0].lower() == "s":
                sim = ControlBoard.exact_skip(sim, int(com[1:]), regressive=ControlBoard.REGRESSIVE)
            elif com[0].lower() == "y":
                sim = ControlBoard.exact_skip(sim, int(com[1:]) * 12, regressive=ControlBoard.REGRESSIVE)
            else:
                sim = ControlBoard.exact_skip(sim, 1)

        return sim

    @staticmethod
    def info_search(sim: Simulation, id: int, info='basic'):
        """
            Search for specific information within a simulation.

            Args:
                sim: The current simulation object
                id: The ID of the person for whom information is requested
                info: A string indicating the specific information to retrieve. 'attitudes' to obtain the person's relationship list with others, and 'locations' to view their location history.

            Returns:
                The requested information in the form of a dictionary or list
            """
        if info == 'attitudes':
            return sim.get_attitudes(id)
        elif info == 'locations':
            return sim.get_historical_figure(id)[0].brain.get_location_history()
        person, history = sim.get_historical_figure(id)
        return (f'\n{person.display()}'
                f'\n{history}')

    @staticmethod
    def exact_skip(sim: Simulation, num: int, failsafe: bool = False, regressive='none', take_enough=False):
        """
        The function that handles the advancement of a Simulation over time.

        :param take_enough: a boolean. if set to True will disregard Simulation.SELECTED_COUPLES.
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

            # number of children born. used to access a Simulation's success
            print(sim.get_number_of_children())
            print(sim.divide_by_generation())

            if sim.Time < dest_time:
                if failsafe:
                    best_minds, male_lst, female_lst = sim.find_best_minds(
                        sim.evaluate(),
                        children_bearers='enough' if take_enough else 'best')
                    processed_best_minds, unified_lst = Simulation.prepare_best_for_reprocess(best_minds,
                                                                                              male_lst[:, 1:],
                                                                                              female_lst[:, 1:])

                    if regressive == 'full':
                        # This learning algorithm always chooses the best across all Simulations. It takes the best from
                        # the last simulation and compares it to the former best
                        best_minds_lst.extend(processed_best_minds)
                        quality_lst = np.append(quality_lst, unified_lst, axis=0)

                        best_minds, male_lst, female_lst = sim.find_best_minds(
                            [best_minds_lst, quality_lst[:, 0], quality_lst[:, 1], quality_lst[:, 2]],
                            children_bearers='enough' if take_enough else 'best')
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
                        (len(best_minds_lst) // 2, 2, 2)),
                                         )

                    del sim
                    sim = new_sim
                else:
                    sim.display()
                    return sim
            else:
                return sim


class ProgressBar(tqdm.tqdm):
    """
        A wrapper class for the imported `tqdm` class. It is designed to display a custom progress bar.
        This class provides the following features:

        * A custom progress bar format
        * The ability to run in a `for` loop
    """

    def __init__(self, iterations):
        """
            Initialize the progress bar.

            Args:
                iterations: The total number of iterations for the progress bar.

            Returns:
                None
        """
        super().__init__(range(iterations), ncols=80, ascii="░▒▓█", unit="m", colour="blue")
