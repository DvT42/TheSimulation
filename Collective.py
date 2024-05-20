from threading import Lock
import numpy as np


class Collective:
    """
        Represents the collective population and global human-level data for the simulation.

        **Constants:**

        BASIC_POPULATION: A large number representing the maximum expected population size in the simulation. Larger values may trade off memory usage for faster runtime, as long as the population does not exceed this limit.

        INHERITANCE_RATIO: A user-defined value representing the extent to which an individual inherits traits from their parents. For instance, a value of 0.5 indicates that traits are inherited equally from both parents.

        MUTATION_NORMALIZATION_RATIO: A number inversely proportional to the degree of mutation in inherited brain traits from parents to offspring.

        CHOICE_RANDOMIZER: A random number between 0 and 1 representing the probability of an individual making a choice that deviates from their brain's preference. Each month, a random number between 0 and 1 is generated for each individual.

        CHOICE_NUM: The number of choices available to individuals.

        SHOULD_PROBABLY_BE_DEAD: A number of months initialized as an improbable lifespan. Similar in concept to `BASIC_POPULATION`.

        ATTITUDE_IMPROVEMENT_BONUS: The extent to which relationships improve when individuals choose to be friendly (see flowchart).

        SELF_ATTITUDE_IMPROVEMENT_BONUS: The extent to which individuals improve their self-attitude when they are not friendly (see flowchart).

        **Class Attributes:**

        population_size: An integer representing the total number of individuals who have ever lived in the simulation.

        _dead: An integer representing the total number of individuals who have died in the simulation.

        world_attitudes: A 2D NumPy array of size (population_size, population_size) storing the attitudes of all individuals towards each other.

        arranged_indexes: An auxiliary array containing all ordinal numbers up to `BASIC_POPULATION`.

        historical_population: A list containing all individuals who have ever lived in the simulation.

        _lock: A `Lock` object preventing concurrent modification of values in the `Collective` instance.
        """
    # Collective Constants:
    BASIC_POPULATION = 10000

    # brainpart constants:
    INHERITANCE_RATIO = 0.3
    MUTATION_NORMALIZATION_RATIO = 0.3

    # PFC constants
    CHOICE_RANDOMIZER = 0.0
    CHOICE_NUM = 10

    # HPC constants
    SHOULD_PROBABLY_BE_DEAD = 120 * 12
    ATTITUDE_IMPROVEMENT_BONUS = 0.05
    SELF_ATTITUDE_IMPROVEMENT_BONUS = 0.1

    def __init__(self):
        """
            Initializes the instance of the class.
        """
        self.population_size = 0
        self._dead = 0
        self.world_attitudes = np.zeros((Collective.BASIC_POPULATION, Collective.BASIC_POPULATION), dtype=float)
        self.arranged_indexes = np.arange(Collective.BASIC_POPULATION)
        self.historical_population = []
        self._lock = Lock()

    def add_person(self, person):
        """
            Adds a newly born person to the collective population.

            Args:
                person (Person): The Person object representing the newly born individual.
        """
        with self._lock:
            self.historical_population.append(person)

            if self.population_size >= Collective.BASIC_POPULATION:
                new_world_attitudes = np.zeros((self.population_size + 1, self.population_size + 1))
                new_world_attitudes[:self.population_size, :self.population_size] = self.world_attitudes
                self.world_attitudes = new_world_attitudes

                self.arranged_indexes = np.append(self.arranged_indexes, self.population_size)

            self.population_size += 1

    @property
    def dead(self):
        return self._dead

    def remove_person(self):
        """
            Increments the `_dead` counter by 1, indicating the death of an individual.

            This method does not perform any specific actions related to the removal of a person from the simulation. It simply updates the `_dead` count.
        """
        with self._lock:
            self._dead += 1
