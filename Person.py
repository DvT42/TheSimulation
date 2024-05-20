import enum
import math

from Brain import *


class Person:
    """
    Represents an individual person within the simulation environment.

    This class encapsulates the attributes and behaviors of a single person, including their age, gender, health, relationships, and actions within the simulated world.

    Attributes:
        isAlive (bool): Indicates whether the person is alive or not.
        isManual (bool): Indicates whether the person was created manually (by parents) or automatically (at the start of the simulation).
        action_flag (bool): A flag to track whether the person has performed an action in the current month, preventing multiple actions per month.
        id (int): A unique identifier for the person.
        collective (Collective): The collective to which the person belongs.
        gender (str): The person's gender ('male' or 'female').
        readiness (int): The age in months at which the person becomes capable of reproduction.
        brain (Brain): The person's brain object, representing their cognitive abilities.
        partner (Person): The person's current partner (if any).
        child_num (int): The number of children the person has fathered/birthed.
        age (int): The person's age in months.
        fatherID (int): The ID of the person's father (if applicable).
        motherID (int): The ID of the person's mother (if applicable).
        father_history (list): A list of the father's action history, potentially used for intergenerational learning.
        mother_history (list): A list of the mother's action history, potentially used for intergenerational learning.
        starting_strength (int): The person's initial strength, determined genetically.
        strength (int): The person's current strength, inversely related to their mortality risk.
        location (Region): The region in which the person is currently located.
        generation (int): The person's generation. For manual individuals, this is 0; for others, it's 1 + the maximum generation of their parents.
        pregnancy (int): A value representing the progress of pregnancy for female individuals, measured in months.
        father_of_child (Person): A reference to the father of the unborn child (for pregnant females).
        youth (int): A value that represents the remaining reproductive years for female individuals. As long as this value is greater than 0, they can bear children.

    Class Constants:
        AGING_STARTING_AGE (int): The age in months when aging starts to affect an individual.
        DRASTIC_AGING_AGE (int): The age in months when aging becomes significantly impactful.
        STRENGTH_MODIFIER (int): The amount by which an individual's strength increases if they focus on it in a given month.
        DIFF_AGE (int): The maximum age difference allowed for marriage.
        GIVE_UP_DISTANCE (int): The maximum distance between partners that the simulation allows for separation.
        DEATH_NORMALIZER (float): A scaling factor for mortality risk.
        INITIAL_AGE (int): The starting age for automatically generated individuals.
        PREGNANCY_LENGTH (int): The duration of pregnancy in months.
        LOVE_BAR (int): The minimum love level required for marriage.
        runningID (int): A counter for generating unique IDs for new individuals.
        _lock (Lock): A lock object to prevent concurrent modifications of person attributes.
    """
    AGING_STARTING_AGE = 40 * 12
    DRASTIC_AGING_AGE = 75 * 12
    STRENGTH_MODIFIER = 0.75
    DIFF_AGE = 15 * 12
    GIVE_UP_DISTANCE = 1
    DEATH_NORMALIZER = 0.3
    INITIAL_AGE = 0 * 12
    PREGNANCY_LENGTH = 12
    LOVE_BAR = 0.7
    runningID = 0
    _lock = Lock()

    def __init__(self, collective, father=None, mother=None, brain=None, properties=None):
        """
            Initializes a Person object with the specified attributes and properties.

            This constructor sets up the fundamental characteristics of a person, including their collective affiliation, family relationships, cognitive abilities, and initial state.

            Args:
                collective (Collective): The collective to which the person belongs.
                father (Person.Person): The person's father (if applicable).
                mother (Person.Person): The person's mother (if applicable).
                brain (Brain): The person's brain object, representing their cognitive abilities.
                properties (list): A list containing optional properties for the person, such as starting strength.
            """

        # ID assignment
        self.isAlive = True
        self.isManual = properties is not None
        self.action_flag = False
        with self._lock:
            self.id = Person.runningID
            Person.runningID += 1

        self.collective = collective

        # Attributes that don't depend on the parents
        self.gender = random.choice(list(Gender))

        self.readiness = 12 * 12 + int(random.normalvariate(0, 12))

        self.partner = None
        self.child_num = 0

        # Attributes that depend on the parents:
        # - standard creation
        if not self.isManual:
            self.age = 0

            self.fatherID = father.id
            self.motherID = mother.id

            self.father_history = father.brain.get_history(simplified=False)
            self.mother_history = mother.brain.get_history(simplified=False)

            # strength - represents physical ability and health. {mean strength of parents}/2 + {normal distribution}/2
            self.starting_strength = (
                    (father.starting_strength + mother.starting_strength) // 4 + int(random.normalvariate(50, 10)))

            self.strength = self.starting_strength // 10

            # get born in the mother's place
            self.location = np.copy(mother.location)

            self.generation = max(father.generation, mother.generation) + 1

        # - Manual creation
        else:
            # [0]
            self.gender = properties[0]
            # [1]
            self.strength = properties[1]
            self.starting_strength = self.strength
            # [2]
            self.location = np.copy(properties[2])
            self.generation = 0
            self.age = Person.INITIAL_AGE

        self.brain = brain.copy() if brain else Brain(self, father, mother)
        self.brain.update_location_history(self.location, 0)

        # procreation availability.
        if self.gender == Gender.Female:
            self.pregnancy = 0
            self.father_of_child = None
            self.youth = 30 * 12 + int(random.normalvariate(0, 24))
            if self.isManual:
                self.youth -= (Person.INITIAL_AGE - 12 * 12) if Person.INITIAL_AGE > 12 * 12 else 0

    def prepare_next_generation(self, other):
        """
            Initiates the pregnancy process for the female Person object.

            This method handles the initial stages of pregnancy, including tracking progress, updating population counts, and potentially involving the father.

            Args:
                other (Person): The partner involved in the pregnancy process.
            """
        other: Person
        if self.gender == Gender.Female:
            if self.pregnancy == 0 and self.father_of_child is None and self.youth > 0:
                self.father_of_child = other
                self.partner = other
                other.partner = self
        else:
            if other.pregnancy == 0 and other.father_of_child is None and other.youth > 0:
                other.father_of_child = self
                self.partner = other
                other.partner = self

    def birth(self):
        """
            Delivers the baby and creates a new Person object representing the newborn.

            This method handles the final stages of pregnancy, including creating a new Person object, initializing its attributes, and updating relationships and population counts.
            """
        f: Person = self.father_of_child
        self.father_of_child = None

        self.pregnancy = 0
        self.child_num += 1
        f.child_num += 1

        return Person(collective=self.collective, father=f, mother=self)

    def natural_death_chance(self, region):
        """
            Calculates the probability of natural death for the Person object.

            This method assesses the person's mortality risk based on their current strength and other factors.

            Returns:
                bool: whether the person died.
            """
        death_chance = region.risk * 0.06 * math.exp(-0.02 * self.strength)
        random_number = random.random()
        return random_number < death_chance

    # noinspection PyTypeChecker
    def action(self, region):
        """
            Determines and executes the person's action for the current month.

            This method serves as the core of the person's decision-making process, utilizing the brain to make choices and updating the person's attributes accordingly.

            Args:
                region (Region.Region): The region in which the person is currently located.
            """
        decision = self.brain.call_decision_making(region=region)

        if decision == 0:
            # should improve attitudes/merge
            self.brain.set_history(self.age, 1)
        if decision == 1:
            self.strength += Person.STRENGTH_MODIFIER * region.resource_distribution()
            self.brain.set_history(self.age, 2)
        if decision in np.arange(2, 10):
            if decision == 2:
                self.location += (-1, 1)
            elif decision == 3:
                self.location += (0, 1)
            elif decision == 4:
                self.location += (1, 1)
            elif decision == 5:
                self.location += (-1, 0)
            elif decision == 6:
                self.location += (1, 0)
            elif decision == 7:
                self.location += (-1, -1)
            elif decision == 8:
                self.location += (0, -1)
            elif decision == 9:
                self.location += (1, -1)

            self.location[0] %= 1600

            self.brain.update_location_history()
            self.brain.set_history(self.age, decision + 1)
        return decision

    def aging(self):
        """
            Simulates the effects of aging on the Person object.

            This method handles the physical and cognitive changes associated with aging, including strength modifications and potential brain deterioration.
            """
        if self.age < 15 * 12:
            self.strength += 0.25

        elif self.age > Person.AGING_STARTING_AGE:
            self.strength -= 1
            if self.age > Person.DRASTIC_AGING_AGE:
                self.strength -= 1

    def is_possible_partner(self, other):
        """
            Determines whether the specified Person object is a potential romantic partner.

            This method considers various factors, including age, strength, love bar, and gender, to assess compatibility.

            Args:
                other (Person): The other Person object to evaluate as a potential partner.

            Returns:
                bool: True if the other person is a potential partner, False otherwise.
        """
        other: Person
        if other.isAlive:
            if (self.location == other.location).all():
                if other.age > other.readiness:
                    if (self.gender != other.gender and not other.partner and
                            abs(self.age - other.age) < Person.DIFF_AGE):
                        if (self.brain.get_attitudes(other) > Person.LOVE_BAR and
                                other.brain.get_attitudes(self) > Person.LOVE_BAR):
                            return True
        return False

    def partner_selection(self, region):
        """
            Selects a potential romantic partner from the people in the current region.

            This method evaluates all eligible individuals in the region using the `is_possible_partner()` method and chooses the most compatible one.

            Args:
                region (Region.Region): The region in which the person is currently located.
            """
        if self.age > self.readiness:
            available_ids = region.pop_id.copy()
            pop_size = min(np.max(available_ids) + 1, self.collective.population_size)
            arr: np.ndarray = self.collective.world_attitudes[self.id, :pop_size]
            available = np.array([arr[i] for i in available_ids if i < pop_size])
            while (available > 0).any():
                other = self.collective.historical_population[available_ids[np.argmax(available)]]
                if self.is_possible_partner(other):
                    return other
                available[np.argmax(available)] = 0
            return None

    def year(self):
        """
            Returns the person's age in years.

            This method simply calculates the age based on the person's current `age` attribute.

            Returns:
                int: The person's age in years.
            """
        return self.age // 12

    @staticmethod
    def Person_reset():
        """
            Resets the runningID counter for Person objects.

            This method ensures that each newly created Person object receives a unique ID.
            """
        Person.runningID = 0

    def __repr__(self):
        """
            Returns a string representation of the Person object.

            This method provides a concise and informative representation of the person's attributes.

            Returns:
                str: The string representation of the Person object.
            """
        # basic information
        txt = f"{self.gender} {self.id}"
        return txt

    def display(self):
        """
            Prints all relevant information about the Person object.

            This method provides a comprehensive overview of the person's attributes.
        """
        # basic information
        if not self.isManual:
            txt = f"{self.id}: \n" \
                  f"parents: [{self.fatherID}, {self.motherID}], generation: {self.generation} \n"
        else:
            txt = (f"{self.id}: \n"
                   f"generation: {self.generation} \n")

        txt += f"gender: {self.gender.name}, age: {self.year()}, children: {self.child_num}\n" \
               f"strength: {self.strength}\n" \
               f"last action: {Brain.get_action_from_history(self.age, self.brain.get_history())}"

        # pregnancy data
        if self.gender == Gender.Female:
            if self.pregnancy != 0:
                txt += f"\n pregnancy: {self.pregnancy}, mate: {self.father_of_child.id}"
            else:
                txt += f"\n timer: {self.youth}"

        return txt


class Gender(enum.IntEnum):
    Female = 0
    Male = 1
