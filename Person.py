import math
import enum
from Brain import *



class Person:
    MAX_POPULATION = 1000000
    AGING_STARTING_AGE = 40 * 12
    DRASTIC_AGING_AGE = 75 * 12
    STRENGTH_MODIFIER = 0.75
    DIFF_AGE = 15 * 12
    GIVE_UP_DISTANCE = 1
    DEATH_NORMALIZER = 0.3
    INITIAL_AGE = 0 * 12
    PREGNANCY_LENGTH = 12
    runningID = 0
    ages = np.zeros(MAX_POPULATION, dtype=int)
    _lock = Lock()

    def __init__(self, collective, father, mother=None, brain=None):
        # ID assignment
        self.isAlive = True
        self.isManual = type(father) is list
        self.action_flag = False
        with self._lock:
            self.id = Person.runningID
            Person.runningID += 1

        self.collective = collective

        # Attributes that don't depend on the parents
        self.gender = random.choice(list(Gender))
        if Person.MAX_POPULATION < Person.runningID:
            print("Reached Max Person.runningID")

        self.readiness = 12 * 12 + int(random.normalvariate(0, 12))

        self.brain = brain if brain else Brain(self, father, mother, collective)
        self.partner = None
        self.child_num = 0

        # Attributes that depend on the parents:
        # - standard creation
        if type(father) is Person:
            self.fatherID = father.id
            self.motherID = mother.id

            self.father_history = father.brain.get_history()
            self.mother_history = mother.brain.get_history()

            # strength - represents physical ability and health. {mean strength of parents}/2 + {normal distribution}/2
            self.starting_strength = (
                    (father.starting_strength + mother.starting_strength) // 4 + int(random.normalvariate(50, 10)))

            self.strength = self.starting_strength // 10

            # get born in the mother's place
            self.location = np.copy(mother.location)

            self.generation = max(father.generation, mother.generation)
            self.brain.update_location_history()

        # - Manual creation
        else:
            # [0]
            self.gender = father[0]
            # [1]
            self.strength = father[1]
            self.starting_strength = self.strength
            # [2]
            self.location = np.copy(father[2])
            self.generation = 0
            self.brain.update_location_history(self.location, 0)


        # procreation availability.
        if self.gender == Gender.Female:
            self.pregnancy = 0
            self.father_of_child = None
            self.youth = 30 * 12 + int(random.normalvariate(0, 24))
            if self.isManual:
                self.youth -= 8 * 12

    def prepare_next_generation(self, other):
        other: Person
        if self.gender == Gender.Female:
            if self.pregnancy == 0 and self.father_of_child is None and self.age() >= self.readiness and self.youth > 0:
                self.father_of_child = other
                self.partner = other
                other.partner = self
        else:
            if other.pregnancy == 0 and other.father_of_child is None and other.age() >= other.readiness and other.youth > 0:
                other.father_of_child = self
                self.partner = other
                other.partner = self

    def birth(self):
        f: Person = self.father_of_child
        self.father_of_child = None

        self.pregnancy = 0
        self.child_num += 1
        f.child_num += 1

        return Person(collective=self.collective, father=f, mother=self)

    def natural_death_chance(self):
        death_chance = Person.DEATH_NORMALIZER * 0.06 * math.exp(-0.02 * self.strength)
        random_number = random.random()
        return random_number < death_chance

    # noinspection PyTypeChecker
    @profile
    def action(self, region):
        decision = self.brain.call_decision_making(region=region)

        if decision == 0:
            # should improve attitudes/merge
            self.brain.set_history(self.age(), 1)
        if decision == 1:
            self.strength += Person.STRENGTH_MODIFIER
            self.brain.set_history(self.age(), 2)
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
            self.brain.set_history(self.age(), 3)
        return decision

    def aging(self):
        if self.age() > Person.AGING_STARTING_AGE:
            self.strength -= 1
            if self.age() > Person.DRASTIC_AGING_AGE:
                self.strength -= 1

    def is_possible_partner(self, other):
        other: Person
        if other.isAlive:
            if (self.location == other.location).all():
                if other.age() > other.readiness:
                    if self.gender != other.gender and not other.partner and abs(
                            self.age() - other.age()) < Person.DIFF_AGE:
                        if self.brain.get_attitudes(other) > 0.7 and other.brain.get_attitudes(self) > 0.7:
                            return True
        return False

    def partner_selection(self, region):
        if self.age() > self.readiness:
            available_ids = region.pop_id
            arr: np.ndarray = self.collective.world_attitudes[self.id, :self.collective.population_size]
            available = np.array([arr[i] for i in available_ids])
            while available.any():
                other = self.collective.historical_population[available_ids[np.argmax(available)]]
                if self.is_possible_partner(other):
                    return other
                available[np.argmax(available)] = 0
            return None

    # noinspection PyTypeChecker
    def age(self):
        return Person.ages[self.id]

    def year(self):
        return self.age() // 12

    @staticmethod
    def Person_reset(initial_couples):
        Person.runningID = 0
        Person.ages = np.zeros(Person.MAX_POPULATION, dtype=int)
        Person.ages[:initial_couples * 2] = Person.INITIAL_AGE

    # Override of the conversion to string
    def __repr__(self):
        # basic information
        txt = f"{self.id}: " \
              f"gender: {self.gender.name}, age: {self.year()}, children: {self.child_num} " \
              f"str: {self.strength}"

        # pregnancy data
        if self.gender == Gender.Female and self.pregnancy != 0:
            txt += f", pregnancy: {self.pregnancy}, mate: {self.father_of_child.id}"

        return txt + ";"

    def display(self):
        # basic information
        if not self.isManual:
            txt = f"{self.id}: \n" \
                  f"parents: [{self.fatherID}, {self.motherID}], generation: {self.generation} \n"
        else:
            txt = (f"{self.id}: \n"
                   f"generation: {self.generation} \n")

        txt += f"gender: {self.gender.name}, age: {self.year()}, children: {self.child_num}\n" \
               f"strength: {self.strength}\n" \
               f"last action: {Brain.get_action_from_history(self.age(), self.brain.get_history())}"

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
