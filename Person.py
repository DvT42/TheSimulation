import math
import enum
from Brain import *


class Person:
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
        f: Person = self.father_of_child
        self.father_of_child = None

        self.pregnancy = 0
        self.child_num += 1
        f.child_num += 1

        return Person(collective=self.collective, father=f, mother=self)

    def natural_death_chance(self, region):
        death_chance = region.risk * 0.06 * math.exp(-0.02 * self.strength)
        random_number = random.random()
        return random_number < death_chance

    # noinspection PyTypeChecker
    def action(self, region):
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
        if self.age < 15 * 12:
            self.strength += 0.25

        elif self.age > Person.AGING_STARTING_AGE:
            self.strength -= 1
            if self.age > Person.DRASTIC_AGING_AGE:
                self.strength -= 1

    def is_possible_partner(self, other):
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
        if self.age > self.readiness:
            available_ids = region.pop_id.copy()
            pop_size = min(np.max(available_ids)+1, self.collective.population_size)
            arr: np.ndarray = self.collective.world_attitudes[self.id, :pop_size]
            available = np.array([arr[i] for i in available_ids if i < pop_size])
            while (available > 0).any():
                other = self.collective.historical_population[available_ids[np.argmax(available)]]
                if self.is_possible_partner(other):
                    return other
                available[np.argmax(available)] = 0
            return None

    def year(self):
        return self.age // 12

    @staticmethod
    def Person_reset():
        Person.runningID = 0

    # Override of the conversion to string
    def __repr__(self):
        # basic information
        txt = f"{self.gender} {self.id}"
        return txt

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
