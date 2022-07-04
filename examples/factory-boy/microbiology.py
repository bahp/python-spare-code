"""
Factory Boy
-----------

"""


import factory
import inspect

# Specific
from datetime import timedelta
from random import randrange


MICROORGANISMS = ['O%s'%i for i in range(10)]
ANTIMICROBIALS = ['A%s'%i for i in range(10)]
SPECIMENS = ['BLDCUL', 'URICUL']
METHODS = ['DD']
SENSITIVITIES = ['S', 'I', 'R']

# -----------------------
# Classes
# -----------------------
class Base:
    """"""
    def __init__(self, **attributes):
        for key, value in attributes.items():
            self.__dict__[key] = value

    def __str__(self):
        attrs = [i
                 for i in inspect.getmembers(self)
                 if not i[0].startswith('_') and
                 not inspect.ismethod(i[1])
        ]
        return ' | '.join(['%s:%s' % (k,v) for k,v in attrs])

class Sample(Base):
    pass

class Test(Base):
    pass


# -----------------------
# Factories
# -----------------------
class SampleFactory(factory.Factory):
    """Factory to create samples."""
    class Meta:
        model = Sample

    date_collected = factory.Faker('date_time')
    laboratory_number = factory.Sequence(lambda n: 'LAB%08d' % n)
    specimen = factory.Iterator(SPECIMENS)


class TestFactory(factory.Factory):
    """Factory to create susceptibility tests."""
    sample = factory.SubFactory(SampleFactory)
    microorganism = factory.Iterator(MICROORGANISMS)
    antimicrobial = factory.Iterator(ANTIMICROBIALS)
    method = factory.Iterator(METHODS)
    sensitivity = factory.Iterator(SENSITIVITIES)

    @factory.lazy_attribute
    def date_received(self):
        return self.sample.date_collected + timedelta(hours=self.offset1)

    @factory.lazy_attribute
    def date_outcome(self):
        return self.date_received + timedelta(hours=self.offset2)

    class Meta:
        model = Test

    class Params:
        offset1 = randrange(3, 24)
        offset2 = randrange(30, 60)



        
if __name__ == '__main__':

    # Libraries
    import random
    from random import randrange

    def create_sample_random(norgs, nabxs):
        """Creates samples (randomly)"""
        s = SampleFactory()
        for o in random.sample(MICROORGANISMS, norgs):
            for a in random.sample(ANTIMICROBIALS, nabxs):
                t = TestFactory(sample=s, antimicrobial=a, microorganism=o)
                print(t)

    def create_sample_seq(norgs, nabxs):
        """Creates samples (Iterator)

        .. note: Ensure that N_MAX_ORGS and N_MAX_ABXS are
                 lower than the lengths of the corresponding
                 items.
        """
        s = SampleFactory()
        for o in range(norgs):
            t = TestFactory.create_batch(nabxs, sample=s)
            for e in t:
                print(e)


    # Constants
    N_SAMPLES = 2
    N_MAX_ORGS = 4
    N_MAX_ABXS = 6

    print("\nSEQUENTIAL:")
    for ni in range(N_SAMPLES):
       create_sample_seq(
           norgs=randrange(N_MAX_ORGS),
           nabxs=randrange(N_MAX_ABXS)
       )

    print("\nRANDOM:")
    for ni in range(N_SAMPLES):
       create_sample_random(
           norgs=randrange(N_MAX_ORGS),
           nabxs=randrange(N_MAX_ABXS)
       )




