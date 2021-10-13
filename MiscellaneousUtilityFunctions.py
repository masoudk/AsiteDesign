import sys
import re
import copy
from numpy import exp, sum
from numpy.random import randint
from io import StringIO
from mpi4py import MPI
from collections.abc import Mapping, Container
from sys import getsizeof
import psutil
import pyrosetta as pr
from pyrosetta.rosetta.core.scoring import score_type_from_name

    # Local imports
from Constants import DEBUG

    # Biotite imports
from biotite.sequence.align import Alignment, SubstitutionMatrix

def consesueSequence(sequences):
    pass

# Helper functions
def killProccesses(msg=''):
    """
    Helper functions that it is called when an error in the code occurs

    OUTPUT
    ------
    Error message and the code is exited
    """
    print('Error >>> {}'.format(msg), file=sys.stderr, flush=True)
    if MPI.COMM_WORLD.size > 1:
        MPI.COMM_WORLD.Abort(1)
    else:
        exit(1)

def printDEBUG(msg, rank=''):
    caller = getCallerName(depth=2)
    callerCaller = getCallerName(depth=3)
    print('Rank: {}, Function: {}: called from {}: >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(rank, caller, callerCaller), file=sys.stderr, flush=True)
    print(msg, file=sys.stderr, flush=True)
    print('Rank: {}, Function: {}: called from {}: >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(rank, caller, callerCaller), file=sys.stderr, flush=True)

def getKT(x, Th, Tl, N, k):
    """
    Helper functions that it is called when the temperature for MC sampling is needed
    """

    # If only one step, there is no point in calculating the temperature
    if N == 1:
        return Tl

    # Compute linear T
    if not k:
        deltaT = Th-Tl
        dT = deltaT/(N-1)
        T = Th - (dT * x)
    # compute exponential T
    else:
        decay = 1/(N * 0.1)
        T = Tl + (Th * exp(-decay*x))

    # Return with respect to the boundary
    if T > Th:
        return Th
    elif T < Tl or x == (N-1):
        return Tl
    else:
        return T

def getCallerName(depth=2):
    import inspect
    return  inspect.stack()[depth].function

def deep_getsizeof(o, ids):
    """Find the memory footprint of a Python object

    ref: https://code.tutsplus.com/tutorials/understand-how-much-memory-your-python-objects-use--cms-25609

    This is a recursive function that drills down a Python object graph
    like a dictionary holding nested dictionaries with lists of lists
    and tuples and sets.

    The sys.getsizeof function does a shallow size of only. It counts each
    object inside a container as pointer only regardless of how big it
    really is.

    :param o: the object
    :param ids:
    :return:
    """
    d = deep_getsizeof
    if id(o) in ids:
        return 0

    r = getsizeof(o)
    ids.add(id(o))

    if isinstance(o, str):
        return r

    if isinstance(o, Mapping):
        return r + sum(d(k, ids) + d(v, ids) for k, v in o.iteritems())

    if isinstance(o, Container):
        return r + sum(d(x, ids) for x in o)

    return r

def availableMemory():
    mem =  psutil.virtual_memory()
    return mem.available

def getScoreFunction(mode='fullAtomWithConstraints'):
    """
    A wraper function to handel score function generation
    :param mode: str: 'full', 'fullWithConstraints', 'constraints'
    :return scorefxn: Pyrosetta score function
    """
    if mode not in ['fullAtom', 'fullAtomWithConstraints', 'onlyConstraints']:
        killProccesses('bad score function definition: {}'.format(mode))

    scorefxn = None
    if mode == 'fullAtom':
        scorefxn = pr.get_fa_scorefxn()

    elif mode == 'fullAtomWithConstraints':
        scorefxn = pr.get_fa_scorefxn()
        scorefxn.set_weight(score_type_from_name('atom_pair_constraint'), 1.0)
        scorefxn.set_weight(score_type_from_name('res_type_constraint'), 1.0)

    elif mode == 'onlyConstraints':
        scorefxn = pr.ScoreFunction()
        scorefxn.set_weight(score_type_from_name('atom_pair_constraint'), 1.0)
        scorefxn.set_weight(score_type_from_name('res_type_constraint'), 1.0)

    return scorefxn

