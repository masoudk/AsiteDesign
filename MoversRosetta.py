# Global imports
import os, sys
import re
import itertools
import copy, time
from numpy import exp, sum, array, where, sqrt, any, std, mean
from numpy.random import randint, normal, uniform
from numpy.linalg import norm
from io import StringIO
from mpi4py import MPI
from itertools import cycle

from random import choice

# PyRosetta import
import pyrosetta as pr

# PyRosetta mover and pakcer imports
from pyrosetta.rosetta.core.kinematics import MoveMap
from pyrosetta.rosetta.core.pack.task.operation import ReadResfile
from pyrosetta.rosetta.core.pack.task import TaskFactory
from pyrosetta.rosetta.protocols.minimization_packing import PackRotamersMover, MinMover
from pyrosetta.rosetta.protocols.constraint_movers import ClearConstraintsMover
from pyrosetta.rosetta.protocols.relax import FastRelax
from pyrosetta.rosetta.protocols.denovo_design.movers import FastDesign
from pyrosetta.rosetta.core.scoring.constraints import BoundFunc, AtomPairConstraint, AmbiguousConstraint, ResidueTypeConstraint
from pyrosetta.rosetta.core.scoring import ScoreType
from pyrosetta.rosetta.utility import vector1_std_shared_ptr_const_core_conformation_Residue_t
from pyrosetta.rosetta.std import ostringstream, istringstream
from pyrosetta.rosetta.protocols.geometry import centroid_by_residues, centroid_by_chain
from pyrosetta.rosetta.core.scoring import score_type_from_name
from pyrosetta.rosetta.protocols.docking import setup_foldtree
from pyrosetta.rosetta.protocols.rigid import RigidBodyPerturbMover
from pyrosetta.rosetta.protocols.simple_moves import ShearMover
from pyrosetta.rosetta.core.kinematics import FoldTree
from pyrosetta.rosetta.numeric import xyzVector_double_t
from pyrosetta.rosetta.core.select.residue_selector import ChainSelector, NeighborhoodResidueSelector, \
    ResidueIndexSelector
from pyrosetta import Pose
# Biotite import
from biotite.structure.io.pdb import PDBFile
from biotite.structure import sasa, annotate_sse, apply_residue_wise

# BIO Python imports
from Bio.PDB.Polypeptide import one_to_three

# Local imports
import Constants
from BaseClases import BaseConstraint, Residue, AminoAcids, DesignDomainWildCards
from MiscellaneousUtilityFunctions import killProccesses, getKT, printDEBUG, getScoreFunction
from Docking import LigandMover


# This is for testing
# pr.init(''' -out:level 0 ''')
# This is for production run
# pr.init(''' -out:level 0 -no_his_his_pairE -extrachi_cutoff 1 -multi_cool_annealer 10 -ex1 -ex2 -use_input_sc ''')

# TODO LOGGING

# reff:
#       http://www.programmersought.com/article/97651668888/
#       https://colab.research.google.com/github/RosettaCommons/PyRosetta.notebooks/blob/master/notebooks/06.02-Packing-design-and-regional-relax.ipynb#scrollTo=9DTnXbppN9OT
#       https://www.rosettacommons.org/docs/latest/full-options-list

class DesignPose(object):
    """
    DesignPose is combines a pose with necessary methods to keep track of state of
    pose during active site design.
    """

    def __init__(self):
        # if you add
        self.pose = pr.Pose()
        self.designInitiated = False
        self.design = list()
        self.designDict = dict()
        self.nDesignResidues = 0
        self.catalyticInitiated = False
        self.catalytic = list()
        self.catalyticTargetAA = dict()
        self.catalyticDict = dict()
        self.catalyticDesign = False
        self.nCatalyticResidues = 0
        self.nNoMutateCatalyticResidues = 0
        self.catalyticAction = dict()
        self.activeResidues = list()
        self.constraintInitiated = False
        self.constraints = list()
        self.noMutate = dict()  # Hold the catalytic index of non-mutatable residues
        self.currentNoneCatalyticMutants = list()
        self.originalSequence = list()

    def __iter__(self):
        for res in self.design:
            yield res

    def assign(self, other):
        """
        This method copies the attributes of one DesignDomainMover object
        to another one.
        """
        self.pose.assign(other.pose)
        self.designInitiated = copy.deepcopy(other.designInitiated)
        self.design = copy.deepcopy(other.design)
        self.designDict = copy.deepcopy(other.designDict)
        self.nDesignResidues = copy.deepcopy(other.nDesignResidues)
        self.catalyticInitiated = copy.deepcopy(other.catalyticInitiated)
        self.catalytic = copy.deepcopy(other.catalytic)
        self.catalyticTargetAA = copy.deepcopy(other.catalyticTargetAA)
        self.catalyticDict = copy.deepcopy(other.catalyticDict)
        self.catalyticDesign = copy.deepcopy(other.catalyticDesign)
        self.nCatalyticResidues = copy.deepcopy(other.nCatalyticResidues)
        self.nNoMutateCatalyticResidues = copy.deepcopy(other.nNoMutateCatalyticResidues)
        self.catalyticAction = copy.deepcopy(other.catalyticAction)
        self.activeResidues = copy.deepcopy(other.activeResidues)
        self.constraintInitiated = copy.deepcopy(other.constraintInitiated)
        self.constraints = copy.deepcopy(other.constraints)
        self.noMutate = copy.deepcopy(other.noMutate)
        self.currentNoneCatalyticMutants = copy.deepcopy(other.currentNoneCatalyticMutants)
        self.originalSequence = copy.deepcopy(other.originalSequence)

    def initiateDesign(self, residues, pose):
        if not all(map(lambda x: isinstance(x, Residue), residues)):
            raise ValueError('Failed during initiating the design residues. Only list of Residue(s) is allowed.')

        # set the pose
        self.pose.assign(pose)
        self.design = residues
        self.nDesignResidues = len(self.design)

        # Set the design dictionary for rapid access (needed during non-catalytic design constraints update)
        # The keys are the (ID, chain) tuples. The values are the res index in the design list
        for index, res in enumerate(self.design):
            self.designDict[(res.ID, res.chain)] = index

        # Set the pose index of the design residues
        self.setPoseIndex()

        # get the original sequence for design domain
        self.originalSequence = self.getPdbSequence()

        # Set the allowed aa for each design position. This is independent of job type and only
        # Fill the positions that are not defined by user 'X'.
        self.setAllowedAA()

        # Set the nativeAA of design residues.
        self.setNativeAAtoPdbAA()

        # Initiate the currentAA of design residues.
        # Try to initiate close to PDB structure
        self.mutateNonCatalyticAll(keepNativeAA=True)

        self.designInitiated = True

    def initiateCatalytic(self, catalytic):

        if catalytic is None:
            self.catalyticInitiated = False
            return

        if not self.designInitiated:
            raise ValueError('Failed during initiating the catalytic residues. Design residues are not initiated yet.')

        self.catalyticTargetAA = dict()
        for resName, aa in catalytic:

            # Check the non-mutable residues
            designIndex = self.designDict.get(resName, False)

            # If resName is already in design domain, the residue is non-mutable.
            if designIndex is not False:
                self.noMutate[resName] = designIndex
                self.design[designIndex].mutate = False

            # Assign the targetAA
            if resName in self.noMutate.keys():
                poseIndex = self.design[designIndex].poseIndex
                nativeAA = self.pose.residue(poseIndex).name1()

                # Use native amino acid and native rotamer
                if aa[0] in DesignDomainWildCards('frozen'):
                    self.catalyticTargetAA[resName] = list(nativeAA)
                    self.design[designIndex].allowedAA = list(nativeAA)
                    self.catalyticAction[resName] = 'NATRO'
                    self.design[designIndex].designAction = 'NATRO'
                    self.design[designIndex].currentAction = 'NATRO'

                # Use native amino acid but calculate rotamer
                elif aa[0] in DesignDomainWildCards('noneMutablePackable'):
                    self.catalyticTargetAA[resName] = list(nativeAA)
                    self.design[designIndex].allowedAA = list(nativeAA)
                    self.catalyticAction[resName] = 'PIKAA'
                    self.design[designIndex].designAction = 'PIKAA'
                    self.design[designIndex].currentAction = 'PIKAA'

                # New amino acid and calculate rotamer
                else:
                    self.catalyticTargetAA[resName] = list(aa)
                    self.design[designIndex].allowedAA = list(aa)
                    self.catalyticAction[resName] = 'PIKAA'
                    self.design[designIndex].designAction = 'PIKAA'
                    self.design[designIndex].currentAction = 'PIKAA'

            # Ambiguous residues where catalytic residues that are not defined yet (RES1, ...)
            # Presence of Ambiguous residues indicate catalytic design
            else:
                self.catalyticDesign = True
                self.catalyticTargetAA[resName] = list(aa)
                self.catalyticAction[resName] = 'PIKAA'

                # Check for non-mutable that are not in design residues
                if re.match('ZZ', aa[0]):
                    raise ValueError('residue {} is declared non mutable but could not be found in the design domain'.
                                     format(resName))

        # Assign the number of catalytic residues
        self.nCatalyticResidues = len(self.catalyticTargetAA)
        self.nNoMutateCatalyticResidues = len(self.noMutate)

        # initiate
        if self.catalyticDesign:
            # If it is a catalytic design, initial choice of the Ambiguous catalytic residues
            # is assigned by random
            self.mutateCatalyticAll()

            # If no Ambiguous catalytic residues is given, the positions are already known
            # just initiate catalytic list, catalyticDict and set the currentAA
        else:
            self.catalytic = list()
            self.catalyticDict = dict()

            for catalyticResName, catalyticResAA in self.catalyticTargetAA.items():
                index = self.designDict[catalyticResName]
                self.catalytic.append(index)
                self.design[index].catalyticName = copy.deepcopy(catalyticResName)
                self.design[index].currentAA = copy.deepcopy(catalyticResAA)

            for index in self.catalytic:
                # Get the catalytic residues Names
                name = copy.deepcopy(self.design[index].catalyticName)
                self.catalyticDict[name] = index

        self.catalyticInitiated = True

    def initiateConstraints(self, constraints):

        if constraints == None:
            self.constraintInitiated = False
            return
        elif type(constraints) != list:
            raise ValueError('failed during initiating constraints. Only list of constraint(s) is allowed.')
        elif not all(map(lambda constraint: isinstance(constraint, BaseConstraint), constraints)):
            raise ValueError('failed during initiating constraints. Only list of constraint(s) is allowed.')
        else:
            self.constraints = constraints

        self.constraintInitiated = True

    def setPoseIndex(self):
        posInfo = self.pose.pdb_info()
        for res in self.design:
            res.poseIndex = posInfo.pdb2pose(res.chain, res.ID)
            # TODO ADD LOGGING
            # print((res.chain, res.poseIndex, res.ID), pose.residue(res.poseIndex).name1())

    def getNonCatalyticPoseIndex(self):

        nonCatalyticPoseIndex = list()
        for res in self.design:
            if res.catalyticName:
                continue
            else:
                nonCatalyticPoseIndex.append(res.poseIndex)
        return nonCatalyticPoseIndex

    def setAllowedAA(self):
        """
        Calculates acceptable AA composition for a given site(s) based on
        solvent-accessible surface area (SASA) ans secondary structure (SSE)
        :param pdbFile:
        :param residueList:
        :return:
        """
        # Ref: https://github.com/sarisabban/RosettaDesign
        # Ref: https://doi.org/10.1371/journal.pone.0080635
        # Ref: The Molecules of Life: Physical and Chemical Principles. Page 216.

        # Get the pdb file out of the pose
        posePdb = ostringstream()
        self.pose.dump_pdb(posePdb)
        pdbFile = StringIO(posePdb.str())

        # Get the PDB file
        pdb = PDBFile.read(pdbFile)
        pdbStruc = pdb.get_structure()[0]

        # Amino acids properties
        aminoAcids = AminoAcids()

        # Calculate SASA
        atomsSASA = sasa(pdbStruc, vdw_radii="Single")
        strucSASA = apply_residue_wise(pdbStruc, atomsSASA, sum)

        # Calculate SSE
        strucSSE = annotate_sse(pdbStruc, pdbStruc.chain_id)

        # Calculate the AA compositions
        for res in self.design:
            # Get the pdb restype if XX+ or ZX.
            # The allowedAA will be updated later for catalytic residues.
            nativeAA = None
            if res.allowedAA[0] in DesignDomainWildCards('nativePlus'):
                nativeAA = self.pose.residue(res.poseIndex).name1()

            # Calculate the aa compositions
            if res.allowedAA[0] in DesignDomainWildCards('mutable'):
                resIndex = res.poseIndex - 1  # The index in Biotite starts from 0, while in PyRosetta starts from 1
                resName = pdbStruc.res_name[0]
                resRASA = strucSASA[resIndex] / aminoAcids.maxSASA3(resName)  # RASA refers to the relative SASA
                resSSE = strucSSE[resIndex]
                # label res based on RASA
                if resRASA <= 0.25:
                    resRASA = 'C'
                elif 0.25 < resRASA < 0.75:
                    resRASA = 'B'
                elif resRASA >= 0.75:
                    resRASA = 'S'

                # get the AA composition
                if resRASA == 'S' and resSSE == 'c':
                    aa = list('PGNQSTDERKH')
                elif resRASA == 'S' and resSSE == 'a':
                    aa = list('EHKQR')
                elif resRASA == 'S' and resSSE == 'b':
                    aa = list('DEGHKNPQRST')
                elif resRASA == 'B' and resSSE == 'c':
                    aa = list('ADEFGHIKLMNPQRSTVWY')
                elif resRASA == 'B' and resSSE == 'a':
                    aa = list('ADEHIKLMNQRSTVWY')
                elif resRASA == 'B' and resSSE == 'b':
                    aa = list('DEFHIKLMNQRSTVWY')
                elif resRASA == 'C' and resSSE == 'c':
                    aa = list('AFGILMPVWY')
                elif resRASA == 'C' and resSSE == 'a':
                    aa = list('AFILMVWY')
                elif resRASA == 'C' and resSSE == 'b':
                    aa = list('FILMVWY')
                res.allowedAA = aa
                res.designAction = "PIKAA"
                res.currentAction = "PIKAA"

                # Add the native AA if asked
                if nativeAA is not None and nativeAA not in res.allowedAA:
                    res.allowedAA.append(nativeAA)

            # Assign the native
            elif res.allowedAA[0] in ['ZX']:
                res.allowedAA = [nativeAA]
                res.designAction = "PIKAA"
                res.currentAction = "PIKAA"

            elif res.allowedAA[0] in ['ZZ']:
                res.allowedAA = [nativeAA]
                res.designAction = "NATRO"
                res.currentAction = "NATRO"

            elif res.allowedAA[0] in DesignDomainWildCards('aminoAcids'):
                res.allowedAA = aminoAcids.selection(res.allowedAA[0])
                res.designAction = "PIKAA"
                res.currentAction = "PIKAA"

            else:
                res.designAction = "PIKAA"
                res.currentAction = "PIKAA"

    def setNativeAAtoPdbAA(self):
        """ Set the native AA based on given pose"""
        for res in self.design:
            res.nativeAA = self.pose.residue(res.poseIndex).name1()

    def setCurrentAAtoPdbAA(self):
        for res in self.design:
            res.currentAA = self.pose.residue(res.poseIndex).name1()

    def mutateNonCatalyticToNativeAA(self):
        #self.activeResidues = list()
        for index, res in enumerate(self.design):
            if res.catalyticName:
                continue
            res.currentAA = res.nativeAA
            res.currentAction = res.designAction
            res.active = True
            self.activeResidues.append(index)

    def mutateNonCatalyticToAllowedAA(self, poseIndices=()):
        #self.activeResidues = list()
        designIndex = list()
        for poseIndex in poseIndices:
            resID, resChain = self.pose.pdb_info().pose2pdb(poseIndex).split()
            designIndex.append(self.designDict[(int(resID), resChain)])

        #print(MPI.COMM_WORLD.Get_rank(), 'previous: ', self.currentNoneCatalyticMutants, 'current: ', designIndex)

        # first stage the previous mutated residue(s) for mutation again. Gives them a chance to recover if possible.
        for index in self.currentNoneCatalyticMutants:
            res = self.design[index]
            # Skip the previous ones that are now assigned to catalytic
            if res.catalyticName:
                continue
            res.currentAA = ''.join(res.allowedAA)
            res.currentAction = res.designAction
            res.active = True
            self.activeResidues.append(index)

        # reset the current mutants list and stage the given residues for mutation
        self.currentNoneCatalyticMutants = list()
        for index, res in enumerate(self.design):
            if res.catalyticName:
                continue
            if index not in designIndex:
                continue

            self.currentNoneCatalyticMutants.append(index)
            res.currentAA = ''.join(res.allowedAA)
            res.currentAction = res.designAction
            res.active = True
            self.activeResidues.append(index)
        #string = ''
        #string += '{}  after non cat: {}\n'.format(MPI.COMM_WORLD.Get_rank(), self.activeResidues)
        #print(string)

    def mutateNonCatalyticToAla(self):
        """
        Set the currentAA of Non-catalytic residues to AA
        """
        #self.activeResidues = list()
        for index, res in enumerate(self.design):
            if res.catalyticName:
                continue
            res.currentAA = ['A']
            res.currentAction = 'PIKAA'
            res.designAction = 'PIKAA'
            res.active = True
            self.activeResidues.append(index)

    def mutateNonCatalyticAll(self, keepNativeAA=True, setNonCatalyticTo='One'):
        """
        Initiate the currentAA of nonCatalytic resides randomly from allowedAA.
        If keepNativeAA is True, the pdb res type is chosen if it is among allowed res.
        :param keepNativeAA:
        """
        if setNonCatalyticTo not in ['One', 'All']:
            raise ValueError('Bad setNonCatalyticTo parameter, expected one of [One, All] received {}'.format(setNonCatalyticTo))

        # reset the list of current mutants at the beginning of initiation
        self.currentNoneCatalyticMutants = list()
        for index, res in enumerate(self.design):
            # Ignore catalytic residues.
            # Note: In the design initiation all residue are non catalytic.
            if res.catalyticName:
                continue

            res.active = True
            self.activeResidues.append(index)

            # Try to set the currentAA to keepNativeAA if it is allowed.
            if keepNativeAA and res.nativeAA in res.allowedAA:
                res.currentAA = list(res.nativeAA[0])
                res.currentAction = res.designAction

            # if not keepNativeAA or not found in allowedAA mutate
            else:
                self.currentNoneCatalyticMutants.append(index)
                if setNonCatalyticTo == 'One':
                    nallowedAA = len(res.allowedAA)
                    indexAA = randint(0, nallowedAA, 1)[0]
                    res.currentAA = list(res.allowedAA[indexAA])
                    res.currentAction = res.designAction

                elif setNonCatalyticTo == 'All':
                    res.currentAA = list(res.allowedAA)
                    res.currentAction = res.designAction

    def mutateCatalyticAll(self):

        # If Ambiguous catalytic residues present, randomly assign them
        if self.catalyticDesign:

            # Reset the name of the catalytic residues if exist
            for index in self.catalytic:
                self.design[index].catalyticName = None

            # Clear the previous residues if exist
            self.catalytic = list()
            self.catalyticDict = dict()

            # Pick up random design indices to assign initial choice of the catalytic residues
            indices = list()
            for i in range(self.nCatalyticResidues - self.nNoMutateCatalyticResidues):
                for j in range(10000):
                    index = randint(0, self.nDesignResidues, 1)[0]
                    if index not in self.noMutate.values() and index not in indices:
                        indices.append(index)
                        break
                if j == 9999:
                    raise ValueError('Failed in randomizeAll. Could not randomize catalytic residue.')

            for catalyticResName, catalyticResAA in self.catalyticTargetAA.items():
                # If none mutable the residue is already known
                if catalyticResName in self.noMutate.keys():
                    index = self.noMutate[catalyticResName]
                    # Otherwise, assign a random residue
                else:
                    index = indices.pop()

                # Select a the index of a random target AA from catalyticResAA
                indexAA = randint(0, len(catalyticResAA), 1)[0]

                # Set up the catalytic res info
                self.catalytic.append(index)
                self.design[index].catalyticName = catalyticResName
                self.design[index].currentAA = list(catalyticResAA[indexAA])
                self.design[index].currentAction = self.catalyticAction[catalyticResName]
                self.design[index].active = True
                self.activeResidues.append(index)

            # Initiate a dictionary which its values point to the catalytic res for fast access
            for index in self.catalytic:
                # Get the catalytic residues Names
                name = self.design[index].catalyticName
                self.catalyticDict[name] = index

    def mutateNonCatalyticOne(self, setNonCatalyticTo='All', poseIndex=None):
        """
        Randomly select a non-catalytic residue and sets its currentAA to one of its allowedAA by random
        """

        if setNonCatalyticTo not in ['One', 'All']:
            raise ValueError('Bad setNonCatalyticTo parameter, expected one of [One, All] received {}'.
                           format(setNonCatalyticTo))

        # clear the previous active residues
        #for index in self.activeResidues:
        #    self.design[index].active = False
        #self.activeResidues = list()

        # Choose a random residue in the design domain
        if poseIndex is None:
            for i in range(1000):
                #nDesignRes = len(self.design)
                indexDesign = randint(0, self.nDesignResidues, 1)[0]

                # Skip the catalytic residues, only applies to DesignNoneCatalytic
                if indexDesign in self.catalytic:
                    continue
                else:
                    break

        # otherwise get the indexDesign from poseIndex
        else:
            resID, resChain = self.pose.pdb_info().pose2pdb(poseIndex).split()
            indexDesign = self.designDict[(int(resID), resChain)]


        res = self.design[indexDesign]
        nallowedAA = len(res.allowedAA)
        self.activeResidues.append(indexDesign)


        # Choose a random AA and set the current AA
        if nallowedAA > 1:
            # print('BBB nonCatalytic: ', self.design[indexDesign].name)
            if setNonCatalyticTo == 'One':
                indexAA = randint(0, nallowedAA, 1)[0]
                res.currentAA = res.allowedAA[indexAA]
                res.currentAction = res.designAction
                re.active = True

            elif setNonCatalyticTo == 'All':
                res.currentAA = list(res.allowedAA)
                res.currentAction = res.designAction
                re.active = True
        # Just set it to be active
        else:
            res.currentAction = res.designAction
            re.active = True

        if Constants.DEBUG:
            printDEBUG(msg=self.state(), rank=MPI.COMM_WORLD.Get_rank())

    # TODO frozen non-mutatable residues are treated as PIKAA. No correct. The non-mute is not picked if it has only one targetAA
    # TODO All is not active
    def mutateCatalyticOne(self, replacePreviousCatalyticWith='A', setPreviousCatalyticNativeAAtoAllowedAA=True):
        """
        Randomly select a non-catalytic residue and sets it switch it with one the catalytic residues
        by random.
        :param replacePreviousCatalyticWith if set to 'A' the currentAA of old catalytic residue is set to Ala
                                            if set to 'One', the currentAA of old catalytic residue is set to randomly
                                                chosen aa from allowedAA
                                            if set to 'All', the currentAA of old catalytic residue is set to allowedAA
        """

        # return if nothing to do
        if not self.catalyticDesign:
            return
        #elif self.nNoMutateCatalyticResidues == self.nCatalyticResidues:
        #    return

        # clear the previous active residues
        #for index in self.activeResidues:
        #    self.design[index].active = False
        #self.activeResidues = list()

        if replacePreviousCatalyticWith not in ['A', 'One', 'All']:
            raise ValueError('Bad replacePreviousCatalyticWith parameter, expected one of [A, One, All] received {}'.
                           format(replacePreviousCatalyticWith))

        #string = ''
        #string += '{}  before cat: {}\n'.format(MPI.COMM_WORLD.Get_rank(), [(i, res.currentAA) for i, res in enumerate(self.design) if res.catalyticName])
        # Pick a random index in the catalytic list
        # and get its index in the design list
        old_index_Found = False
        for i in range(1000):
            oldCatalyticIndex = randint(0, self.nCatalyticResidues, 1)[0]
            oldDesignIndex = self.catalytic[oldCatalyticIndex]

            # Accept the move if it is not part of fixed position
            if oldDesignIndex not in self.noMutate.values():
                catalyticName = self.design[oldDesignIndex].catalyticName
                targetAA = self.catalyticTargetAA[catalyticName]
                old_index_Found = True
                break

            # If it is fixed position make sure has multiple target values
            elif oldDesignIndex in self.noMutate.values():
                catalyticName = self.design[oldDesignIndex].catalyticName
                targetAA = self.catalyticTargetAA[catalyticName]
                if len(targetAA) > 1:
                    old_index_Found = True
                    break

        if not old_index_Found:
            raise ValueError(' failed in randomizeOne. Could not randomize catalytic residue.')

        # Choose a random targetAA in case multiple target AA is given for catalytic residues
        #indexAA = randint(0, len(targetAA), 1)[0]
        #currentAA = targetAA[indexAA]

        # Only mutate the amino acid type if a fixed position is chosen
        if oldDesignIndex in self.noMutate.values():
            self.design[oldDesignIndex].currentAA = targetAA
            self.design[oldDesignIndex].currentAction = 'PIKAA'
            self.design[oldDesignIndex].designAction = 'PIKAA'
            self.design[oldDesignIndex].active = True
            self.activeResidues .append(oldDesignIndex)

        else:
            new_index_Found = False
            for i in range(10000):
                newDesignIndex = randint(0, self.nDesignResidues, 1)[0]
                if newDesignIndex not in self.catalytic:
                    new_index_Found = True
                    break

            if not (new_index_Found and old_index_Found):
                raise ValueError(' failed in randomizeOne. Could not randomize catalytic residue.')

            #string += '{} mutate {}->{}\n'.format(MPI.COMM_WORLD.Get_rank(), oldDesignIndex, newDesignIndex)

            # print('BBB Catalytic', catalyticName, ' from ', self.design[oldDesignIndex].name, ' to ', self.design[newDesignIndex].name)
            # Update the catalytic res names
            self.design[newDesignIndex].catalyticName = catalyticName
            self.design[oldDesignIndex].catalyticName = None

            # Update currentAA and currentAction
            self.design[newDesignIndex].currentAction = self.catalyticAction[catalyticName]
            self.design[newDesignIndex].currentAA = ''.join(targetAA)

            # Set the new active residue list
            self.design[newDesignIndex].active = True
            self.design[oldDesignIndex].active = True
            self.activeResidues.append(oldDesignIndex)
            self.activeResidues.append(newDesignIndex)

            if replacePreviousCatalyticWith == 'A':
                self.design[oldDesignIndex].currentAA = ['A']
                self.design[oldDesignIndex].currentAction = 'PIKAA'
                self.design[oldDesignIndex].designAction = 'PIKAA'


            elif replacePreviousCatalyticWith == 'One':
                nAllowedAA = len(self.design[oldDesignIndex].allowedAA)
                indexAA = randint(0, nAllowedAA, 1)[0]
                self.design[oldDesignIndex].currentAA = list(self.design[oldDesignIndex].allowedAA[indexAA])
                self.design[oldDesignIndex].currentAction = 'PIKAA'
                self.design[oldDesignIndex].designAction = 'PIKAA'

            elif replacePreviousCatalyticWith == 'All':
                self.design[oldDesignIndex].currentAA = list(self.design[oldDesignIndex].allowedAA)
                self.design[oldDesignIndex].currentAction = 'PIKAA'
                self.design[oldDesignIndex].designAction = 'PIKAA'

            # This used during recovering the native residues,
            if setPreviousCatalyticNativeAAtoAllowedAA:
                self.design[oldDesignIndex].nativeAA = list(self.design[oldDesignIndex].allowedAA)


            # Update the list of catalytic residues with the index of the new residue
            self.catalytic[oldCatalyticIndex] = newDesignIndex

            # Update the dictionary
            self.catalyticDict[catalyticName] = newDesignIndex
        #string += '{}  before cat: {}\n'.format(MPI.COMM_WORLD.Get_rank(), [(i, res.currentAA) for i, res in enumerate(self.design) if res.catalyticName])
        #print(string)

    def clearActive(self):
        for index in self.activeResidues:
            self.design[index].active = False
        self.activeResidues = list()

    def getCurrentAASequence(self):
        seq = list()
        for res in self.design:
            seq.extend(res.currentAA)
        return ''.join(seq)

    def getPdbSequence(self):
        seq = list()
        for res in self.design:
            seq.extend(self.pose.residue(res.poseIndex).name1())
        return ''.join(seq)

    def getSeuenceDiff(self):
        diff = list()
        seqCurrent = list()
        for res in self.design:
            seqCurrent.extend(self.pose.residue(res.poseIndex).name1())

        seqCurrent = ''.join(seqCurrent)
        #print(self.originalSequence, seqCurrent)
        diff = [i for i in range(len(self.originalSequence)) if self.originalSequence[i] != seqCurrent[i]]
        return diff

    def getCatalyticPoseIndex(self):
        catalyticPoseIndex = list()
        for index in self.catalytic:
            catalyticPoseIndex.append(self.design[index].poseIndex)
        return catalyticPoseIndex

    def getDesignPoseIndex(self):
        designPoseIndex = list()
        for res in self.design:
            designPoseIndex.append(res.poseIndex)
        return designPoseIndex

    def isPdbCatalyticSynced(self):
        for cataltyName, catalyticDesignIndex in self.catalyticDict.items():
            res = self.design[catalyticDesignIndex]
            if self.pose.residue(res.poseIndex).name1() not in self.catalyticTargetAA[cataltyName]:
                return False
        return True

    def isPdbNonCatalyticSynced(self):
        for res in self.design:
            # Ignore catalytic
            if res.catalyticName:
                continue
            if self.pose.residue(res.poseIndex).name1() not in res.currentAA:
                return False
        return True

    def isPdbSynced(self):
        for res in self.design:
            if self.pose.residue(res.poseIndex).name1() not in res.currentAA:
                return False
        return True

    def updateConstraints(self):

        # Clear the previous constraints
        cst_remover = ClearConstraintsMover()
        cst_remover.apply(self.pose)

        # Over the constrains
        for cst in self.constraints:
            if re.match('B', cst.type, re.IGNORECASE):
                self._getBondConstraints(cst)

            elif re.match('S', cst.type, re.IGNORECASE):
                self._getSequenceConstraints(cst)

            else:
                raise ValueError("{} constraint is not implemented yet. Coming soon".format(cst.type))

    def _getBondConstraints(self, cst):

        # Create ambiguous constraints for each constraints group
        ambiguous_constraint = AmbiguousConstraint()

        # Get the residue index "i"
        if cst.res_i in self.catalyticDict.keys():
            designIndex_res_i = self.catalyticDict[cst.res_i]
        elif cst.res_i in self.designDict.keys():
            designIndex_res_i = self.designDict[cst.res_i]
        else:  # To allow for constraints with residues outside domains
            designIndex_res_i = None
            # raise ValueError("Could not find res_i {} in constraint {}.".format(cst.res_i, cst.tag))

            # Get the residue index "j"
        if cst.res_j in self.catalyticDict.keys():
            designIndex_res_j = self.catalyticDict[cst.res_j]

        elif cst.res_j in self.designDict.keys():
            designIndex_res_j = self.designDict[cst.res_j]

        else:  # To allow for constraints with residues outside domains
            designIndex_res_j = None
            # raise ValueError("Could not find res_j {} in constraint {}.".format(cst.res_j, cst.tag))

        if designIndex_res_i is not None:
            i_resID = self.design[designIndex_res_i].poseIndex
        else:
            i_resID = self.pose.pdb_info().pdb2pose(cst.res_i[1], cst.res_i[0])

        if designIndex_res_j is not None:
            j_resID = self.design[designIndex_res_j].poseIndex
        else:
            j_resID = self.pose.pdb_info().pdb2pose(cst.res_j[1], cst.res_j[0])

        # Over all possible atom combinations
        for atomName_i in cst.atom_i_list:
            for atomName_j in cst.atom_j_list:

                try:
                    # Try to add cst only if atoms are available
                    if self.pose.residue(i_resID).has(atomName_i) and self.pose.residue(j_resID).has(atomName_j):
                        i_atomID = self.pose.residue(i_resID).atom_index(atomName_i)
                        j_atomID = self.pose.residue(j_resID).atom_index(atomName_j)

                        # Set the new constraints
                        i = pr.AtomID(i_atomID, i_resID)
                        j = pr.AtomID(j_atomID, j_resID)
                        func = BoundFunc(cst.lb, cst.hb, cst.sd, cst.tag)
                        distance_constraint = AtomPairConstraint(i, j, func)

                        # Add individual constraint to the ambiguous_constraint
                        ambiguous_constraint.add_individual_constraint(distance_constraint)
                except Exception as e:
                    pass

        self.pose.add_constraint(ambiguous_constraint)

    def _getSequenceConstraints(self, cst):

        for residue, targetAA in cst.res.items():
            targetAA = one_to_three(targetAA)
            designIndex = self.designDict[residue]
            poseIndex = self.design[designIndex].poseIndex
            seqCst = ResidueTypeConstraint(poseIndex, str(poseIndex), targetAA, cst.weight * 1)
            self.pose.add_constraint(seqCst)

    def writeResfile(self, resfileName='.resfile', activeResidues=False):
        resfileString = ''
        action = ''
        with open(resfileName, 'w') as resfile:
            resfile.write('NATRO\n')
            resfile.write('START\n')
            for res in self.design:
                # if active residues are True write only active residues
                if activeResidues and not res.active:
                    continue

                action = res.currentAction
                # Write to the file
                if action == 'NATRO':
                    resfile.write('{}  {}  {}    \n'.format(res.ID, res.chain, action))
                else:
                    resfile.write('{}  {}  {}  {}\n'.format(res.ID, res.chain, action, ''.join(res.currentAA)))

                if Constants.DEBUG:
                    if action == 'NATRO':
                        resfileString += '{}  {}  {}    \n'.format(res.ID, res.chain, action)
                    else:
                        resfileString += '{}  {}  {}  {}\n'.format(res.ID, res.chain, action, ''.join(res.currentAA))
        if Constants.DEBUG:
            printDEBUG(msg=resfileString, rank=MPI.COMM_WORLD.Get_rank())

    def state(self):
        state = ''
        # state += 'Type: {}\n'.format(self.type)
        state += 'Catalytic indices: {}\n'.format(self.catalytic)
        state += 'Catalytic Dict: {}\n'.format(self.catalyticDict)
        state += 'Catalytic noMutate: {}\n'.format(self.noMutate)
        state += 'Catalytic Target AA: {}\n'.format(self.catalyticTargetAA)
        state += 'Catalytic Actions AA: {}\n'.format(self.catalyticAction)
        state += '------------------------------------------------------------------------------------------------\n'
        state += 'Design residues: \n'
        for res in self.design:
            state += 'ID: {}, chain: {}, designAction: {}, currentAction: {}, CatalyticName: {}, pose Index: {} \n'. \
                format(res.ID, res.chain, res.designAction, res.currentAction, res.catalyticName, res.poseIndex)
            state += 'Current residues: {}\n'.format(''.join(res.currentAA))
            state += 'Native  residues: {}\n'.format(''.join(res.nativeAA))
            state += 'Allowed residues: {}\n'.format(''.join(res.allowedAA))
        state += '------------------------------------------------------------------------------------------------\n'
        print('Constraints:')
        for index, cst in enumerate(self.constraints):
            print('{}, {}'.format(index, cst.show()))
        state += '------------------------------------------------------------------------------------------------\n'
        return state

    def statePretty(self):

        print("Design domain with catalytic assignments: ")
        for index, res in enumerate(self.design):
            resname = (res.ID, res.chain)
            print('position: {}'.format(index))
            print('     residue: {}'.format(':'.join(map(str, resname))))
            print('     Pose Index: {}'.format(res.poseIndex))
            if res.catalyticName:
                print('     catalytic assignment: {}'.format(res.catalyticName))
                if res.catalyticName in self.noMutate:
                    print('     mutate: False')
                else:
                    print('     mutate: True')

                if self.catalyticAction[res.catalyticName] == 'NATRO':
                    print('     frozen: True')
                else:
                    print('     frozen: False')

                print('     allowedAA: {}'.format(''.join(self.catalyticTargetAA[res.catalyticName])))
                print('     currentAA: {}'.format(''.join(res.currentAA)))
                print('     PDB AA: {}'.format(self.pose.residue(res.poseIndex).name1()))

            else:
                print('     catalytic assignment: None')
                print('     mutate: True')
                if res.designAction == 'NATRO':
                    print('     frozen: True')
                else:
                    print('     frozen: False')
                print('     allowedAA: {}'.format(''.join(res.allowedAA)))
                print('     currentAA: {}'.format(''.join(res.currentAA)))

        print('Constraints:')
        for index, cst in enumerate(self.constraints):
            print('{}, {}'.format(index, cst.show()))

    def stateCatalytic(self):
        state = ''
        state += 'Catalytic indices: {}\n'.format(self.catalytic)
        state += 'Catalytic Dict: {}\n'.format(self.catalyticDict)
        state += 'Catalytic noMutate: {}\n'.format(self.noMutate)
        state += 'Catalytic Target AA: {}\n'.format(self.catalyticTargetAA)
        state += 'Catalytic Actions AA: {}\n'.format(self.catalyticAction)
        for designIndex, res in enumerate(self.design):
            if not res.catalyticName:
                continue
            state += 'ID: {}, chain: {}, designAction: {}, currentAction: {}, CatalyticName: {}, pose Index: {} , designIndex {}\n'. \
                format(res.ID, res.chain, res.designAction, res.currentAction, res.catalyticName, res.poseIndex, designIndex)
            state += 'Current residues: {}\n'.format(''.join(res.currentAA))
            state += 'Native  residues: {}\n'.format(''.join(res.nativeAA))
            state += 'Allowed residues: {}\n'.format(''.join(res.allowedAA))
            state += 'Current PDB AA: {}\n'.format(self.pose.residue(res.poseIndex).name1())
        state += '------------------------------------------------------------------------------------------------\n'
        return state


class ActiveSiteMover(object):
    """Active Site Mover implement medium level algorithms for perturbing the active site (catalytic, non-catalytic) residues"""

    def __init__(self, activeSiteDesignMode='MC', mimimizeBackbone=False, activeSiteLoops=1, nNoneCatalytic=5, scratch=''):

        # Initiate Rosetta classes
        self.taskFactory = TaskFactory()
        self.scorefxn = getScoreFunction(mode='fullAtomWithConstraints')
        self.scorefxn.set_weight(score_type_from_name('fa_dun'), 0.1)

        self.packer = PackRotamersMover()
        self.packer.score_function(self.scorefxn)

        self.fastRelax = FastRelax(self.scorefxn)
        self.activeSiteMinMover = MinMover()
        self.activeSiteMinMoveMap = MoveMap()

        self.cst_remover = ClearConstraintsMover()

        self.proccessName = '{}-{}.resfile'.format(MPI.COMM_WORLD.Get_rank(), os.getpid())
        self.resfileName = os.path.join(scratch, self.proccessName)

        # Temporary design pose
        self.dPoseTemp = DesignPose()
        self.dPoseOrig = DesignPose()

        self.activeSiteLoops = activeSiteLoops
        self.nNoneCatalytic = nNoneCatalytic

        # Design mode
        self.acticeSiteDesignMode = activeSiteDesignMode  # ['MIN', 'MC', 'None']
        self.mimimizeBackbone = mimimizeBackbone # [True, False]

    '''
    def moveCatalyticPositionsWithNonCatalyticAsAla(self, dPose: DesignPose, countInitial=False):
        """
        Mutates position of one catalytic residue while converting the non catalytic res to Ala.
        The non catalytic residues are then recovered the to initial pdb AA type in the end.
        :param dPose:
        :return bool: True if move successful, False otherwise
        """

        foundMutant = False

        if not countInitial:
            energyPrevious = float('inf')
            energyCurrent = energyPrevious
        else:
            energyPrevious = self.scorefxn(dPose.pose)
            energyCurrent = energyPrevious

        # make a copy of the original one
        self.dPoseOrig.assign(dPose)

        for i in range(self.catalyticLoops):
            self.dPoseTemp.assign(self.dPoseOrig)

            # Clear the old cst
            #self.cst_remover.apply(self.dPoseTemp.pose)


            # Set non-catalytics to Ala, this automatically set currentAction and designAction to PIKAA.
            # Once a residue is mutated to Ala, the rotamer information is lost
            self.dPoseTemp.mutateNonCatalyticToAla()

            # Mutate the catalytic
            self.dPoseTemp.writeResfile(self.resfileName, activeResidues=True)
            resfileCatalytic = ReadResfile(self.resfileName)
            self.taskFactory.clear()
            self.taskFactory.push_back(resfileCatalytic)
            self.packer.task_factory(self.taskFactory)

            # Make a quick packing to mutate the corresponding residues
            self.packer.nloop(1)
            self.packer.apply(self.dPoseTemp.pose)

            # randomize one catalytic residue, while setting the currentAA of the previous catalytic residue to Ala.
            # The nativeAA of the previous residue should be set to allowedAA because the AA identity of this residue
            # is lost. As such to recover the AA of the previous catalytic (now non-catalytic) residue, it should be
            # chosen from allowedAA
            self.dPoseTemp.mutateCatalyticOne(replacePreviousCatalyticWith='A', setPreviousCatalyticNativeAAtoAllowedAA=True)
            # Mutate the catalytic
            self.dPoseTemp.writeResfile(self.resfileName, activeResidues=True)
            resfileCatalytic = ReadResfile(self.resfileName)
            self.taskFactory.clear()
            self.taskFactory.push_back(resfileCatalytic)
            self.packer.task_factory(self.taskFactory)

            # Make a quick packing to mutate the corresponding residues
            self.packer.nloop(1)
            self.packer.apply(self.dPoseTemp.pose)

            self.dPoseTemp.updateConstraints()

            # Do a full packing
            self.packer.nloop(self.packerLoops)
            self.packer.apply(self.dPoseTemp.pose)
            self.dPoseTemp.clearActive()

            # if mutations are not accepted, skip this round and continue to a new round
            if not self.dPoseTemp.isPdbCatalyticSynced() or not self.dPoseTemp.isPdbNonCatalyticSynced():
                continue

            # Recover the original AA identity. For this, the currentAA ar set to native.
            # In the case of the old catalytic res, currentAA include allowedAA
            self.dPoseTemp.mutateNonCatalyticToNativeAA()

            # Mutate None-catalytic
            self.dPoseTemp.writeResfile(self.resfileName, activeResidues=True)
            resfileNoneCatalytic = ReadResfile(self.resfileName)
            self.taskFactory.clear()
            self.taskFactory.push_back(resfileNoneCatalytic)
            self.packer.task_factory(self.taskFactory)

            # Make a quick packing to mutate the corresponding residues
            self.packer.nloop(1)
            self.packer.apply(self.dPoseTemp.pose)
            self.dPoseTemp.clearActive()

            self.dPoseTemp.updateConstraints()

            # Do a full packing
            self.activeSiteMinMoveMap.clear()
            for res in self.dPoseTemp.design:
                self.activeSiteMinMoveMap.set_chi(res.poseIndex)

            self.dPoseTemp.writeResfile(self.resfileName, activeResidues=False)
            resfileCatalytic = ReadResfile(self.resfileName)
            self.taskFactory.clear()
            self.taskFactory.push_back(resfileCatalytic)

            self.fastRelax.set_task_factory(self.taskFactory)
            self.fastRelax.set_movemap(self.activeSiteMinMoveMap)
            self.fastRelax.set_scorefxn(self.scorefxn)
            self.fastRelax.apply(self.dPoseTemp.pose)

            # if mutations are not accepted, skip this round and continue to a new round
            if not self.dPoseTemp.isPdbCatalyticSynced() or not self.dPoseTemp.isPdbNonCatalyticSynced():
                continue

            # Set the currentAA and nativeAA of all residues to current pdbAA to reflect
            # the current state of the system
            self.dPoseTemp.setNativeAAtoPdbAA()
            self.dPoseTemp.setCurrentAAtoPdbAA()


            # Check the energy, if lower assign to dPose
            energyCurrent = self.scorefxn(self.dPoseTemp.pose)
            if energyCurrent < energyPrevious:
                foundMutant = True
                energyPrevious = energyCurrent
                dPose.assign(self.dPoseTemp)

        return foundMutant
    '''

    '''
    def moveCatalyticPositionsDirect(self, dPose: DesignPose, countInitial=False):
        """
        # mutate one catalytic res while converting the the AA of previous catalytic (now non-catalytic)
        is chosen at random from allowed AA.

        :param dPose:
        :return bool: True if move successful, False otherwise
        """

        foundMutant = False

        if not countInitial:
            energyPrevious = float('inf')
            energyCurrent = energyPrevious
        else:
            energyPrevious = self.scorefxn(dPose.pose)
            energyCurrent = energyPrevious

        # make a copy of the original one
        self.dPoseOrig.assign(dPose)

        for i in range(self.catalyticLoops):

            # Make a copy
            self.dPoseTemp.assign(self.dPoseOrig)

            # Clear the old cst
            #self.cst_remover.apply(self.dPoseTemp.pose)

            # randomize one catalytic residue, while setting the currentAA of the previous catalytic residue to
            # allowedAA.
            self.dPoseTemp.mutateCatalyticOne(replacePreviousCatalyticWith='All', setPreviousCatalyticNativeAAtoAllowedAA=True)


            # Mutate the catalytic
            self.dPoseTemp.writeResfile(self.resfileName, activeResidues=True)
            resfileCatalytic = ReadResfile(self.resfileName)
            self.taskFactory.clear()
            self.taskFactory.push_back(resfileCatalytic)
            self.packer.task_factory(self.taskFactory)

            # Make a quick packing to mutate the corresponding residues
            self.packer.nloop(1)
            self.packer.apply(self.dPoseTemp.pose)
            self.dPoseTemp.clearActive()

            self.dPoseTemp.updateConstraints()

            # Do a full packing
            self.dPoseTemp.writeResfile(self.resfileName, activeResidues=False)
            resfileCatalytic = ReadResfile(self.resfileName)
            self.taskFactory.clear()
            self.taskFactory.push_back(resfileCatalytic)

            self.fastRelax.set_task_factory(self.taskFactory)
            self.fastRelax.set_movemap(self.activeSiteMinMoveMap)
            self.fastRelax.set_scorefxn(self.scorefxn)
            self.fastRelax.apply(self.dPoseTemp.pose)


            # if mutations are not possible, skip this round and continue to a new round
            if not self.dPoseTemp.isPdbCatalyticSynced() or not self.dPoseTemp.isPdbNonCatalyticSynced():
                continue

            # Set the currentAA and nativeAA of all residues to current pdbAA to reflect
            # the current state of the system
            self.dPoseTemp.setNativeAAtoPdbAA()
            self.dPoseTemp.setCurrentAAtoPdbAA()

            # Check the energy, if lower assign to dPose
            #if MPI.COMM_WORLD.Get_rank() == 1:
            #    print('BBB', MPI.COMM_WORLD.Get_rank(), 'attempts', self.dPoseTemp.getCatalyticPoseIndex(), 'orig   ', self.dPoseOrig.getCatalyticPoseIndex())

            energyCurrent = self.scorefxn(self.dPoseTemp.pose)
            if energyCurrent < energyPrevious:
                foundMutant = True
                energyPrevious = energyCurrent
                dPose.assign(self.dPoseTemp)

        return foundMutant
    '''

    '''
    def moveNoneCatalyticRestype(self, dPose: DesignPose, countInitial=False):
        """
        Mutate one non-catalytic residue to one of AA chosen randomly from allowedAA

        :param dPose:
        :return bool: True if move successful, False otherwise
        """
        foundMutant = False

        if not countInitial:
            energyPrevious = float('inf')
            energyCurrent = energyPrevious
        else:
            energyPrevious = self.scorefxn(dPose.pose)
            energyCurrent = energyPrevious

        # make a copy of the original one
        self.dPoseOrig.assign(dPose)

        for i in range(self.noneCatalyticLoops):
            self.dPoseTemp.assign(self.dPoseOrig)
            #print('BBBB', self.dPoseTemp.nDesignResidues)
            # Clear the old cst
            #self.cst_remover.apply(self.dPoseTemp.pose)

            # randomize one non-catalytic residue. This could be done by choosing one AA from allowedAA
            # or all AA from allowedAA
            self.dPoseTemp.mutateNonCatalyticOne(setNonCatalyticTo=self.nonCatalyticDesignMode)

            # Mutate the catalytic
            self.dPoseTemp.writeResfile(self.resfileName, activeResidues=True)
            resfileCatalytic = ReadResfile(self.resfileName)
            self.taskFactory.clear()
            self.taskFactory.push_back(resfileCatalytic)
            self.packer.task_factory(self.taskFactory)

            # Make a quick packing to mutate the corresponding residues
            self.packer.nloop(1)
            self.packer.apply(self.dPoseTemp.pose)
            self.dPoseTemp.clearActive()

            self.dPoseTemp.updateConstraints()

            # Do a full packing
            self.dPoseTemp.writeResfile(self.resfileName, activeResidues=False)
            resfileCatalytic = ReadResfile(self.resfileName)
            self.taskFactory.clear()
            self.taskFactory.push_back(resfileCatalytic)

            self.fastRelax.set_task_factory(self.taskFactory)
            self.fastRelax.set_movemap(self.activeSiteMinMoveMap)
            self.fastRelax.set_scorefxn(self.scorefxn)
            self.fastRelax.apply(self.dPoseTemp.pose)

            # if mutations are not accepted, skip this round and continue to a new round
            if not self.dPoseTemp.isPdbCatalyticSynced() or not self.dPoseTemp.isPdbNonCatalyticSynced():
                continue

            # Set the currentAA and nativeAA of all residues to current pdbAA to reflect
            # the current state of the system
            self.dPoseTemp.setNativeAAtoPdbAA()
            self.dPoseTemp.setCurrentAAtoPdbAA()

            # Check the energy, if lower assign to dPose
            energyCurrent = self.scorefxn(self.dPoseTemp.pose)
            if energyCurrent < energyPrevious:
                foundMutant = True
                energyPrevious = energyCurrent
                dPose.assign(self.dPoseTemp)

        return foundMutant
    '''

    '''
    def minimizeCatalyticPositionsWithNonCatalyticAsAla(self, dPose: DesignPose, countInitial=False):
        """
        Minimizes non-catalytic residue by mutating one res at a time.

        :param dPose:
        :return bool: True if move successful, False otherwise
        """

        foundMutant = False

        # TODO TEST this
        if not countInitial:
            energyPrevious = float('inf')
            energyCurrent = energyPrevious
        else:
            energyPrevious = self.scorefxn(dPose.pose)
            energyCurrent = energyPrevious

        # make a copy of the original one
        self.dPoseOrig.assign(dPose)

        for i in range(self.catalyticLoops):
            self.dPoseTemp.assign(dPose)

            # Clear the old cst
            #self.cst_remover.apply(self.dPoseTemp.pose)

            # Set non-catalytics to Ala, this automatically set currentAction and designAction to PIKAA.
            # Once a residue is mutated to Ala, the rotamer information is lost
            self.dPoseTemp.mutateNonCatalyticToAla()

            # Mutate to Ala
            self.dPoseTemp.writeResfile(self.resfileName, activeResidues=True)
            resfileCatalytic = ReadResfile(self.resfileName)
            self.taskFactory.clear()
            self.taskFactory.push_back(resfileCatalytic)
            self.packer.task_factory(self.taskFactory)

            # Make a quick packing to mutate the corresponding residues
            self.packer.nloop(1)
            self.packer.apply(self.dPoseTemp.pose)
            self.dPoseTemp.clearActive()
            # randomize one catalytic residue, while setting the currentAA of the previous catalytic residue to Ala.
            # The nativeAA of the previous residue should be set to allowedAA because the AA identity of this residue
            # is lost. As such to recover the AA of the previous catalytic (now non-catalytic) residue, it should be
            # chosen from allowedAA
            self.dPoseTemp.mutateCatalyticOne(replacePreviousCatalyticWith='A', setPreviousCatalyticNativeAAtoAllowedAA=True)

            self.dPoseTemp.writeResfile(self.resfileName, activeResidues=True)
            resfileCatalytic = ReadResfile(self.resfileName)
            self.taskFactory.clear()
            self.taskFactory.push_back(resfileCatalytic)
            self.packer.task_factory(self.taskFactory)

            # Make a quick packing to mutate the corresponding residues
            self.packer.nloop(1)
            self.packer.apply(self.dPoseTemp.pose)
            self.dPoseTemp.clearActive()

            self.dPoseTemp.updateConstraints()

            # Do a full packing
            self.packer.nloop(self.packerLoops)
            self.packer.apply(self.dPoseTemp.pose)

            # if mutations are not accepted, skip this round and continue to a new round
            if not self.dPoseTemp.isPdbCatalyticSynced() or not self.dPoseTemp.isPdbNonCatalyticSynced():
                continue

            # Recover the original AA identity. For this, the currentAA ar set to native.
            # In the case of the old catalytic res, currentAA include allowedAA
            self.dPoseTemp.mutateNonCatalyticToNativeAA()

            # Mutate None-catalytic
            self.dPoseTemp.writeResfile(self.resfileName, activeResidues=True)
            resfileNoneCatalytic = ReadResfile(self.resfileName)
            self.taskFactory.clear()
            self.taskFactory.push_back(resfileNoneCatalytic)
            self.packer.task_factory(self.taskFactory)

            # Make a quick packing to mutate the corresponding residues
            self.packer.nloop(1)
            self.packer.apply(self.dPoseTemp.pose)
            self.dPoseTemp.clearActive()

            self.dPoseTemp.updateConstraints()

            # Do a full packing
            self.dPoseTemp.writeResfile(self.resfileName, activeResidues=False)
            resfileCatalytic = ReadResfile(self.resfileName)
            self.taskFactory.clear()
            self.taskFactory.push_back(resfileCatalytic)

            self.fastRelax.set_task_factory(self.taskFactory)
            self.fastRelax.set_movemap(self.activeSiteMinMoveMap)
            self.fastRelax.set_scorefxn(self.scorefxn)
            self.fastRelax.apply(self.dPoseTemp.pose)

            # if mutations are not accepted, skip this round and continue to a new round
            if not self.dPoseTemp.isPdbCatalyticSynced() or not self.dPoseTemp.isPdbNonCatalyticSynced():
                continue

            # Set the currentAA and nativeAA of all residues to current pdbAA to reflect
            # the current state of the system
            self.dPoseTemp.setNativeAAtoPdbAA()
            self.dPoseTemp.setCurrentAAtoPdbAA()

            # Check the energy, if lower assign to dPose
            energyCurrent = self.scorefxn(self.dPoseTemp.pose)
            if energyCurrent < energyPrevious:
                foundMutant = True
                energyPrevious = energyCurrent
                dPose.assign(self.dPoseTemp)

        return foundMutant
    '''

    '''
    def minimizeCatalyticPositionsDirect(self, dPose: DesignPose, countInitial=False):
        """
        Minimizes catalytic residues by mutating one res at a time.

        :param dPose:
        :return bool: True if move successful, False otherwise
        """
        foundMutant = False

        if not countInitial:
            energyPrevious = float('inf')
            energyCurrent = energyPrevious
        else:
            energyPrevious = self.scorefxn(dPose.pose)
            energyCurrent = energyPrevious

        # make a copy of the original one
        #self.dPoseOrig.assign(dPose)

        for i in range(self.catalyticLoops):
            self.dPoseTemp.assign(dPose)

            # Clear the old cst
            #self.cst_remover.apply(self.dPoseTemp.pose)

            # randomize one catalytic residue, while setting the currentAA of the previous catalytic residue to
            # allowedAA.
            self.dPoseTemp.mutateCatalyticOne(replacePreviousCatalyticWith='All', setPreviousCatalyticNativeAAtoAllowedAA=True)

            # Mutate the catalytic
            self.dPoseTemp.writeResfile(self.resfileName, activeResidues=True)
            resfileCatalytic = ReadResfile(self.resfileName)
            self.taskFactory.clear()
            self.taskFactory.push_back(resfileCatalytic)
            self.packer.task_factory(self.taskFactory)

            # Make a quick packing to mutate the corresponding residues
            self.packer.nloop(1)
            self.packer.apply(self.dPoseTemp.pose)
            self.dPoseTemp.clearActive()

            self.dPoseTemp.updateConstraints()

            # Do a full packing
            self.dPoseTemp.writeResfile(self.resfileName, activeResidues=False)
            resfileCatalytic = ReadResfile(self.resfileName)
            self.taskFactory.clear()
            self.taskFactory.push_back(resfileCatalytic)

            self.fastRelax.set_task_factory(self.taskFactory)
            self.fastRelax.set_movemap(self.activeSiteMinMoveMap)
            self.fastRelax.set_scorefxn(self.scorefxn)
            self.fastRelax.apply(self.dPoseTemp.pose)

            # if mutations are not accepted, skip this round and continue to a new round
            if not self.dPoseTemp.isPdbCatalyticSynced() or not self.dPoseTemp.isPdbNonCatalyticSynced():
                continue

            # Set the currentAA and nativeAA of all residues to current pdbAA to reflect
            # the current state of the system
            self.dPoseTemp.setNativeAAtoPdbAA()
            self.dPoseTemp.setCurrentAAtoPdbAA()

            # Check the energy, if lower assign to dPose
            energyCurrent = self.scorefxn(self.dPoseTemp.pose)
            if energyCurrent < energyPrevious:
                foundMutant = True
                energyPrevious = energyCurrent
                dPose.assign(self.dPoseTemp)

        return foundMutant
    '''

    '''
    def minimizeNoneCatalyticRestype(self, dPose: DesignPose, countInitial=False):
        """
        Minimizes non-catalytic residue by mutating one res at a time.

        :param dPose:
        :return bool: True if move successful, False otherwise
        """
        foundMutant = False

        if not countInitial:
            energyPrevious = float('inf')
            energyCurrent = self.scorefxn(dPose.pose)
        else:
            energyPrevious = self.scorefxn(dPose.pose)
            energyCurrent = energyPrevious

        # get the non catalytic residues energies
        nonCatalyticPoseIndices = dPose.getNonCatalyticPoseIndex()
        nonCatalyticEnergies = [(poseIndex, dPose.pose.energies().residue_total_energy(poseIndex)) for poseIndex in nonCatalyticPoseIndices]
        nonCatalyticEnergies.sort(key=lambda element: element[1], reverse=True)
        poseIndices = [element[0] for element in nonCatalyticEnergies[0:self.noneCatalyticLoops]]
        #nonCatalyticEnergy = cycle(nonCatalyticEnergies)

        # make a copy of the original one
        #self.dPoseOrig.assign(dPose)

        self.dPoseTemp.assign(dPose)
        # Clear the old cst
        #self.cst_remover.apply(self.dPoseTemp.pose)

        # randomize non-catalytic residues. This could be done by choosing one AA from allowedAA
        # or all AA from allowedAA. The randomization is done first on high energy residues
        #poseIndex, energy = next(nonCatalyticEnergy)
        self.dPoseTemp.mutateNonCatalyticToAllowedAA(poseIndex=poseIndices)

        # Mutate non catalytic
        self.dPoseTemp.writeResfile(self.resfileName, activeResidues=True)
        resfileCatalytic = ReadResfile(self.resfileName)
        self.taskFactory.clear()
        self.taskFactory.push_back(resfileCatalytic)
        self.packer.task_factory(self.taskFactory)

        # Make a quick packing to mutate the corresponding residues
        self.packer.nloop(1)
        self.packer.apply(self.dPoseTemp.pose)
        self.dPoseTemp.clearActive()

        self.dPoseTemp.updateConstraints()

        # Do a full packing
        self.dPoseTemp.writeResfile(self.resfileName, activeResidues=False)
        resfileCatalytic = ReadResfile(self.resfileName)
        self.taskFactory.clear()
        self.taskFactory.push_back(resfileCatalytic)

        self.fastRelax.set_task_factory(self.taskFactory)
        self.fastRelax.set_movemap(self.activeSiteMinMoveMap)
        self.fastRelax.set_scorefxn(self.scorefxn)
        self.fastRelax.apply(self.dPoseTemp.pose)

        # if mutations are not accepted, skip this round and continue to a new round
        if not self.dPoseTemp.isPdbCatalyticSynced() or not self.dPoseTemp.isPdbNonCatalyticSynced():
            foundMutant = False
            return foundMutant

        # Set the currentAA and nativeAA of all residues to current pdbAA to reflect
        # the current state of the system
        self.dPoseTemp.setNativeAAtoPdbAA()
        self.dPoseTemp.setCurrentAAtoPdbAA()

        # Check the energy, if lower assign to dPose
        energyCurrent = self.scorefxn(self.dPoseTemp.pose)
        if energyCurrent < energyPrevious:
            foundMutant = True
            energyPrevious = energyCurrent
            dPose.assign(self.dPoseTemp)

        return foundMutant
    '''

    def moveActiveSite(self, dPose: DesignPose, countInitial=False):
        """
        # mutate Active site res according to the input file
        :param dPose:
        :return bool: True if move successful, False otherwise
        """

        foundMutant = False

        if not countInitial:
            energyPrevious = float('inf')
            energyCurrent = energyPrevious
        else:
            energyPrevious = self.scorefxn(dPose.pose)
            energyCurrent = energyPrevious

        sequenceOrig = dPose.getPdbSequence()
        # get the non catalytic residues energies
        nonCatalyticPoseIndices = dPose.getNonCatalyticPoseIndex()
        nonCatalyticEnergies = [(poseIndex, dPose.pose.energies().residue_total_energy(poseIndex)) for poseIndex in nonCatalyticPoseIndices]
        nonCatalyticEnergies.sort(key=lambda element: element[1], reverse=True)
        poseIndices = [element[0] for element in nonCatalyticEnergies[0:self.nNoneCatalytic]]

        # make a copy of the original one
        self.dPoseOrig.assign(dPose)

        for i in range(self.activeSiteLoops):
            # Make a copy
            self.dPoseTemp.assign(self.dPoseOrig)

            # Clear the old cst
            #self.cst_remover.apply(self.dPoseTemp.pose)

            # Set to mutate one catalytic residue, while setting the currentAA of the previous catalytic residue to allowedAA.
            self.dPoseTemp.mutateCatalyticOne(replacePreviousCatalyticWith='All', setPreviousCatalyticNativeAAtoAllowedAA=True)

            # Set to mutate the high energy non catalytic residues
            self.dPoseTemp.mutateNonCatalyticToAllowedAA(poseIndices)

            # Mutate the active site
            self.dPoseTemp.writeResfile(self.resfileName, activeResidues=True)
            resfileCatalytic = ReadResfile(self.resfileName)
            self.taskFactory.clear()
            self.taskFactory.push_back(resfileCatalytic)
            self.packer.task_factory(self.taskFactory)

            # Make a quick packing to mutate the corresponding residues
            self.packer.nloop(1)
            self.packer.apply(self.dPoseTemp.pose)
            self.dPoseTemp.clearActive()

            self.dPoseTemp.updateConstraints()

            # Do a full packing
            self.dPoseTemp.writeResfile(self.resfileName, activeResidues=False)
            resfileCatalytic = ReadResfile(self.resfileName)
            self.taskFactory.clear()
            self.taskFactory.push_back(resfileCatalytic)

            self.activeSiteMinMoveMap.clear()
            for res in self.dPoseTemp.design:
                self.activeSiteMinMoveMap.set_chi(res.poseIndex)

            self.fastRelax.set_task_factory(self.taskFactory)
            self.fastRelax.set_movemap(self.activeSiteMinMoveMap)
            self.fastRelax.set_scorefxn(self.scorefxn)
            self.fastRelax.apply(self.dPoseTemp.pose)

            #print(MPI.COMM_WORLD.Get_rank(), self.dPoseTemp.isPdbCatalyticSynced(), '\n', self.dPoseTemp.stateCatalytic())
            # if mutations are not accepted, skip this round and continue to a new round
            if not self.dPoseTemp.isPdbCatalyticSynced() or not self.dPoseTemp.isPdbNonCatalyticSynced():
                foundMutant = False
                continue

            sequenceCurrent = self.dPoseTemp.getPdbSequence()
            if sequenceCurrent == sequenceOrig:
                foundMutant = False
            else:
                foundMutant = True

            # Set the currentAA and nativeAA of all residues to current pdbAA to reflect
            # the current state of the system
            self.dPoseTemp.setNativeAAtoPdbAA()
            self.dPoseTemp.setCurrentAAtoPdbAA()

            # Minimize the new structure
            self.activeSiteMinMoveMap.clear()
            for res in self.dPoseTemp.design:
                self.activeSiteMinMoveMap.set_chi(res.poseIndex)
                if self.mimimizeBackbone:
                    # print('adding: ', res.name)
                    self.activeSiteMinMoveMap.set_bb(res.poseIndex)

            self.activeSiteMinMover.set_movemap(self.activeSiteMinMoveMap)
            self.activeSiteMinMover.score_function(self.scorefxn)
            self.activeSiteMinMover.apply(self.dPoseTemp.pose)

            # Check the energy, if lower assign to dPose
            energyCurrent = self.scorefxn(self.dPoseTemp.pose)
            if energyCurrent < energyPrevious:
                #foundMutant = True
                energyPrevious = energyCurrent
                dPose.assign(self.dPoseTemp)

            return foundMutant

    def minimizeActiveite(self, dPose: DesignPose, countInitial=False):
        """
        # minimize active site by mutate Active site res according to the input file
        :param dPose:
        :return bool: True if move successful, False otherwise
        """

        foundMutant = False

        if not countInitial:
            energyPrevious = float('inf')
            energyCurrent = energyPrevious
        else:
            energyPrevious = self.scorefxn(dPose.pose)
            energyCurrent = energyPrevious

        sequenceOrig = dPose.getPdbSequence()

        # get the non catalytic residues energies
        nonCatalyticPoseIndices = dPose.getNonCatalyticPoseIndex()
        nonCatalyticEnergies = [(poseIndex, dPose.pose.energies().residue_total_energy(poseIndex)) for poseIndex in nonCatalyticPoseIndices]
        nonCatalyticEnergies.sort(key=lambda element: element[1], reverse=True)
        poseIndices = [element[0] for element in nonCatalyticEnergies[0:self.nNoneCatalytic]]

        # make a copy of the original one
        #self.dPoseOrig.assign(dPose)

        for i in range(self.activeSiteLoops):
            # Make a copy
            self.dPoseTemp.assign(dPose)

            # Clear the old cst
            #self.cst_remover.apply(self.dPoseTemp.pose)

            # Set to mutate one catalytic residue, while setting the currentAA of the previous catalytic residue to allowedAA.
            self.dPoseTemp.mutateCatalyticOne(replacePreviousCatalyticWith='All', setPreviousCatalyticNativeAAtoAllowedAA=True)

            # Set to mutate the high energy non catalytic residues
            self.dPoseTemp.mutateNonCatalyticToAllowedAA(poseIndices)

            # Mutate the active site
            self.dPoseTemp.writeResfile(self.resfileName, activeResidues=True)
            resfileCatalytic = ReadResfile(self.resfileName)
            self.taskFactory.clear()
            self.taskFactory.push_back(resfileCatalytic)
            self.packer.task_factory(self.taskFactory)

            # Make a quick packing to mutate the corresponding residues
            self.packer.nloop(1)
            self.packer.apply(self.dPoseTemp.pose)
            self.dPoseTemp.clearActive()

            self.dPoseTemp.updateConstraints()

            # Do a full packing
            self.dPoseTemp.writeResfile(self.resfileName, activeResidues=False)
            resfileCatalytic = ReadResfile(self.resfileName)
            self.taskFactory.clear()
            self.taskFactory.push_back(resfileCatalytic)

            self.activeSiteMinMoveMap.clear()
            for res in self.dPoseTemp.design:
                self.activeSiteMinMoveMap.set_chi(res.poseIndex)

            self.fastRelax.set_task_factory(self.taskFactory)
            self.fastRelax.set_movemap(self.activeSiteMinMoveMap)
            self.fastRelax.set_scorefxn(self.scorefxn)
            self.fastRelax.apply(self.dPoseTemp.pose)

            # if mutations are not accepted, skip this round and continue to a new round
            if not self.dPoseTemp.isPdbCatalyticSynced() or not self.dPoseTemp.isPdbNonCatalyticSynced():
                foundMutant = False
                continue

            sequenceCurrent = self.dPoseTemp.getPdbSequence()
            if sequenceCurrent == sequenceOrig:
                foundMutant = False
            else:
                foundMutant = True

            # Set the currentAA and nativeAA of all residues to current pdbAA to reflect
            # the current state of the system
            self.dPoseTemp.setNativeAAtoPdbAA()
            self.dPoseTemp.setCurrentAAtoPdbAA()

            # Minimize the new structure
            self.activeSiteMinMoveMap.clear()
            for res in self.dPoseTemp.design:
                self.activeSiteMinMoveMap.set_chi(res.poseIndex)
                if self.mimimizeBackbone:
                    # print('adding: ', res.name)
                    self.activeSiteMinMoveMap.set_bb(res.poseIndex)

            self.activeSiteMinMover.set_movemap(self.activeSiteMinMoveMap)
            self.activeSiteMinMover.score_function(self.scorefxn)
            self.activeSiteMinMover.apply(self.dPoseTemp.pose)

            # Check the energy, if lower assign to dPose
            energyCurrent = self.scorefxn(self.dPoseTemp.pose)
            if energyCurrent < energyPrevious:
                #foundMutant = True
                energyPrevious = energyCurrent
                dPose.assign(self.dPoseTemp)

            return foundMutant

    def initateCatalytic(self, dPose: DesignPose):

        # Clear the old cst
        self.cst_remover.apply(dPose.pose)

        # Set the design pose state
        dPose.mutateCatalyticAll()

        # Mutate the catalytic
        dPose.writeResfile(self.resfileName)
        resfileCatalytic = ReadResfile(self.resfileName)
        self.taskFactory.clear()
        self.taskFactory.push_back(resfileCatalytic)
        self.packer.task_factory(self.taskFactory)

        # Make a quick packing to mutate the corresponding residues
        self.packer.nloop(1)
        self.packer.apply(dPose.pose)

        # Fold
        dPose.updateConstraints()
        dPose.writeResfile(self.resfileName, activeResidues=False)
        resfileCatalytic = ReadResfile(self.resfileName)
        self.taskFactory.clear()
        self.taskFactory.push_back(resfileCatalytic)

        self.fastRelax.set_task_factory(self.taskFactory)
        self.fastRelax.set_movemap(self.activeSiteMinMoveMap)
        self.fastRelax.set_scorefxn(self.scorefxn)

        self.fastRelax.apply(dPose.pose)
        dPose.clearActive()

    def initateNoneCatalytic(self, dPose: DesignPose):

        # Clear the old cst
        self.cst_remover.apply(dPose.pose)

        # Set the design pose state
        dPose.mutateNonCatalyticAll(keepNativeAA=True, setNonCatalyticTo='All')

        # Mutate the catalytic
        dPose.writeResfile(self.resfileName)
        resfileCatalytic = ReadResfile(self.resfileName)
        self.taskFactory.clear()
        self.taskFactory.push_back(resfileCatalytic)
        self.packer.task_factory(self.taskFactory)

        # Make a quick packing to mutate the corresponding residues
        self.packer.nloop(1)
        self.packer.apply(dPose.pose)

        # Update constraint
        dPose.updateConstraints()

        dPose.writeResfile(self.resfileName, activeResidues=False)
        resfileCatalytic = ReadResfile(self.resfileName)
        self.taskFactory.clear()
        self.taskFactory.push_back(resfileCatalytic)

        self.fastRelax.set_task_factory(self.taskFactory)
        self.fastRelax.set_movemap(self.activeSiteMinMoveMap)
        self.fastRelax.set_scorefxn(self.scorefxn)

        self.fastRelax.apply(dPose.pose)
        dPose.clearActive()

    '''
    def applyCatalytic(self, dPose: DesignPose, countInitial=False):
        """
        Performs one round of catalytic design
        """
        # Move/Minimize catalytic
        moved = False
        if self.catalyticDesignMode == 'MC' and self.catalyticMutationMode == 'Direct':
            moved = self.moveCatalyticPositionsDirect(dPose, countInitial)

        elif self.catalyticDesignMode == 'MIN' and self.catalyticMutationMode == 'Direct':
            moved = self.minimizeCatalyticPositionsDirect(dPose, countInitial)

        return moved
    '''

    '''
    def applyNonCatalytic(self, dPose: DesignPose, countInitial=False):
        """
        Performs one round of non-catalytic design
        """
        # Move/Minimize non-catalytic
        moved = False
        if self.nonCatalyticDesignMode == 'MC':
            moved = self.moveNoneCatalyticRestype(dPose, countInitial=countInitial)

        elif self.nonCatalyticDesignMode == 'MIN':
            moved = self.minimizeNoneCatalyticRestype(dPose, countInitial=countInitial)

        return moved
    '''

    def apply(self, dPose: DesignPose, countInitial=False):
        """
        Performs one round of catalytic/non-catalytic design
        """
        # Move/Minimize catalytic
        moved = False
        if self.acticeSiteDesignMode == 'MC':
            moved = self.moveActiveSite(dPose, countInitial)
        elif self.acticeSiteDesignMode == 'MIN':
            moved = self.minimizeActiveite(dPose, countInitial)

        return moved

# TODO soft repulsion force the system to converge to a local minima. It is better to
# TODO be performed at iteration level. But it should be excluded from spawning and ranking
# TODO energies.
class ActiveSiteSampler(object):
    """
    This class gets a pose and a mover and sample it for n trials and
    return best pose and DesignDomainMover.
    """

    def __init__(self, softRepulsion: bool=True, dynamicSideChainCoupling: bool= False, activeSiteSampling: str='MC',
                 ligandSampling: str='Coupled', kT: float=1.0, nSteps: int=1000, anneal: bool=True, kT_high: float=1000,
                 kT_low: float=1, kT_decay: bool=True):

        # Movers for different design stages
        #self.ligandPackingMover = None
        #self.ligandRigidBodyMover = None
        #self.ligandNeighborsPackingMover = None
        self.ligandMovers = list()
        self.activeSiteMover = None
        self.scorefxn = None

        self.dPose = DesignPose()
        self.dPose_run_tm = DesignPose()
        self.dPose_move_tm = DesignPose()
        #self.E_final = float('inf')

        self.kT = kT
        self.nSteps = nSteps
        self.acceptanceRatio = 0
        self.ligandAcceptedRatio = 0
        self.activeSiteAcceptedRatio = 0
        self.nonCatalyticAcceptedRatio = 0

        #self.mode = mode  # ['LigandFirst', 'ActiveSiteFirst']

        self.anneal = anneal
        self.kT_high = kT_high
        self.kT_low = kT_low
        self.kT_decay = kT_decay

        self.softRepulsion = softRepulsion
        self.repulsionMin = 0.2
        self.repulsionMax = 0.55

        self.dynamicSideChainCoupling =  dynamicSideChainCoupling

        self.activeSiteSampling = activeSiteSampling          # ['MIN', 'MC', 'Coupled']
        self.ligandSampling = ligandSampling                # ['MIN', 'MC', 'Coupled']

    def setLigandMover(self, mover: LigandMover):
        if not isinstance(mover, LigandMover):
            raise ValueError('Failed setting LigandMover, wrong mover is given.')
        else:
            self.ligandMovers.append(mover)

    def setLigandMovers(self, movers: list):
        if not isinstance(movers, list):
            raise ValueError('Failed setting LigandMover, Expecting a list of LigandMovers.')
        else:
            for mover in movers:
                self.setLigandMover(mover)

    def setActiveSiteMover(self, mover: ActiveSiteMover):
        if not isinstance(mover, ActiveSiteMover):
            raise ValueError('Failed setting ActiveSiteMover, wrong mover is given.')
        else:
            self.activeSiteMover = mover

    def setScoreFunction(self, scorefxn):
        self.scorefxn = scorefxn

    def setInitialdPose(self, dPose: DesignPose, nAttempts=20, scratch=''):
        """
        Initiate the Design Pose. Once the dPose is generated, the catalytic are not synched with
        the pose sequence, The initial assignment of catalytic residues should be mutated in the
        pose. This would be done here.

        :param dPose:
        """
        # Try to randomize the pose
        activeSiteMover = ActiveSiteMover(activeSiteLoops=5, nNoneCatalytic=5, scratch=scratch)
        for i in range(nAttempts):

            self.dPose.assign(dPose)
            activeSiteMover.initateCatalytic(self.dPose)
            activeSiteMover.initateNoneCatalytic(self.dPose)
            if self.dPose.isPdbCatalyticSynced() and self.dPose.isPdbNonCatalyticSynced():
                break

        if not self.dPose.isPdbCatalyticSynced() or not self.dPose.isPdbNonCatalyticSynced():
            raise ValueError('Could not randomize the catalytic/non-catalytic of the initial structure. This could be '
                           'due to strict design parameters.')
        self.dPose.setCurrentAAtoPdbAA()
        self.dPose.setNativeAAtoPdbAA()

    #TODO remove this method its redundant
    def apply(self, dPose: DesignPose):
        """ The designed is performed by first perturbing the ligand and then adapting the
        active site to the ligand.
        :param dPose:
        """
        # get current Catalytic for updating the sideChainCouplingExcludedPoseIndex list of ligand movers
        #currentCatalytic = dPose.getCatalyticPoseIndex()

        # 1)  Move the ligand(s)
        for ligandMover in self.ligandMovers:

            # exclude the side chain of catalytic from side chain coupling
            ligandMover.setSideChainCouplingExcludedPoseIndex(dPose.getCatalyticPoseIndex())

            # exclude the design resides from neighbours packing
            ligandMover.setExcludedResiduePoseIndex(dPose.getDesignPoseIndex())

            if self.ligandSampling == 'MC':
                # In this case the ligand move moves are subject to Metropolis criterion separately.
                # This is done to allow the ligand remian stable and AS change and adapt

                # Make a Copy of design pose
                self.dPose_move_tm.assign(dPose)

                # Get the energy full
                E_old = self.scorefxn(self.dPose_move_tm.pose)
                E_old += self.getSASAConstraint(self.dPose_move_tm.pose)

                # perturb the system
                ligandMover.apply(self.dPose_move_tm.pose, countInitial=False)

                # Get he new energies
                E_new = self.scorefxn(self.dPose_move_tm.pose)
                E_new += self.getSASAConstraint(self.dPose_move_tm.pose)

                if self.acceptMove(E_new, E_old):
                    dPose.assign(self.dPose_move_tm)

            elif self.ligandSampling == 'MIN':
                ligandMover.apply(dPose.pose, countInitial=True)

            elif self.ligandSampling == 'Coupled':
                ligandMover.apply(dPose.pose, countInitial=False)

        # 2) Move active site catalytic residues by either mover or minimizer. Active site move do
        # include the SASA penalty since the side chain packing is should not be function of ligand SASA
        if self.activeSiteMover and dPose.catalyticInitiated:
            if self.catalyticSampling == 'MC':
                # In this case the catalytic moves are subject to
                # Metropolis criterion separately. This is done to allow the AS to remain
                # stable so that the ligand has the chance to adapt
                self.dPose_move_tm.assign(dPose)
                E_old = self.scorefxn(self.dPose_move_tm.pose)
                self.activeSiteMover.applyCatalytic(self.dPose_move_tm, countInitial=False)
                E_new = self.scorefxn(self.dPose_move_tm.pose)
                if self.acceptMove(E_new, E_old):
                    dPose.assign(self.dPose_move_tm)

            elif self.catalyticSampling == 'MIN':
                # In this case the catalytic and non-catalytic moves may not change the pose
                # if they can not find a lower energy move (countInitial=True)
                self.activeSiteMover.applyCatalytic(dPose, countInitial=True)

                # In this case the system is forced to change
            elif self.catalyticSampling == 'Coupled':
                self.activeSiteMover.applyCatalytic(dPose, countInitial=False)

        # 3) Move active site non-catalytic residues by either mover or minimizer.
        if self.activeSiteMover and dPose.designInitiated:
            # In this case the catalytic moves are subject to
            # Metropolis criterion separately. This is done to allow the AS to remain
            # stable so that the ligand has the chance to adapt
            if self.nonCatalyticSampling == 'MC':
                self.dPose_move_tm.assign(dPose)
                E_old = self.scorefxn(self.dPose_move_tm.pose)
                self.activeSiteMover.applyNonCatalytic(self.dPose_move_tm, countInitial=False)
                E_new = self.scorefxn(self.dPose_move_tm.pose)
                if self.acceptMove(E_new, E_old):
                    dPose.assign(self.dPose_move_tm)

            # In this case the catalytic and non-catalytic moves may not change the pose
            # if they can not find a lower energy move (countInitial=True)
            elif self.nonCatalyticSampling == 'MIN':
                self.activeSiteMover.applyNonCatalytic(dPose, countInitial=True)

            # In this case the system is forced to change
            elif self.catalyticSampling == 'Coupled':
                self.activeSiteMover.applyNonCatalytic(dPose, countInitial=False)

        # 4) Re-minimize ligand.
        for ligandMover in self.ligandMovers:
            # exclude the side chain of catalytic from side chain coupling
            ligandMover.setSideChainCouplingExcludedPoseIndex(dPose.getCatalyticPoseIndex())
            # exclude the design resides from neighbours packing
            ligandMover.setExcludedResiduePoseIndex(dPose.getDesignPoseIndex())
            ligandMover.relaxLigand(dPose.pose)


        # 5) Minimize Active site
        if self.activeSiteMover:
            self.activeSiteMover.minimizeActiveSite(dPose)

    def getdPose(self):
        dPose = DesignPose()
        dPose.assign(self.dPose)
        return dPose

    def setdPose(self, dPose):
        self.dPose.assign(dPose)

    def setState(self, dPose, method=None):
        """
        Sets the state of sampler (dPose). If method is None, the pose is set directly. If method is MIN,
        the dPose is set only if the energy of dPose is lower than current energy of sampler. If the method is MC
        the dPose exchange is su.
        :param dPose:
        :param method: None, MIN, or MC
        """
        if method not in ['MC', 'MN', None]:
            raise ValueError('Bad set state method, expecting MC, MIN, or None')

        if method is None:
            self.setdPose(dPose)

        elif method == 'MIN':
            E_old = self.getStateEnergy()
            E_new = self.scorefxn(dPose.pose)
            if E_new < E_old:
                self.setdPose(dPose)

        elif method == 'MIC':
            E_old = self.getStateEnergy()
            E_new = self.scorefxn(dPose.pose)
            if self.acceptMove(E_new, E_old):
                self.setdPose(dPose)

    def getSequence(self):
        return self.dPose.getPdbSequence()

    def getSequenceDiff(self):
        return self.dPose.getSeuenceDiff()

    def getStateEnergy(self):
        energy = self.scorefxn(self.dPose.pose)
        #energy += self.getStateEnergy()

    def getSASAConstraint(self, pose: Pose):
        sasaEnergy = 0
        if self.ligandMovers:
            for ligandMover in self.ligandMovers:
                sasaEnergy += ligandMover.getSASAConstraint(pose)
        return sasaEnergy

    def getLigandEnergy(self, pose: Pose):
        ligandEnergy = 0
        if self.ligandMovers:
            for ligandMover in self.ligandMovers:
                ligandEnergy += ligandMover.getLigandEnergy(pose)
        return ligandEnergy

    def run(self):

        # Set local variables
        E_old = self.scorefxn(self.dPose.pose)
        E_old += self.getSASAConstraint(self.dPose.pose)

        E_new = E_old

        step = 0
        nAccepted = 0
        nLigandAccepted = 0
        nActiveSiteAccepted = 0
        while step < self.nSteps:

            #print(MPI.COMM_WORLD.Get_rank(), step, self.dPose.currentNoneCatalyticMutants)
            #print("BBBB", step, flush=True)
            # reset the temp pose
            self.dPose_run_tm.assign(self.dPose)

            # Set the T for the current step
            if self.anneal:
                self.kT = getKT(x=step, Th=self.kT_high, Tl=self.kT_low, N=self.nSteps, k=self.kT_decay)

            # Set the repulsion
            if self.softRepulsion:
                self.scorefxn.set_weight(ScoreType.fa_rep, self.stepRepulsion(step))


            #print('core:', MPI.COMM_WORLD.Get_rank(), 'step', step)

            # Update SideChainCoupling of ligand movers if asked.
            if self.dynamicSideChainCoupling:
                for ligandMover in self.ligandMovers:
                    ligandMover.updateSideChainCouplingByStep(currentStep=step, totalSteps=self.nSteps)

            # Move the temp pose
            #self.apply(self.dPose_run_tm)
            ##########################################################################################################
            # get current Catalytic for updating the sideChainCouplingExcludedPoseIndex list of ligand movers
            #currentCatalytic = dPose.getCatalyticPoseIndex()
            ligandsMoved = False
            activeSiteMoved = False

            # 1)  Move the ligand(s)
            for ligandMover in self.ligandMovers:
                #print('BBB activating ligand mover.')
                thisLigandMoved = False
                # exclude the side chain of catalytic from side chain coupling
                ligandMover.setSideChainCouplingExcludedPoseIndex(self.dPose_run_tm.getCatalyticPoseIndex())

                # exclude the design resides from neighbours packing
                ligandMover.setExcludedResiduePoseIndex(self.dPose_run_tm.getDesignPoseIndex())

                if self.ligandSampling == 'MC':
                    # In this case the ligand move moves are subject to Metropolis criterion separately.
                    # This is done to allow the ligand remian stable and AS change and adapt

                    # Make a Copy of design pose
                    self.dPose_move_tm.assign(self.dPose_run_tm)

                    # Get the energy full
                    E_old_ligand = self.scorefxn(self.dPose_move_tm.pose)
                    E_old_ligand += self.getSASAConstraint(self.dPose_move_tm.pose)

                    # perturb the system
                    thisLigandMoved = ligandMover.apply(self.dPose_move_tm.pose, countInitial=False)

                    # If the ligand moved check for energy Get he new energies
                    if thisLigandMoved:
                        E_new_ligand = self.scorefxn(self.dPose_move_tm.pose)
                        E_new_ligand += self.getSASAConstraint(self.dPose_move_tm.pose)
                        if self.acceptMove(E_new_ligand, E_old_ligand):
                            self.dPose_run_tm.assign(self.dPose_move_tm)
                        else:
                            # Move not accepted reset the moved flag
                            thisLigandMoved = False

                elif self.ligandSampling == 'MIN':
                    thisLigandMoved = ligandMover.apply(self.dPose_run_tm.pose, countInitial=True)

                elif self.ligandSampling == 'Coupled':
                    thisLigandMoved = ligandMover.apply(self.dPose_run_tm.pose, countInitial=False)

                if thisLigandMoved:
                    ligandsMoved = True

            #self.dPose_run_tm.pose.dump_pdb('{:d}.pdb'.format(step))
            # 2) Move active site by either a mover or minimizer. Active site move do not include the SASA
            #  penalty since the side chain packing should not be function of ligand SASA
            if self.activeSiteMover:
                if self.activeSiteSampling == 'MC':
                    # In this case the moves are subject to Metropolis criterion separately. This is done to
                    # allow the AS to remain stable so that the ligand has the chance to adapt
                    self.dPose_move_tm.assign(self.dPose_run_tm)
                    E_old_catalytic = self.scorefxn(self.dPose_move_tm.pose)
                    activeSiteMoved = self.activeSiteMover.apply(self.dPose_move_tm, countInitial=False)

                    # If catalyticMoved check for energy
                    if activeSiteMoved:
                        E_new_catalytic = self.scorefxn(self.dPose_move_tm.pose)
                        if self.acceptMove(E_new_catalytic, E_old_catalytic):
                            self.dPose_run_tm.assign(self.dPose_move_tm)
                        else:
                            # Move not accepted reset the catalytic flag
                            activeSiteMoved = False

                elif self.activeSiteSampling == 'MIN':
                    # In this case the catalytic and non-catalytic moves may not change the pose
                    # if they can not find a lower energy move (countInitial=True)
                    activeSiteMoved = self.activeSiteMover.apply(self.dPose_run_tm, countInitial=True)

                    # In this case the system is forced to change
                elif self.activeSiteSampling == 'Coupled':
                    activeSiteMoved = self.activeSiteMover.apply(self.dPose_run_tm, countInitial=False)


            # 3) Re-minimize ligand.
            for ligandMover in self.ligandMovers:
                # exclude the side chain of catalytic from side chain coupling
                ligandMover.setSideChainCouplingExcludedPoseIndex(self.dPose_run_tm.getCatalyticPoseIndex())
                ligandMover.relaxLigand(self.dPose_run_tm.pose)

            ##########################################################################################################
            # Compute the current E
            E_new = self.scorefxn(self.dPose_run_tm.pose)
            E_new += self.getSASAConstraint(self.dPose_run_tm.pose)

            #if MPI.COMM_WORLD.Get_rank() == 1:
            #print(MPI.COMM_WORLD.Get_rank(), step, E_new)
            #    self.dPose_run_tm.pose.dump_pdb('C1_{}_{:.1f}.pdb'.format(step, E_new))

            #self.dPose_run_tm.pose.dump_pdb('catalytic-{}-{:.1f}.pdb'.format(step, E_new))
            # reject if nothing is moved
            if not ligandsMoved and not activeSiteMoved and E_new == E_old:
                step += 1
                continue
            # print('diff: ', self.dPose_run_tm.getSeuenceDiff())
            if self.acceptMove(E_new, E_old):
                # if move accepted update the dPose and energy
                #if MPI.COMM_WORLD.Get_rank() == 1:
                #    print('-------------------------------------------------step Accepted')
                self.dPose.assign(self.dPose_run_tm)
                E_old = E_new
                nAccepted += 1
                if ligandsMoved:
                    nLigandAccepted += 1
                if activeSiteMoved:
                    nActiveSiteAccepted += 1
                #self.dPose.pose.dump_pdb('catalytic-{}-{:.1f}.pdb'.format(step, E_old))
                #print('Accepted at Step; {}, kt {} with Energy: {}'.format(step, self.kT, E_old))

            step += 1


        # Finish the run
        if step != 0:
            self.acceptanceRatio = nAccepted/step
            self.ligandAcceptedRatio = nLigandAccepted/step
            self.activeSiteAcceptedRatio = nActiveSiteAccepted/step

        else:
            self.acceptanceRatio = 0.0
            self.ligandAcceptedRatio = 0
            self.activeSiteAcceptedRatio = 0

        #self.E_final = E_old

    def acceptMove(self, E_new, E_old):
        dV = (E_new - E_old)
        if dV < 0:
            W = 1
        else:
            W = exp(-dV / self.kT)

        if W > uniform(0, 1):
            return True
        else:
            return False

    def stepRepulsion(self, step):
        """
        Return a value of repulsion corresponds to the current step
        """
        ix = step / self.nSteps
        a = 1 + exp(-10 * (ix - (1 - 0.8)))
        s = ((1 / a) * ix) ** (1 - ix)
        return self.repulsionMin * (1 - s) + self.repulsionMax * s

    def setRepulstion(self, value: float):
        self.scorefxn.set_weight(ScoreType.fa_rep, value)


# TODO  ###############################################################################
# TODO Should remove these, the interface design still inheretes from DesignDomainMover
# TODO  ###############################################################################
'''
class DesignDomainMover(object):
    """
    This class is the one with the attributes and methods
    to either design the non-catalytic residues around an
    active site (DesignNoneCatalytic) or design a
    specific type of active site (DesignCatalytic).
    """

    def __init__(self, type):
        self.type = type
        self.design = list()
        self.designDict = dict()
        self.designLength = 0
        self.catalytic = list()
        self.catalyticTargetAA = dict()
        self.catalyticDict = dict()
        self.catalyticLength = 0
        self.catalyticAction = dict()
        self.constraints = list()
        self.noMutate = dict()  # Hold the catalytic index of non-mutatable residues

    def __iter__(self):
        for res in self.design:
            yield res

    def assign(self, other):
        """
        This method copies the attributes of one DesignDomainMover object
        to another one.
        """
        self.type = other.type
        self.design = copy.deepcopy(other.design)
        self.designDict = copy.deepcopy(other.designDict)
        self.designLength = copy.deepcopy(other.nDesignResidues)
        self.catalytic = copy.deepcopy(other.catalytic)
        self.catalyticTargetAA = copy.deepcopy(other.catalyticTargetAA)
        self.catalyticDict = copy.deepcopy(other.catalyticDict)
        self.catalyticAction = copy.deepcopy(other.catalyticAction)
        self.catalyticLength = copy.deepcopy(other.nCatalyticResidues)
        self.noMutate = copy.deepcopy(other.noMutate)
        self.constraints = copy.deepcopy(other.constraints)

    def initiateDesign(self, residues, pose):
        if not all(map(lambda x: isinstance(x, Residue), residues)):
            raise ValueError('Failed during initiating the design residues. Only list of Residue(s) is allowed.')

        self.design = residues
        self.designLength = len(self.design)

        # Set the design dictionary for rapid access (needed during non-catalytic design constraints update)
        # The keys are the (ID, chain) tuples. The values are the res index in the design list
        for index, res in enumerate(self.design):
            self.designDict[(res.ID, res.chain)] = index

            # Set the pose index of the design residues
        self.setPoseIndex(pose)

        # Set the allowed aa for each design position. This is independent of job type and only
        # Fill the positions that are not defined by user 'X'.
        self.setAllowedAA(pose)

        self.setCurrentAA()
        self.randomizeAll()

    def initiateCatalytic(self, catalytic, pose):
        for resName, aa in catalytic:
            # print('BBB', resName, aa)
            # Check the non-mutable residues
            designIndex = self.designDict.get(resName, False)

            # If resName is already in design domain, the residue is non mutable.
            if designIndex is not False:
                self.noMutate[resName] = designIndex
                self.design[designIndex].mutate = False

            else:
                # All catalytic residues must be part of design domain and non-mutable (i.e. no ambiguous resname) if it is DesignNoneCatalytic
                if re.match('DesignNoneCatalytic', self.type, re.IGNORECASE):
                    raise ValueError(
                        'The catalytic residue {} was not found in design residues. This is not allowed in {} jobType.'.format(
                            resName, self.type))

                # Assign the targetAA
            if resName in self.noMutate.keys() and re.match('DesignCatalytic', self.type, re.IGNORECASE):

                poseIndex = self.design[designIndex].poseIndex
                nativeAA = pose.residue(poseIndex).name1()
                # print('BBB', nativeAA, aa[0], DesignDomainWildCards('frozen'))
                if aa[0] in DesignDomainWildCards('frozen'):  # Use native amino acid and native rotamer
                    # TODO This is redundant, for non-mutated catalytic residues catalyticTargetAA and design.allowed AA
                    # TODO hold same info.
                    self.catalyticTargetAA[resName] = nativeAA
                    self.design[designIndex].allowedAA = nativeAA
                    self.catalyticAction[resName] = 'NATRO'
                    self.design[designIndex].action = 'NATRO'

                    # Use native amino acid but calculate rotamer
                elif aa[0] in DesignDomainWildCards('noneMutablePackable'):
                    self.catalyticTargetAA[resName] = nativeAA
                    self.design[designIndex].allowedAA = nativeAA
                    self.catalyticAction[resName] = 'PIKAA'
                    self.design[designIndex].action = 'PIKAA'

                else:  # New amino acid and calculate rotamer
                    self.catalyticTargetAA[resName] = aa
                    self.design[designIndex].allowedAA = aa
                    self.catalyticAction[resName] = 'PIKAA'
                    self.design[designIndex].action = 'PIKAA'

                # All catalytic residues are frozen in DesignNoneCatalytic
            elif resName in self.noMutate.keys() and re.match('DesignNoneCatalytic', self.type, re.IGNORECASE):

                poseIndex = self.design[designIndex].poseIndex
                nativeAA = pose.residue(poseIndex).name1()

                if not (aa[0] == nativeAA or aa[0] in DesignDomainWildCards('frozen')):
                    print(
                        'Warning >>> Residue {} is catalytic and should have native target amino acid ({} or ZZ), instead found {}. Setting the target to frozen {}.'.format(
                            resName, nativeAA, aa[0], nativeAA), file=sys.stderr, flush=True)
                self.catalyticTargetAA[resName] = nativeAA
                self.design[designIndex].allowedAA = nativeAA
                self.catalyticAction[resName] = 'NATRO'

            # Ambiguous residues where catalytic residues that are not defined yet (RES1, ...)
            else:
                self.catalyticTargetAA[resName] = aa
                self.catalyticAction[resName] = 'PIKAA'

                # Check for non-mutable that are not in design residues
                if re.match('ZZ', aa[0]):
                    raise ValueError(
                        'residue {} is declared non mutable but could not be found in the design domain'.format(
                            resName))

            # Assign the number of catalytic residues
        self.catalyticLength = len(self.catalyticTargetAA)

        # Assign initial choice of the catalytic residues for designCatalytic, This also set all the currentAA to Ala  for designCatalytic.
        # For others it assign catalytic residues and sets the current AA randomly.
        self.randomizeAll()

    def initiateConstraints(self, constraints):

        if not all(map(lambda constraint: isinstance(constraint, BaseConstraint), constraints)):
            raise ValueError('failed during initiating constraints. Only list of constraint(s) is allowed.')
        else:
            self.constraints = constraints

    def randomizeOne(self):

        if re.match('DesignCatalytic', self.type, re.IGNORECASE):

            # Pick a random index in the catalytic list and get its index in the design list
            old_index_Found = False
            for i in range(10000):
                catalyticIndex = randint(0, self.catalyticLength, 1)[0]
                old_index = self.catalytic[catalyticIndex]

                if old_index not in self.noMutate.values():
                    # Accept the move if it is not part of fixed position
                    catalyticName = self.design[old_index].catalyticName
                    targetAA = self.catalyticTargetAA[catalyticName]
                    old_index_Found = True
                    break

                    # If it is fixed position make sure has multiple target values
                elif old_index in self.noMutate.values():
                    catalyticName = self.design[old_index].catalyticName
                    targetAA = self.catalyticTargetAA[catalyticName]
                    if len(targetAA) > 1:
                        old_index_Found = True
                        break

            indexAA = randint(0, len(targetAA), 1)[0]
            currentAA = targetAA[indexAA]

            # Only mutate the amino acid type if a fixed position is chosen
            if old_index in self.noMutate.values():

                if not old_index_Found:
                    raise ValueError(' failed in randomizeOne. Could not randomize catalytic residue.')

                self.design[old_index].currentAA = currentAA

            else:
                new_index_Found = False
                for i in range(10000):
                    new_index = randint(0, self.designLength, 1)[0]
                    if new_index not in self.catalytic:
                        new_index_Found = True
                        break

                if not (new_index_Found and old_index_Found):
                    raise ValueError(' failed in randomizeOne. Could not randomize catalytic residue.')

                    # Update the catalytic res names
                self.design[new_index].catalyticName = catalyticName
                self.design[new_index].currentAA = currentAA

                self.design[old_index].catalyticName = None
                self.design[old_index].currentAA = 'A'

                # Update the list of catalytic residues with the index of the new residue
                self.catalytic[catalyticIndex] = new_index

                # Update the dictionary
                self.catalyticDict[catalyticName] = new_index

        elif re.match('DesignNoneCatalytic', self.type, re.IGNORECASE) or re.match('Optimize', self.type,
                                                                                   re.IGNORECASE):

            # Choose a random residue in the design domain
            for i in range(100):
                nDesignRes = len(self.design)
                indexDesign = randint(0, nDesignRes, 1)[0]

                # Skip the catalytic residues, only applies to DesignNoneCatalytic
                if indexDesign in self.catalytic:
                    continue

                res = self.design[indexDesign]
                nallowedAA = len(res.allowedAA)

                # choose a random AA and set the current AA
                if nallowedAA > 1:
                    indexAA = randint(0, nallowedAA, 1)[0]
                    res.currentAA = res.allowedAA[indexAA]
                    break


        else:
            raise ValueError('randomizeOne not implemented for job type {}'.format(self.type))

        if Constants.DEBUG:
            printDEBUG(msg=self.state(), rank=MPI.COMM_WORLD.Get_rank())

    def randomizeAll(self):

        if re.match('DesignCatalytic', self.type, re.IGNORECASE):
            # Set all currentAA of the design res to Ala.
            self.setCurrentAA()

            # Reset the name of the catalytic residues if exist
            for index in self.catalytic:
                self.design[index].catalyticName = None

            # Clear the previous residues if exist
            self.catalytic = list()
            self.catalyticDict = dict()

            # Pick up random design indices to assign initial choice of the catalytic residues
            indices = list()
            for i in range(self.catalyticLength - len(self.noMutate)):
                for j in range(10000):
                    index = randint(0, self.designLength, 1)[0]
                    if index not in self.noMutate.values() and index not in indices:
                        indices.append(index)
                        break
                if j == 9999:
                    raise ValueError('Failed in randomizeAll. Could not randomize catalytic residue.')

            for catalyticResName, catalyticResAA in self.catalyticTargetAA.items():
                # If none mutable the residue is already known
                if catalyticResName in self.noMutate.keys():
                    index = self.noMutate[catalyticResName]
                    # Otherwise, assign a random residue
                else:
                    index = indices.pop()

                    # Select a the index of a random target AA from catalyticResAA
                indexAA = randint(0, len(catalyticResAA), 1)[0]

                # Set up the catalytic res info
                self.catalytic.append(index)
                self.design[index].catalyticName = catalyticResName
                self.design[index].currentAA = [catalyticResAA[indexAA]]

                # Initiate a dictionary which its values point to the catalytic res for fast access
            for index in self.catalytic:
                # Get the catalytic residues Names
                name = self.design[index].catalyticName
                self.catalyticDict[name] = index

        elif re.match('DesignNoneCatalytic', self.type, re.IGNORECASE):
            # Clear the previous residues if exist
            self.catalytic = list()
            self.catalyticDict = dict()

            # If it is DesignNoneCatalytic, the catalytic residues are already known
            for catalyticResName, catalyticResAA in self.catalyticTargetAA.items():
                index = self.designDict[catalyticResName]
                self.catalytic.append(index)
                self.design[index].catalyticName = catalyticResName
                self.design[index].currentAA = catalyticResAA

            for index in self.catalytic:
                # Get the catalytic residues Names
                name = self.design[index].catalyticName
                self.catalyticDict[name] = index

            # Set the currentAA of catalytic residues to the catalyticTargetAA and
            # the rest of design domain to allowedAA
            self.setCurrentAA()

            # Randomize the current AA for each residue. No need to read from catalyticTargetAA for catalytic ones.
            # catalytic res don't change and target AAs are already assigned to allowedAA.
            for index, res in enumerate(self.design):
                # Skip the catalytic residues
                if index in self.catalytic:
                    continue
                nallowedAA = len(res.allowedAA)
                if nallowedAA == 0:
                    raise ValueError('randomizeAll failed, residue {} has no allowed AA'.format(res.name))
                else:
                    indexAA = randint(0, nallowedAA, 1)[0]
                    res.currentAA = res.allowedAA[indexAA]


        elif re.match('Optimize', self.type, re.IGNORECASE):

            self.setCurrentAA()
            for index, res in enumerate(self.design):
                nallowedAA = len(res.allowedAA)
                if nallowedAA == 0:
                    raise ValueError('randomizeAll failed, residue {} has no allowed AA'.format(res.name))
                elif nallowedAA == 1:
                    continue  # No point doing anything
                else:
                    indexAA = randint(0, nallowedAA, 1)[0]
                    res.currentAA = res.allowedAA[indexAA]


        else:
            raise ValueError('randomizeAll not implemented for job type {}'.format(self.type))

        if Constants.DEBUG:
            printDEBUG(msg=self.state(), rank=MPI.COMM_WORLD.Get_rank())

    def updateConstraints(self, pose):

        # Clear the previous constraints
        cst_remover = ClearConstraintsMover()
        cst_remover.apply(pose)

        # Over the constrains
        for cst in self.constraints:
            if re.match('B', cst.type, re.IGNORECASE):
                self._getBondConstraints(pose, cst)

            elif re.match('S', cst.type, re.IGNORECASE):
                self._getSequenceConstraints(pose, cst)

            else:
                raise ValueError("{} constraint is not implemented yet. Coming soon".format(cst.type))

    def _getBondConstraints(self, pose, cst):

        # Create ambiguous constraints for each constraints group
        ambiguous_constraint = AmbiguousConstraint()

        # Get the residue index "i"
        if cst.res_i in self.catalyticDict.keys():
            designIndex_res_i = self.catalyticDict[cst.res_i]
        elif cst.res_i in self.designDict.keys():
            designIndex_res_i = self.designDict[cst.res_i]
        else:  # To allow for constraints with residues outside domains
            designIndex_res_i = None
            # raise ValueError("Could not find res_i {} in constraint {}.".format(cst.res_i, cst.tag))

            # Get the residue index "j"
        if cst.res_j in self.catalyticDict.keys():
            designIndex_res_j = self.catalyticDict[cst.res_j]

        elif cst.res_j in self.designDict.keys():
            designIndex_res_j = self.designDict[cst.res_j]

        else:  # To allow for constraints with residues outside domains
            designIndex_res_j = None
            # raise ValueError("Could not find res_j {} in constraint {}.".format(cst.res_j, cst.tag))

        if designIndex_res_i is not None:
            i_resID = self.design[designIndex_res_i].poseIndex
        else:
            i_resID = pose.pdb_info().pdb2pose(cst.res_i[1], cst.res_i[0])

        if designIndex_res_j is not None:
            j_resID = self.design[designIndex_res_j].poseIndex
        else:
            j_resID = pose.pdb_info().pdb2pose(cst.res_j[1], cst.res_j[0])

        # Over all possible atom combinations
        for atomName_i in cst.atom_i_list:
            for atomName_j in cst.atom_j_list:

                try:
                    # Try to add cst only if atoms are available
                    if pose.residue(i_resID).has(atomName_i) and pose.residue(j_resID).has(atomName_j):
                        i_atomID = pose.residue(i_resID).atom_index(atomName_i)
                        j_atomID = pose.residue(j_resID).atom_index(atomName_j)

                        # Set the new constraints
                        i = pr.AtomID(i_atomID, i_resID)
                        j = pr.AtomID(j_atomID, j_resID)
                        func = BoundFunc(cst.lb, cst.hb, cst.sd, cst.tag)
                        distance_constraint = AtomPairConstraint(i, j, func)

                        # Add individual constraint to the ambiguous_constraint
                        ambiguous_constraint.add_individual_constraint(distance_constraint)
                except Exception as e:
                    pass

        pose.add_constraint(ambiguous_constraint)

    def _getSequenceConstraints(self, pose, cst):

        for residue, targetAA in cst.res.items():
            targetAA = one_to_three(targetAA)
            designIndex = self.designDict[residue]
            poseIndex = self.design[designIndex].poseIndex
            seqCst = ResidueTypeConstraint(poseIndex, str(poseIndex), targetAA, cst.weight * 1)
            pose.add_constraint(seqCst)

    def writeResfile(self, resfileName='.resfile'):
        resfileString = ''
        action = ''
        with open(resfileName, 'w') as resfile:
            resfile.write('NATRO\n')
            resfile.write('START\n')
            for res in self.design:
                if re.match('DesignCatalytic', self.type, re.IGNORECASE):
                    # Get the residue based action
                    if res.catalyticName:
                        action = self.catalyticAction[res.catalyticName]
                    else:
                        action = 'PIKAA'

                elif re.match('DesignNoneCatalytic', self.type, re.IGNORECASE) or re.match('Optimize', self.type,
                                                                                           re.IGNORECASE):

                    # Get the residue based action
                    if res.catalyticName:
                        action = self.catalyticAction[res.name]
                    else:
                        action = res.designAction

                else:
                    raise ValueError("writeResfile is not defined for {} job.".format(self.type))

                    # Write to the file
                if action == 'NATRO':
                    resfile.write('{}  {}  {}    \n'.format(res.ID, res.chain, action))
                else:
                    resfile.write('{}  {}  {}  {}\n'.format(res.ID, res.chain, action, ''.join(res.currentAA)))

                if Constants.DEBUG:
                    if action == 'NATRO':
                        resfileString += '{}  {}  {}    \n'.format(res.ID, res.chain, action)
                    else:
                        resfileString += '{}  {}  {}  {}\n'.format(res.ID, res.chain, action, ''.join(res.currentAA))
        if Constants.DEBUG:
            printDEBUG(msg=resfileString, rank=MPI.COMM_WORLD.Get_rank())

    def getSequence(self, pose):
        sequence = list()
        for res in self.design:
            sequence.append(pose.residue(res.poseIndex).name1())
        return ''.join(sequence)

    def setAllowedAA(self, pose):
        """
        This Function calculates acceptable AA composition for a given site(s) based on
        solvent-accessible surface area (SASA) ans secondary structure (SSE)
        :param pdbFile:
        :param residueList:
        :return:
        """
        # Ref: https://github.com/sarisabban/RosettaDesign
        # Ref: https://doi.org/10.1371/journal.pone.0080635
        # Ref: The Molecules of Life: Physical and Chemical Principles. Page 216.

        # Get the pdb file out of the pose
        posePdb = ostringstream()
        pose.dump_pdb(posePdb)
        pdbFile = StringIO(posePdb.str())

        # Get the PDB file
        pdb = PDBFile.read(pdbFile)
        pdbStruc = pdb.get_structure()[0]

        # Amino acids properties
        aminoAcids = AminoAcids()

        # Calculate SASA
        atomsSASA = sasa(pdbStruc, vdw_radii="Single")
        strucSASA = apply_residue_wise(pdbStruc, atomsSASA, sum)
        # Calculate SSE
        strucSSE = annotate_sse(pdbStruc, pdbStruc.chain_id)

        # Calculate the AA compositions
        for res in self.design:
            # Get the pdb AA if XX+ or ZX. Do it here since allowedAA will be overwritten later.
            nativeAA = None
            if res.allowedAA[0] in DesignDomainWildCards('nativePlus'):
                nativeAA = pose.residue(res.poseIndex).name1()

            # Calculate the aa compositions
            if res.allowedAA[0] in DesignDomainWildCards('mutable'):
                resIndex = res.poseIndex - 1  # The index in Biotite starts from 0, while in PyRosetta starts from 1
                resName = pdbStruc.res_name[0]
                resRASA = strucSASA[resIndex] / aminoAcids.maxSASA3(resName)  # RASA refers to the relative SASA
                resSSE = strucSSE[resIndex]
                # label res based on RASA
                if resRASA <= 0.25:
                    resRASA = 'C'
                elif 0.25 < resRASA < 0.75:
                    resRASA = 'B'
                elif resRASA >= 0.75:
                    resRASA = 'S'

                # get the AA composition
                if resRASA == 'S' and resSSE == 'c':
                    aa = list('PGNQSTDERKH')
                elif resRASA == 'S' and resSSE == 'a':
                    aa = list('EHKQR')
                elif resRASA == 'S' and resSSE == 'b':
                    aa = list('DEGHKNPQRST')
                elif resRASA == 'B' and resSSE == 'c':
                    aa = list('ADEFGHIKLMNPQRSTVWY')
                elif resRASA == 'B' and resSSE == 'a':
                    aa = list('ADEHIKLMNQRSTVWY')
                elif resRASA == 'B' and resSSE == 'b':
                    aa = list('DEFHIKLMNQRSTVWY')
                elif resRASA == 'C' and resSSE == 'c':
                    aa = list('AFGILMPVWY')
                elif resRASA == 'C' and resSSE == 'a':
                    aa = list('AFILMVWY')
                elif resRASA == 'C' and resSSE == 'b':
                    aa = list('FILMVWY')
                res.allowedAA = aa
                res.designAction = "PIKAA"

                # Add the native AA if asked
                if nativeAA is not None and nativeAA not in res.allowedAA:
                    res.allowedAA.append(nativeAA)

            # Assign the native
            elif res.allowedAA[0] in ['ZX']:
                res.allowedAA = [nativeAA]
                res.designAction = "PIKAA"

            elif res.allowedAA[0] in ['ZZ']:
                res.allowedAA = [nativeAA]
                res.designAction = "NATRO"

            #

            elif res.allowedAA[0] in DesignDomainWildCards('aminoAcids'):
                res.allowedAA = aminoAcids.selection(res.allowedAA[0])
                res.designAction = "PIKAA"
            else:
                res.designAction = "PIKAA"

    def setCurrentAA(self):

        for res in self.design:

            catalyticAA = self.catalyticTargetAA.get(res.name, False)
            # Set the currentAA to 'A' in the case of Designcatalytic. This will be overwritten by randomize functions
            if re.match('DesignCatalytic', self.type, re.IGNORECASE):
                if catalyticAA:
                    # This is redundant, for catalytic residues catalyticTargetAA is used
                    res.currentAA = catalyticAA
                else:
                    res.currentAA = ['A']


            elif re.match('DesignNoneCatalytic', self.type, re.IGNORECASE) or re.match('Optimize', self.type,
                                                                                       re.IGNORECASE):
                # Try finding the res in catalytic set
                if catalyticAA:
                    res.currentAA = catalyticAA
                    res.catalyticName = res.name
                else:
                    res.currentAA = res.allowedAA
                    res.catalyticName = None

    def setPoseIndex(self, pose):

        posInfo = pose.pdb_info()
        for res in self.design:
            res.poseIndex = posInfo.pdb2pose(res.chain, res.ID)
            # TODO ADD LOGGING
            # print((res.chain, res.poseIndex, res.ID), pose.residue(res.poseIndex).name1())

    def finalize(self, pose):

        # Update current residues
        if re.match('DesignCatalytic', self.type, re.IGNORECASE):
            # Set the current AA from the pose (in case multiple residues were allowed for a catalytic site)
            for res in self.design:
                if res.catalyticName:
                    res.currentAA = list(pose.residue(res.poseIndex).name1())
                else:
                    # Check if it is frozen
                    if res.designAction == 'NATRO':
                        res.currentAA = DesignDomainWildCards('frozen')
                    else:
                        res.currentAA = res.allowedAA

                # update the atom of the residues, especially ambiguous residues
            # invalidCstList = list()
            for cstIndex, cst in enumerate(self.constraints):

                if cst.type == 'B':
                    # validCst = True
                    # res_i_old = cst.res_i
                    # res_j_old = cst.res_j

                    # Get the residue index.
                    if cst.res_i in self.catalyticDict.keys():
                        designIndex_res_i = self.catalyticDict[cst.res_i]

                    elif cst.res_i in self.designDict.keys():
                        designIndex_res_i = self.designDict[cst.res_i]

                    else:
                        designIndex_res_i = None
                        # raise ValueError("failed to finalize. Could not find res_i {} in constraint {}.".format(cst.res_i, cst.tag))

                        # Get the residue index. order matters here
                    if cst.res_j in self.catalyticDict.keys():
                        designIndex_res_j = self.catalyticDict[cst.res_j]

                    elif cst.res_j in self.designDict.keys():
                        designIndex_res_j = self.designDict[cst.res_j]

                    else:
                        designIndex_res_j = None
                        # raise ValueError("failed to finalize. Could not find res_j {} in constraint {}.".format(cst.res_j, cst.tag))

                    if designIndex_res_i is not None:
                        res_i = self.design[designIndex_res_i]
                        cst.res_i = res_i.name
                    else:  # In case it is not part of any domain
                        cst.res_i = cst.res_i

                    if designIndex_res_j is not None:
                        res_j = self.design[designIndex_res_j]
                        cst.res_j = res_j.name
                    else:  # In case it is not part of any domain
                        cst.res_j = cst.res_j

                    if False: """
                    # Check the atoms of resi
                    missingAtoms = list()
                    presentAtoms = list()
                    for atom_i in cst.atom_i_list:
                        if pose.residue(res_i.poseIndex).has(atom_i):
                            presentAtoms.append(atom_i)
                        else:
                            missingAtoms.append(atom_i)

                    # Remove the cst if no longer valid, or update the atom list if still valid
                    if len(presentAtoms) == 0:
                        invalidCstList.append(cst.tag)
                        print("Warning >>> restraint {} is no longer valid. {} is assigned to {} which do not possess the defined atoms .".format(cst.tag, ':'.join(map(lambda x: str(x), res_i_old)), ':'.join(map(lambda x: str(x), (res_i.ID, res_i.chain)))), file=sys.stderr, flush=True)
                        continue
                    else:
                        cst.atom_i_list = presentAtoms

                    # Check the atoms of resj
                    missingAtoms = list()
                    presentAtoms = list()
                    for atom_j in cst.atom_j_list:
                        if pose.residue(res_j.poseIndex).has(atom_j):
                            presentAtoms.append(atom_j)
                        else:
                            missingAtoms.append(atom_j)

                    # Remove the cst if no longer valid, or update the atom list i still valid
                    if len(presentAtoms) == 0:
                        invalidCstList.append(cst.tag)
                        print("Warning >>> restraint {} is no longer valid. {} is assigned to {} which do not possess the defined atoms .".format(cst.tag, ':'.join(map(lambda x: str(x), res_j_old)), ':'.join(map(lambda x: str(x), (res_j.ID, res_j.chain)))), file=sys.stderr, flush=True)
                        continue
                    else:
                        cst.atom_j_list = presentAtoms

                # Remove the invalids
            newConstraints = list()
            for constraint in self.constraints:
                if constraint.tag not in invalidCstList:
                    newConstraints.append(constraint)
                # Replace
            self.constraints = newConstraints
            """
                # Change the type
            self.type = 'DesignNoneCatalytic'


        elif re.match('DesignNoneCatalytic', self.type, re.IGNORECASE):

            for res in self.design:

                # Check for the fronzen residues
                if res.designAction == 'NATRO':
                    res.currentAA = DesignDomainWildCards('frozen')

                    # Get all residues from pose. design finished here.
                else:
                    res.currentAA = list(pose.residue(res.poseIndex).name1())

                # Change the type
            self.type = 'Optimize'

        else:
            raise ValueError('finalize can only be called for "DesignCatalytic" and "DesignNoneCatalytic" job types.')

    def writeYaml(self, fileName, pdbName, prefixName, nIterations=1, nSteps=1, nPoses=1, kT=1.0, anneal=True,
                  kT_high=1000, kT_low=1.0, kT_decay=True):

        # writeYaml should only be applied to the finalized movers, where the jobType is changed.
        if not (re.match('DesignNoneCatalytic', self.type, re.IGNORECASE) or re.match('Optimize', self.type,
                                                                                      re.IGNORECASE)):
            raise ValueError(
                'writeYalm can only be called for DesignNoneCatalytic and Optimize job type, not {}.'.format(self.type))

        with open(fileName, 'w') as f:

            # Write down the type of Job being DesignNoneCatalytic
            if re.match('DesignNoneCatalytic', self.type, re.IGNORECASE):
                f.write('JobType: DesignNoneCatalytic\n')

            # Write down the type of Job being Optimize
            elif re.match('Optimize', self.type, re.IGNORECASE):
                f.write('JobType: Optimize\n')

                # Write down the type of input filename and the prefixName for the job
            f.write('PDB: {}\n'.format(pdbName))
            f.write('Name: {}\n'.format(prefixName))

            # Write down the design residues.
            f.write('DesignResidues:\n')
            for res in self.design:
                f.write('  {}-{}: {}\n'.format(res.ID, res.chain, '-'.join(res.currentAA)))

                # Catalytic residues are only for DesignNoneCatalytic jobs
            if re.match('DesignNoneCatalytic', self.type, re.IGNORECASE):
                f.write('CatalyticResidues:\n')
                for resIndex in self.catalytic:
                    res = self.design[resIndex]
                    f.write('  {}-{}: ZZ        # Assigned to {}.\n'.format(res.ID, res.chain, '-'.join(res.currentAA)))

                # Write down the distance constraints for the catalytic residues in the yaml file
            f.write('Constraints:\n')
            for cst in self.constraints:
                if cst.type == 'B':

                    # Get the residue index. order matters here
                    # if cst.res_i in self.catalyticDict.keys():
                    #    designIndex_res_i = self.catalyticDict[cst.res_i]
                    # At this stage all csts are updated
                    if cst.res_i in self.designDict.keys():
                        designIndex_res_i = self.designDict[cst.res_i]
                        resiID, resiChain = self.design[designIndex_res_i].ID, self.design[designIndex_res_i].chain
                    else:
                        resiID, resiChain = cst.res_i[0], cst.res_i[1]
                        # raise ValueError("failed to write yaml. Could not find res_i {} in constraint {}.".format(cst.res_i, cst.tag))

                    # Get the residue index. order matters here
                    # if cst.res_j in self.catalyticDict.keys():
                    #    designIndex_res_j = self.catalyticDict[cst.res_j]

                    if cst.res_j in self.designDict.keys():
                        designIndex_res_j = self.designDict[cst.res_j]
                        resjID, resjChain = self.design[designIndex_res_j].ID, self.design[designIndex_res_j].chain
                    else:
                        resjID, resjChain = cst.res_j[0], cst.res_j[1]
                        # raise ValueError("failed to write yaml. Could not find res_j {} in constraint {}.".format(cst.res_j, cst.tag))

                    f.write('  {}:\n'.format(cst.tag))
                    f.write('    type: {}\n'.format(cst.type))
                    f.write('    resi: {}-{}\n'.format(resiID, resiChain))
                    f.write('    atomi: {}\n'.format('-'.join(cst.atom_i_list)))
                    f.write('    resj: {}-{}\n'.format(resjID, resjChain))
                    f.write('    atomj: {}\n'.format('-'.join(cst.atom_j_list)))
                    f.write('    lb: {}\n'.format(cst.lb))
                    f.write('    hb: {}\n'.format(cst.hb))
                    f.write('    sd: {}\n'.format(cst.sd))

                elif cst.type == 'S':
                    f.write('  {}:\n'.format(cst.tag))
                    f.write('    type: {}\n'.format(cst.type))
                    f.write('        weight: {}\n'.format(cst.weight))
                    # The target residues are already known so No need for the ref
                    # f.write('    reference: {}\n'.format(cst.ref))
                    f.write('    residues: \n')
                    for res, targetAA in cst.res.items():
                        f.write('      {}: {}\n'.format('-'.join(map(lambda x: str(x), res)), targetAA))

                else:
                    raise ValueError(
                        'failed writing nonCatalyricYaml file. Writing constraint {} type is not implemented.')

            # Write down the iterations, number of poses and steps, and the temperature in the yaml file
            f.write('nIterations: {}\n'.format(nIterations))
            f.write('nPoses: {}\n'.format(nPoses))
            f.write('nSteps: {}\n'.format(10))
            if anneal:
                f.write('Anneal: {}\n'.format(anneal))
                f.write('kT_high: {}\n'.format(kT_high))
                f.write('kT_low: {}\n'.format(kT_low))
                f.write('kT_decay: {}\n'.format(kT_decay))
            else:
                f.write('kT: {}\n'.format(kT))

    def state(self):
        state = ''
        state += 'Type: {}\n'.format(self.type)
        state += 'Catalytic indices: {}\n'.format(self.catalytic)
        state += 'Catalytic Dict: {}\n'.format(self.catalyticDict)
        state += 'Catalytic noMutate: {}\n'.format(self.noMutate)
        state += 'Catalytic Target AA: {}\n'.format(self.catalyticTargetAA)
        state += 'Catalytic Actions AA: {}\n'.format(self.catalyticAction)
        state += '------------------------------------------------------------------------------------------------\n'
        state += 'Design residues: \n'
        for res in self.design:
            state += 'ID: {}, chain: {}, Action: {}, CatalyticName: {}, pose Index: {} \n'.format(res.ID, res.chain,
                                                                                                  res.designAction,
                                                                                                  res.catalyticName,
                                                                                                  res.poseIndex)
            state += 'Current residues: \n'.format(''.join(res.currentAA))
            state += 'Allowed residues: \n'.format(''.join(res.allowedAA))
        state += '------------------------------------------------------------------------------------------------\n'
        state += 'Constraint: \n'
        for cst in self.constraints:

            # If this is a DesignCatalytic job, the residue index can be found in the catalyticDict which holds the index of the
            # current catalytic residues
            if cst.type == 'B':
                # Get the residue index
                if cst.res_i in self.catalyticDict.keys():
                    designIndex_res_i = self.catalyticDict[cst.res_i]

                elif cst.res_i in self.designDict.keys():
                    designIndex_res_i = self.designDict[cst.res_i]

                else:
                    raise ValueError("Could not find res_i {} in constraint {}.".format(cst.res_i, cst.tag))

                # Get the residue index
                if cst.res_j in self.catalyticDict.keys():
                    designIndex_res_j = self.catalyticDict[cst.res_j]

                elif cst.res_j in self.designDict.keys():
                    designIndex_res_j = self.designDict[cst.res_j]

                else:
                    raise ValueError("Could not find res_i {} in constraint {}.".format(cst.res_i, cst.tag))

                state += '{},  {}({}):{},  {}({}):{},  {} {} {} {}\n'.format(cst.type,
                                                                             cst.res_i, (
                                                                                 self.design[designIndex_res_i].ID,
                                                                                 self.design[designIndex_res_i].chain),
                                                                             cst.atom_i_list,
                                                                             cst.res_j, (
                                                                                 self.design[designIndex_res_j].ID,
                                                                                 self.design[designIndex_res_j].chain),
                                                                             cst.atom_j_list,
                                                                             cst.lb, cst.hb, cst.sd, cst.tag)

            elif cst.type == 'S':
                state += '{},  {},  {}\n'.format(cst.type, cst.weight, cst.res)
            else:
                raise ValueError('constraint {} type is not implemented in state function.')
        state += '------------------------------------------------------------------------------------------------\n'
        return state

    def statePretty(self):

        print("Design domain with catalytic assignments: ")
        for index, res in enumerate(self.design):
            resname = (res.ID, res.chain)
            print('position: {}'.format(index))
            print('     residue: {}'.format(':'.join(map(str, resname))))
            print('     Pose Index: {}'.format(res.poseIndex))
            if res.catalyticName:
                print('     catalytic assignment: {}'.format(res.catalyticName))
                if res.catalyticName in self.noMutate:
                    print('     mutate: False')
                else:
                    print('     mutate: True')

                if self.catalyticAction[res.catalyticName] == 'NATRO':
                    print('     frozen: True')
                else:
                    print('     frozen: False')

                print('     allowedAA: {}'.format(''.join(self.catalyticTargetAA[res.catalyticName])))
                print('     currentAA: {}'.format(''.join(res.currentAA)))

            else:
                print('     catalytic assignment: None')
                print('     mutate: True')
                if res.designAction == 'NATRO':
                    print('     frozen: True')
                else:
                    print('     frozen: False')
                print('     allowedAA: {}'.format(''.join(res.allowedAA)))
                print('     currentAA: {}'.format(''.join(res.currentAA)))

        print('Constraints:')
        for index, cst in enumerate(self.constraints):
            print('{}, {}'.format(index, cst.show()))


class CatalyticMoverByResfile(pr.rosetta.protocols.moves.Mover):
    """
    This class contains the tools to perform MC sampling of the different residues
    to find those that stabilize the protein the best.
    """

    def __init__(self, DesignDomainMover, scorefxn, packer_nloop=1000, prefixName=0, scratch=''):
        """Design Mover"""

        pr.rosetta.protocols.moves.Mover.__init__(self)

        # Initiate the residue groups
        self.domain = DesignDomainMover

        # Initiate the Rosetta classes
        self.taskFactory = TaskFactory()
        self.packer = PackRotamersMover()
        self.scorefxn = scorefxn
        self.cst_remover = ClearConstraintsMover()
        self.packer_nloop = packer_nloop
        self.counter = 0
        self.resfileName = os.path.join(scratch, '{}.resfile'.format(prefixName))

    def getName(self):
        """Return name of class."""

        return self.__class__.__name__

    def apply(self, pose):
        """Applies move to pose."""

        # remove the constraints
        self.cst_remover.apply(pose)

        # Select new catalytic_residues out of the design_residues randomly
        self.domain.randomizeOne()

        # Write the resfile for task factory
        self.domain.writeResfile(self.resfileName)

        # Make a task from resfile
        resfile = ReadResfile(self.resfileName)
        self.taskFactory.clear()
        self.taskFactory.push_back(resfile)
        os.remove(self.resfileName)

        # Initiate the packer
        self.packer.task_factory(self.taskFactory)
        self.packer.score_function(self.scorefxn, )

        # Make the initial Mutations
        self.packer.nloop(1)
        self.packer.apply(pose)

        # Update the restraints in the pose
        self.domain.updateConstraints(pose)

        # Search rotamers
        self.packer.nloop(self.packer_nloop)
        self.packer.apply(pose)


class CatalyticSampler(object):
    """
    This class gets a pose and a mover and sample it for n trials and
    return best pose and DesignDomainMover.
    """

    def __init__(self, pose, mover, scorefxn, kT=1.0, nSteps=1000, anneal=True, kT_high=1000, kT_low=1, kT_decay=True):

        if not isinstance(mover, CatalyticMoverByResfile):
            raise ValueError(
                'Error >>> CatalyticSampler only accepts CatalyticMoverByResfile or CatalyticMoverByResfileWithDocking')

        self.pose = pose
        self.mover = mover
        self.scorefxn = scorefxn
        self.kT = kT
        self.nSteps = nSteps

        self.anneal = anneal
        self.kT_high = kT_high
        self.kT_low = kT_low
        self.kT_decay = kT_decay

        # Initiate the best results
        self.final_pose = pr.Pose()
        self.final_pose.assign(self.pose)
        self.initial_pose = pr.Pose()
        self.initial_pose.assign(self.pose)
        self.final_score = 999999.9
        self.final_domain = DesignDomainMover(mover.domain.type)
        self.final_domain.assign(mover.domain)

        # Initiate MC object
        self.mc = pr.MonteCarlo(self.pose, self.scorefxn, self.kT)

    def run(self):

        for step in range(self.nSteps):
            # Set the T for the current step
            if self.anneal:
                self.kT = getKT(x=step, Th=self.kT_high, Tl=self.kT_low, N=self.nSteps, k=self.kT_decay)
                self.mc.set_temperature(self.kT)

            # Make a move
            self.mover.apply(self.pose)

            # Evaluate the move with the Metropolis criterion
            if self.mc.boltzmann(self.pose):
                self.final_pose.assign(self.pose)
                self.final_domain.assign(self.mover.domain)
                self.final_score = self.mc.last_score()


class NoneCatalyticMoverByResfile(pr.rosetta.protocols.moves.Mover):

    def __init__(self, DesignDomainMover, scorefxn, packer_nloop=10, prefixName=0, scratch=''):
        """NoneCatalytic Desing Mover"""
        pr.rosetta.protocols.moves.Mover.__init__(self)

        self.domain = DesignDomainMover
        self.taskFactory = TaskFactory()
        self.movemap = MoveMap()
        self.scorefxn = scorefxn
        self.cst_remover = ClearConstraintsMover()
        self.packer = PackRotamersMover()
        self.fastRelax = FastRelax()
        self.fastDesign = FastDesign()
        self.packer_nloop = packer_nloop
        self.counter = 0
        self.resfileName = os.path.join(scratch, '{}.resfile'.format(prefixName))

        # Initiate taskFactory
        self.domain.writeResfile(self.resfileName)
        resfile = ReadResfile(self.resfileName)
        self.taskFactory.push_back(resfile)
        os.remove(self.resfileName)

        # Initiate movemap
        for res in self.domain.design:
            self.movemap.set_bb(res.poseIndex, True)
            self.movemap.set_chi(res.poseIndex, True)

        # Initiate Packer mover
        self.packer.task_factory(self.taskFactory)
        self.packer.score_function(self.scorefxn, )
        self.packer.nloop(self.packer_nloop)

        # Initiate fastRealx mover
        self.fastRelax.set_task_factory(self.taskFactory)
        self.fastRelax.set_movemap(self.movemap)
        self.fastRelax.set_scorefxn(self.scorefxn)

    def getName(self):
        """Return name of class."""
        return self.__class__.__name__

    def apply(self, pose):
        """Applies move to pose."""

        # remove the constraints
        self.cst_remover.apply(pose)

        self.domain.randomizeOne()

        # Write the resfile for task factory
        self.domain.writeResfile(self.resfileName)

        # Make a task from resfile
        resfile = ReadResfile(self.resfileName)
        self.taskFactory.clear()
        self.taskFactory.push_back(resfile)
        os.remove(self.resfileName)

        # Update the restraints in the pose
        self.domain.updateConstraints(pose)

        # Make the initial Mutations
        self.packer.nloop(1)
        self.packer.apply(pose)

        # Update the restraints in the pose
        self.domain.updateConstraints(pose)

        # Search rotamers
        self.packer.nloop(self.packer_nloop)
        self.packer.apply(pose)

        # DO Relax
        self.fastRelax.apply(pose)


class NoneCatalyticSampler(object):
    """
    This class gets a pose and a mover and sample it for n trials and return
    best pose and DesignDomainMover.
    """

    def __init__(self, pose, mover, scorefxn, kT=1.0, nSteps=10, anneal=True, kT_high=1000, kT_low=1, kT_decay=True):

        if not isinstance(mover, NoneCatalyticMoverByResfile):
            raise ValueError(
                'Error >>> NoneCatalyticSampler only accepts NoneCatalyticMover or NoneCatalyticMoverByResfileWithDocking')

        self.pose = pose
        self.mover = mover
        self.scorefxn = scorefxn
        self.kT = kT
        self.nSteps = nSteps

        self.anneal = anneal
        self.kT_high = kT_high
        self.kT_low = kT_low
        self.kT_decay = kT_decay

        # Initiate the best results
        self.final_pose = pr.Pose()
        self.final_pose.assign(self.pose)
        self.initial_pose = pr.Pose()
        self.initial_pose.assign(self.pose)
        self.final_score = self.scorefxn(self.pose)
        self.final_domain = DesignDomainMover(mover.domain.type)
        self.final_domain.assign(mover.domain)

        # Initiate MC object
        self.mc = pr.MonteCarlo(self.pose, self.scorefxn, self.kT)

    def run(self):
        for step in range(self.nSteps):
            # Reset the structure for each step
            if self.anneal:
                self.kT = getKT(x=step, Th=self.kT_high, Tl=self.kT_low, N=self.nSteps, k=self.kT_decay)
                self.mc.set_temperature(self.kT)

                # Make a move
            self.mover.apply(self.pose)

            # Evaluate the move with the Metropolis criterion
            if self.mc.boltzmann(self.pose):
                self.final_pose.assign(self.pose)
                self.final_domain.assign(self.mover.domain)
                self.final_score = self.mc.last_score()
'''