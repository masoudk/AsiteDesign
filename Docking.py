import os
import sys
import yaml
import re
import math
import itertools
import copy, time
import numpy as np
from numpy import exp, sum, array, where, sqrt, any, std, mean, dot, zeros, ones
from numpy.random import randint, normal
from numpy.linalg import norm
from io import StringIO
from mpi4py import MPI
from random import choice, sample
from math import ceil
from multiprocessing import Pool, cpu_count
# PyRosetta import
import pyrosetta as pr

# PyRosetta mover and pakcer imports
from pyrosetta import Pose
from pyrosetta.rosetta.core.id import AtomID
from pyrosetta.rosetta.core.scoring import EMapVector
from pyrosetta.rosetta.core.kinematics import MoveMap
from pyrosetta.rosetta.core.pack.task.operation import ReadResfile
from pyrosetta.rosetta.core.pack.task import TaskFactory
from pyrosetta.rosetta.protocols.minimization_packing import PackRotamersMover, MinMover
from pyrosetta.rosetta.protocols.constraint_movers import ClearConstraintsMover
from pyrosetta.rosetta.protocols.relax import FastRelax
from pyrosetta.rosetta.protocols.denovo_design.movers import FastDesign
from pyrosetta.rosetta.core.scoring.constraints import BoundFunc, AtomPairConstraint, AmbiguousConstraint, \
    ResidueTypeConstraint
from pyrosetta.rosetta.core.scoring import ScoreType
from pyrosetta.rosetta.std import ostringstream, istringstream
from pyrosetta.rosetta.protocols.geometry import centroid_by_residues, centroid_by_chain
from pyrosetta.rosetta.core.scoring import score_type_from_name
from pyrosetta.rosetta.protocols.docking import setup_foldtree
from pyrosetta.rosetta.protocols.grafting import delete_region, replace_region, return_region, insert_pose_into_pose
from pyrosetta.rosetta.protocols.rigid import RigidBodyPerturbMover
from pyrosetta.rosetta.protocols.toolbox.pose_manipulation import rigid_body_move
from pyrosetta.rosetta.protocols.simple_moves import ShearMover
from pyrosetta.rosetta.core.kinematics import FoldTree
from pyrosetta.rosetta.numeric import xyzVector_double_t
from pyrosetta.rosetta.core.select.residue_selector import ChainSelector, NeighborhoodResidueSelector, \
    ResidueIndexSelector

# PyRosetta vectors
from pyrosetta.rosetta.utility import vector1_std_shared_ptr_const_core_conformation_Residue_t
from pyrosetta.rosetta.numeric import xyzVector_double_t
from pyrosetta.rosetta.utility import vector1_core_id_AtomID
from pyrosetta.rosetta.utility import vector1_numeric_xyzVector_double_t
from pyrosetta.rosetta.utility import vector1_bool

from scipy.spatial.distance import pdist, cdist, squareform

# Biotite import
from biotite.structure.io.pdb import PDBFile
from biotite.structure import sasa, annotate_sse, apply_residue_wise

# BIO Python imports
from Bio.PDB.Polypeptide import one_to_three

# Local imports
import Energy
import Constants
from BaseClases import BaseConstraint, Residue, AminoAcids, DesignDomainWildCards, Ligand
from MiscellaneousUtilityFunctions import killProccesses, getKT, printDEBUG, getScoreFunction


class LigandPackingMover(object):

    def __init__(self, ligand: Ligand = None,
                 pose: Pose = None,
                 nonStandardNeighbors: list = None,
                 neighborDistCutoff: float = 15.0,
                 mainChainCoupling: float = 1.0,
                 sideChainCoupling: float = 0.0,
                 sideChainCouplingMax: float = 1.0,
                 sideChainCouplingExcludedPoseIndex: list = None,
                 sideChainsGridPoints: list = None,
                 sideChainsGridLimit: list = None,
                 maxGrid: int = 12,
                 minGrid: int = 6,
                 gridInterval: float = 360.0,
                 numberOfGridNeighborhood=2,
                 packingLoops: int = 1,
                 minimization: bool = True,
                 fullEnergy=False,
                 nRandomTorsionPurturbation= 1,
                 nproc=1):

        # self.aminoAcids = AminoAcids()
        self.sideChainsGridRanges = None
        self.neighborDistCutoff = None
        self.neighborsPoseIndex = None
        self.neighborsAtomId = vector1_core_id_AtomID()
        self.ligandAtomId = vector1_core_id_AtomID()
        self.neighborsCharges = None
        self.neighborsLjRadii = None
        self.neighborsLjEps = None
        self.neighborsLkdG = None
        self.neighborsLkLam = None
        self.neighborsLkVol = None

        self.self_qij = None
        self.self_dljr = None
        self.self_dljep = None
        self.self_self_sol_prefactor = None
        self.pair_qij = None
        self.pair_dljr = None
        self.pair_dljep = None
        self.pair_sol_prefactor = None

        self.neighborsSideChainFlagVector = None
        self.neighborsBackboneFlagVector = None

        self.neighborsXYZVector1 = vector1_numeric_xyzVector_double_t()
        self.ligandXYZVector1 = vector1_numeric_xyzVector_double_t()

        self.neighborsXYZ = None
        self.ligandXYZ = None

        self.ligandNonBondedXYZIndex = None
        self.ligandNonBondedWeights = None

        self.ligand = None
        self.nonStandardNeighborsDict = None

        self.minimization = minimization
        self.sideChainCoupling = sideChainCoupling
        self.sideChainCouplingMax = sideChainCouplingMax
        self.mainChainCoupling = mainChainCoupling
        self.sideChainCouplingExcludedPoseIndex = sideChainCouplingExcludedPoseIndex

        self.solWeight = 0.035
        self.atrWeight = 1.000
        self.elcWeight = 1.000
        self.repWeight_self_0 = 0.500
        self.repWeight_self_1 = 0.500
        self.repWeight_self_2 = 0.500
        self.repWeight_pair_0 = 0.150
        self.repWeight_pair_1 = 0.300
        self.repWeight_pair_2 = 0.500

        self.bondConstraintForce = 1000
        self.angleConstraintForce = 500

        self.bondXYZIndex = None
        self.bondsLenghts = None
        self.bondConstraints = None

        self.angleConstraints = None

        self._packingLoops = None
        self.packingLoops = packingLoops

        self.fullEnergy = fullEnergy
        self.maxGrid = maxGrid
        self.minGrid = minGrid
        self.gridInterval = gridInterval
        self.numberOfGridNeighborhood = numberOfGridNeighborhood
        self.neighborDistCutoff = neighborDistCutoff

        self.sideChainPackingScore = pr.get_fa_scorefxn()
        self.ligandPackingScore = pr.get_fa_scorefxn()
        self.minimizer = MinMover()
        self.movemap = MoveMap()
        self.minimizationCutoff = 0.1
        self.max_cycle = 100
        self.max_line_search_step = 0.25
        self.nRandomTorsionPurturbation = nRandomTorsionPurturbation
        self.nproc = nproc
        os.environ["OMP_NUM_THREADS"] = str(nproc)

        if ligand:
            self.initiate(ligand, pose, nonStandardNeighbors, neighborDistCutoff, sideChainsGridPoints,
                          sideChainsGridLimit)

    def initiate(self, ligand,
                 pose,
                 nonStandardNeighbors,
                 neighborDistCutoff=15.0,
                 sideChainsGridPoints=None,
                 sideChainsGridLimit=None,
                 sideChainCouplingExcludedPoseIndex=None):

        self.ligand = ligand
        self.neighborDistCutoff = neighborDistCutoff

        # initiate grid ranges for each torsion level of each side chain
        if ligand.numberOfSideChains > 0:
            self.getSideChainsGridRanges(sideChainsGridPoints, sideChainsGridLimit)

        self.sideChainCouplingExcludedPoseIndex = []
        if sideChainCouplingExcludedPoseIndex is not None:
            self.sideChainCouplingExcludedPoseIndex = sideChainCouplingExcludedPoseIndex

        if pose is not None:
            # initiate the score function
            self.sideChainPackingScore(pose)
            self.getLigandPoseIndex(pose)
            self.getNonStandardNeighborsDict(nonStandardNeighbors)
            self.getligandProperties(pose)
            self.getNeighborsPoseIndex(pose)
            self.getNeighborsProperties(pose)
            self.getMininizerMovemap()

    @property
    def packingLoops(self):
        return self._packingLoops

    @packingLoops.setter
    def packingLoops(self, packingLoops):
        self._packingLoops = packingLoops

        # Allocate a list of poses for keeping the results of sidechain packing for each packing loop
        self.packingFinalPoses = list()
        for i in range(self.packingLoops):
            self.packingFinalPoses.append(Pose())

    def getMininizerMovemap(self):
        self.movemap.clear()

        # Add the neighbors
        # for index in self.neighborsPoseIndex:
        #    if self.backboneMinimization:
        #        self.movemap.set_bb(index)
        #    self.movemap.set_chi(index)

        # Add the ligand
        self.movemap.set_bb(self.ligand.poseIndex)
        self.movemap.set_chi(self.ligand.poseIndex)

        self.minimizer.set_movemap(self.movemap)
        self.minimizer.score_function(self.ligandPackingScore)

    def getSideChainsGridRanges(self, sideChainsGridPoints=None, sideChainsGridLimit=None):
        """
        sideChainsGridPoints is a dictionaries that contains chi index as key and a tuple containing
        number of grid and interval (n, I) as values, e.g.
           {ChiIndex: (n,I), ChiIndex: (n,I), ChiIndex: (n,I)}

        sideChainsGridLimit is a list of tuple for each side chain containing (MaxGrid, MinGrid, Interval)
           [(Max, Min, Interval),
           [(Max, Min, Interval),
           [(Max, Min, Interval)]

        If neither is given, maxGrid, minGrid, gridInterval is used for all side chains.
        """

        # Check the sideChainsGridLimit
        if sideChainsGridLimit is not None:
            if len(sideChainsGridLimit) != self.ligand.numberOfSideChains:
                raise ValueError(
                    'Bad sideChainsGridLimit, Ligand {}{} contains {} side chains, {} GridLimits are given'.
                    format(self.ligand.chain, self.ligand.ID, self.ligand.numberOfSideChains,
                           len(sideChainsGridPoints)))

        # for each side chain compute a gridRangeMatrix.
        self.sideChainsGridRanges = []
        self.sideChainsRandomInterval = []
        for sideChainIndex in range(self.ligand.numberOfSideChains):
            if sideChainsGridLimit is not None:
                sideChainGridLimit = sideChainsGridLimit[sideChainIndex]
            else:
                sideChainGridLimit = None

            self.getGridRanges(sideChainIndex, sideChainsGridPoints, sideChainGridLimit)

        # print(self.sideChainsGridRanges)

    def getGridRanges(self, sideChainIndex, sideChainGridPoints=None, sideChainGridLimit=None):
        """
        Initiates the grid Ranges
        :param sideChainIndex: index of ligand side chain
        :param sideChainGridPoints: list of tup
        :param gridsPoints: list of grid points for torsion level
        """
        # self.gridRanges = list()
        # MaxDepth = 10

        # Sort out limit if given.
        if sideChainGridLimit is None:
            maxGrid = self.maxGrid
            minGrid = self.minGrid
            gridInterval = self.gridInterval

        else:
            maxGrid = sideChainGridLimit[0]
            minGrid = sideChainGridLimit[1]
            gridInterval = sideChainGridLimit[2]

        # Make sure all gridPoints should be even
        maxGrid += maxGrid % 2
        minGrid += minGrid % 2

        # Generate the side chain grid ranges
        gridRanges = []
        randomInterval = []
        currentGrid = maxGrid
        for levelIndex, level in enumerate(self.ligand.sideChainsOrderedIndex[sideChainIndex]):
            for torsionIndex, torsionPoseIndex in enumerate(level):
                result = []
                # Read if given explicitly
                if sideChainGridPoints is not None and torsionPoseIndex in sideChainGridPoints.keys():
                    gridpoint, interval = sideChainGridPoints[torsionPoseIndex]
                # Or set by limits
                else:
                    gridpoint, interval = currentGrid, gridInterval

                # Set the random interval torsions if interval number is given
                if interval < 0:
                    # save the interval number
                    randomInterval.append(-1 * interval)

                    # convert interval number to interval in degrees
                    interval = -360/interval
                else:
                    randomInterval.append(0)

                # Get the grid range
                self._getGridRangeRecursivly(gridpoint, interval, result)
                gridRanges.append(result)

                # Update current grid for each level
                currentGrid -= 2
                if currentGrid < minGrid:
                    currentGrid = minGrid

        # Convert it to a list of arrays
        sideChainGridRanges = []
        gridRanges = array(gridRanges, dtype=object).transpose()
        for packingLevel in gridRanges:
            sideChainGridRanges.append([])
            for gridRange in packingLevel:
                sideChainGridRanges[-1].append(array(gridRange))

        # print('BBBB', sideChainGridRanges)
        # for i in sideChainGridRanges:
        #    for j in i:
        #        print(len(j), ['{:.1f}'.format(k) for k in j])
        self.sideChainsGridRanges.append(sideChainGridRanges)
        self.sideChainsRandomInterval.append(array(randomInterval, dtype=np.int))

    def _getGridRangeRecursivly(self, numberOfGridPoints, interval=360.0, result=[], level=0):
        # Max minimization level
        MaxLevel = 2
        spacing = interval / numberOfGridPoints

        if level == 0:  # The first level interval corresponds to 360 degrees, hence 180 == -180
            lowerBound = -int(numberOfGridPoints / 2)
            higherBound = int(numberOfGridPoints / 2)
        else:
            lowerBound = -int(numberOfGridPoints / 2)
            higherBound = int(numberOfGridPoints / 2) + 1

        result.append([spacing * i for i in range(lowerBound, higherBound)])
        if level < MaxLevel:
            level += 1
            numberOfGridPoints = int(numberOfGridPoints / 1.5)
            self._getGridRangeRecursivly(numberOfGridPoints, 2 * spacing, result, level)

    def getSideChainGrid(self, sideChainIndex, minimizationLevel, torsionCenters=None):

        sideChainChiIndices = []
        sideChainGrid = None
        gridRanges = []
        # print('torsionCenters', torsionCenters)
        # print('sideChainOrderedIndex', sideChainOrderedIndex)

        torsionIndex = 0
        for levelIndex, level in enumerate(self.ligand.sideChainsOrderedIndex[sideChainIndex]):
            for torsionLevelIndex, torsionPoseIndex in enumerate(level):
                # Keep indices in order
                sideChainChiIndices.append(torsionPoseIndex)
                # get the center of grid range
                if torsionCenters is not None:
                    center = torsionCenters[torsionIndex]
                else:
                    center = 0.0
                # print('****center', center, torsionIndex)
                gridRanges.append(self.sideChainsGridRanges[sideChainIndex][minimizationLevel][torsionIndex] + center)
                torsionIndex += 1

        # wrap the angles
        for gridRangeIndex in range(len(gridRanges)):
            for angleIndex in range(len(gridRanges[gridRangeIndex])):
                angle = gridRanges[gridRangeIndex][angleIndex]
                if angle >= 180:
                    gridRanges[gridRangeIndex][angleIndex] = angle - 360
                elif angle < -180:
                    gridRanges[gridRangeIndex][angleIndex] = angle + 360
            # print(minimizationLevel, gridRanges[gridRangeIndex])

        # Add one more column for energies
        gridRanges.append([0.0])

        # print(gridRanges)
        sideChainGrid = array(np.meshgrid(*gridRanges), dtype=np.float).T.reshape(-1, len(gridRanges))

        # print(gridRanges)
        # print(sideChainGrid, len(sideChainGrid))

        return array(sideChainChiIndices, dtype=np.intc), sideChainGrid

    def updateProperties(self, pose):
        self.getLigandPoseIndex(pose)
        self.getNeighborsPoseIndex(pose)

        self.getligandProperties(pose)
        self.getNeighborsProperties(pose)

    def apply(self, pose, sideChainIndex=None):
        # Check n side chains

        self.updateProperties(pose)
        self.getLigandXYZ(pose)
        self.getNeighborsXYZ(pose)

        if sideChainIndex is None:
            for i in range(self.packingLoops):
                # it packs the side chains on the same structure an alternative approach
                # would be to keep the starting pose and repack starting with that !!!
                for sideChainIndex in sample(range(self.ligand.numberOfSideChains), self.ligand.numberOfSideChains):
                    #print("BBB Activating min and loop: ", i, " sc", sideChainIndex)
                    self.minimumEnergyNeighborhoodSearch(pose, sideChainIndex)
                self.packingFinalPoses[i].assign(pose)

            # minimize ang the the energies
            poseFinalEnergies = list()
            # self.getMininizerMovemap()
            # counter = 0
            for packingFinalPose in self.packingFinalPoses:
                if self.minimization:
                    # packingFinalPose.dump_pdb('transformed-min1.pdb')
                    # E_min1 = self.ligandPackingScore(packingFinalPose)
                    self.ligandMinimization(packingFinalPose)
                    # packingFinalPose.dump_pdb('transformed-min2.pdb')
                    # E_min2 = self.ligandPackingScore(packingFinalPose)
                    # print('E_min1: ', E_min1, 'E_min2: ', E_min2)

                poseFinalEnergies.append(self.ligandPackingScore(packingFinalPose))

            # assign the best pose
            index = np.argmin(poseFinalEnergies)
            pose.assign(self.packingFinalPoses[index])

        else:
            self.minimumEnergyNeighborhoodSearch(pose, sideChainIndex)
            if self.minimization:
                self.ligandMinimization(pose)

    def ligandMinimization(self, pose):
        """
        Minimizes using the reduced potential
        :param pose:
        """
        self.updateProperties(pose)
        self.getLigandXYZ(pose)
        self.getNeighborsXYZ(pose)

        weights_self = array([self.repWeight_self_2, self.atrWeight, self.elcWeight, self.solWeight])
        weights_pair = array([self.repWeight_pair_2, self.atrWeight, self.elcWeight, self.solWeight])
        # print('xyz IN', self.ligandXYZ)

        if len(self.neighborsXYZ) != 0:
            Energy.ligand_steep_decent(self.ligandXYZ,
                                       self.neighborsXYZ,
                                       self.self_qij,
                                       self.self_dljr,
                                       self.self_dljep,
                                       self.self_self_sol_prefactor,
                                       self.ligand.lkLam,
                                       self.ligandNonBondedXYZIndex,
                                       self.ligandNonBondedWeights,
                                       self.mainChainCoupling,
                                       weights_self,

                                       self.bondXYZIndex,
                                       self.bondsLenghts,
                                       self.bondConstraints,

                                       self.angleXYZIndex,
                                       self.anglesLenghts,
                                       self.angleConstraints,

                                       self.ligand.ligandMasks,
                                       self.pair_qij,
                                       self.pair_dljr,
                                       self.pair_dljep,
                                       self.pair_sol_prefactor,
                                       self.neighborsLkLam,
                                       self.neighborsSideChainFlagVector,
                                       self.neighborsBackboneFlagVector,
                                       self.sideChainCoupling,
                                       self.mainChainCoupling,
                                       weights_pair,
                                       self.minimizationCutoff,
                                       self.max_cycle,
                                       self.max_line_search_step)
        # print('xyz out', self.ligandXYZ)
        self.setLigandXYZ(pose)

    def minimumEnergyNeighborhoodSearch(self, pose, sideChainIndex):
        """
        Performs ligand side chain packing with minimum neighbor search
        :param pose:
        """
        # Initiate torsion with either by their current center from pose or from a random interval if it is set
        # The torsion centers has one extra column
        #print(self.sideChainsRandomInterval[sideChainIndex].nonzero())
        sideChainRandomIntervalIndices = list(self.sideChainsRandomInterval[sideChainIndex].nonzero()[0])

        if self.nRandomTorsionPurturbation > len(sideChainRandomIntervalIndices):
            sampleSize = len(sideChainRandomIntervalIndices)
        else:
            sampleSize = self.nRandomTorsionPurturbation

        selectedSideChainRandomIntervalIndices = sample(sideChainRandomIntervalIndices, sampleSize)

        sideChainCenters = []
        ChiIndex = 0
        for levelIndex, level in enumerate(self.ligand.sideChainsOrderedIndex[sideChainIndex]):
            for torsionLevelIndex, torsionPoseIndex in enumerate(level):
                if ChiIndex in selectedSideChainRandomIntervalIndices:

                    # Choose a random interval in the neighborhood of current angle
                    numberOfInterval = self.sideChainsRandomInterval[sideChainIndex][ChiIndex]
                    interval = 360/numberOfInterval
                    randomIntervalNumber = np.random.randint(-1, 2)

                    # get current torsion center
                    torsionCenter = pose.chi(torsionPoseIndex, self.ligand.poseIndex)

                    # Move the current center to the neighboring interval (-1, 1), or don't change (0)
                    # for angle<0 forward direction is counter clock wise
                    if torsionCenter < 0:
                        torsionCenter = torsionCenter - (randomIntervalNumber * interval)
                    # for angle>=0 forward direction is clock wise
                    elif torsionCenter >= 0:
                        torsionCenter = torsionCenter + (randomIntervalNumber * interval)

                    #print("BBBB", torsionCenter, randomIntervalNumber, ChiIndex)
                    if torsionCenter >= 180:
                        torsionCenter = torsionCenter - 360
                    elif torsionCenter < -180:
                        torsionCenter = torsionCenter + 360

                else: # Choose the current value as the torsion center
                    torsionCenter = pose.chi(torsionPoseIndex, self.ligand.poseIndex)

                sideChainCenters.append(pose.chi(torsionPoseIndex, self.ligand.poseIndex))
                ChiIndex += 1

        sideChainCenters.append(0)
        # print("Docking line 493 in minimumEnergyNeighborhoodSearch: ", sideChainCenters)
        # torsionCenters_curt should be list of lists for neighbourhood search
        torsionCenters_curt = [sideChainCenters]
        #sideChainMask = self.ligand.sideChainMasks[sideChainIndex]

        for minimizationLevel in range(3):
            torsionCenters_next = []

            ####################################
            # print('minimizationLevel', minimizationLevel)
            # for torsion in torsionCenters_curt:
            #    print(['{:.1f}'.format(i) for i in torsion])
            ####################################

            for torsionCenter in torsionCenters_curt:
                # get grids, ignore the energy column (last one)
                chis, grid = self.getSideChainGrid(sideChainIndex, minimizationLevel, torsionCenter[:-1])

                if self.fullEnergy:
                    if self.nproc > 1: # and grid.shape[0] > 5500:
                        torsionCenters_next.extend(self._gridSearch_parallel_fullEnergy(pose, sideChainIndex, chis, grid, minimizationLevel))
                    else:
                        torsionCenters_next.extend(self._gridSearch_fullEnergy(pose, sideChainIndex, chis, grid, minimizationLevel))

                else :
                    if self.nproc > 1:
                        torsionCenters_next.extend(self._gridSearch_reduced_parallel(pose, sideChainIndex, chis, grid, minimizationLevel))
                        #torsionCenters_next.extend(self._gridSearch_parallel(pose, sideChainIndex, chis, grid, minimizationLevel))
                    else:
                        torsionCenters_next.extend(self._gridSearch_reduced(pose, sideChainIndex, chis, grid, minimizationLevel))
                        #torsionCenters_next.extend(self._gridSearch(pose, sideChainIndex, chis, grid, minimizationLevel))

            torsionCenters_curt = torsionCenters_next

        ####################################
        # print('minimizationLevel   Final')
        # for torsion in torsionCenters_curt:
        #    print(['{:.1f}'.format(i) for i in torsion])
        ####################################

        # Sort by energy torsionCenters are python list here.
        torsionCenters_curt.sort(key=lambda element: element[-1])

        # for i, center in enumerate(torsionCenters_curt):
        #    for j in range(len(center)-1):
        #        pose.set_chi(chis[j], self.ligand.poseIndex, center[j])
        #    energy = self.score(pose)
        # pose.dump_pdb('pack_cluster_{}_{}_{}.pdb'.format(minimizationLevel, i, energy))
        # print('{}  {} {:.1f}'.format(minimizationLevel, str(['{:.1f}'.format(i) for i in center]), energy))

        center = torsionCenters_curt[0]

        # Set the pose according to the best result.
        for j in range(len(center) - 1):
            pose.set_chi(chis[j], self.ligand.poseIndex, center[j])

    def _gridSearch_reduced(self, pose, sideChainIndex, chis, grid, minimizationLevel):
        """
        Performs a grid search with EDesign energy function
        :param chis:
        :param grid:
        :param minimizationLevel:
        :return:
        """
        # update weights
        if minimizationLevel == 0:
            weights_self = array([self.repWeight_self_0, self.atrWeight, self.elcWeight, self.solWeight])
            weights_pair = array([self.repWeight_pair_0, self.atrWeight, self.elcWeight, self.solWeight])
        elif minimizationLevel == 1:
            weights_self = array([self.repWeight_self_1, self.atrWeight, self.elcWeight, self.solWeight])
            weights_pair = array([self.repWeight_pair_1, self.atrWeight, self.elcWeight, self.solWeight])
        elif minimizationLevel == 2:
            weights_self = array([self.repWeight_self_2, self.atrWeight, self.elcWeight, self.solWeight])
            weights_pair = array([self.repWeight_pair_2, self.atrWeight, self.elcWeight, self.solWeight])
        else:
            raise ValueError('Max 3 minimum energy neighborhood level is allowed.')

        self.getLigandXYZ(pose)
        #print('BBBBB ->>>>>', grid.shape)
        Energy.grid_search(grid,
                           self.ligandXYZ,
                           self.neighborsXYZ,

                           self.self_qij,
                           self.self_dljr,
                           self.self_dljep,
                           self.self_self_sol_prefactor,
                           self.ligand.lkLam,
                           self.ligandNonBondedXYZIndex,
                           self.ligandNonBondedWeights,
                           self.mainChainCoupling,
                           weights_self,

                           self.ligand.sideChainMasks[sideChainIndex],
                           self.ligand.sideChainsChiXYZIndices[sideChainIndex],
                           self.ligand.sideChainsSubTreeMasks[sideChainIndex],

                           self.pair_qij,
                           self.pair_dljr,
                           self.pair_dljep,
                           self.pair_sol_prefactor,
                           self.neighborsLkLam,
                           self.neighborsSideChainFlagVector,
                           self.neighborsBackboneFlagVector,
                           self.sideChainCoupling,
                           self.mainChainCoupling,
                           weights_pair)

            # print([i for i in grid[i]])
        grid = grid[grid[:, -1].argsort()]

        nGridstoCluster = 50
        clusters = Energy.select_torsions_by_distance(grid[:nGridstoCluster, :], K=self.numberOfGridNeighborhood,
                                                      cutoff_percent=0.02)

        #print('BBBB ----E', clusters[0, :], minimizationLevel)
        if minimizationLevel == 0:
            return clusters[:, :]
        else:
            return [clusters[0, :]]
        # print(torsionCenters)

    def _gridSearch_reduced_parallel(self, pose, sideChainIndex, chis, grid, minimizationLevel):
        """
        Performs a grid search with EDesign energy function
        :param chis:
        :param grid:
        :param minimizationLevel:
        :return:
        """
        # update weights
        if minimizationLevel == 0:
            weights_self = array([self.repWeight_self_0, self.atrWeight, self.elcWeight, self.solWeight])
            weights_pair = array([self.repWeight_pair_0, self.atrWeight, self.elcWeight, self.solWeight])
        elif minimizationLevel == 1:
            weights_self = array([self.repWeight_self_1, self.atrWeight, self.elcWeight, self.solWeight])
            weights_pair = array([self.repWeight_pair_1, self.atrWeight, self.elcWeight, self.solWeight])
        elif minimizationLevel == 2:
            weights_self = array([self.repWeight_self_2, self.atrWeight, self.elcWeight, self.solWeight])
            weights_pair = array([self.repWeight_pair_2, self.atrWeight, self.elcWeight, self.solWeight])
        else:
            raise ValueError('Max 3 minimum energy neighborhood level is allowed.')

        self.getLigandXYZ(pose)
        #print('BBBBB ->>>>>', grid.shape)
        Energy.grid_search_parallel(grid,
                           self.ligandXYZ,
                           self.neighborsXYZ,

                           self.self_qij,
                           self.self_dljr,
                           self.self_dljep,
                           self.self_self_sol_prefactor,
                           self.ligand.lkLam,
                           self.ligandNonBondedXYZIndex,
                           self.ligandNonBondedWeights,
                           self.mainChainCoupling,
                           weights_self,

                           self.ligand.sideChainMasks[sideChainIndex],
                           self.ligand.sideChainsChiXYZIndices[sideChainIndex],
                           self.ligand.sideChainsSubTreeMasks[sideChainIndex],

                           self.pair_qij,
                           self.pair_dljr,
                           self.pair_dljep,
                           self.pair_sol_prefactor,
                           self.neighborsLkLam,
                           self.neighborsSideChainFlagVector,
                           self.neighborsBackboneFlagVector,
                           self.sideChainCoupling,
                           self.mainChainCoupling,
                           weights_pair,
                           self.nproc)
        #print('--------------------------------', minimizationLevel)
            # print([i for i in grid[i]])
        grid = grid[grid[:, -1].argsort()]

        nGridstoCluster = 50
        clusters = Energy.select_torsions_by_distance(grid[:nGridstoCluster, :], K=self.numberOfGridNeighborhood,
                                                      cutoff_percent=0.02)

        #print('BBBB ----E', clusters[0, :], minimizationLevel)
        if minimizationLevel == 0:
            return clusters[:, :]
        else:
            return [clusters[0, :]]
        # print(torsionCenters)

    def _gridSearch_fullEnergy(self, pose, sideChainIndex, chis, grid, minimizationLevel):
        """
        Performs a grid search with EDesign energy function
        :param chis:
        :param grid:
        :param minimizationLevel:
        :return:
        """
        # update weights
        if minimizationLevel == 0:
            weights_self = array([self.repWeight_self_0, self.atrWeight, self.elcWeight, self.solWeight])
            weights_pair = array([self.repWeight_pair_0, self.atrWeight, self.elcWeight, self.solWeight])
        elif minimizationLevel == 1:
            weights_self = array([self.repWeight_self_1, self.atrWeight, self.elcWeight, self.solWeight])
            weights_pair = array([self.repWeight_pair_1, self.atrWeight, self.elcWeight, self.solWeight])
        elif minimizationLevel == 2:
            weights_self = array([self.repWeight_self_2, self.atrWeight, self.elcWeight, self.solWeight])
            weights_pair = array([self.repWeight_pair_2, self.atrWeight, self.elcWeight, self.solWeight])
        else:
            raise ValueError('Max 3 minimum energy neighborhood level is allowed.')

        for i in range(grid.shape[0]):
            energy = 0.0

            # generate the pose
            for j in range(len(chis)):
                # print(chis[j], grid[i, j])
                pose.set_chi(chis[j], self.ligand.poseIndex, grid[i, j])
                #if i == 0:
                #    print(grid[i, j], '==', pose.chi(chis[j], self.ligand.poseIndex))

            self.sideChainPackingScore.set_weight(ScoreType.fa_rep, weights_pair[0])
            energy = self.sideChainPackingScore(pose)

            grid[i, -1] = energy

        #print('BBBB ----E', grid[0, :])
        grid = grid[grid[:, -1].argsort()]
        nGridstoCluster = 50
        clusters = Energy.select_torsions_by_distance(grid[:nGridstoCluster, :], K=self.numberOfGridNeighborhood,
                                                      cutoff_percent=0.02)

        #print('BBBB ----E', clusters[0, :])
        if minimizationLevel == 0:
            return clusters[:, :]
        else:
            return [clusters[0, :]]
        # print(torsionCenters)

    def _gridSearch_parallel_fullEnergy(self, pose, sideChainIndex, chis, grid, minimizationLevel):
        """
        Performs a grid search with EDesign energy function
        :param chis:
        :param grid:
        :param minimizationLevel:
        :return:
        """

        # update weights
        if minimizationLevel == 0:
            weights_self = array([self.repWeight_self_0, self.atrWeight, self.elcWeight, self.solWeight])
            weights_pair = array([self.repWeight_pair_0, self.atrWeight, self.elcWeight, self.solWeight])
        elif minimizationLevel == 1:
            weights_self = array([self.repWeight_self_1, self.atrWeight, self.elcWeight, self.solWeight])
            weights_pair = array([self.repWeight_pair_1, self.atrWeight, self.elcWeight, self.solWeight])
        elif minimizationLevel == 2:
            weights_self = array([self.repWeight_self_2, self.atrWeight, self.elcWeight, self.solWeight])
            weights_pair = array([self.repWeight_pair_2, self.atrWeight, self.elcWeight, self.solWeight])
        else:
            raise ValueError('Max 3 minimum energy neighborhood level is allowed.')

        args = list()
        #self.nproc
        step = int(grid.shape[0] / self.nproc)
        lb = 0
        for i in range(self.nproc):
            if i == self.nproc - 1:  # end partition
                hb = grid.shape[0]
            else:
                hb = i * step + step
            args.append((pose, self.ligand.poseIndex, chis, grid[lb: hb], weights_pair))
            lb = hb

        # print(type(chis), type(grid), type(sideChainCoupling), type(mainChainCoupling))
        with Pool(self.nproc) as pool:
            energies = pool.starmap(LigandPackingMover._gridPointEvaluation_fullEnergy, args)

        grid[:, -1] = [energy for chunk in energies for energy in chunk]
        grid = grid[grid[:, -1].argsort()]

        nGridstoCluster = 50
        clusters = Energy.select_torsions_by_distance(grid[:nGridstoCluster, :], K=self.numberOfGridNeighborhood,
                                                      cutoff_percent=0.02)
        if minimizationLevel == 0:
            return clusters[:, :]
        else:
            return [clusters[0, :]]
        # print(torsionCenters)

    @staticmethod
    def _gridPointEvaluation_fullEnergy(pose, ligandPoseIndex, chis, grid, weights_pair):
        energies = list()
        score = pr.get_fa_scorefxn()
        for i in range(grid.shape[0]):

            for j in range(len(chis)):
                pose.set_chi(chis[j], ligandPoseIndex, grid[i, j])

            score.set_weight(ScoreType.fa_rep, weights_pair[0])
            energy = score(pose)

            energies.append(energy)
        return energies

    def getNeighborsXYZ(self, pose):
        """
        Get the coordinate of neighbor residues
        :param pose:
        """
        self.neighborsXYZVector1.clear()
        pose.batch_get_xyz(self.neighborsAtomId, self.neighborsXYZVector1)
        self.neighborsXYZ = array([[atom.x, atom.y, atom.z] for atom in self.neighborsXYZVector1])

    def getLigandXYZ(self, pose):
        """
        Get the coordinate of neighbor residues
        :param pose:
        """
        self.ligandXYZVector1.clear()
        pose.batch_get_xyz(self.ligandAtomId, self.ligandXYZVector1)
        self.ligandXYZ = array([[atom.x, atom.y, atom.z] for atom in self.ligandXYZVector1])

    def setLigandXYZ(self, pose):

        self.ligandXYZVector1.clear()
        for ri in self.ligandXYZ:
            self.ligandXYZVector1.append(xyzVector_double_t(*ri))

        pose.batch_set_xyz(self.ligandAtomId, self.ligandXYZVector1)

    def getLigandXYZ_direct(self, pose):
        """
        Get the coordinate of neighbor residues
        :param pose:
        """
        ligandRes = pose.residue(self.ligand.poseIndex)
        self.ligandXYZ = array(
            [[ligandRes.xyz(i).x, ligandRes.xyz(i).y, ligandRes.xyz(i).z] for i in range(1, ligandRes.natoms() + 1)])

    def getLigandPoseIndex(self, pose):
        self.ligand.poseIndex = pose.pdb_info().pdb2pose(self.ligand.chain, self.ligand.ID)

    def getligandProperties(self, pose):
        """
        Computes the atom ID vector of the ligand. Other properties
        are already precomputed in the ligand object.
        :param pose:
        """

        self.getLigandPoseIndex(pose)

        self.ligandAtomId.clear()

        # Get atom id vector
        # couter = 0
        for atomIndex in self.ligand.atoms:
            self.ligandAtomId.append(AtomID(atomIndex, self.ligand.poseIndex))
            # print('BBB ligand atom name', couter, pose.residue(self.ligand.poseIndex).atom_name(atomIndex))
            # couter += 1

        # get non bonded indices
        self.ligandNonBondedXYZIndex = array(self.ligand.nonBondedAtomsPairs, dtype=np.int32) - 1
        self.ligandNonBondedWeights = array(self.ligand.nonBondedAtomsPairsWeights, dtype=np.float64)

        # get bond indices, equilibrium length
        self.bondXYZIndex = array(self.ligand.bonds, dtype=np.int32) - 1
        self.bondsLenghts = array(self.ligand.bondsLengths, dtype=np.float64)
        self.bondConstraints = ones(len(self.ligand.bonds)) * self.bondConstraintForce

        # get bond indices, equilibrium length
        self.angleXYZIndex = array(self.ligand.angles, dtype=np.int32) - 1
        self.anglesLenghts = array(self.ligand.anglesLengths, dtype=np.float64)
        self.angleConstraints = ones(len(self.ligand.angles)) * self.angleConstraintForce

        dimension = self.ligandNonBondedXYZIndex.shape[0]

        self.self_qij = zeros(dimension)
        self.self_dljr = zeros(dimension)
        self.self_dljep = zeros(dimension)
        self.self_self_sol_prefactor = zeros((dimension, 2))

        # precompute the energy matrices
        Energy.ligand_ligand_matrices(self.ligand.charges, self.self_qij,
                                      self.ligand.ljRadii, self.self_dljr,
                                      self.ligand.ljEps, self.self_dljep,
                                      self.ligand.lkdG, self.ligand.lkLam, self.ligand.lkVol,
                                      self.self_self_sol_prefactor,
                                      self.ligandNonBondedXYZIndex)

    def getNeighborsProperties(self, pose):
        """
        Compute the Atom ID, charges and Vdw vectors for neighbors and neighbors backbone. These
        are needed for batch coordinate retrieval
        :param pose:
        """
        # clean the previous ones
        self.neighborsAtomId.clear()
        self.neighborsCharges = list()
        self.neighborsLjRadii = list()
        self.neighborsLjEps = list()
        self.neighborsLkdG = list()
        self.neighborsLkLam = list()
        self.neighborsLkVol = list()
        self.neighborsSideChainFlagVector = list()
        self.neighborsBackboneFlagVector = list()
        self.neighborsNumberOfAtoms = 0
        # self.neighborsElements = list()

        # Reassign new atom ids
        # couter = 0
        for resIndex in self.neighborsPoseIndex:
            res = pose.residue(resIndex)
            id, chain = pose.pdb_info().pose2pdb(resIndex).split()
            resName = (int(id), chain)
            # Construct neighbors vectors
            for atomIndex in range(1, res.natoms() + 1):

                # Skip virtual atoms
                if res.atom_type(atomIndex).is_virtual():
                    continue
                # print('BBB enviromet atom name', couter, pose.pdb_info().pose2pdb(resIndex), pose.residue(resIndex).atom_name(atomIndex))
                # couter += 1

                self.neighborsNumberOfAtoms += 1

                # Get the atomIDs
                self.neighborsAtomId.append(AtomID(atomIndex, resIndex))

                # get charges
                self.neighborsCharges.append(res.atomic_charge(atomIndex))

                # get Vdw
                # vdw = self.aminoAcids.VdwRadii[res.atom_type(atomIndex).element()]
                self.neighborsLjRadii.append(res.atom_type(atomIndex).lj_radius())
                self.neighborsLjEps.append(res.atom_type(atomIndex).lj_wdepth())

                # get solvation parameters
                self.neighborsLkdG.append(res.atom_type(atomIndex).lk_dgfree())
                self.neighborsLkLam.append(res.atom_type(atomIndex).lk_lambda())
                self.neighborsLkVol.append(res.atom_type(atomIndex).lk_volume())

                # Get the side chain/backbone flag,
                # first check if the residue is excluded from side chain coupling.
                if resIndex in self.sideChainCouplingExcludedPoseIndex:
                    self.neighborsSideChainFlagVector.append(0)
                    self.neighborsBackboneFlagVector.append(1)

                # print('BBBB', res.atom_name(atomIndex), res.atom_is_backbone(atomIndex))
                # if the residue poseIndex is in the dict use the ligand coreAtom definition
                elif resName in self.nonStandardNeighborsDict.keys():
                    # print('BBBB in coreatom')
                    nonStandardNeighborCoreAtom = self.nonStandardNeighborsDict.get(resName, [])
                    if atomIndex in nonStandardNeighborCoreAtom:
                        # print('BBBBB, backbone, ligand')
                        self.neighborsSideChainFlagVector.append(0)
                        self.neighborsBackboneFlagVector.append(1)
                    else:
                        self.neighborsSideChainFlagVector.append(1)
                        self.neighborsBackboneFlagVector.append(0)

                # metals are treated as backbone
                elif res.is_metal():
                    # print('BBBBB, backbone metal')
                    self.neighborsSideChainFlagVector.append(0)
                    self.neighborsBackboneFlagVector.append(1)

                # otherwise use library definition
                else:
                    # print('BBBB in else')
                    if res.atom_is_backbone(atomIndex):
                        # print('BBBBB, backbone other')
                        self.neighborsSideChainFlagVector.append(0)
                        self.neighborsBackboneFlagVector.append(1)
                    else:
                        self.neighborsSideChainFlagVector.append(1)
                        self.neighborsBackboneFlagVector.append(0)

        # save in correct format
        self.neighborsCharges = array(self.neighborsCharges)
        self.neighborsLjRadii = array(self.neighborsLjRadii)
        self.neighborsLjEps = array(self.neighborsLjEps)
        self.neighborsLkdG = array(self.neighborsLkdG)
        self.neighborsLkLam = array(self.neighborsLkLam)
        self.neighborsLkVol = array(self.neighborsLkVol)
        self.neighborsSideChainFlagVector = array(self.neighborsSideChainFlagVector, dtype=np.int32)
        self.neighborsBackboneFlagVector = array(self.neighborsBackboneFlagVector, dtype=np.int32)

        # print('BBBBB')
        # for i in range(self.neighborsCharges.shape[0]):
        #    print(i, self.neighborsBackboneFlagVector[i], self.neighborsSideChainFlagVector[i], self.neighborsCharges[i])

        dimension = (self.ligand.numberOfAtoms, self.neighborsNumberOfAtoms)
        self.pair_qij = np.zeros(dimension)
        self.pair_dljr = np.zeros(dimension)
        self.pair_dljep = np.zeros(dimension)

        dimension = (self.ligand.numberOfAtoms, self.neighborsNumberOfAtoms, 2)
        self.pair_sol_prefactor = np.zeros(dimension)

        # precompute the energy matrices
        Energy.ligand_environment_matrices(self.ligand.charges, self.neighborsCharges, self.pair_qij,
                                           self.ligand.ljRadii, self.neighborsLjRadii, self.pair_dljr,
                                           self.ligand.ljEps, self.neighborsLjEps, self.pair_dljep,
                                           self.ligand.lkdG, self.neighborsLkdG,
                                           self.ligand.lkLam, self.neighborsLkLam,
                                           self.ligand.lkVol, self.neighborsLkVol,
                                           self.pair_sol_prefactor)

    def getNeighborsPoseIndex(self, pose):
        pose.update_residue_neighbors()
        ligandSelector = ResidueIndexSelector(self.ligand.poseIndex)
        nbr_selector = NeighborhoodResidueSelector(ligandSelector, self.neighborDistCutoff, False)
        nbrBoolVector = nbr_selector.apply(pose)
        nbrBoolVector = array(nbrBoolVector, dtype=bool)
        self.neighborsPoseIndex = list(where(nbrBoolVector)[0] + 1)

    def getNonStandardNeighborsDict(self, nonStandardNeighbors):
        """
        Creates a dict
        :param nonStandardNeighbors: list of Ligand object
        """
        self.nonStandardNeighborsDict = dict()
        if nonStandardNeighbors is not None:
            for ligand in nonStandardNeighbors:
                self.nonStandardNeighborsDict[ligand.name] = ligand.coreAtoms

    def addNonStandardNeighborDict(self, nonStandardNeighbor):
        """
        Creates a dict
        :param nonStandardNeighbor: Ligand object
        """
        # Create a dictionay if it does not exist
        if not self.nonStandardNeighborsDict:
            self.nonStandardNeighborsDict = dict()

        if nonStandardNeighbor is not None:
            self.nonStandardNeighborsDict[nonStandardNeighbor.name] = nonStandardNeighbor.coreAtoms

    def show(self):
        string = ''
        string = '---------------------------------------Ligand(s) grids------------------------------------------\n'
        string += 'Ligand: {}\n'.format(str(self.ligand.name))
        string += '     Side Chain Coupling: {}\n'.format(self.sideChainCoupling)
        string += '     Main Chain Coupling: {}\n'.format(self.mainChainCoupling)
        if not self.sideChainsGridRanges is None:
            for i, j in enumerate(self.sideChainsGridRanges):
                string += '     Side Chain: {}\n'.format(i)
                for l, m in enumerate(j):
                    string += '         Min Level: {}\n'.format(l)
                    for s, t in enumerate(m):
                        string += '             Chi {}: Random Interval: {} Range: {}\n'.format(s, self.sideChainsRandomInterval[i][s], list(t))

        string += '------------------------------------------------------------------------------------------------\n'
        return string


class LigandNeighborsPackingMover(object):
    """A Simple class that repack the protein side chains during docking
    """

    def __init__(self, ligand: Ligand = None,
                 excludedResiduePoseIndex: list = None,
                 neighborDistCutoff: float = 15.0,
                 scratch=''):

        # Initiate Rosetta classes
        self.taskFactory = TaskFactory()
        self.scorefxn = getScoreFunction(mode='fullAtomWithConstraints')

        self.packer = PackRotamersMover()
        self.packer.score_function(self.scorefxn)

        self.neighborDistCutoff = neighborDistCutoff
        self.excludedResiduePoseIndex = excludedResiduePoseIndex
        self.neighborsPoseIndex = None

        if ligand:
            self.initiate(ligand, excludedResiduePoseIndex, neighborDistCutoff, scratch)

    def initiate(self, ligand: Ligand, excludedResiduePoseIndex: list = None, neighborDistCutoff: float = 15.0,
                 scratch=''):
        self.ligand = ligand
        self.neighborDistCutoff = neighborDistCutoff
        self.proccessName = '{}-{}.resfile'.format(MPI.COMM_WORLD.Get_rank(), os.getpid())
        self.resfileName = os.path.join(scratch, self.proccessName)
        self.excludedResiduePoseIndex = excludedResiduePoseIndex

    def writeResfile(self, pose, resfileName=''):
        if self.neighborsPoseIndex is None:
            raise ValueError('Failed pack side chains in ligand docking. No reside pose index is found.')

        resfileString = ''
        with open(resfileName, 'w') as resfile:
            resfile.write('NATRO\n')
            resfile.write('START\n')
            for poseIndex in self.neighborsPoseIndex:
                # skip the excluded ones
                if poseIndex in self.excludedResiduePoseIndex:
                    continue
                # get the pdb info and write
                ID, Chain = pose.pdb_info().pose2pdb(poseIndex).split()
                resfile.write('{}  {}  NATAA\n'.format(ID, Chain))

                if Constants.DEBUG:
                    resfileString += '{}  {}  NATAA\n'.format(ID, Chain)
        if Constants.DEBUG:
            printDEBUG(msg=resfileString, rank=MPI.COMM_WORLD.Get_rank())

    def apply(self, pose):
        self.getLigandPoseIndex(pose)
        self.getNeighborsPoseIndex(pose)
        if not self.neighborsPoseIndex:
            return
        self.writeResfile(pose, self.resfileName)
        resfileCatalytic = ReadResfile(self.resfileName)
        self.taskFactory.clear()
        self.taskFactory.push_back(resfileCatalytic)
        self.packer.task_factory(self.taskFactory)
        self.packer.apply(pose)

    def getLigandPoseIndex(self, pose):
        self.ligand.poseIndex = pose.pdb_info().pdb2pose(self.ligand.chain, self.ligand.ID)

    def getNeighborsPoseIndex(self, pose):
        pose.update_residue_neighbors()
        ligandSelector = ResidueIndexSelector(self.ligand.poseIndex)
        nbr_selector = NeighborhoodResidueSelector(ligandSelector, self.neighborDistCutoff, False)
        nbrBoolVector = nbr_selector.apply(pose)
        nbrBoolVector = array(nbrBoolVector, dtype=bool)
        self.neighborsPoseIndex = list(where(nbrBoolVector)[0] + 1)


class LigandRigidBodyMover(object):
    def __init__(self, ligand: Ligand = None,
                 pose: Pose = None,
                 dockingCenter: list = None,
                 simulationCenter: list = None,
                 simulationRadius: float = 15.0,
                 nonStandardNeighbors: list = None,
                 neighborDistCutoff: float = 15.0,
                 translationSTD: float = 1.0,
                 rotationSTD: float = 10.0,
                 translationLoops: int = 10,
                 rotationLoops: int = 50,
                 sidechainCoupling: float = 0.0,
                 sidechainCouplingMax: float = 1.0,
                 sideChainCouplingExcludedPoseIndex: list = None,
                 backboneCoupling: float = 1.0,
                 overlap: float = 0.8,
                 sasaScaling: bool = True,
                 sasaCutoff: float = 0.2,
                 translationScale=0.5,
                 rotationScale=0.5):

        self.neighborDistCutoff = neighborDistCutoff
        self.neighborsPoseIndex = None

        self.neighborsAtomId = vector1_core_id_AtomID()
        self.ligandCoreAtomId = vector1_core_id_AtomID()
        self.neighborsLjRadii = None
        self.neighborsSideChainFlagVector = None
        self.neighborsBackboneFlagVector = None

        self.neighborsXYZVector1 = vector1_numeric_xyzVector_double_t()
        self.ligandCoreXYZVector1 = vector1_numeric_xyzVector_double_t()
        self.poseBoolVector = vector1_bool()
        self.translationVector = xyzVector_double_t()
        self.rotationVector = xyzVector_double_t()
        self.centerOfRotationVector = xyzVector_double_t()

        self.neighborsXYZ = None
        self.ligandCoreXYZ = None

        self.ligand = None
        self.nonStandardNeighborsDict = None

        self.translationSTD = translationSTD
        self.rotationSTD = rotationSTD
        self.translationLoops = translationLoops
        self.rotationLoops = rotationLoops

        self.backboneCoupling = backboneCoupling
        self.sideChainCoupling = sidechainCoupling
        self.sideChainCouplingMax = sidechainCouplingMax
        self.sideChainCouplingExcludedPoseIndex = sideChainCouplingExcludedPoseIndex

        self.overlap = overlap
        self._tempPose = Pose()
        self.dockingCenter = dockingCenter
        self.simulationCenter = simulationCenter
        self.simulationRadius = simulationRadius

        self.sasaScaling = sasaScaling
        self.sasaCutoff = sasaCutoff
        self.sasaCurrent = float('inf')
        self.translationScale = translationScale
        self.rotationScale = rotationScale

        # Initiate
        if ligand is not None and pose is not None:
            self.initiate(ligand, dockingCenter, simulationCenter, simulationRadius, pose, nonStandardNeighbors,
                          sideChainCouplingExcludedPoseIndex, neighborDistCutoff)

    def initiate(self, ligand, dockingCenter=None, simulationCenter=None, simulationRadius=15.0, pose=None,
                 nonStandardNeighbors=None, sideChainCouplingExcludedPoseIndex=None, neighborDistCutoff=15.0,
                 sasaScaling=True, sasaCutoff=0.2, translationScale=0.5, rotationScale=0.5):

        self.neighborDistCutoff = neighborDistCutoff
        self.ligand = ligand
        self.sideChainCouplingExcludedPoseIndex = []

        self.sasaScaling = sasaScaling
        self.sasaCutoff = sasaCutoff
        self.translationScale = translationScale
        self.rotationScale = rotationScale

        if simulationCenter is not None:
            self.simulationCenter = array(simulationCenter)

        if sideChainCouplingExcludedPoseIndex is not None:
            self.sideChainCouplingExcludedPoseIndex = sideChainCouplingExcludedPoseIndex

        if dockingCenter is not None:
            self.dockingCenter = dockingCenter

        self.simulationRadius = simulationRadius
        self.getNonStandardNeighborsDict(nonStandardNeighbors)

        if pose is not None:
            self.getLigandPoseIndex(pose)
            self.getligandProperties(pose)
            self.getNeighborsPoseIndex(pose)
            self.getNeighborsProperties(pose)
            self.getNeighborsXYZ(pose)
            self.getLigandCoreXYZ(pose)

            # Initiate the docking center and simulation centers from
            # the current position of the ligand
            if self.simulationCenter is None:
                self.simulationCenter = self.getLigandCoreCentroid()

            if self.dockingCenter is None:
                self.dockingCenter = self.getLigandCoreCentroid()

    def getNonStandardNeighborsDict(self, nonStandardNeighbors):
        """
        Creates a dict
        :param nonStandardNeighbors: list of Ligand object
        """
        self.nonStandardNeighborsDict = dict()
        if nonStandardNeighbors is not None:
            for ligand in nonStandardNeighbors:
                self.nonStandardNeighborsDict[ligand.name] = ligand.coreAtoms

    def addNonStandardNeighborDict(self, nonStandardNeighbor):
        """
        Creates a dict
        :param nonStandardNeighbor: Ligand object
        """
        # Create a dictionay if it does not exist
        if not self.nonStandardNeighborsDict:
            self.nonStandardNeighborsDict = dict()

        if nonStandardNeighbor is not None:
            self.nonStandardNeighborsDict[nonStandardNeighbor.name] = nonStandardNeighbor.coreAtoms

    def getLigandPoseIndex(self, pose):
        # print('BBB 3', type(pose), pose.total_residue())
        self.ligand.poseIndex = pose.pdb_info().pdb2pose(self.ligand.chain, self.ligand.ID)

    def getligandProperties(self, pose):
        # getLigandCoreAtomId
        self.ligandCoreAtomId.clear()

        # Get id vector of core atoms
        for atomIndex in self.ligand.coreAtoms:
            self.ligandCoreAtomId.append(AtomID(atomIndex, self.ligand.poseIndex))

        # getLigandPoseBoolVector
        self.poseBoolVector.clear()
        for index, residue in enumerate(pose.residues):
            if index + 1 == self.ligand.poseIndex:
                self.poseBoolVector.append(1)
            else:
                self.poseBoolVector.append(0)

    def getNeighborsPoseIndex(self, pose):
        pose.update_residue_neighbors()
        ligandSelector = ResidueIndexSelector(self.ligand.poseIndex)
        # print('BBBB', self.neighborDistCutoff)
        nbr_selector = NeighborhoodResidueSelector(ligandSelector, self.neighborDistCutoff, False)
        nbrBoolVector = nbr_selector.apply(pose)
        nbrBoolVector = array(nbrBoolVector, dtype=bool)
        self.neighborsPoseIndex = list(where(nbrBoolVector)[0] + 1)
        # print('BBBB', self.neighborsPoseIndex)

    def getNeighborsProperties(self, pose):
        # Neighbors AtomId and Neighbors LjRadii
        self.neighborsAtomId.clear()
        self.neighborsLjRadii = list()
        self.neighborsSideChainFlagVector = list()
        self.neighborsBackboneFlagVector = list()
        # self.neighborsNumberOfAtoms = 0
        # self.neighborsElements = list()

        # Reassign new atom ids
        for resIndex in self.neighborsPoseIndex:
            res = pose.residue(resIndex)
            id, chain = pose.pdb_info().pose2pdb(resIndex).split()
            resName = (int(id), chain)
            # Construct neighbors vectors
            for atomIndex in range(1, res.natoms() + 1):
                # Skip virtual atoms
                if res.atom_type(atomIndex).is_virtual():
                    continue

                # Get the atomIDs
                self.neighborsAtomId.append(AtomID(atomIndex, resIndex))

                # get Vdw
                # vdw = self.aminoAcids.VdwRadii[res.atom_type(atomIndex).element()]
                self.neighborsLjRadii.append(res.atom_type(atomIndex).lj_radius())

                # Get the side chain/backbone flag,
                # first check if the residue is excluded from side chain coupling.
                if resIndex in self.sideChainCouplingExcludedPoseIndex:
                    self.neighborsSideChainFlagVector.append(0)
                    self.neighborsBackboneFlagVector.append(1)

                # print('BBBB', res.atom_name(atomIndex), res.atom_is_backbone(atomIndex))
                # if the residue poseIndex is in the dict use the ligand coreAtom definition
                elif resName in self.nonStandardNeighborsDict.keys():
                    # print('BBBB in coreatom')
                    nonStandardNeighborCoreAtom = self.nonStandardNeighborsDict.get(resName, [])
                    if atomIndex in nonStandardNeighborCoreAtom:
                        # print('BBBBB, backbone, ligand')
                        self.neighborsSideChainFlagVector.append(0)
                        self.neighborsBackboneFlagVector.append(1)
                    else:
                        self.neighborsSideChainFlagVector.append(1)
                        self.neighborsBackboneFlagVector.append(0)

                # The residue is not declared above it should be non flexible
                elif res.is_ligand():
                    self.neighborsSideChainFlagVector.append(0)
                    self.neighborsBackboneFlagVector.append(1)
                # metals are treated as backbone
                elif res.is_metal():
                    # print('BBBBB, backbone metal')
                    self.neighborsSideChainFlagVector.append(0)
                    self.neighborsBackboneFlagVector.append(1)

                # otherwise use library definition
                else:
                    # print('BBBB in else')
                    if res.atom_is_backbone(atomIndex):
                        # print('BBBBB, backbone other')
                        self.neighborsSideChainFlagVector.append(0)
                        self.neighborsBackboneFlagVector.append(1)
                    else:
                        self.neighborsSideChainFlagVector.append(1)
                        self.neighborsBackboneFlagVector.append(0)

        # save in correct format
        self.neighborsLjRadii = array(self.neighborsLjRadii)
        self.neighborsSideChainFlagVector = array(self.neighborsSideChainFlagVector, dtype=np.int32)
        self.neighborsBackboneFlagVector = array(self.neighborsBackboneFlagVector, dtype=np.int32)

    def getNeighborsXYZ(self, pose):
        self.neighborsXYZVector1.clear()
        pose.batch_get_xyz(self.neighborsAtomId, self.neighborsXYZVector1)
        self.neighborsXYZ = array([[atom.x, atom.y, atom.z] for atom in self.neighborsXYZVector1])

    def getLigandCoreXYZ(self, pose):
        self.ligandCoreXYZVector1.clear()
        pose.batch_get_xyz(self.ligandCoreAtomId, self.ligandCoreXYZVector1)
        self.ligandCoreXYZ = array([[atom.x, atom.y, atom.z] for atom in self.ligandCoreXYZVector1])

    def getRandomUnitVector(self):
        v = normal(size=3)
        return v / norm(v)

    def getLigandCoreCentroid(self):
        return self.ligandCoreXYZ.sum(axis=0) / self.ligandCoreXYZ.shape[0]

    def updateProperties(self, pose):
        self.getLigandPoseIndex(pose)
        self.getNeighborsPoseIndex(pose)
        self.getligandProperties(pose)
        self.getNeighborsProperties(pose)

    def translate(self, pose):

        # get Translation Unit Vector
        self.translationVector.assign(*self.getRandomUnitVector())

        # get Dummy rotation Unit Vector
        self.rotationVector.assign(1, 1, 1)

        # get Dummy center of rotation
        self.centerOfRotationVector.assign(1, 1, 1)

        # get Translation Magnitude
        translationMagnitude = normal(scale=self.translationSTD)

        if self.sasaScaling and self.sasaCurrent < self.sasaCutoff:
            # if scale is negative use sasa as scale
            if self.translationScale < 0:
                translationMagnitude *= self.sasaCurrent
            else:
                translationMagnitude *= self.translationScale

        # apply
        rigid_body_move(self.rotationVector, 0.0, self.translationVector, translationMagnitude, pose,
                        self.poseBoolVector, self.centerOfRotationVector)

    def rotate(self, pose):

        # get dummy Translation Vector
        self.translationVector.assign(1, 1, 1)

        # get rotation Unit Vector
        self.rotationVector.assign(*self.getRandomUnitVector())

        # get center of rotation make sure the xyz vectors are updated
        self.centerOfRotationVector.assign(*self.getLigandCoreCentroid())

        # get Translation Magnitude
        rotationMagnititude = normal(scale=self.rotationSTD)

        if self.sasaScaling and self.sasaCurrent < self.sasaCutoff:
            if self.rotationScale < 0:
                rotationMagnititude *= self.sasaCurrent
            else:
                rotationMagnititude *= self.rotationScale

        # apply
        rigid_body_move(self.rotationVector, rotationMagnititude, self.translationVector, 0.0, pose,
                        self.poseBoolVector, self.centerOfRotationVector)

    def apply(self, pose):
        "Apply rigid body transformation."

        # update sysem properties
        # print('BBB 2 ', type(pose), pose.total_residue())
        self.updateProperties(pose)
        # Compute SASA
        if self.sasaScaling:
            self.sasaCurrent = self.getSASA(pose)
        for i in range(self.translationLoops):
            self._tempPose.assign(pose)
            self.translate(self._tempPose)
            self.getLigandCoreXYZ(self._tempPose)
            # check ligand is in simulation sphere
            d = norm(self.simulationCenter - self.getLigandCoreCentroid())
            if d > self.simulationRadius:
                continue
            # Update enviroment
            self.getNeighborsPoseIndex(self._tempPose)
            self.getNeighborsProperties(self._tempPose)
            self.getNeighborsXYZ(self._tempPose)
            # Tests rotations
            for j in range(self.rotationLoops):
                self.rotate(self._tempPose)
                self.getLigandCoreXYZ(self._tempPose)

                # Make sure the ligand has neighbors
                if len(self.neighborsXYZ) == 0:
                    collision = 0
                else:
                    collision = Energy.ligand_environment_collision(self.ligandCoreXYZ,
                                                                    self.neighborsXYZ,
                                                                    self.ligand.ljRadii,
                                                                    self.neighborsLjRadii,
                                                                    self.neighborsSideChainFlagVector,
                                                                    self.neighborsBackboneFlagVector,
                                                                    self.sideChainCoupling,
                                                                    self.backboneCoupling,
                                                                    self.overlap)
                if collision == 1:
                    continue
                else:
                    pose.assign(self._tempPose)
                    #print('RGB found', self.sasaCurrent, self.sasaCutoff)
                    # print('BBBB i, j count: ', i, j)
                    return True

        # if gets here moved failed
        return False

    def isLigandInPose(self, pose):
        return pose.pdb_info().pdb2pose(self.ligand.chain, self.ligand.ID)

    def centerLigand(self, pose):

        if not self.isLigandInPose(pose):
            raise ValueError(
                'Can not center ligand. Ligand {}-{} was not found.'.format(self.ligand.ID, self.ligand.chain))

        self.getLigandPoseIndex(pose)
        ligandPosition = self.getLigandCoreCentroid()
        translationVector = self.dockingCenter - ligandPosition
        translationMagnitude = norm(translationVector)

        # get Translation Unit Vector in rosetta vector format
        translationVector = translationVector / translationMagnitude
        self.translationVector.assign(*translationVector)

        # get Dummy rotation Unit Vector
        self.rotationVector.assign(1, 1, 1)

        # get Dummy center of rotation
        self.centerOfRotationVector.assign(1, 1, 1)

        # apply
        rigid_body_move(self.rotationVector, 0.0, self.translationVector, translationMagnitude, pose,
                        self.poseBoolVector, self.centerOfRotationVector)

    def dockLigand(self, pose):

        # add the ligand if it does not exist.
        if not self.isLigandInPose(pose):
            # append
            pose.append_pose_by_jump(self.ligand.pose)

            # Update Foldtree

            # First get the chains' boundary
            chainsBoundary = dict()
            for i in range(1, pose.num_chains() + 1):
                chainsBoundary[i] = (pose.chain_begin(i), pose.chain_end(i))

            # Set the fold tree. All non Ligand residues are fixed relative to res 1.
            ft = FoldTree()
            jumpCount = 1
            for chain, boundary in chainsBoundary.items():
                # First chain is the reff. set frozen
                if chain == 1:
                    ft.add_edge(chainsBoundary[1][0], boundary[1], -1)
                else:
                    # for the rest the jump is movable if ligand
                    if pose.residue(boundary[0]).is_ligand():
                        ft.add_edge(chainsBoundary[1][0], boundary[0], jumpCount)
                        jumpCount += 1
                    # Otherwise the jump is frozen relative to reff
                    else:
                        ft.add_edge(chainsBoundary[1][0], boundary[0], -1)
                    # If the chain has multiple res add frozen edge
                    if boundary[0] != boundary[1]:
                        ft.add_edge(boundary[0], boundary[1], -1)
            pose.fold_tree(ft)

        # center it
        self.centerLigand(pose)

    def getSASA(self, pose):

        # Calculate bound Ligand SASA
        self.getLigandPoseIndex(pose)

        posePdb = ostringstream()
        pose.dump_pdb(posePdb)
        pdbFile = StringIO(posePdb.str())

        # Get the PDB file
        pdb = PDBFile.read(pdbFile)
        pdbStruc = pdb.get_structure()[0]

        atomsSASA = sasa(pdbStruc, vdw_radii="Single")
        strucSASA = apply_residue_wise(pdbStruc, atomsSASA, sum)
        boundLigandSASA = strucSASA[self.ligand.poseIndex - 1]

        # Calculate free bound (with current conformation) Ligand SASA
        ligPose = return_region(pose, self.ligand.poseIndex, self.ligand.poseIndex)
        ligPosePdb = ostringstream()
        ligPose.dump_pdb(ligPosePdb)
        ligpdbFile = StringIO(ligPosePdb.str())

        # Get the PDB file
        ligpdb = PDBFile.read(ligpdbFile)
        ligpdbStruc = ligpdb.get_structure()[0]

        ligatomsSASA = sasa(ligpdbStruc, vdw_radii="Single")
        ligstrucSASA = apply_residue_wise(ligpdbStruc, ligatomsSASA, sum)
        freeLigandSASA = ligstrucSASA[0]

        # print("Docking line 1489 in getSASA: ", boundLigandSASA, freeLigandSASA, boundLigandSASA/freeLigandSASA)
        return boundLigandSASA / freeLigandSASA


class LigandMover(object):

    def __init__(self, ligand: Ligand = None, ligandRigidBodyMover: LigandRigidBodyMover = None, doRigidBody=True,
                 ligandPackingMover: LigandPackingMover = None, doPacking=True,
                 ligandNeighborsPackingMover: LigandNeighborsPackingMover = None, doNeighborPacking=True,
                 ligandPurturbationMode='MC', ligandPurturbationLoops: int = 1, sasaConstraint: int=0):

        self.ligandRigidBodyMover = ligandRigidBodyMover
        self.doRigidBody = doRigidBody
        self.ligandPackingMover = ligandPackingMover
        self.doPacking = doPacking
        self.ligandNeighborsPackingMover = ligandNeighborsPackingMover
        self.doNeighborPacking = doNeighborPacking
        self.ligandPurturbationMode = ligandPurturbationMode  # 'MC', 'MIN'
        self.ligandPurturbationLoops = ligandPurturbationLoops
        self.ligand = ligand
        self.sasaConstraint = sasaConstraint
        # Temporary design pose
        self.poseTemp = Pose()
        self.poseOrig = Pose()

        self.scorefxn = getScoreFunction(mode='fullAtomWithConstraints')

    def setLigand(self, ligand: Ligand):
        self.ligand = ligand

    def setLigandRigidBodyMover(self, ligandRigidBodyMover: LigandRigidBodyMover):
        self.ligandRigidBodyMover = ligandRigidBodyMover

    def setLigandPackingMover(self, ligandPackingMover: LigandPackingMover):
        self.ligandPackingMover = ligandPackingMover

    def setLigandNeighborsPackingMover(self, ligandNeighborsPackingMover: LigandNeighborsPackingMover):
        self.ligandNeighborsPackingMover = ligandNeighborsPackingMover

    def setSideChainCouplingExcludedPoseIndex(self, sideChainCouplingExcludedPoseIndex: list):
        if self.ligandRigidBodyMover:
            self.ligandRigidBodyMover.sideChainCouplingExcludedPoseIndex = sideChainCouplingExcludedPoseIndex
        if self.ligandPackingMover:
            self.ligandPackingMover.sideChainCouplingExcludedPoseIndex = sideChainCouplingExcludedPoseIndex

    def setExcludedResiduePoseIndex(self, excludedResiduePoseIndex: list):
        if self.ligandNeighborsPackingMover:
            self.ligandNeighborsPackingMover.excludedResiduePoseIndex = excludedResiduePoseIndex

    def setSideChainCoupling(self, coupling):

        if self.ligandRigidBodyMover:
            self.ligandRigidBodyMover.sideChainCoupling = coupling

        if self.ligandPackingMover:
            self.ligandPackingMover.sideChainCoupling = coupling

    def updateSideChainCouplingByStep(self, currentStep: int, totalSteps: int):
        """
        Compute value of coupling corresponds to the current step
        """
        ix = currentStep / totalSteps
        a = 1 + exp(-10 * (ix - (1 - 0.05)))
        s = ((1 / a) * ix) ** (1 - ix)

        if self.ligandPackingMover:
            coupling = self.ligandPackingMover.sideChainCouplingMax * s
        elif self.ligandRigidBodyMover:
            coupling = self.ligandRigidBodyMover.sideChainCouplingMax * s
        else:
            raise ValueError('Failed updating SideChainCoupling. No mover found.')

        self.setSideChainCoupling(coupling)

    def moveLigandPose(self, pose: Pose, countInitial=False):
        """
        Performs a MC move n number of times and return  the lowest energy MC move, if countInitial is
        False, it is guaranteed that one MC move is performed.
        :param pose:
        :param countInitial:
        :return: ligandMoveAccepted
        """
        if not countInitial:
            energyPrevious = float('inf')
            energyCurrent = energyPrevious
        else:
            energyPrevious = self.scorefxn(pose)
            energyPrevious += self.getSASAConstraint(self.poseTemp)
            energyCurrent = energyPrevious

        # make a copy of the original one
        self.poseOrig.assign(pose)

        rgbMoved = False
        packMoved = False
        ligandMoveAccepted = False
        for i in range(self.ligandPurturbationLoops):

            self.poseTemp.assign(self.poseOrig)

            # 1) RGB move

            if self.ligandRigidBodyMover and self.doRigidBody:
                # print('BBB 1', type(dPose.pose), dPose.pose.total_residue())
                #start = time.time()
                rgbMoved = self.ligandRigidBodyMover.apply(self.poseTemp)
                #end = time.time()
                #print("BBB RGB in : ", end - start, rgbMoved, flush=True)
                # Don't proceed if RGB and not moved
                if not rgbMoved:
                    continue


            # 2) Ligand Side Chain packing. Respect Catalytic Residues, i.e. coupling 1.0.
            #print(self.ligandPackingMover, self.doPacking)
            if self.ligandPackingMover and self.doPacking:
                #print("BBB Pack", flush=True)
                self.ligandPackingMover.apply(self.poseTemp)
                packMoved = True

            # 3) repack the non design neighbors.
            if self.ligandNeighborsPackingMover and self.doNeighborPacking:
                self.ligandNeighborsPackingMover.apply(self.poseTemp)

            # Check the energy, if lower assign to pose
            energyCurrent = self.scorefxn(self.poseTemp)
            energyCurrent += self.getSASAConstraint(self.poseTemp)

            # If RGB and not rgbMoved, we never get here
            if energyCurrent < energyPrevious:
                # First check if rgb is present and moved
                if self.ligandRigidBodyMover:
                    if rgbMoved:
                        ligandMoveAccepted = True
                        energyPrevious = energyCurrent
                        pose.assign(self.poseTemp)
                else:
                    if packMoved: # This is always true
                        ligandMoveAccepted = True
                        energyPrevious = energyCurrent
                        pose.assign(self.poseTemp)

        return ligandMoveAccepted

    def minimizeLigandPose(self, pose: Pose, countInitial=False):

        if not countInitial:
            energyPrevious = float('inf')
            energyCurrent = energyPrevious
        else:
            energyPrevious = self.scorefxn(pose)
            energyPrevious += self.getSASAConstraint(self.poseTemp)
            energyCurrent = energyPrevious

        # make a copy of the original one
        # self.poseOrig.assign(pose)
        rgbMoved = False
        packMoved = False
        ligandMoveAccepted = False
        for i in range(self.ligandPurturbationLoops):

            self.poseTemp.assign(pose)

            # 1) RGB move
            if self.ligandRigidBodyMover:
                # print('BBB 1', type(dPose.pose), dPose.pose.total_residue())
                rgbMoved = self.ligandRigidBodyMover.apply(self.poseTemp)

                # Don't proceed if RGB and not moved
                if not rgbMoved:
                    continue

            # 2) Ligand Side Chain packing. Respect Catalytic Residues, i.e. coupling 1.0.
            if self.ligandPackingMover:
                self.ligandPackingMover.apply(self.poseTemp)

            # 3) repack the neighbors.
            if self.ligandNeighborsPackingMover:
                self.ligandNeighborsPackingMover.apply(self.poseTemp)

            # Check the full energy, if lower assign to pose
            energyCurrent = self.scorefxn(self.poseTemp)
            energyCurrent += self.getSASAConstraint(self.poseTemp)

            if energyCurrent < energyPrevious:
                # First check if rgb is present and moved
                if self.ligandRigidBodyMover:
                    if rgbMoved:
                        ligandMoveAccepted = True
                        energyPrevious = energyCurrent
                        pose.assign(self.poseTemp)

                else:
                    if packMoved: # This is always true
                        ligandMoveAccepted = True
                        energyPrevious = energyCurrent
                        pose.assign(self.poseTemp)

        return ligandMoveAccepted

    def relaxLigand(self, pose: Pose):
        self.ligandPackingMover.ligandMinimization(pose)

    def getLigandPoseIndex(self, pose):
        self.ligand.poseIndex = pose.pdb_info().pdb2pose(self.ligand.chain, self.ligand.ID)

    def getSASA(self, pose):

        # Calculate bound Ligand SASA
        self.getLigandPoseIndex(pose)

        posePdb = ostringstream()
        pose.dump_pdb(posePdb)
        pdbFile = StringIO(posePdb.str())

        # Get the PDB file
        pdb = PDBFile.read(pdbFile)
        pdbStruc = pdb.get_structure()[0]

        atomsSASA = sasa(pdbStruc, vdw_radii="Single")
        strucSASA = apply_residue_wise(pdbStruc, atomsSASA, sum)
        boundLigandSASA = strucSASA[self.ligand.poseIndex - 1]

        # Calculate free bound (with current conformation) Ligand SASA
        ligPose = return_region(pose, self.ligand.poseIndex, self.ligand.poseIndex)
        ligPosePdb = ostringstream()
        ligPose.dump_pdb(ligPosePdb)
        ligpdbFile = StringIO(ligPosePdb.str())

        # Get the PDB file
        ligpdb = PDBFile.read(ligpdbFile)
        ligpdbStruc = ligpdb.get_structure()[0]

        ligatomsSASA = sasa(ligpdbStruc, vdw_radii="Single")
        ligstrucSASA = apply_residue_wise(ligpdbStruc, ligatomsSASA, sum)
        freeLigandSASA = ligstrucSASA[0]

        # print("Docking line 1489 in getSASA: ", boundLigandSASA, freeLigandSASA, boundLigandSASA/freeLigandSASA)
        return boundLigandSASA / freeLigandSASA

    def getLigandEnergy(self, pose):

        # Update the pose index
        self.getLigandPoseIndex(pose)

        # bound energy
        self.scorefxn(pose)
        energyBound =  pose.energies().residue_total_energy(self.ligand.poseIndex)
        #energyBound += self.scorefxn.score_by_scoretype(pose, score_type_from_name('atom_pair_constraint'))

        # Unbound energy
        ligPose = return_region(pose, self.ligand.poseIndex, self.ligand.poseIndex)
        energyUnbound = self.scorefxn(ligPose)
        return energyBound - energyUnbound

    def getSASAConstraint(self, pose):
        if self.sasaConstraint != 0:
            return self.getSASA(pose) * self.sasaConstraint
        else:
            return 0.0

    def apply(self, pose: Pose, countInitial=False):

        ligandMoved = False
        if self.ligandPurturbationMode == 'MC':
            ligandMoved = self.moveLigandPose(pose, countInitial)

        elif self.ligandPurturbationMode == 'MIN':
            ligandMoved = self.minimizeLigandPose(pose, countInitial)

        return ligandMoved

# TODO Fish up the docking sampler for normal docking.
class LigandDockingSampler(object):
    def __init__(self, pose=None, ligand=None, boxCenter=None, boxRadius=10, neighborDistCutoff=15.0):

        # domain attribute
        # A copy of the protein pose with no ligand
        self.dockingDomainPose = None
        self.dockingDomainResidues = None
        self.dockingDomainCenter = None
        self.dockingDomainRadius = None
        self.dockingDomainResidues = None
        # self.dockingDomainBakboneAtomIDs = None
        # self.dockingDomainBakboneAtomIDs = None

        # ligand attributes
        # A copy of the isolated ligand pose (with out any other molecule)
        self.ligandPose = None
        self.ligandChain = None
        self.ligandID = None
        self.ligandPoseIndex = None

        # core atoms define the rigid part of molecule
        self.ligandAtoms = None
        self.ligandCoreAtoms = None
        self.ligandSideChainsAtoms = None

        # The chi angles of the ligand.
        self.ligandChis = None

        # The chis are grouped into side chains.
        self.ligandSideChains = None
        self.ligandSideChainsIndex = None

        # The side chains are ordered in terms of tree level (different grid size for different level)
        self.ligandSideChainsOrdered = None
        self.ligandSideChainsOrderedIndex = None

        if pose is not None and ligand is not None and boxCenter is not None:
            self.initiate(pose, ligand, boxCenter, boxRadius, neighborDistCutoff)

    def initiate(self, pose, ligand, boxCenter, boxRadius, neighborDistCutoff):

        # self.initiateLigand(pose, ligand)
        self.initiateDomain(pose, boxCenter, boxRadius, neighborDistCutoff)

    def initiateDomain(self, pose, boxCenter, boxRadius, neighborDistCutoff):
        """
        Initiates the docking box parameters. Domain should be initiated after ligand.
        :param boxCenter: list xyz coordinates
        :param boxRadius: float
        :param neighborDistCutoff: float
        """
        # keep a copy of the protein with no ligand
        self.dockingDomainPose = Pose().assign(pose)
        self.dockingDomainPose.delete_residue_slow(self.ligandPoseIndex)

        # get docking domain metrics
        self.dockingDomainCenter = array(boxCenter).reshape((1, 3))
        self.dockingDomainRadius = boxRadius

        # Get docking domain
        self.getdockingDomain()

    def getdockingDomain(self):
        """
        Gets the residues within distance of (boxRadius + lignad.nbr) of the boxCenter.
        """
        self.dockingDomainResidues = list()
        boxCenter = xyzVector_double_t(*self.dockingDomainCenter[0, :])
        boxRadius = self.dockingDomainRadius + self.ligandPose.residue(1).nbr_radius()
        residueVector = vector1_std_shared_ptr_const_core_conformation_Residue_t()

        for poseIndex, residue in enumerate(self.dockingDomainPose.residues):
            if residue.is_protein():
                residueVector.clear()
                residueVector.append(residue)
                residueCentroid = centroid_by_residues(residueVector)
                if boxCenter.distance(residueCentroid) <= boxRadius:
                    # create new residue obj
                    dockingDomainResidue = Residue()
                    ID, Chain = self.dockingDomainPose.pdb_info().pose2pdb(poseIndex + 1).split()
                    dockingDomainResidue.ID = int(ID)
                    dockingDomainResidue.chain = Chain
                    dockingDomainResidue.currentAA = self.dockingDomainPose.residue(poseIndex + 1).name1()

                    # add it to the list
                    self.dockingDomainResidues.append(dockingDomainResidue)

    # TODO remove ############################
    def initiateLigand(self, pose, ligand):
        """
        Initiates the ligand parameters. The order of operation is important.
        :param pose:
        :param ligand:
        """
        # get ligand info
        self.ligandID = ligand.ID
        self.ligandChain = ligand.chain
        self.ligandPoseIndex = pose.pdb_info().pdb2pose(self.ligandChain, self.ligandID)
        self.ligandPointer = pose.residue(self.ligandPoseIndex)
        self.ligandPose = return_region(pose, self.ligandPoseIndex, self.ligandPoseIndex)

        # get the ligand chi angles (includes only rotatable bonds)
        self.getLigandChis(ligand.excludedTorsions)

        # Compute side chains
        self.getLigandSideChains()

        # Compute ordered side chain
        self.getLigandSideChainsOrdered()

        # Get the atom index of each side chain for collision detection
        self.getLigandSideChainsAtoms()

        # Get the core atoms if not given
        self.getLigandCoreAtoms()

        # Check ligand atoms
        self.getLigandAtoms()

    # TODO remove ############################
    def getLigandAtoms(self):
        """
        Gets the atom indices of the ligand
        """
        ligandResidue = self.ligandPose.residue(1)
        self.ligandAtoms = list(range(1, ligandResidue.natoms() + 1))
        ligandSetsAtoms = []
        for i in self.ligandCoreAtoms:
            if i in self.ligandAtoms:
                ligandSetsAtoms.append(i)
        for sideChain in self.ligandSideChainsAtoms:
            for i in sideChain:
                if i in self.ligandAtoms:
                    ligandSetsAtoms.append(i)
        if len(ligandSetsAtoms) != len(self.ligandAtoms):
            raise ValueError('Ligand could not be initialized. The total atoms in ligand core and side chains differ '
                             'from ligand total number of atoms.')

    # TODO remove ############################
    def getLigandChis(self, excludedTorsions):
        """
        Get the ligand chis excluding the ones in the excludedTorsions list
        :param excludedTorsions: list of torsion tuples with atom name
        """
        self.ligandChis = list(tuple(tors) for tors in self.ligandPose.residue(1).chi_atoms())
        ligandChisNames = [tuple(self.ligandPose.residue(1).atom_name(i).split()[0] for i in chi) for chi in
                           self.ligandChis]
        for excludedTorsion in excludedTorsions:
            if excludedTorsion not in ligandChisNames:
                raise ValueError('Bad excluded torsion. {} is not part of ligand chis {}'
                                 .format(str(excludedTorsion), str(ligandChisNames)))
            else:
                index = ligandChisNames.index(excludedTorsion)
                self.ligandChis.pop(index)

    # TODO remove ############################
    def getLigandSideChains(self):
        """
        Groups the chi angles of the ligand into side chains by traversing
        the chis of the ligand.
        """
        self.ligandSideChains = []
        # Get all possible side-chain in the ligand.
        # This would result in redundant sub-trees
        sideChains = list()
        for root in self.ligandChis:
            sideChain = set()
            sideChain.add(root)
            self._getSideChainsForward(root, self.ligandChis, sideChain)
            sideChains.append(sideChain)

        # Get unique side-chain. i.e. remove the sub-trees
        sideChainsUnique = list()
        for i in sideChains:
            keepSideChain = True
            for j in sideChains:
                if i != j and i.issubset(j):
                    keepSideChain = False
            if keepSideChain:
                if i not in sideChainsUnique:
                    sideChainsUnique.append(i)

        # Construct the set of atoms involved in each site chain. These set will be
        # used to identify gaps in ligand chis (non rotatable bonds). Two primitive side chains
        # should be combined into one if the atom set of one of them is a subset of the other
        sideChainsAtoms = list()
        for sideChain in sideChainsUnique:
            sideChainAtoms = list()
            for torsion in sideChain:
                root = torsion[2]
                # add atoms atoms j and k to stop backward travers
                visited = [torsion[1], torsion[2]]
                self._getAtomsIndicesRecursively(root, self.ligandPose.residue(1), sideChainAtoms, visited)
            sideChainsAtoms.append(set(sideChainAtoms))

        # Find the index of superset side chains
        indexOfSideChainsFinal = list()
        for i, iSet in enumerate(sideChainsAtoms):
            keep = True
            for j, jSet in enumerate(sideChainsAtoms):
                if iSet.issubset(jSet) and iSet != jSet:
                    sideChainsUnique[j].update(sideChainsUnique[i])
                    keep = False
            if keep:
                indexOfSideChainsFinal.append(i)

        for i in indexOfSideChainsFinal:
            self.ligandSideChains.append(sideChainsUnique[i])

        # get the side chain indices too
        self.getLigandSideChainsIndex()

    # TODO remove ############################
    def getLigandSideChainsIndex(self):
        """
        Get the chi index of torsions in each side chain. This would allow for
        fast access of ligand torsions.
        """
        # Get the indices
        sideChainsIndex = list()
        for sideChain in self.ligandSideChains:
            sideChainIndex = set()
            for torsion in sideChain:
                sideChainIndex.add(self.ligandChis.index(torsion) + 1)
            sideChainsIndex.append(sideChainIndex)
        self.ligandSideChainsIndex = sideChainsIndex

    # TODO remove ############################
    def _getSideChainsForward(self, root, torsions, sideChain, visited=[]):
        """
        detects side chains recursively in forward direction, i.e. if atom k of root torsion
        is identical to atom j of the other. these two torsions belong to the same side chain
        :param root: torsion (i, j, k, l)
        :param torsions: list of all torsions
        :return sideChain: set of torsions belong to the same side chain
        """
        for torsion in torsions:
            if root[2] == torsion[1]:
                if torsion not in visited and tuple(reversed(torsion)) not in visited:
                    visited.append(torsion)
                    visited = self._getSideChainsForward(torsion, torsions, sideChain)
                if torsion in self.ligandChis:
                    sideChain.add(torsion)
                elif tuple(reversed(torsion)) in self.ligandChis:
                    sideChain.add(tuple(reversed(torsion)))
                return visited

    # TODO remove ############################
    def getLigandSideChainsOrdered(self):
        """
        Get the level of each torsion in each side chains. This is used for level
        dependant grid size generation in Minimum Energy Neighborhood side chain packing.
        the ordered side chains are encoded as list of list of list, where the first index is
        the side chain and the second index is the level of torsions . The final index refers
        to torsion, i.e. [0][1][2] is the third level 1 torsion in side chain 0.
        [[(i,j,k,l)], [(i,j,k,l), (i,j,k,l)]]
        """
        self.ligandSideChainsOrdered = list()
        for sideChain in self.ligandSideChains:
            # The torsions for each side chain are ordered in terms of number of atoms in
            # their sub tree.
            sideChain = list(sideChain)
            torsionsNumberOfAtoms = list()
            for index, torsion in enumerate(sideChain):
                subTreeAtoms = list()
                root = torsion[2]
                # add atoms atoms j and k to stop backward travers
                visited = [torsion[1], torsion[2]]
                self._getAtomsIndicesRecursively(root, self.ligandPose.residue(1), subTreeAtoms, visited)
                torsionsNumberOfAtoms.append((index, len(subTreeAtoms)))

            # Group torsions based on their atom count
            torsionsNumberOfAtoms.sort(key=lambda x: x[1], reverse=True)
            currentAtomCount = 9999
            sideChainOrdered = []
            for index, torsionNumberOfAtoms in torsionsNumberOfAtoms:
                if torsionNumberOfAtoms < currentAtomCount:
                    currentAtomCount = torsionNumberOfAtoms
                    sideChainOrdered.append([sideChain[index]])
                else:
                    sideChainOrdered[-1].extend(sideChain[index])

            # Add the ordered side chain list to the list of all side chains
            self.ligandSideChainsOrdered.append(sideChainOrdered)

        # get the side chain indices too
        self.getLigandsideChainsOrderedIndex()

    # TODO remove ############################
    def getLigandsideChainsOrderedIndex(self):
        """
        Get the chi index of torsions in each ordered side chain. This would allow for
        fast access of ligand torsions. The sideChainsOrderedIndex has same structure as
        sideChainsOrdered
        """

        # get the indices of ordered side chains
        sideChainsOrderedIndex = list()
        for sideChainOrdered in self.ligandSideChainsOrdered:
            sideChainOrderedIndex = list()
            for level in sideChainOrdered:
                levelIndex = list()
                for torsion in level:
                    levelIndex.append(self.ligandChis.index(torsion) + 1)
                sideChainOrderedIndex.append(levelIndex)
            sideChainsOrderedIndex.append(sideChainOrderedIndex)

        self.ligandSideChainsOrderedIndex = sideChainsOrderedIndex

    # TODO remove ############################
    def _getSideChainsOrderedForward(self, root, torsions, sideChain):
        """
        detects side chains recursively in forward direction, i.e. if atom k of root torsion
        is identical to atom j of the other. these two torsions belong to the same side chain
        :param root: torsion (i, j, k, l)
        :param torsions: list of all torsions
        :return sideChain: set of torsions belong to the same side chain
        """
        for i in torsions:
            if root[2] == i[1]:
                self._getSideChainsForward(i, torsions, sideChain)
                sideChain.add(i)

    # TODO remove ############################
    def getLigandSideChainsAtoms(self):
        """
        Get the atoms involved in each side chain using the l atom of
        level 0 torsion as root by traversing the molecule toward terminal atoms
        """
        self.ligandSideChainsAtoms = list()
        residue = self.ligandPose.residue(1)
        for sideChain in self.ligandSideChainsOrdered:
            sideChainAtoms = list()
            root = sideChain[0][0][2]
            # add atoms atoms j and k to stop backward travers
            visited = [sideChain[0][0][1], sideChain[0][0][2]]
            self._getAtomsIndicesRecursively(root, residue, sideChainAtoms, visited)
            self.ligandSideChainsAtoms.append(sideChainAtoms)

    # TODO remove ############################
    def getLigandCoreAtoms(self):
        """
        Compute core atoms using the nbr atom as the root atom and k atoms of all level 0 torsions
        as the search boundary.
        """
        self.ligandCoreAtoms = list()
        residue = self.ligandPose.residue(1)
        visited = list()

        # add k atom of all level 0 torsions in all side chains to the visited list
        visited.extend([sideChain[0][0][2] for sideChain in self.ligandSideChainsOrdered])

        # add root atom to visited
        visited.append(residue.nbr_atom())

        # the root and k atoms are part of core atoms by default
        self.ligandCoreAtoms.extend(visited)

        # Perform a recursive search to get the rest of core atoms
        self._getAtomsIndicesRecursively(root=residue.nbr_atom(), residue=residue, results=self.ligandCoreAtoms,
                                         visited=visited)

    # TODO remove ############################
    def _getAtomsIndicesRecursively(self, root, residue, results, visited=[]):
        """
        Recursively find all core atoms in a residue object, starting from root atom.
        :param root: int root atom number
        :param residue: pyrosetta residue
        :param visited: list
        :return results: list of atoms indices
        """
        for neighbor in list(residue.bonded_neighbor(root)):
            if neighbor not in visited:
                visited.append(neighbor)
                results.append(neighbor)
                self._getAtomsIndicesRecursively(neighbor, residue, results, visited)

    # TODO remove ############################
    def isAnyStericClashWithLigand(self, pose, all_ligand_atoms=False, backbone=False, sidechain=False, overlap=1.0):
        """
        Method that finds if there are any clashes between the core atoms or all atoms
        of the ligand and the neighbouring residues (with all their atoms,
        the backbone, or the side chain). An overlap factor can be inserted to allow
        small clashes (the value must be reduced, by default is 1.0).

        OUTPUT
        ------
        boolean value regarding where there are clashes or not
        """

        # Get the coordinates and WdV radii of selected atoms in the target ligand
        ligAtomCoords = list()
        ligAtomWdvRadii = list()
        if all_ligand_atoms:
            for LigAtomIndex in range(1, self.lig.natoms() + 1):
                ligAtomCoords.append(self.lig.xyz(LigAtomIndex))
                element = self.lig.atom_type(LigAtomIndex).element()
                ligAtomWdvRadii.append(AminoAcids().VdwRadii[element])
        else:
            for CoreAtomNames in self.ligCoreAtomNames:
                for resAtomName in CoreAtomNames:
                    ligAtomCoords.append(self.lig.xyz(resAtomName))
                    element = self.lig.atom_type(self.lig.atom_index(resAtomName)).element()
                    ligAtomWdvRadii.append(AminoAcids().VdwRadii[element])

            # Get the coordinates and WdV radii for the neighbouring residues to the ligand
        neighborsAtomCoords = list()
        neighborsAtomWdvRadii = list()
        for resIndex in self.ligNeighbors:
            RES = pose.residue(resIndex)
            for atomIndex in range(1, RES.natoms() + 1):
                if backbone:
                    if RES.atom_is_backbone(atomIndex):
                        neighborsAtomCoords.append(RES.xyz(atomIndex))
                        neighborsAtomWdvRadii.append(
                            AminoAcids().VdwRadii[RES.atom_type(atomIndex).element()] * overlap)
                    else:
                        pass
                elif sidechain:
                    if not RES.atom_is_backbone(atomIndex):
                        neighborsAtomCoords.append(RES.xyz(atomIndex))
                        neighborsAtomWdvRadii.append(
                            AminoAcids().VdwRadii[RES.atom_type(atomIndex).element()] * overlap)
                    else:
                        pass
                else:
                    neighborsAtomCoords.append(RES.xyz(atomIndex))
                    neighborsAtomWdvRadii.append(AminoAcids().VdwRadii[RES.atom_type(atomIndex).element()] * overlap)

        ligAtomCoords = array(ligAtomCoords)
        ligAtomWdvRadii = array(ligAtomWdvRadii)
        neighborsAtomCoords = array(neighborsAtomCoords)
        neighborsAtomWdvRadii = array(neighborsAtomWdvRadii)

        # Calculate the current distances between the selected ligand atoms and the neigbouhring residue atoms
        # Compare it with the minimal WdV distance and check if some has been surpassed (being negative, indicating clash)
        currentDistanceMatrix = sqrt(sum((ligAtomCoords[None, :] - neighborsAtomCoords[:, None]) ** 2, -1))
        wdvDistanceMatrix = neighborsAtomWdvRadii[:, None] + ligAtomWdvRadii[None, :]
        differenceMatrix = (currentDistanceMatrix - wdvDistanceMatrix)

        return any(differenceMatrix < 0)

    def updateLigNeighborsList(self, pose):
        """
        Method that gets the current neighbouring
        residues of the ligand.

        OUTPUT
        ------
        The ligNeighbors attribute of the class is updated
        with the current neighbouring residues
        """

        ligandSelector = ResidueIndexSelector(self.ligPoseIndex)
        nbr_selector = NeighborhoodResidueSelector(ligandSelector, self.distanceCutoff, False)
        nbrBoolVector = nbr_selector.apply(pose)
        nbrBoolVector = array(nbrBoolVector, dtype=bool)
        self.ligNeighbors = where(nbrBoolVector)[0] + 1

    def setFoldTree(self, pose, ligChainId, ligChainName):

        # Reset lig data
        self.ligJumpNumber = None
        self.ligChainNumber = None
        ft = FoldTree()
        nodes = []
        currentChain = 0
        # Get the nodes (N-terminal resides)
        for i, res in enumerate(pose.residues):
            if res.chain() != currentChain:
                currentChain = res.chain()
                nodes.append(i + 1)
                if i + 1 == self.ligPoseIndex and self.ligChainNumber is None:
                    self.ligChainNumber = res.chain()
                    # Catch repeated
                elif res == self.ligPoseIndex and self.ligChainNumber is not None:
                    # raise ValueError('failed setting the fold tree, ligand is repeated.')
                    pass
        for index in range(1, len(nodes)):
            ref, res_i, res_j = nodes[0], nodes[index - 1], nodes[index]
            res_jp = res_j - 1
            if res_jp - res_i != 0:
                ft.add_edge(res_i, res_jp, -1)
            if res_j == self.ligPoseIndex:
                self.ligJumpNumber = index
                # Set the jumps
            ft.add_edge(ref, res_j, index)
        pose.fold_tree(ft)
        if self.ligJumpNumber is None:
            # raise ValueError('failed setting the fold tree, ligand jump number could not be found.')
            print('failed setting the fold tree, ligand jump number could not be found.')


def sideChainRMSD(pose1, pose2, poseIndex, atomIndex):
    natom = pose1.residue(poseIndex).natoms()
    rmsd = 0.0
    count = 0
    for i in range(1, natom + 1):
        if pose1.residue(poseIndex).atom_is_hydrogen(i):
            continue
        elif i not in atomIndex:
            continue
        else:
            rmsd = (pose1.residue(poseIndex).xyz(i).x - pose2.residue(poseIndex).xyz(i).x) ** 2 + \
                   (pose1.residue(poseIndex).xyz(i).y - pose2.residue(poseIndex).xyz(i).y) ** 2 + \
                   (pose1.residue(poseIndex).xyz(i).z - pose2.residue(poseIndex).xyz(i).z) ** 2
            count += 1
    return sqrt(rmsd / count)
