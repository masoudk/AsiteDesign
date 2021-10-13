# Global imports
import sys
import re
import copy
import numpy as np
from numpy import exp, sum, array, zeros, arccos, dot, ones
from numpy.linalg import norm
from numpy.random import randint
from io import StringIO
from mpi4py import MPI

# PyRosetta imports
from pyrosetta.rosetta.protocols.grafting import delete_region, replace_region, return_region, insert_pose_into_pose
from pyrosetta.rosetta.std import ostringstream, istringstream
from pyrosetta.rosetta.core.scoring import score_type_from_name, ScoreType
from pyrosetta import Pose
import pyrosetta as pr

# Biotite import
from biotite.structure.io.pdb import PDBFile
from biotite.structure import sasa, annotate_sse, apply_residue_wise

# Local imports
import Constants
from MiscellaneousUtilityFunctions import killProccesses

# Residue class
class Residue(object):
    """
    This class contains the attributes of a residue from a protein chain.
    The residue number (self.ID), the protein chain (self.chain), and 
    many more. 
    """

    def __init__(self, resID: int = None, chain: str = None):
        self.ID = resID
        self.chain = chain
        self.catalyticName = None
        self.allowedAA = None
        self.nativeAA = None
        self.currentAA = None
        self.poseIndex = None
        self.active = False
        self.designAction = None  # Pick Amino Acid (PIKAA), Native Rotamer (NATRO)
        self.currentAction = None  # Pick Amino Acid (PIKAA), Native Rotamer (NATRO)
        self.mutate = True

    @property
    def name(self):
        return (self.ID, self.chain)

    def show(self):
        return self.ID, self.chain, self.catalyticName, self.allowedAA, self.currentAA, self.nativeAA, self.poseIndex, \
               self.designAction, self.currentAction, self.mutate


class Ligand(Residue):
    def __init__(self, pose: Pose = None, resID: int = None, chain: str = None, excludedTorsions=None):
        super().__init__(resID, chain)

        # A copy of the isolated ligand pose (with out any other molecule)
        self.pose = None
        self.chain = None
        self.ID = None
        self.poseIndex = None
        self.nbrRadii = None
        self.nbrAtomName = None
        self.sasaMAX = None

        self.ljRadii = None
        self.ljEps = None
        self.charges = None
        self.lkdG = None
        self.lkLam = None
        self.lkVol = None

        # core atoms define the rigid part of molecule
        self.atoms = None
        self.numberOfAtoms = 0
        self.coreAtoms = None
        self.sideChainsAtoms = None
        self.sideChainMasks = None
        self.sideChainsSubTreeMasks = None
        self.sideChainsChiXYZIndices = None
        self.ligandMasks = None

        self.nonBondedAtomsPairs = None
        self.nonBondedAtomsPairsWeights = None
        self.bonds = None
        self.bondsLengths = None

        self.angles = None
        self.anglesLengths = None

        # The chi angles of the ligand.
        self.chis = None
        self.chisFull = None

        # The chis are grouped into side chains.
        self.sideChains = None
        self.sideChainsSizes = None
        self.sideChainsIndex = None
        self.numberOfSideChains = None

        # The side chains are ordered in terms of tree level (different grid size for different level)
        self.sideChainsOrdered = None
        self.sideChainsOrderedIndex = None

        # Constructor
        if pose is not None and resID is not None and chain is not None:
            self.initiate(pose, resID, chain, excludedTorsions)

    def initiate(self, pose, resID, chain, excludedTorsions):
        """
        Initiates the ligand parameters. The order of operation is important.
        :param pose:
        :param ligand:
        """
        # get ligand info
        self.ID = resID
        self.chain = chain
        self.poseIndex = pose.pdb_info().pdb2pose(self.chain, self.ID)
        # print(self.poseIndex)
        self.pose = return_region(pose, self.poseIndex, self.poseIndex)
        self.resName = self.pose.residue(1).name3()
        self.nbrRadii = self.pose.residue(1).nbr_radius()
        self.nbrAtomName = self.pose.residue(1).atom_name(self.pose.residue(1).nbr_atom()).split()[0]

        self.getLigandAtomsProperties()

        # get the ligand chi angles (includes only rotatable bonds)
        self.getChis(excludedTorsions)

        # Compute side chains
        self.getSideChains()

        # Compute ordered side chain
        self.getSideChainsOrdered()

        # Get the atom index of each side chain for collision detection
        self.getSideChainsAtoms()

        # Get the core atoms if not given
        self.getCoreAtoms()

        # get ligand atoms
        self.getLigandAtoms()

        self.getSideChainsMask()

        self.getSideChainsSubTreeMasks()

        # Get the nonBonded atoms
        self.getNonBondedAtoms()

        self.checkLigandAtoms()

        self.getBonded()

        self.getSASA()

    def getLigandAtomsProperties(self):
        """
        Gets Van der Waals Radii of ligand atoms
        """
        charges = list()
        ljRadii = list()
        ljEps = list()
        lkdG = list()
        lkLam = list()
        lkVol = list()
        for index in range(1, self.pose.residue(1).natoms() + 1):
            # vdwRadii.append(AminoAcids().VdwRadii[self.pose.residue(1).atom_type(index).element()])
            charges.append(self.pose.residue(1).atomic_charge(index))
            ljRadii.append(self.pose.residue(1).atom_type(index).lj_radius())
            ljEps.append(self.pose.residue(1).atom_type(index).lj_wdepth())
            lkdG.append(self.pose.residue(1).atom_type(index).lk_dgfree())
            lkLam.append(self.pose.residue(1).atom_type(index).lk_lambda())
            lkVol.append(self.pose.residue(1).atom_type(index).lk_volume())

        self.charges = array(charges)
        self.ljRadii = array(ljRadii)
        self.ljEps = array(ljEps)
        self.lkdG = array(lkdG)
        self.lkLam = array(lkLam)
        self.lkVol = array(lkVol)

    def getLigandAtoms(self):
        """
        Gets the atom indices of the ligand
        """
        self.atoms = list(range(1, self.pose.residue(1).natoms() + 1))
        self.numberOfAtoms = len(self.atoms)

    def checkLigandAtoms(self):
        ligandSetsAtoms = []
        for i in self.coreAtoms:
            if i in self.atoms:
                ligandSetsAtoms.append(i)
        for sideChain in self.sideChainsAtoms:
            for i in sideChain:
                if i in self.atoms:
                    ligandSetsAtoms.append(i)
        if len(ligandSetsAtoms) != len(self.atoms):
            raise ValueError('Ligand could not be initialized. The total atoms in ligand core and side chains differ '
                             'from ligand total number of atoms.')

    def getChis(self, excludedTorsions):
        """
        Get the ligand chis excluding the ones in the excludedTorsions list
        :param excludedTorsions: list of torsion tuples with atom name
        """
        self.chisFull = list(tuple(tors) for tors in self.pose.residue(1).chi_atoms())
        self.chis = list()
        nbrAtom = self.pose.residue(1).nbr_atom()
        for torsion in self.chisFull:
            if torsion[1] != nbrAtom and torsion[2] != nbrAtom:
                self.chis.append(torsion)

        # remove excluded torsions if a list is given
        if type(excludedTorsions) is list:
            for excludedTorsion in excludedTorsions:
                excludedTorsion = tuple(excludedTorsion)
                ligandChisNames = [tuple(self.pose.residue(1).atom_name(i).split()[0] for i in chi) for chi in
                                   self.chis]
                if excludedTorsion not in ligandChisNames:
                    raise ValueError('Bad excluded torsion. {} is not part of ligand chis: \n   {}'
                                     .format(str(excludedTorsion), '\n   '.join([str(i) for i in ligandChisNames])))
                else:
                    index = ligandChisNames.index(excludedTorsion)
                    self.chis.pop(index)

        # remove all if All is asked
        elif excludedTorsions == 'All':
            self.chis = list()

    def getSideChains(self):
        """
        Groups the chi angles of the ligand into side chains by traversing
        the chis of the ligand.
        """
        self.sideChains = []
        self.sideChainsSizes = []
        self.numberOfSideChains = 0
        # Get all possible side-chain in the ligand.
        # This would result in fragmented sub-trees if
        # side chain contains non rotatable bonds
        sideChains = list()
        for root in self.chis:
            sideChain = set()
            sideChain.add(root)
            self._getSideChainsForward(root, self.chis, sideChain)
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
                self._getAtomsIndicesRecursively(root, self.pose.residue(1), sideChainAtoms, visited)
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
            self.sideChains.append(sideChainsUnique[i])
            self.sideChainsSizes.append(len(sideChainsUnique[i]))

        self.numberOfSideChains = len(self.sideChains)

        # get the side chain indices too
        self.getSideChainsIndex()

    def getSideChainsIndex(self):
        """
        Get the chi index of torsions in each side chain. This would allow for
        fast access of ligand torsions.
        """
        # Get the indices
        sideChainsIndex = list()
        for sideChain in self.sideChains:
            sideChainIndex = set()
            for torsion in sideChain:
                sideChainIndex.add(self.chisFull.index(torsion) + 1)
            sideChainsIndex.append(sideChainIndex)
        self.sideChainsIndex = sideChainsIndex

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
                if torsion in self.chis:
                    sideChain.add(torsion)
                elif tuple(reversed(torsion)) in self.chis:
                    sideChain.add(tuple(reversed(torsion)))
                return visited

    def getSideChainsOrdered(self):
        """
        Get the level of each torsion in each side chains. This is used for level
        dependant grid generation in Minimum Energy Neighborhood Search side chain packing.
        the ordered side chains are encoded as list of list of list, where the first index is
        the side chain and the second index is the level of torsions . The final index refers
        to torsion, i.e. [0][1][2] is the third level 1 torsion in side chain 0.
        [[(i,j,k,l)], [(i,j,k,l), (i,j,k,l)]]
        """
        self.sideChainsOrdered = list()
        for sideChain in self.sideChains:
            # The torsions for each side chain are ordered in terms of number of atoms in
            # their sub tree.
            sideChain = list(sideChain)
            torsionsNumberOfAtoms = list()
            for index, torsion in enumerate(sideChain):
                subTreeAtoms = list()
                root = torsion[2]
                # add atoms atoms j and k to stop backward travers
                visited = [torsion[1], torsion[2]]
                self._getAtomsIndicesRecursively(root, self.pose.residue(1), subTreeAtoms, visited)
                torsionsNumberOfAtoms.append((index, len(subTreeAtoms)))

            # Group torsions based on their atom count
            torsionsNumberOfAtoms.sort(key=lambda x: x[1], reverse=True)
            currentAtomCount = float('inf')
            sideChainOrdered = []
            for index, torsionNumberOfAtoms in torsionsNumberOfAtoms:
                if torsionNumberOfAtoms < currentAtomCount:
                    currentAtomCount = torsionNumberOfAtoms
                    sideChainOrdered.append([sideChain[index]])
                else:
                    sideChainOrdered[-1].extend([sideChain[index]])

            # Add the ordered side chain list to the list of all side chains
            self.sideChainsOrdered.append(sideChainOrdered)
        """
        print('ligandChis', [[self.pose.residue(1).atom_name(i) for i in chi] for chi in self.chisFull])
        print('ligandChis', [[self.pose.residue(1).atom_name(i) for i in chi] for chi in self.chis])
        print('ligandSideChains',
              [[[self.pose.residue(1).atom_name(i) for i in chi] for chi in sidechain] for sidechain in
               self.sideChains])
        print('ligandSideChainsIndex', self.sideChainsIndex)
        print('ligandSideChainsOrdered', self.sideChainsOrdered)
        print('ligandSideChainsOrdered',
              [[[tuple(self.pose.residue(1).atom_name(i) for i in chi) for chi in level] for level in sidechain] for
               sidechain in self.sideChainsOrdered])
        for sidechain in self.sideChainsOrdered:
            print('side chain')
            for i, level in enumerate(sidechain):
                print('level: ', i, level)

        """
        # get the side chain indices too
        self.getSideChainsOrderedIndex()

    def getSideChainsOrderedIndex(self):
        """
        Get the chi index of torsions in each ordered side chain. This would allow for
        fast access of ligand torsions. The sideChainsOrderedIndex has same structure as
        sideChainsOrdered
        """

        # get the indices of ordered side chains
        sideChainsOrderedIndex = list()
        for sideChainOrdered in self.sideChainsOrdered:
            sideChainOrderedIndex = list()
            for level in sideChainOrdered:
                levelIndex = list()
                for torsion in level:
                    levelIndex.append(self.chisFull.index(torsion) + 1)
                sideChainOrderedIndex.append(levelIndex)
            sideChainsOrderedIndex.append(sideChainOrderedIndex)

        self.sideChainsOrderedIndex = sideChainsOrderedIndex

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

    def getSideChainsAtoms(self):
        """
        Get the atoms involved in each side chain using the l atom of
        level 0 torsion as root by traversing the molecule toward terminal atoms
        """
        self.sideChainsAtoms = list()
        for sideChain in self.sideChainsOrdered:
            sideChainAtoms = list()
            root = sideChain[0][0][2]
            # add atoms j and k to stop backward travers
            visited = [sideChain[0][0][1], sideChain[0][0][2]]
            self._getAtomsIndicesRecursively(root, self.pose.residue(1), sideChainAtoms, visited)
            self.sideChainsAtoms.append(sideChainAtoms)

    def getSideChainsMask(self):
        self.sideChainMasks = list()
        self.ligandMasks = ones(self.numberOfAtoms)
        for sidechainAtoms in self.sideChainsAtoms:

            sideChainMask = zeros(self.numberOfAtoms)
            for sidechainAtom in sidechainAtoms:
                sideChainMask[sidechainAtom - 1] = 1
            self.sideChainMasks.append(sideChainMask)

    def getSideChainsSubTreeMasks(self):
        self.sideChainsSubTreeMasks = list()
        self.sideChainsChiXYZIndices = list()
        for sideChain in self.sideChainsOrdered:
            sideChainSubTreeMasks = list()
            sideChainChiXYZIndices = list()
            for level in sideChain:
                for chi in level:
                    chiSubTreeMasks = zeros(self.numberOfAtoms)
                    #print('         CHI: {}'.format(str([self.pose.residue(1).atom_name(j).split()[0] for j in chi])))
                    sideChainSubTreeAtoms = list()
                    root = chi[2]
                    visited = [chi[0], chi[1], chi[2]]

                    self._getAtomsIndicesRecursively(root, self.pose.residue(1), sideChainSubTreeAtoms, visited)
                    for sideChainSubTreeAtom in sideChainSubTreeAtoms:
                        chiSubTreeMasks[sideChainSubTreeAtom - 1] = 1
                    #print('         ATOMS: {}'.format(str([self.pose.residue(1).atom_name(j).split()[0] for j in sideChainSubTreeAtoms])))
                    #print(chiSubTreeMasks)
                    sideChainSubTreeMasks.append(chiSubTreeMasks)
                    sideChainChiXYZIndices.append(np.array([chi]) - 1)
                    #print(chiSubTreeMasks)

            self.sideChainsSubTreeMasks.append(np.array(sideChainSubTreeMasks, dtype=np.intc).reshape(len(sideChainSubTreeMasks), self.numberOfAtoms))
            self.sideChainsChiXYZIndices.append(np.array(sideChainChiXYZIndices, dtype=np.intc).reshape(len(sideChainChiXYZIndices), 4))

        #for i, sideChainSubTreeMasks in enumerate(self.sideChainsSubTreeMasks):
            #print(i, ': ', sideChainSubTreeMasks.shape)
            #print('indix', self.sideChainsChiXYZIndices[i])
            #print('   ', sideChainSubTreeMasks)

        #print('ALL')
        #print(self.sideChainMasks)

    def getCoreAtoms(self):
        """
        Compute core atoms using the nbr atom as the root atom and k atoms of all level 0 torsions
        as the search boundary.
        """
        self.coreAtoms = list()
        visited = list()

        # add k atom of all level 0 torsions in all side chains to the visited list
        visited.extend([sideChain[0][0][2] for sideChain in self.sideChainsOrdered])

        # add root atom to visited
        visited.append(self.pose.residue(1).nbr_atom())

        # the root and k atoms are part of core atoms by default
        self.coreAtoms.extend(visited)

        # Perform a recursive search to get the rest of core atoms
        self._getAtomsIndicesRecursively(self.pose.residue(1).nbr_atom(), self.pose.residue(1), self.coreAtoms, visited)

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

    def getNonBondedAtoms(self):
        nonBondedAtomsPairs = list()
        nonBondedAtomsPairsWeights = list()
        for i in range(1, self.pose.residue(1).natoms() + 1):
            for j in range(i + 1, self.pose.residue(1).natoms() + 1):
                # check for 1-2, 1-3 connection
                if not self._nighbors_depthwise(i, j, self.pose.residue(1), max_depth=2):
                    if [i, j] not in nonBondedAtomsPairs and [j, i] not in nonBondedAtomsPairs:
                        # Check for 1-4 connection
                        if self._nighbors_depthwise(i, j, self.pose.residue(1), max_depth=3):
                            w = 0.2
                        else:
                            w = 1
                        nonBondedAtomsPairsWeights.append(w)
                        nonBondedAtomsPairs.append([i, j])

        # find the pairs that belong to aromatic rings
        nonBondedAtomsPairsNoRing = list()
        nonBondedAtomsPairsWeightsNoRing = list()
        for index, pair in enumerate(nonBondedAtomsPairs):
            i, j = pair
            # find the rings <= 6 atoms involving atom_i
            iRings = list()
            keepPair = True
            self._find_rings(target=i, root=i, res=self.pose.residue(1), rings=iRings)
            for ring in iRings:
                if j in ring:
                    keepPair = False
            # print(self.pose.residue(1).atom_name(i), self.pose.residue(1).atom_name(j), keepPair)
            if keepPair:
                nonBondedAtomsPairsNoRing.append([i, j])
                nonBondedAtomsPairsWeightsNoRing.append(nonBondedAtomsPairsWeights[index])

        self.nonBondedAtomsPairs = nonBondedAtomsPairsNoRing
        self.nonBondedAtomsPairsWeights = nonBondedAtomsPairsWeightsNoRing

    def _nighbors_depthwise(self, root, target, res, max_depth, depth=0):
        depth += 1
        found = False
        if depth <= max_depth:
            for nbr in res.bonded_neighbor(root):
                if nbr == target:
                    found = True
                    break
                else:
                    found = self._nighbors_depthwise(nbr, target, res, max_depth, depth)
                    if found:
                        break
            return found

    def _find_rings(self, target, root, res, rings, visited=[], depth=-1):
        depth += 1
        myvisited = visited.copy()
        myvisited.append(root)
        for nbr in res.bonded_neighbor(root):
            if nbr == target and depth > 1:
                if len(myvisited) <= 6:
                    ring = set(myvisited)
                    if ring not in rings:
                        rings.append(ring)
            if nbr not in myvisited:
                # print(depth, ': ', root, nbr, myvisited)
                self._find_rings(target, nbr, res, rings, myvisited, depth)

    def getBonded(self):
        self.getBonds()
        self.getAngles()

    def getBonds(self):
        """
        Finds all i, j of ligand bonds
        """
        self.bonds = list()
        self.bondsLengths = list()

        # Find all bonds
        for atomIndex in range(1, self.pose.residue(1).natoms() + 1):
            for neighborIndex in list(self.pose.residue(1).bonded_neighbor(atomIndex)):
                if [atomIndex, neighborIndex] not in self.bonds and [neighborIndex, atomIndex] not in self.bonds:
                    self.bonds.append([atomIndex, neighborIndex])

        # Find the equilibrium distance
        for i, j in self.bonds:
            ri = self.pose.residue(1).xyz(i)
            rj = self.pose.residue(1).xyz(j)
            d0 = ri.distance(rj)
            self.bondsLengths.append(d0)

    def getAngles(self):
        """
        Finds all i, j, k of ligand Angles
        """
        self.angles = list()
        self.anglesLengths = list()

        # Find all angles
        for i, j in self.bonds:
            for k, l in self.bonds:
                # ignore itself
                if i == k and j == l or i == l and j == k:
                    continue
                elif j == k and [i, j, l] not in self.angles and [l, j, i] not in self.angles:
                    self.angles.append([i, j, l])
                elif j == l and [i, j, k] not in self.angles and [k, j, i] not in self.angles:
                    self.angles.append([i, j, k])
                elif i == k and [j, i, l] not in self.angles and [l, i, j] not in self.angles:
                    self.angles.append([j, i, l])
                elif i == l and [j, i, k] not in self.angles and [k, i, j] not in self.angles:
                    self.angles.append([j, i, k])

        # find the equilibrium angle
        for i, j, k in self.angles:

            ri = array(self.pose.residue(1).xyz(i), dtype=np.float32)
            rj = array(self.pose.residue(1).xyz(j), dtype=np.float32)
            rk = array(self.pose.residue(1).xyz(k), dtype=np.float32)
            rji = ri - rj
            rjk = rk - rj
            scp = dot(rji, rjk) / (norm(rji) * norm(rjk))

            # This is to avoid numerical error due to precision
            if scp > 1: scp = 1
            if scp < -1: scp = 1

            teta0 = arccos(scp)
            self.anglesLengths.append(teta0)

    def getSASA(self):

        # Get the pdb file out of the pose
        posePdb = ostringstream()
        self.pose.dump_pdb(posePdb)
        pdbFile = StringIO(posePdb.str())

        # Get the PDB file
        pdb = PDBFile.read(pdbFile)
        pdbStruc = pdb.get_structure()[0]

        # Calculate SASA
        atomsSASA = sasa(pdbStruc, vdw_radii="Single")
        strucSASA = apply_residue_wise(pdbStruc, atomsSASA, sum)
        self.sasaMAX = strucSASA[0]

    # TODO return ligand attributes
    def show(self):
        string = ''
        string += 'Ligand: {}-{}, PoseIndex: {}\n'.format(self.ID, self.chain, self.poseIndex)
        string += ' NBR Atom: {}, NBR Radii {}\n'.format(self.nbrAtomName, self.nbrRadii)
        string += ' Chis Angles (Full): \n'
        for i, chi in enumerate(self.chisFull):
            string += '     {}: {} \n'.format(i + 1, str([self.pose.residue(1).atom_name(j).split()[0] for j in chi]))
        string += ' Chis Angles (Final): \n'

        for chi in self.chis:
            i = self.chisFull.index(chi) + 1
            string += '     {}: {} \n'.format(i, str([self.pose.residue(1).atom_name(j).split()[0] for j in chi]))

        string += ' SideChains Ordered: \n'
        for i, sidechain in enumerate(self.sideChainsOrdered):
            for j, level in enumerate(sidechain):
                for chi in level:
                    k = self.chisFull.index(chi) + 1
                    string += '     SC: {}  level:  {}  Chi:  {}, {} \n'.format(i, j, k, str(
                        [self.pose.residue(1).atom_name(l).split()[0] for l in chi]))

        string += ' Atom Names: \n     '
        for i, atom in enumerate(self.atoms):
            if (i + 1) % 20 == 0:
                string += '{:4s} \n     '.format(str(self.pose.residue(1).atom_name(atom).split()[0]))
            else:
                string += '{:4s} '.format(str(self.pose.residue(1).atom_name(atom).split()[0]))
        string += '\n'

        string += ' Core Atom Names: \n     '
        for i, atom in enumerate(self.coreAtoms):
            if (i + 1) % 20 == 0:
                string += '{:4s}\n     '.format(str(self.pose.residue(1).atom_name(atom).split()[0]))
            else:
                string += '{:4s} '.format(str(self.pose.residue(1).atom_name(atom).split()[0]))
        string += '\n'

        string += ' Side Chains Atom Names: \n'
        for i, sidechain in enumerate(self.sideChainsAtoms):
            string += '   SC: {} \n     '.format(i)
            for j, atom in enumerate(sidechain):
                if (j + 1) % 20 == 0:
                    string += '{:4s} \n     '.format(str(self.pose.residue(1).atom_name(atom).split()[0]))
                else:
                    string += '{:4s} '.format(str(self.pose.residue(1).atom_name(atom).split()[0]))
            string += '\n'
        string += '\n'

        return string


# Constraint classes
class BaseConstraint(object):
    def __init__(self):
        self.type = None


class GeometricConstraint(BaseConstraint):
    """
    This class contains the parameters for the distance constraints around
    a set of atoms. In the case of DesignCatalytic, the catalytic distances
    of the active site.
    """

    def __init__(self):
        super().__init__()
        self.type = None
        self.res_i = None
        self.res_j = None
        self.res_k = None
        self.atom_i_list = []
        self.atom_j_list = []
        self.atom_k_list = []
        self.lb = 0.0
        self.hb = 0.0
        self.sd = 0.0
        self.tag = None

    def show(self):
        if self.type == 'B':
            return 'type: {}, i: {}:{}, j: {}:{}, lb: {}, hb: {}, sd: {:.2f}, tag: {}'.format(self.type,
                                                                                          ':'.join(map(lambda x: str(x),
                                                                                                       self.res_i)),
                                                                                          (','.join(self.atom_i_list)),
                                                                                          ':'.join(map(lambda x: str(x),
                                                                                                       self.res_j)),
                                                                                          (','.join(self.atom_j_list)),
                                                                                          self.lb, self.hb, self.sd,
                                                                                          self.tag)
        else:
            raise ValueError('show() for constraint {} type is not implemented'.format(self.type))


class SequenceConstraint(BaseConstraint):
    """
    This class contains the parameters for the sequence constraints of 
    a residue or set of residues.
    """

    def __init__(self):
        super().__init__()
        self.type = None
        self.weight = 0
        self.tag = None
        self.ref = None
        self.res = dict()  # It is a dict of (resname, chain): 'aa'

    def show(self):
        if self.type == 'S':
            string = ''
            string += 'type: {}, weight: {}, tag: {}\n'.format(self.type, self.weight, self.tag)
            string += '   constraint list:\n   '
            for i, element in enumerate(
                    [('{}-{}'.format(residue[0], residue[1]), aa) for residue, aa in self.res.items()]):
                if (i + 1) % 5 == 0:
                    string += '{:6s}: {}\n   '.format(element[0], element[1])
                else:
                    string += '{:6s}: {}   '.format(element[0], element[1])

            return string

        else:
            raise ValueError('show() for constraint {} type is not implemented'.format(self.type))


# Input and Result classes
class Inputs(object):
    """
    Servers as a dictionary for input data.
    """
    pass


class Result(object):
    """
    Servers as a dictionary for the results.
    """
    pass


class AminoAcids():
    def __init__(self):
        self.wildCardSelection = {'ALLAA': 'ARNDCEQGHILKMFPSTWYV',  # All amino acids
                                  'ALLA*': 'ARNDCEQHILKMFSTWYV',  # All amino acids excluding P and G
                                  'APOLAR': 'AILMFWV',
                                  'POLAR': 'RNDCEQHKSTY',
                                  'POLARN': 'NCQHST',
                                  'POLARA': 'DE',
                                  'POLARB': 'RK'}

        # MAX SASA for normalization of SASA
        self.MaxSASAAminoAcid3Dict = {'ALA': 129, 'PRO': 159, 'ASN': 195, 'HID': 224,
                                      'VAL': 174, 'TYR': 263, 'CYS': 167, 'LYS': 236,
                                      'ILE': 197, 'PHE': 240, 'GLN': 225, 'SER': 155,
                                      'LEU': 201, 'TRP': 285, 'GLU': 223, 'THR': 172,
                                      'MET': 224, 'ARG': 274, 'GLY': 104, 'ASP': 193}

        self.VdwRadii = {'Ac': 2.00, 'Al': 2.00, 'Am': 2.00, 'Sb': 2.00, 'Ar': 1.88,
                         'As': 1.85, 'At': 2.00, 'Ba': 2.00, 'Bk': 2.00, 'Be': 2.00,
                         'Bi': 2.00, 'Bh': 2.00, 'B': 2.00, 'Br': 1.85, 'Cd': 1.58,
                         'Cs': 2.00, 'Ca': 2.00, 'Cf': 2.00, 'C': 1.70, 'Ce': 2.00,
                         'Cl': 1.75, 'Cr': 2.00, 'Co': 2.00, 'Cu': 1.40, 'Cm': 2.00,
                         'Ds': 2.00, 'Db': 2.00, 'Dy': 2.00, 'Es': 2.00, 'Er': 2.00,
                         'Eu': 2.00, 'Fm': 2.00, 'F': 1.47, 'Fr': 2.00, 'Gd': 2.00,
                         'Ga': 1.87, 'Ge': 2.00, 'Au': 1.66, 'Hf': 2.00, 'Hs': 2.00,
                         'He': 1.40, 'Ho': 2.00, 'H': 1.09, 'In': 1.93, 'I': 1.98,
                         'Ir': 2.00, 'Fe': 2.00, 'Kr': 2.02, 'La': 2.00, 'Lr': 2.00,
                         'Pb': 2.02, 'Li': 1.82, 'Lu': 2.00, 'Mg': 1.73, 'Mn': 2.00,
                         'Mt': 2.00, 'Md': 2.00, 'Hg': 1.55, 'Mo': 2.00, 'Nd': 2.00,
                         'Ne': 1.54, 'Np': 2.00, 'Ni': 1.63, 'Nb': 2.00, 'N': 1.55,
                         'No': 2.00, 'Os': 2.00, 'O': 1.52, 'Pd': 1.63, 'P': 1.80,
                         'Pt': 1.72, 'Pu': 2.00, 'Po': 2.00, 'K': 2.75, 'Pr': 2.00,
                         'Pm': 2.00, 'Pa': 2.00, 'Ra': 2.00, 'Rn': 2.00, 'Re': 2.00,
                         'Rh': 2.00, 'Rb': 2.00, 'Ru': 2.00, 'Rf': 2.00, 'Sm': 2.00,
                         'Sc': 2.00, 'Sg': 2.00, 'Se': 1.90, 'Si': 2.10, 'Ag': 1.72,
                         'Na': 2.27, 'Sr': 2.00, 'S': 1.80, 'Ta': 2.00, 'Tc': 2.00,
                         'Te': 2.06, 'Tb': 2.00, 'Tl': 1.96, 'Th': 2.00, 'Tm': 2.00,
                         'Sn': 2.17, 'Ti': 2.00, 'W': 2.00, 'U': 1.86, 'V': 2.00,
                         'Xe': 2.16, 'Yb': 2.00, 'Y': 2.00, 'Zn': 1.39, 'Zr': 2.00,
                         'X': 1.60}

    def selection(self, wildcard):
        return list(self.wildCardSelection[wildcard])

    def maxSASA3(self, AminoAcid3):
        return self.MaxSASAAminoAcid3Dict[AminoAcid3]


def DesignDomainWildCards(wildcard=all):
    aaWildcards = list(AminoAcids().wildCardSelection.keys())
    allWildcards = ['ZZ', 'ZX', 'XX', 'XX+']
    allWildcards.extend(aaWildcards)

    wildCardSelection = {'all': allWildcards,
                         'aminoAcids': aaWildcards,
                         'domain': ['ZZ', 'ZX', 'XX', 'XX+'],
                         'frozen': ['ZZ'],
                         'noneMutable': ['ZZ', 'ZX'],
                         'noneMutablePackable': ['ZX'],
                         'mutable': ['XX', 'XX+'],
                         'nativePlus': ['ZZ', 'ZX', 'XX+']}

    return wildCardSelection[wildcard]
