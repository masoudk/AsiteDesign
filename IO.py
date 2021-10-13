# Global imports
import os
import sys
import re
import copy
from numpy import exp, sum, array, zeros, sqrt
from numpy.random import randint, uniform
from numpy.linalg import norm
from io import StringIO
from mpi4py import MPI
from collections import namedtuple

# Local imports
import Constants
from MiscellaneousUtilityFunctions import killProccesses, printDEBUG
#from MoversRosetta import InterfaceSymmetric, InterfaceAsymmetric, InterfaceDomainMover
from MoversRosetta import getScoreFunction
from BaseClases import DesignDomainWildCards, GeometricConstraint, SequenceConstraint #, InterfaceBoundary
from BaseClases import Residue, Ligand


# Bio Python import
from Bio.PDB.Polypeptide import one_to_three

# PyRosetta import
from pyrosetta import Pose, pose_from_pdb, Vector1, pose_from_file


class InputBase(object):
    """
    Contains the methods needed for input handeling
    """
    def __init__(self):
        #self.comm = MPI.COMM_WORLD
        self.nProccess = MPI.COMM_WORLD.Get_size()

        self._jobtype: str = ''
        self._name: str= ''
        self._pose: Pose = None
        self._parameterFiles: list = None
        self._nPoses: int = 0
        self._nSteps: int = 0
        self._nIterations: int = 0
        self._kT: bool = False
        self._simulationTime: bool = False
        self._simulationTimeFrequency: int = 10
        self._writeALL: bool = False
        self._packerLoops: int = 10

        # output path
        self._outPath: str = ''
        self._bestPosesPath: str = ''
        self._scratch: str = ''

        # to be overwritten by subclasses
        self._designResidues: list = list()

        # Constraints
        self._constraints: list = list()

        # Anneal Parameters
        self._anneal: bool = True
        self._kT_high: float = 1000.0
        self._kT_low: float = 1.0
        self._kT_decay: bool = True
        self._kT_highScale: bool = True

        # Metrics
        self._rankingMetric = False
        self._spawningMetric = False
        self._spawningMetricSteps = False
        self._spawningMethod = None

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, name: str):
        # Make the pose
        if name:
            self._name = name
        else:
            self._name = self.jobType

    @property
    def outPath(self):
        return self._outPath

    @outPath.setter
    def outPath(self, name: str):
        if name:
            self._outPath = os.path.join('{}_{}_output'.format(self.jobType, name))
        else:
            self._outPath = os.path.join('{}_output'.format(self.jobType))

        if not os.path.isdir(self._outPath):
            os.makedirs(self._outPath)
        else:
            for i in range(100):
                try:
                    newOutPath = os.path.join('{}_{}'.format(self._outPath, i))
                    os.makedirs(newOutPath)
                    break
                except:
                    pass
            if i == 99:
                raise ValueError('Failed creating output folders.')
            else:
                self._outPath = newOutPath

    @property
    def bestPosesPath(self):
        return self._bestPosesPath

    @bestPosesPath.setter
    def bestPosesPath(self, name: str):
        if name:
            self._bestPosesPath = os.path.join('{}_{}_final_pose'.format(self.jobType, name))
        else:
            self._bestPosesPath = os.path.join('{}_final_pose'.format(self.jobType))

        if not os.path.isdir(self._bestPosesPath):
            os.makedirs(self._bestPosesPath)
        else:
            for i in range(100):
                try:
                    newOutPath = os.path.join('{}_{}'.format(self._bestPosesPath, i))
                    os.makedirs(newOutPath)
                    break
                except:
                    pass
            if i == 99:
                raise ValueError('Failed creating output folders.')
            else:
                self._bestPosesPath = newOutPath

    @property
    def scratch(self):
        return self._scratch

    @scratch.setter
    def scratch(self, name: str):
        if name:
            self._scratch = os.path.join('{}_{}_scratch'.format(self.jobType, name))
        else:
            self._scratch = os.path.join('{}_scratch'.format(self.jobType))

        if not os.path.isdir(self._scratch):
            os.makedirs(self._scratch)
        else:
            for i in range(100):
                try:
                    newOutPath = os.path.join('{}_{}'.format(self._scratch, i))
                    os.makedirs(newOutPath)
                    break
                except:
                    pass
            if i == 99:
                raise ValueError('Failed creating output folders.')
            else:
                self._scratch = newOutPath

    @property
    def parameterFiles(self):
        return self._parameterFiles

    @parameterFiles.setter
    def parameterFiles(self, parameterFiles: list):
        self._parameterFiles = list()

        if parameterFiles is None:
            return
        elif type(parameterFiles) is not list:
            raise ValueError('Bad ParameterFiles, expecting a list')

        for file in parameterFiles:
            if os.path.isfile(file):
                self._parameterFiles.append(file)
            else:
                raise ValueError('Bad ParameterFiles, {} was not found'.format(file))

    @property
    def pose(self):
        return self._pose

    @pose.setter
    def pose(self, pdbFileName: str):
        # Make the pose
        if not pdbFileName:
            raise ValueError('No pdb file name is given')

        if not os.path.isfile(pdbFileName):
            raise ValueError("failed reading the pdb file: {}.".format(pdbFileName))

        self._pose = Pose()
        res_set = self._pose.conformation().modifiable_residue_type_set_for_conf()
        if len(self._parameterFiles) > 0:
            res_set.read_files_for_base_residue_types(Vector1(self._parameterFiles))
        self._pose.conformation().reset_residue_type_set_for_conf(res_set)
        pose_from_file(self._pose, pdbFileName)

    @property
    def jobType(self):
        raise ValueError('jobType method should be overwritten by subclass')

    @jobType.setter
    def jobType(self, jobType: str):
        raise ValueError('jobType method should be overwritten by subclass')

    @property
    def nPoses(self):
        return self._nPoses

    @nPoses.setter
    def nPoses(self, nPoses: int):
            # Check the number of poses
        if not nPoses:
            raise ValueError('no nPoses is given.')

        if self.nProccess > 1 and nPoses > self.nProccess - 1:
            raise ValueError("nPose can not exceed 'nCPUs - 1', found {} ".format(nPoses))
        self._nPoses = nPoses

    @property
    def nIterations(self):
        return self._nIterations

    @nIterations.setter
    def nIterations(self, nIterations: int):
        self._nIterations = nIterations

    @property
    def nSteps(self):
        return self._nSteps

    @nSteps.setter
    def nSteps(self, nSteps: int):
        self._nSteps = nSteps

    @property
    def packerLoops(self):
        return self._packerLoops

    @packerLoops.setter
    def packerLoops(self, packerLoops: int):
        if type(packerLoops) != int:
            raise ValueError('Bad PackerLoops, int is expected.')

        self._packerLoops = packerLoops

    @property
    def kT(self):
        return self._kT

    @kT.setter
    def kT(self, kT: float):
        self._kT = kT

    @property
    def anneal(self):
        return self._anneal

    @anneal.setter
    def anneal(self, anneal: bool):
        # Set the default values.
        if anneal:
            self._anneal = True
            self._kT_high = 1000
            self._kT_low = 1
            self._kT_decay = True
            self._kT_highScale = True
        else:
            self._anneal = False
            self._kT_high = 1
            self._kT_low = 1
            self._kT_decay = False
            self._kT_highScale = False

    @property
    def kT_high(self):
        return self._kT_high

    @kT_high.setter
    def kT_high(self, kT_high: float):
        if not self.anneal:
            raise ValueError("Anneal is not set, can not set kT_high")
        self._kT_high = kT_high

    @property
    def kT_low(self):
        return self._kT_low

    @kT_low.setter
    def kT_low(self, kT_low: float):
        if not self.anneal:
            raise ValueError("Anneal is not set, can not set kT_high")
        self._kT_low = kT_low

    @property
    def kT_decay(self):
        return self._kT_decay

    @kT_decay.setter
    def kT_decay(self, kT_decay: bool):
        if not self.anneal:
            raise ValueError("Anneal is not set, can not set kT_high")
        self._kT_decay = kT_decay

    @property
    def kT_highScale(self):
        return self._kT_highScale

    @kT_highScale.setter
    def kT_highScale(self, kT_highScale: bool):
        if not self.anneal:
            raise ValueError("Anneal is not set, can not set kT_high")
        self._kT_highScale = kT_highScale

    @property
    def rankingMetric(self):
        return self._rankingMetric

    @rankingMetric.setter
    def rankingMetric(self, rankingMetric: str):
        if rankingMetric not in ['FullAtom', 'FullAtomWithConstraints', 'OnlyConstraints', 'SASA', 'Ligand']:
            raise ValueError("bad rankingMetric: {}.".format(rankingMetric))
        else:
            self._rankingMetric = rankingMetric

    @property
    def spawningMetric(self):
        return self._spawningMetric

    @spawningMetric.setter
    def spawningMetric(self, spawningMetric: str):
        if spawningMetric not in ['FullAtom', 'FullAtomWithConstraints', 'OnlyConstraints', 'SASA', 'Split', 'Ligand']:
            raise ValueError('bad spawningMetric {}'.format(spawningMetric))
        else:
            self._spawningMetric = spawningMetric

    @property
    def spawningMetricSteps(self):
        return self._spawningMetricSteps

    @spawningMetricSteps.setter
    def spawningMetricSteps(self, spawningMetricSteps: list):

        if self.spawningMetric == 'Split' and spawningMetricSteps is None:
            raise ValueError('SpawningMetric Split is given without defining SpawningMetricSteps')

        # set to None if not given, to check during update
        if spawningMetricSteps is None:
            return

        spawningMetricStepsTM = list()
        for index, element in enumerate(spawningMetricSteps):
            try:
                iterationRatio, spawningMetricMethod = element.split()
                iterationRatio = float(iterationRatio)
            except Exception:
                raise ValueError('failed reading SpawningMetricSteps {}.'.format(element))

            if iterationRatio <= 0 or iterationRatio > 1:
                raise ValueError(
                    "bad SpawningMetric: {}: {}. Iteration ratio is defined in the interval (0, 1].".format(iterationRatio, spawningMetricMethod))

            elif spawningMetricMethod not in ['FullAtom', 'FullAtomWithConstraints', 'OnlyConstraints', 'SASA', 'Ligand']:
                raise ValueError("bad SpawningMetric: {}: {}. Unknown Metric".format(iterationRatio, spawningMetricMethod))

            if index > 0:
                if iterationRatio <= spawningMetricStepsTM[index - 1][0]:
                    raise ValueError("bad SpawningMetric: {}: {}. Iteration step can not be less than the previous one.".format(iterationRatio, spawningMetricMethod))

                # If pass add it as list
            spawningMetricStepsTM.append([iterationRatio, spawningMetricMethod])

        # Check the last element
        if spawningMetricStepsTM[-1][0] != 1.0:
            print("Warning >>> bad SpawningMetricStep: {}: {}, The iteration ratio of the last step should match 1.0.".format(spawningMetricStepsTM[-1][0], spawningMetricStepsTM[-1][1]))
            print("            Setting the last step to: {}: {}.".format(1, spawningMetricStepsTM[-1][1]))
            spawningMetricStepsTM[-1][0] = 1

        self._spawningMetricSteps = spawningMetricStepsTM

    @property
    def spawningMethod(self):
        return self._spawningMethod

    @spawningMethod.setter
    def spawningMethod(self, method):

        if method is None:
            method = 'Adaptive'

        if method not in ['REM', 'Adaptive']:
            raise ValueError('Bad SpawningMethod, expecting REM or Adaptive.')

        if method == 'Adaptive':
            self._spawningMethod = 'Adaptive'
        elif method == 'REM':
            self._spawningMethod = 'REM'

    @property
    def simulationTime(self):
        return self._simulationTime

    @property
    def simulationTimeFrequency(self):
        return self._simulationTimeFrequency

    @simulationTimeFrequency.setter
    def simulationTimeFrequency(self, simulationTimeFrequency: int):
        self._simulationTimeFrequency = simulationTimeFrequency

    #TODO approximate simulationTimeFrequency
    @simulationTime.setter
    def simulationTime(self, simulationTime: float):

        if simulationTime:
            self._simulationTime = simulationTime * 3600
            self.simulationTimeFrequency = 10

    @property
    def writeALL(self):
        return self._writeALL

    @writeALL.setter
    def writeALL(self, writeALL: bool):
        self._writeALL = writeALL

    @property
    def designResidues(self):
        return self._designResidues

    @designResidues.setter
    def designResidues(self, residues: list):
        self._designResidues = residues

    @property
    def constraints(self):
        return self._constraints

    @constraints.setter
    def constraints(self, constraints: dict):
        if constraints is None:
            self._constraints = None
            return

        if not self.designResidues:
            raise ValueError('Constraints can not be set before defining the designResidues')

        self._constraints = list()
        sequenceConstraintFound = False
        for tag, cst in constraints.items():

            # Read the distance constraints
            type = cst.get('type', None)
            if type not in ['B', 'S']:
                raise ValueError('failed initiating constraint. Constraint {} is of unknown type.'.format(tag))

            if type == 'B':
                constraint = GeometricConstraint()
                constraint.type, constraint.tag = type, tag

                # Read the redi, resj
                res_i = cst.get('resi', None)
                res_j = cst.get('resj', None)
                if res_i is None or res_j is None:
                    raise ValueError('failed initiating constraint. Constraint {} has no resi/resj.'.format(tag))

                # Convert res ID to int only if possible
                res_i = cst['resi'].strip().split('-')
                res_j = cst['resj'].strip().split('-')

                try:
                    res_i[0] = int(res_i[0])
                except:
                    pass

                try:
                    res_j[0] = int(res_j[0])
                except:
                    pass

                    # Make the resNames
                res_i = tuple(res_i)
                res_j = tuple(res_j)

                # get the atomi and atomj
                atom_i_list = cst.get('atomi', None)
                atom_j_list = cst.get('atomj', None)
                if atom_i_list is None or atom_j_list is None:
                    raise ValueError('failed initiating constraint. Constraint {} has no atomi/atomj.'.format(tag))

                atom_i_list = cst['atomi'].strip().split('-')
                atom_j_list = cst['atomj'].strip().split('-')

                # get the lb, hb, and sd
                lb = cst.get('lb', None)
                hb = cst.get('hb', None)
                sd = cst.get('sd', None)

                if lb is None or hb is None or sd is None:
                    raise ValueError('failed initiating constraint. Constraint {} has no lb/hb/sd.'.format(tag))

                # Convert the K to standard deviation
                sd = 1 / sqrt(sd)
                constraint.res_i = res_i
                constraint.res_j = res_j
                constraint.atom_i_list = atom_i_list
                constraint.atom_j_list = atom_j_list
                constraint.lb = lb
                constraint.hb = hb
                constraint.sd = sd

                self._constraints.append(constraint)

            # Read the sequence constraints
            elif type == 'S':

                # Don't allow multiple sequence constraints
                if sequenceConstraintFound:
                    raise ValueError('multiple sequence constraints are not allowed.')
                else:
                    sequenceConstraintFound = True

                constraint = SequenceConstraint()
                constraint.type, constraint.tag = type, tag

                constraint.weight = cst.get('weight', 5)
                if constraint.weight < 0:
                    raise ValueError('Bad sequence constraint. Constraint weight < 0 is not allowed.')

                constraint.ref = cst.get('reference', None)
                if constraint.ref:
                    if not os.path.isfile(constraint.ref):
                        raise ValueError('Bad sequence constraint. The reference file could not be found')

                residues_tm = cst.get('residues', None)

                # If no residues are explicitly is given assign all desig residues
                if residues_tm is None:
                    if constraint.ref is None:
                        raise ValueError(
                            'Bad sequence constraint. Sequence constraint(s) without residue definition requires a reference structure.')
                    else:
                        # Add all design residues to constraint with _ as the place holder
                        for resName in [res.name for res in self.designResidues]:
                            constraint.res[resName] = 'ZZ'

                # if residues are explicitly given add them
                else:
                    for residue, aa in residues_tm.items():
                        # Make the res name tuple
                        try:
                            ID, chain = residue.split('-')
                            ID, chain = int(ID), str(chain)
                            resName = (ID, chain)
                            constraint.res[resName] = aa
                        except:
                            raise ValueError('bad sequence constraint. Residue {} don\'t seems to be valid.'.format(residue))

                            # Check the validity of the amino acid in the restraint
                        if aa != 'ZZ':
                            try:
                                one_to_three(aa)
                            except:
                                raise ValueError('bad sequence constraint. Residue {} don\'t seems to have a valid amino acid {}.'.format(residue, aa))

                            # Check the constraint consistency
                        if resName not in [res.name for res in self.designResidues]:
                            raise ValueError('bad sequence constraint. Residue {} was not found in the design domain.'.format(resName))

                            # Check the presence of ref
                        if re.match('ZZ', aa) and constraint.ref is None:
                            raise ValueError('bad sequence constraint. Target amino acid for residue {} is defined as native and requires a reference structure.')

                    # SORT OUT TARGET AA
                if constraint.ref is not None:
                    # Check reff pdb file exist
                    if not os.path.isfile(constraint.ref):
                        raise ValueError("failed initiating constraints. No reference structure was found: {}.".format(constraint.ref))
                    refStructure = Pose()
                    res_set = refStructure.conformation().modifiable_residue_type_set_for_conf()
                    if len(self._parameterFiles) > 0:
                        res_set.read_files_for_base_residue_types(Vector1(self._parameterFiles))
                    refStructure.conformation().reset_residue_type_set_for_conf(res_set)
                    pose_from_file(refStructure, constraint.ref)

                for residue, aa in constraint.res.items():
                    # If not given, get it from ref
                    if aa == 'ZZ':
                        poseIndex = refStructure.pdb_info().pdb2pose(residue[1], residue[0])
                        if poseIndex == 0:
                            raise ValueError('Bad sequence constraint. Residue {}-{} was not found in the pdb file {}'.format(residue[0], residue[1], constraint.ref))
                        constraint.res[residue] = refStructure.residue(poseIndex).name1()
                        # Test it for validity
                    else:
                        try:
                            one_to_three(aa)
                        except:
                            raise ValueError('Bad sequence constraint. Target amino acid {}, for residue {} don\'t seems to be valid.'.format(aa, residue))

                self._constraints.append(constraint)


class InputActiveSiteDesign(InputBase):

    def __init__(self):
        super().__init__()
        self._jobType = ''
        self._catalyticResidues = None
        self._ligands = None
        self._ligandsParm = None
        self._activeSiteDesignMode = None
        self._activeSiteLoops = None
        self._nNoneCatalytic = None
        self._softRepulsion = None
        self._activeSiteSampling = None
        self._ligandSampling = None
        self._dynamicSideChainCoupling = None
        self._mimimizeBackbone = None
        self._ligandClusterCutoff = 1.0

    @property
    def jobType(self):
        return self._jobType

    @jobType.setter
    def jobType(self, jobType: str):
        if not jobType:
            self._jobType = ''

        self._jobType = jobType

    @property
    def designResidues(self):
        return self._designResidues

    @designResidues.setter
    def designResidues(self, designDict: dict):

        if designDict is None:
            raise ValueError('No designResidues were given.')

        elif self.pose is None:
            raise ValueError('No pose is defined. DesignResidues cant nto be set before settig pose.')

        # Initiate the list
        self._designResidues = list()

        for res, aa in designDict.items():
            id, chain = res.strip().split('-')
            allowedAA = aa.strip().split('-')

            # Check input consistency
            id, chain = int(id), str(chain)
            poseindex = self.pose.pdb_info().pdb2pose(chain, id)
            if poseindex == 0:
                raise ValueError("failed initiating design residues. Residue {} was not found in PDB.".format(res))

            # Check for wildcards
            if len(allowedAA[0]) > 1:

                # Only accept one wildcard at each position
                if len(allowedAA) > 1:
                    raise ValueError("failed initiating design residues {}. More that one wildcard specified".format(res))

                    # Check for the allowed wildcards
                if not allowedAA[0] in DesignDomainWildCards('all'):
                    raise ValueError("failed initiating design residues {}. {} wildcard is not valid ({}).".
                                     format(res, allowedAA[0],DesignDomainWildCards('all')))

            # Check for the amino acid
            elif len(allowedAA[0]) == 1:
                # Check the given amino acids are valid
                dummy = list()
                for aaName in allowedAA:
                    try:
                        one_to_three(aaName)
                    except:
                        raise ValueError(
                            "failed initiating design residues {}. The allowed amino acid {} seems not to be valid.".format(res, aaName))

                        # Check it is not repeating
                    if aaName in dummy:
                        raise ValueError("failed initiating design residues {}. The allowed amino acid {} is repeating.".format(res, aaName))

                    else:
                        dummy.append(aaName)

            else:
                raise ValueError(
                    "failed initiating design residues {}. No allowed amino acid(s) is given for this position.".format(res))

            residue = Residue(id, chain)
            residue.allowedAA = allowedAA
            self._designResidues.append(residue)

    @property
    def catalyticResidues(self):
        return self._catalyticResidues

    @catalyticResidues.setter
    def catalyticResidues(self, catalyticResidues: dict):

        if catalyticResidues is None:
            self._catalyticResidues = None
            return

        elif catalyticResidues is not None and self.pose is None:
            raise ValueError('No pose is defined. CatalyticResidue cant not be set before setting pose.')

        # Initiate the list
        self._catalyticResidues = list()
        for res, aa in catalyticResidues.items():

            # Convert res ID to int only if possible
            res = res.strip().split('-')
            aa = aa.strip().split('-')
            try:
                res[0] = int(res[0])
            except:
                pass

            resName = tuple(res)

            # Check the input consistency
            if len(aa[0]) > 1:

                # Check for the wildcard
                if len(aa) > 1:
                    raise ValueError("failed initiating catalytic residues {}.  More that one wildcard specified.".format(res))

                if aa[0] not in DesignDomainWildCards('noneMutable'):
                    raise ValueError(
                        "failed initiating catalytic residues {}. {} wildcard is not valid {}.".format(res, aa[0],
                                                                                                       DesignDomainWildCards(
                                                                                                           'noneMutable')))

            # Check for normal AA
            elif len(aa[0]) == 1:
                dummy = list()
                for targetAA in aa:
                    # Check it is valid AA
                    try:
                        one_to_three(targetAA)
                    except:
                        raise ValueError(
                            "failed initiating catalytic residues {}. The target amino acid {} seems not to be valid.".format(
                                res, targetAA))

                    # Check it is not repeating
                    if targetAA in dummy:
                        raise ValueError(
                            "failed initiating catalytic residues {}. The target amino acid {} is repeating.".format(
                                res, targetAA))
                    else:
                        dummy.append(targetAA)
            else:
                raise ValueError(
                    "failed initiating catalytic residues {}. No target amino acid(s) is given for this position.".format(
                        res))

            # Check the catalytic residues are defined
            #if re.match('DesignNoneCatalytic', jobType, re.IGNORECASE):
            #    if resName not in domain.designDict.keys():
            #        raise ValueError(
            #            "failed initiating catalytic. Residues {} not found in design domain. Ambiguous residues are not allowed in {}.".format(
            #                res, jobType))

            self._catalyticResidues.append([resName, aa])

    @property
    def ligands(self):
        return self._ligands

    @property
    def ligandsParm(self):
        return self._ligandsParm

    @ligands.setter
    def ligands(self, ligands: list):

        if ligands is None:
            self._ligands = None
            return
        elif ligands is not None and self.pose is None:
            raise ValueError('No pose is defined. Ligands cant not be set before setting pose.')

        # Initiate the ligand and their parameters lists
        self._ligands = list()
        self._ligandsParm = list()
        for ligand in ligands:

            # Check ligand is in the pose
            ligName = str(*ligand.keys())
            ligParms = dict(*ligand.values())
            ID, chain = int(ligName.split('-')[0]), ligName.split('-')[1]
            poseIndex = self.pose.pdb_info().pdb2pose(chain, ID)

            if poseIndex == 0:
                raise ValueError("failed initiating ligand. Residue {} was not found in PDB.".format(ligName))

            # Check the PerturbationMode, if not given set to default
            if ligParms.get('PerturbationMode', None) is None:
                ligParms['PerturbationMode'] = 'MC'
            else:
                if  ligParms.get('PerturbationMode', None) not in ['MC', 'MIN']:
                    raise ValueError("failed initiating ligand {}. Bad PerturbationMode, expecting MC or MIN.".format(ligName))

            # Check the PerturbationLoops, if not given set to default
            if ligParms.get('PerturbationLoops', None) is None:
                ligParms['PerturbationLoops'] = 1
            else:
                if type(ligParms['PerturbationLoops']) != int:
                    raise ValueError("failed initiating ligand {}. Bad PerturbationLoops, expecting int.".format(ligName))

            # Check the nRandomTorsionPurturbation, if not given set to default
            if ligParms.get('nRandomTorsionPurturbation', None) is None:
                ligParms['nRandomTorsionPurturbation'] = 1
            else:
                if type(ligParms['nRandomTorsionPurturbation']) != int:
                    raise ValueError("failed initiating ligand {}. Bad nRandomTorsionPurturbation, expecting int.".format(ligName))

            # Check for the rigid body parameters
            if ligParms.get('RigidBody', None) is None:
                ligParms['RigidBody'] = False
            else:
                if ligParms['RigidBody'] not in [True, False]:
                    raise ValueError(
                        "failed initiating ligand {}. Bad RigidBody, expecting True or False.".format(ligName))

            # if dockingCenter is not defined, it will be computed from ligand centroid
            if ligParms.get('DockingCenter', None) is None:
                ligParms['DockingCenter'] = None
                print('Warning >>> No DockingCenter is defined for ligand {}, will be computed from ligand\'s centroid.'.format(ligName))

            # If it is defined make sure it is defined properly.
            else:
                if type(ligParms['DockingCenter']) is not list:
                    raise ValueError("failed initiating ligand {}. Bad DockingCenter, expecting a list.".format(ligName))
                elif len(ligParms['DockingCenter']) != 3:
                    raise ValueError("failed initiating ligand {}. Bad DockingCenter, expecting a list of three.".format(ligName))
                else:
                    try:
                        dummy = float(ligParms['DockingCenter'][0])
                        dummy = float(ligParms['DockingCenter'][1])
                        dummy = float(ligParms['DockingCenter'][2])
                    except:
                        raise ValueError("failed initiating ligand {}. Bad DockingCenter, expecting a list of floats.".format(ligName))

            # if SimulationCenter is not defined, it will be computed from ligand centroid
            if ligParms.get('SimulationCenter', None) is None:
                ligParms['SimulationCenter'] = None
                print('Warning >>> No SimulationCenter is defined for ligand {}, will be computed from ligand\'s centroid.'.format(ligName))

            # If it is defined make sure it is defined properly.
            else:
                if type(ligParms['SimulationCenter']) is not list:
                    raise ValueError("failed initiating ligand {}. Bad SimulationCenter, expecting a list.".format(ligName))
                elif len(ligParms['SimulationCenter']) != 3:
                    raise ValueError("failed initiating ligand {}. Bad SimulationCenter, expecting a list of three.".format(ligName))
                else:
                    try:
                        dummy = float(ligParms['SimulationCenter'][0])
                        dummy = float(ligParms['SimulationCenter'][1])
                        dummy = float(ligParms['SimulationCenter'][2])
                    except:
                        raise ValueError("failed initiating ligand {}. Bad SimulationCenter, expecting a list of floats.".format(ligName))

            # if SimulationRadius is not defined, it will be set to 15.0 default
            if ligParms.get('SimulationRadius', None) is None:
                ligParms['SimulationRadius'] = 15.0
                print('Warning >>> No SimulationRadius is defined for ligand {}, set to 15.0.'.format(ligName))

            # If it is defined make sure it is defined properly.
            else:
                try:
                    dummy = float(ligParms['SimulationRadius'])
                except:
                    raise ValueError("failed initiating ligand {}. Bad SimulationRadius, expecting a floats.".format(ligName))

            # if SideChainCoupling is not defined, it will be set to 0.0 default
            if ligParms.get('SideChainCoupling', None) is None:
                ligParms['SideChainCoupling'] = 0.0
                print('Warning >>> No SideChainCoupling is defined for ligand {}, set to 0.0.'.format(ligName))

            # If it is defined make sure it is defined properly.
            else:
                try:
                    dummy = float(ligParms['SideChainCoupling'])
                except:
                    raise ValueError("failed initiating ligand {}. Bad SideChainCoupling, expecting a floats.".format(ligName))

            # if SideChainCouplingMax is not defined, it will be set to 0.0 default
            if ligParms.get('SideChainCouplingMax', None) is None:
                ligParms['SideChainCouplingMax'] = 0.0

            # If it is defined make sure it is defined properly.
            else:
                try:
                    dummy = float(ligParms['SideChainCouplingMax'])
                except:
                    raise ValueError("failed initiating ligand {}. Bad SideChainCouplingMax, expecting a floats.".format(ligName))

            # if TranslationSTD is not defined, it will be set to 5.0 default
            if ligParms.get('TranslationSTD', None) is None:
                ligParms['TranslationSTD'] = 5.0
                print('Warning >>> No TranslationSTD is defined for ligand {}, set to 5.0.'.format(ligName))

            # If it is defined make sure it is defined properly.
            else:
                try:
                    dummy = float(ligParms['TranslationSTD'])
                except:
                    raise ValueError("failed initiating ligand {}. Bad TranslationSTD, expecting a floats.".format(ligName))

            # if RotationSTD is not defined, it will be set to 30.0 default
            if ligParms.get('RotationSTD', None) is None:
                ligParms['RotationSTD'] = 30.0
                print('Warning >>> No RotationSTD is defined for ligand {}, set to 30.0.'.format(ligName))

            # If it is defined make sure it is defined properly.
            else:
                try:
                    dummy = float(ligParms['RotationSTD'])
                except:
                    raise ValueError("failed initiating ligand {}. Bad RotationSTD, expecting a floats.".format(ligName))

            # if TranslationLoops is not defined, it will be set to 10 default
            if ligParms.get('TranslationLoops', None) is None:
                ligParms['TranslationLoops'] = 10
                print('Warning >>> No TranslationLoops is defined for ligand {}, set to 10.'.format(ligName))

            # If it is defined make sure it is defined properly.
            else:
                try:
                    dummy = int(ligParms['TranslationLoops'])
                except:
                    raise ValueError("failed initiating ligand {}. Bad TranslationLoops, expecting an int.".format(ligName))

            # if RotationLoops is not defined, it will be set to 50 default
            if ligParms.get('RotationLoops', None) is None:
                ligParms['RotationLoops'] = 50
                print('Warning >>> No RotationLoops is defined for ligand {}, set to 50.'.format(ligName))

            # If it is defined make sure it is defined properly.
            else:
                try:
                    dummy = int(ligParms['RotationLoops'])
                except:
                    raise ValueError("failed initiating ligand {}. Bad RotationLoops, expecting an int.".format(ligName))

            # if ClashOverlap is not defined, it will be set to 0.8 default
            if ligParms.get('ClashOverlap', None) is None:
                ligParms['ClashOverlap'] = 0.8
                print('Warning >>> No ClashOverlap is defined for ligand {}, set to 0.8.'.format(ligName))

            # If it is defined make sure it is defined properly.
            else:
                try:
                    dummy = float(ligParms['ClashOverlap'])
                except:
                    raise ValueError("failed initiating ligand {}. Bad ClashOverlap, expecting a floats.".format(ligName))

            # if NeighbourCutoff is not defined, it will be set to nbr + 4.0 default
            if ligParms.get('NeighbourCutoff', None) is None:
                neighbourCutoff = self.pose.residue(poseIndex).nbr_radius() + 5.0
                ligParms['NeighbourCutoff'] = neighbourCutoff
                print('Warning >>> No NeighbourCutoff is defined for ligand {}, set to {}.'.format(ligName, neighbourCutoff))

            # If it is defined make sure it is defined properly.
            else:
                try:
                    dummy = float(ligParms['NeighbourCutoff'])
                except:
                    raise ValueError("failed initiating ligand {}. Bad NeighbourCutoff, expecting a floats.".format(ligName))

            # if SasaScaling is not defined, it will be set to false, but other parameters should be initiated to
            # be passed to the class constructor, eventhogh they are not used
            if ligParms.get('SasaScaling', None) is None:
                ligParms['SasaScaling'] = False
                ligParms['SasaCutoff'] = 0.5
                ligParms['TranslationScale'] = 0.5
                ligParms['RotationScale'] = 0.5

            # If it is defined set other parameters too.
            elif ligParms['SasaScaling'] is True:

                # if sasaScaling, SasaCutoff should be defined
                if ligParms.get('SasaCutoff', None) is None:
                    ligParms['SasaCutoff'] = 0.5
                else:
                    try:
                        sasaCutoff = float(ligParms['SasaCutoff'])
                    except:
                        raise ValueError("failed initiating ligand {}. Bad SasaCutoff, expecting a floats.".format(ligName))

                    if sasaCutoff > 1.0 or sasaCutoff <= 0.0:
                        raise ValueError("failed initiating ligand {}. Bad SasaCutoff, expecting between (0, 1].".format(ligName))

                # Check for the TranslationScale
                if ligParms.get('TranslationScale', None) is None:
                    ligParms['TranslationScale'] = 0.5
                else:
                    try:
                        dummy = float(ligParms['TranslationScale'])
                    except:
                        raise ValueError("failed initiating ligand {}. Bad TranslationScale, expecting a floats.".format(ligName))

                # Check for the RotationScale
                if ligParms.get('RotationScale', None) is None:
                    ligParms['RotationScale'] = 0.5
                else:
                    try:
                        dummy = float(ligParms['RotationScale'])
                    except:
                        raise ValueError("failed initiating ligand {}. Bad RotationScale, expecting a floats.".format(ligName))

            # Check ligand SASA Constraint
            if ligParms.get('SasaConstraint', None) is None:
                ligParms['SasaConstraint'] = 0.0
            else:
                try:
                    dummy = float(ligParms['SasaConstraint'])
                except:
                    raise ValueError("failed initiating ligand {}. Bad SasaConstraint, expecting a floats.".format(ligName))

            # Check for the packer parameters
            if ligParms.get('Packing', None) is None:
                ligParms['Packing'] = False

            # if Energy is not defined, it will be set to Reduced default
            if ligParms.get('Energy', None) is None:
                ligParms['Energy'] = 'Reduced'
                print('Warning >>> No Energy is defined for ligand {}, set to "Reduced".'.format(ligName))

            # If it is defined make sure it is defined properly.
            else:
                if ligParms['Energy'] not in ['Reduced', 'Full']:
                    raise ValueError("failed initiating ligand {}. Bad Energy, expecting either 'Reduced' or 'Full'.".format(ligName))

            # Set the Full Energy flag for packer
            if ligParms['Energy'] == 'Full':
                ligParms['FullEnergy'] = True
            else:
                ligParms['FullEnergy'] = False


            # if PackingLoops is not defined, it will be set to 1 default
            if ligParms.get('PackingLoops', None) is None:
                ligParms['PackingLoops'] = 1
                print('Warning >>> No PackingLoops is defined for ligand {}, set to 1.'.format(ligName))

            # If it is defined make sure it is defined properly.
            else:
                try:
                    dummy = int(ligParms['PackingLoops'])
                except:
                    raise ValueError("failed initiating ligand {}. Bad PackingLoops, expecting an int.".format(ligName))


            # if NumberOfGridNeighborhoods is not defined, it will be set to 2 default
            if ligParms.get('NumberOfGridNeighborhoods', None) is None:
                ligParms['NumberOfGridNeighborhoods'] = 2
                print('Warning >>> No NumberOfGridNeighborhoods is defined for ligand {}, set to 2.'.format(ligName))

            # If it is defined make sure it is defined properly.
            else:
                try:
                    dummy = int(ligParms['NumberOfGridNeighborhoods'])
                except:
                    raise ValueError("failed initiating ligand {}. Bad NumberOfGridNeighborhoods, expecting an int.".format(ligName))


            # if MaxGrid is not defined, it will be set to 12 default
            if ligParms.get('MaxGrid', None) is None:
                ligParms['MaxGrid'] = 12
                print('Warning >>> No MaxGrid is defined for ligand {}, set to 12.'.format(ligName))

            # If it is defined make sure it is defined properly.
            else:
                try:
                    dummy = int(ligParms['MaxGrid'])
                except:
                    raise ValueError("failed initiating ligand {}. Bad MaxGrid, expecting an int.".format(ligName))

            # if MinGrid is not defined, it will be set to 6 default
            if ligParms.get('MinGrid', None) is None:
                ligParms['MinGrid'] = 6
                print('Warning >>> No MinGrid is defined for ligand {}, set to 6.'.format(ligName))

            # If it is defined make sure it is defined properly.
            else:
                try:
                    dummy = int(ligParms['MinGrid'])
                except:
                    raise ValueError("failed initiating ligand {}. Bad MinGrid, expecting an int.".format(ligName))


            # if GridInterval is not defined, it will be set to 360 default
            if ligParms.get('GridInterval', None) is None:
                ligParms['GridInterval'] = 360
                print('Warning >>> No GridInterval is defined for ligand {}, set to 360.'.format(ligName))

            # If it is defined make sure it is defined properly.
            else:
                try:
                    dummy = int(ligParms['GridInterval'])
                except:
                    raise ValueError("failed initiating ligand {}. Bad GridInterval, expecting an int.".format(ligName))


            # if SideChainsGridLimit is not defined, set to None
            if ligParms.get('SideChainsGridLimit', None) is None:
                ligParms['SideChainsGridLimit'] = None

            # If it is defined make sure it is defined properly.
            else:
                if type(ligParms['SideChainsGridLimit']) is not list:
                    raise ValueError("failed initiating ligand {}. Bad SideChainsGridLimit, expecting a list.".format(ligName))

                for limit in ligParms['SideChainsGridLimit']:
                    if len(limit) != 3:
                        raise ValueError("failed initiating ligand {}. Bad SideChainsGridLimit, expecting a list of size 3.".format(ligName))

                    try:
                        dummy = int(limit[0])
                        dummy = int(limit[1])
                        dummy = float(limit[2])
                    except:
                        raise ValueError("failed initiating ligand {}. Bad GridInterval, expecting int, int, float.".format(ligName))


            # if SideChainsGridLimit is not defined, set to None
            if ligParms.get('TorsionsGridPoints', None) is None:
                ligParms['TorsionsGridPoints'] = None

            # If it is defined make sure it is defined properly.
            else:
                if type(ligParms['TorsionsGridPoints']) is not dict:
                    raise ValueError("failed initiating ligand {}. Bad TorsionsGridPoints, expecting a dict.".format(ligName))

                for torsionIndex, gridpoint in ligParms['TorsionsGridPoints'].items():
                    if len(gridpoint) != 2:
                        raise ValueError("failed initiating ligand {}. Bad TorsionsGridPoints, expecting a dict of  'int: [int, float]'.".format(ligName))

                    try:
                        dummy = int(torsionIndex)
                        dummy = int(gridpoint[0])
                        dummy = float(gridpoint[1])
                    except:
                        raise ValueError("failed initiating ligand {}. Bad TorsionsGridPoints, expecting a dict of  'int: [int, float]'.".format(ligName))


            # Check for the ExcludedTorsions. This depends on operations on this ligand
            if ligParms['Packing'] is True:
                if ligParms.get('ExcludedTorsions', None) is None:
                    ligParms['ExcludedTorsions'] = None
                else:
                    if type(ligParms['ExcludedTorsions']) is not list:
                        raise ValueError("failed initiating ligand {}. Bad ExcludedTorsions, expecting a list.".format(ligName))

                    for torsion in ligParms['ExcludedTorsions']:
                        if len(torsion) != 4:
                            raise ValueError("failed initiating ligand {}. Bad ExcludedTorsions, expecting a list of size 4.".format(ligName))

            # If the ligand has RGB mover but no packing, all torsions should be excluded. This would imply that
            # the core atoms involve whole ligand.
            elif ligParms['Packing'] is False and ligParms['RigidBody'] is True:
                ligParms['ExcludedTorsions'] = 'All'

            elif ligParms['Packing'] is False and ligParms['RigidBody'] is False:
                # No point including the ligand, other ligand will see this as frozen and all atoms are part of core atoms
                print('Warning >>> The RigidBody and Packing are False for Ligand {}. This ligand would be ignored'.format(ligName))

            # create the ligand and parm lists
            if ligParms['Packing'] is True or ligParms['RigidBody'] is True:
                ligand = Ligand(pose=self.pose, resID=ID, chain=chain, excludedTorsions=ligParms['ExcludedTorsions'])
                self._ligands.append(ligand)
                self._ligandsParm.append(ligParms)

    @property
    def activeSiteDesignMode(self):
        return self._activeSiteDesignMode

    @activeSiteDesignMode.setter
    def activeSiteDesignMode(self, mode):

        if mode is None:
            self._activeSiteDesignMode = 'MC'
            return

        if mode not in ['MC', 'MIN']:
            raise ValueError('Bad CatalyticDesignMode, expecting MC or MIN')
        else:
            self._activeSiteDesignMode = mode

    @property
    def mimimizeBackbone(self):
        return self._mimimizeBackbone

    @mimimizeBackbone.setter
    def mimimizeBackbone(self, mimimizeBackbone):
        if mimimizeBackbone is None:
            self._mimimizeBackbone = False
        elif mimimizeBackbone not in [True, False]:
            raise ValueError('Bad MimimizeBackbone, expected boolean.')
        else:
            self._mimimizeBackbone = mimimizeBackbone


    @property
    def ligandClusterCutoff(self):
        return self._ligandClusterCutoff

    @ligandClusterCutoff.setter
    def ligandClusterCutoff(self, ligandClusterCutoff):
        if ligandClusterCutoff is None:
            self._ligandClusterCutoff = 1.0
        elif ligandClusterCutoff <= 0:
            raise ValueError('Bad LigandClusterCutoff, expected a positive float.')
        else:
            self._ligandClusterCutoff = ligandClusterCutoff

    @property
    def activeSiteLoops(self):
        return self._activeSiteLoops

    @activeSiteLoops.setter
    def activeSiteLoops(self, loops: int):
        if type(loops) != int:
            raise ValueError('Bad activeSiteLoops, int is expected.')
        self._activeSiteLoops = loops

    @property
    def nNoneCatalytic(self):
        return self._nNoneCatalytic

    @nNoneCatalytic.setter
    def nNoneCatalytic(self, loops: int):
        if type(loops) != int:
            raise ValueError('Bad nNoneCatalytic, int is expected.')
        self._nNoneCatalytic = loops

    @property
    def softRepulsion(self):
        return self._softRepulsion

    @softRepulsion.setter
    def softRepulsion(self, rep: bool):
        if type(rep) != bool:
            raise ValueError('Bad SoftRepulsion, True/False is expected.')
        self._softRepulsion = rep

    @property
    def activeSiteSampling(self):
        return self._activeSiteSampling

    @activeSiteSampling.setter
    def activeSiteSampling(self, mode):

        if mode is None:
            self._activeSiteSampling = 'MC'

        if mode not in ['MC', 'MIN', 'Coupled']:
            raise ValueError('Bad ActiveSiteSampling, expecting MC, MIN or Coupled')
        else:
            self._activeSiteSampling = mode

    @property
    def ligandSampling(self):
        return self._ligandSampling

    @ligandSampling.setter
    def ligandSampling(self, mode):

        if mode is None:
            self._ligandSampling = 'Coupled'

        if mode not in ['MC', 'MIN', 'Coupled']:
            raise ValueError('Bad LigandSampling, expecting MC, MIN or Coupled')
        else:
            self._ligandSampling = mode

    @property
    def dynamicSideChainCoupling(self):
        return self._dynamicSideChainCoupling

    @dynamicSideChainCoupling.setter
    def dynamicSideChainCoupling(self, mode):

        if mode is None:
            self._dynamicSideChainCoupling = False

        if mode not in [True, False]:
            raise ValueError('Bad BynamicSideChainCoupling, expecting True or False')
        else:
            self._dynamicSideChainCoupling = mode

    @property
    def outPath(self):
        return self._outPath

    @outPath.setter
    def outPath(self, name: str):
        if name:
            self._outPath = os.path.join('{}_output'.format(name))
        else:
            self._outPath = os.path.join('output')

        if not os.path.isdir(self._outPath):
            os.makedirs(self._outPath)
        else:
            for i in range(100):
                try:
                    newOutPath = os.path.join('{}_{}'.format(self._outPath, i))
                    os.makedirs(newOutPath)
                    break
                except:
                    pass
            if i == 99:
                raise ValueError('Failed creating output folders.')
            else:
                self._outPath = newOutPath

    @property
    def bestPosesPath(self):
        return self._bestPosesPath

    @bestPosesPath.setter
    def bestPosesPath(self, name: str):
        if name:
            self._bestPosesPath = os.path.join('{}_final_pose'.format(name))
        else:
            self._bestPosesPath = os.path.join('final_pose')

        if not os.path.isdir(self._bestPosesPath):
            os.makedirs(self._bestPosesPath)
        else:
            for i in range(100):
                try:
                    newOutPath = os.path.join('{}_{}'.format(self._bestPosesPath, i))
                    os.makedirs(newOutPath)
                    break
                except:
                    pass
            if i == 99:
                raise ValueError('Failed creating output folders.')
            else:
                self._bestPosesPath = newOutPath

    @property
    def scratch(self):
        return self._scratch

    @scratch.setter
    def scratch(self, name: str):
        if name:
            self._scratch = os.path.join('{}_scratch'.format(name))
        else:
            self._scratch = os.path.join('scratch')

        if not os.path.isdir(self._scratch):
            os.makedirs(self._scratch)
        else:
            for i in range(100):
                try:
                    newOutPath = os.path.join('{}_{}'.format(self._scratch, i))
                    os.makedirs(newOutPath)
                    break
                except:
                    pass
            if i == 99:
                raise ValueError('Failed creating output folders.')
            else:
                self._scratch = newOutPath

    def show(self):
        # Write some concepts of the Yaml control file in the console or the output file
        print('\n')
        print('-----------------------------------------INPUT SUMMARY------------------------------------------------')
        print('DEBUG: {}'.format(Constants.DEBUG))
        print('Parameter Files: {}'.format('  '.join([str(i) for i in self.parameterFiles])))
        print('PDB: {}'.format(self.pose.pdb_info().name()))
        print('Path to Outputs: {}'.format(self.outPath))
        print('Path to Final Results: {}'.format(self.bestPosesPath))
        print('Path to Scratch: {}'.format(self.scratch))
        print('Simulation Conditions:')
        print('     nPoses: {}\n     writeAll:  {}'.format(self.nPoses, self.writeALL))
        print('     nSteps: {}\n     nIterations: {}'.format(self.nSteps, self.nIterations))
        print('     LigandClusterCutoff: {}'.format(self.ligandClusterCutoff))
        print('     SimulationTime: {} s\n     SimulationTimeFrequency: {}'
                                                      .format(int(self.simulationTime), self.simulationTimeFrequency))
        if self.anneal:
            print('     Anneal: True\n     kT_high: {}\n     kT_low: {}\n     kT_decay: {}\n     kT_highScale: {}'
                                                 .format(self.kT_high, self.kT_low, self.kT_decay, self.kT_highScale))
        else:
            print('     Anneal: False\n     kT: {}'.format(self.kT))

        print('     RankingMetric: {}'.format(self.rankingMetric))
        print('     SpawningMethod: {}'.format(self.spawningMethod))
        print('     SpawningMetric: {}'.format(self.spawningMetric))
        if self.spawningMetricSteps:
            for element in self.spawningMetricSteps:
                print('         Itiration ratio: {}  {}'.format(element[0], element[1]))

        print('Design Modes:')
        print('     ActiveSiteDesignMode: {}'.format(self.activeSiteDesignMode))
        print('     MimimizeBackbone: {}'.format(self.mimimizeBackbone))
        print('     PackerLoops: {}'.format(self.packerLoops))
        print('     ActiveSiteLoops: {}'.format(self.activeSiteLoops))
        print('     nNoneCatalytic: {}'.format(self.nNoneCatalytic))

        print('Sampling Modes:')
        print('     ActiveSiteSampling: {}'.format(self.activeSiteSampling))
        print('     LigandSampling: {}'.format(self.ligandSampling))
        print('     DynamicSideChainCoupling: {}'.format(self._dynamicSideChainCoupling))
        print('     SoftRepulsion: {}'.format(self.softRepulsion))

        print('Design residues:')
        for res in self.designResidues:
            print('     {:7}: {}'.format('{}-{}'.format(res.ID, res.chain), ''.join(res.allowedAA)))

        print('Catalytic residues:')
        if self.catalyticResidues is not None:
            for resName, aa in self._catalyticResidues:
                if len(resName) == 2: # for printing nonmute res
                    print('     {:7}: {}    '.format('{}-{}'.format(resName[0], resName[1]), ''.join(aa)))
                else:
                    print('     {:7}: {}'.format('{}'.format(*resName), ''.join(aa)))
        else:
            print('     None')

        print('Ligands:')
        if self.ligands is not None:
            # Create a list to sort, its confusing without sorting
            for ligand, ligandParm in zip(self.ligands, self.ligandsParm):
                print('   Ligand {}:'.format(ligand.name))
                parmList = list()
                for parmName, parmValue in ligandParm.items():
                    parmList.append([parmName, parmValue])

                parmList.sort(key=lambda element: element[0])
                for parmName, parmValue in parmList:
                    print('         {}: {}'.format(parmName, str(parmValue)))

        print('Constraints:')
        if self.constraints is not None:
            for constraint in self.constraints:
                print(constraint.show())
        print('------------------------------------------------------------------------------------------------------')
        print('\n')
        print('----------------------------------------LIGANDS SUMMARY------------------------------------------------')
        if self.ligands:
            for ligand in self.ligands:
                print(ligand.show())

        else:
            print("Ligand: None")
        print('------------------------------------------------------------------------------------------------------')
        print('\n')

    def read(self, inputFile):
        #self.jobType = inputFile.get('JobType', '')
        print('Initiating pose.')
        self.parameterFiles = inputFile.get('ParameterFiles', None)
        self.pose = inputFile.get('PDB', '')

        print('Initiating simulation parameters.')
        self.nPoses = inputFile.get('nPoses', 0)
        self.nIterations = inputFile.get('nIterations', 10)
        self.nSteps = inputFile.get('nSteps', 100)
        self.kT = inputFile.get('kT', 1.0)
        self.anneal = inputFile.get('Anneal', True)
        self.kT_high = inputFile.get('kT_high', 1000)
        self.kT_low = inputFile.get('kT_low', 1)
        self.kT_decay = inputFile.get('kT_decay', True)
        self.kT_highScale = inputFile.get('kT_highScale', True)
        self.writeALL = inputFile.get('WriteALL', False)
        self.simulationTime = inputFile.get('Time', False)
        self.simulationTimeFrequency = inputFile.get('TimeFrequency', 1)
        self.ligandClusterCutoff = inputFile.get('LigandClusterCutoff', None)

        print('Initiating output paths.')
        self.name = inputFile.get('Name', '')
        self.outPath = inputFile.get('Name', '')
        self.bestPosesPath = inputFile.get('Name', '')
        self.scratch = inputFile.get('Name', '')

        print('Initiating Spawning/Ranking.')
        self.rankingMetric = inputFile.get('RankingMetric', 'FullAtomWithConstraints')
        self.spawningMethod = inputFile.get('SpawningMethod', None)
        self.spawningMetric = inputFile.get('SpawningMetric', 'FullAtomWithConstraints')
        self.spawningMetricSteps = inputFile.get('SpawningMetricSteps', None)

        print('Initiating Design residues.')
        self.designResidues = inputFile.get('DesignResidues', None)

        print('Initiating Catalytic residues.')
        self.catalyticResidues = inputFile.get('CatalyticResidues', None)

        print('Initiating Design Modes.')
        self.activeSiteDesignMode = inputFile.get('ActiceSiteDesignMode', None)
        self.mimimizeBackbone = inputFile.get('MimimizeBackbone', None)
        self.activeSiteLoops = inputFile.get('ActiveSiteLoops', 1)
        self.nNoneCatalytic = inputFile.get('nNoneCatalytic', 100)

        print('Initiating Sampling Modes.')
        self.activeSiteSampling = inputFile.get('ActiveSiteSampling', None)
        self.ligandSampling = inputFile.get('LigandSampling', None)
        self.dynamicSideChainCoupling = inputFile.get('DynamicSideChainCoupling', None)
        self.softRepulsion = inputFile.get('SoftRepulsion', True)

        print('Initiating ligands.')
        self.ligands = inputFile.get('Ligands', None)

        print('Initiating constraints.')
        self.constraints = inputFile.get('Constraints', None)

        # print out a summary
        self.show()


"""
class InputInterfaceDesign(InputBase):

    def __init__(self):
        super().__init__()
        self._jobType: str = ''
        self._interfaceBoundary: list = list()
        self._designDomainBoundary: list = list()
        self._neighborDistanceCutoff: float = 5.0
        self._neighborNumberCutoff: int = 3
        self._targetAA: list = list()
        self._interfaceMutationPosition: str = 'Local'
        self._interfaceMutationRestype: str = 'All'
        self._complexRepacking: str = 'Local'
        self._complexMinimization: str = 'Global'
        self._energyWeight: list = list()
        self._jump: int = 0

    @property
    def jobType(self):
        return self._jobType

    @jobType.setter
    def jobType(self, jobType: str):
        print('\n\nInitiating jobType.')
        if not jobType:
            raise ValueError('No pdb file name is given')
        if jobType not in ['AB', 'AA_BB', 'AA_CD']:
            raise ValueError('bad JobType. Not recognized job type {}'.format(jobType))
        self._jobType = jobType

    @property
    def neighborDistanceCutoff(self):
        return self._neighborDistanceCutoff

    @neighborDistanceCutoff.setter
    def neighborDistanceCutoff(self, neighborDistanceCutoff: float):
        self._neighborDistanceCutoff = neighborDistanceCutoff

    @property
    def neighborNumberCutoff(self):
        return self._neighborNumberCutoff

    @neighborNumberCutoff.setter
    def neighborNumberCutoff(self, neighborNumberCutoff: int):
        self._neighborNumberCutoff = neighborNumberCutoff

    @property
    def targetAA(self):
        return self._targetAA

    @targetAA.setter
    def targetAA(self, targetAA: str):
        self._targetAA = targetAA

    @property
    def interfaceMutationPosition(self):
        return self._interfaceMutationPosition

    @interfaceMutationPosition.setter
    def interfaceMutationPosition(self, interfaceMutationPosition: str):
        if interfaceMutationPosition not in ['Global', 'Local']:
            raise ValueError('bad interfaceMutationPosition {}.'.format(interfaceMutationPosition))
        self._interfaceMutationPosition = interfaceMutationPosition

    @property
    def interfaceMutationRestype(self):
        return self._interfaceMutationRestype

    @interfaceMutationRestype.setter
    def interfaceMutationRestype(self, interfaceMutationRestype: str):
        if interfaceMutationRestype not in ['All', 'One']:
            raise ValueError('bad InterfaceMutationRestype {}.'.format(interfaceMutationRestype))
        self._interfaceMutationRestype = interfaceMutationRestype

    @property
    def complexRepacking(self):
        return self._complexRepacking

    @complexRepacking.setter
    def complexRepacking(self, complexRepacking: str):
        if complexRepacking not in ['Global', 'Local']:
            raise ValueError('bad InterfaceMutationRestype {}.'.format(complexRepacking))
        self._complexRepacking = complexRepacking

    @property
    def complexMinimization(self):
        return self._complexMinimization

    @complexMinimization.setter
    def complexMinimization(self, complexMinimization: str):
        if complexMinimization not in ['Global', 'Local']:
            raise ValueError('bad InterfaceMutationRestype {}.'.format(complexMinimization))
        self._complexMinimization = complexMinimization

    @property
    def energyWeight(self):
        return self._energyWeight

    @energyWeight.setter
    def energyWeight(self, energyWeight: list):
        if not self.jobType:
            raise ValueError('JobTyp is not set. Cant set EnergyWeight.')

        if self.jobType in ['AB', 'AA_BB']:
            if len(energyWeight) != 1:
                raise ValueError('bad EnergyWeight. Expecting 1, received {}'.format(energyWeight))

        elif self.jobType in ['AA_CD']:
            if len(energyWeight) != 3:
                raise ValueError('bad EnergyWeight. Expecting 3, received {}'.format(energyWeight))
        self._energyWeight = list(map(float, energyWeight))

    @property
    def jump(self):
        return self._jump

    @jump.setter
    def jump(self, jump: int):
        self._jump = jump

    @property
    def interfaceBoundary(self):
        return self._interfaceBoundary

    @interfaceBoundary.setter
    def interfaceBoundary(self, interfaceBoundary: dict):
        print('Initiationg InterfaceBoundary')
        if not interfaceBoundary:
            raise ValueError('no InterfaceBoundary is given.')

        if not self._pose:
            raise ValueError('pose if not defined. InterfaceBoundary can not be set.')


            # the keys in ResIntervals are the chain
        interfaceChains = sorted(list(interfaceBoundary.keys()))
        if len(interfaceChains) != 2:
            raise ValueError('InterfaceBoundary should contain 2 chains, found {}'.format(interfaceChains))

            # Get the boundaries.
        A_boundary = [int(i) for i in interfaceBoundary[interfaceChains[0]].split('-')]
        B_boundary = [int(i) for i in interfaceBoundary[interfaceChains[1]].split('-')]

            # check the boundaries exist in pose
        for resID in A_boundary:
            if not self._pose.pdb_info().pdb2pose(interfaceChains[0], resID):
                raise ValueError('bad InterfaceBoundary for chain {} not found in pdb file'.format(interfaceChains[0]))

        for resID in B_boundary:
            if not self._pose.pdb_info().pdb2pose(interfaceChains[1], resID):
                raise ValueError('bad InterfaceBoundary for chain {} not found in pdb file'.format(interfaceChains[0]))

        if (A_boundary[1] - A_boundary[0]) != (B_boundary[1] - B_boundary[0]):
            raise ValueError('bad InterfaceBoundary. The length of boundaries are not equal')
        self._interfaceBoundary = InterfaceBoundary(interfaceChains, A_boundary, B_boundary)

    @property
    def designDomainBoundary(self):
        return self._designDomainBoundary

    @designDomainBoundary.setter
    def designDomainBoundary(self, designDomainBoundary: dict):

        if not self.interfaceBoundary:
            raise ValueError('InterfaceBoundary is not set, DesignDomain can not be set.')

            # Make the dictionary of chain: [intervals][]
        chainIntervals = designDomainBoundary.get('ChainIntervals', dict())
        if not chainIntervals:
            print('not detected')
            chainIntervals[self.interfaceBoundary.interfaceChains[0]] = [self.interfaceBoundary.A_boundary]
            chainIntervals[self.interfaceBoundary.interfaceChains[1]] = [self.interfaceBoundary.B_boundary]
        else:
            for chain in chainIntervals.keys():
                chainIntervals[chain] = [list(map(int, interval.split('-'))) for interval in chainIntervals[chain].split(',')]

            # Convert the interval dictionary to a list if resNames
        designDomainResName = list()
        for chain, intervals in chainIntervals.items():
            if chain not in self.interfaceBoundary.interfaceChains:
                raise ValueError('bad ResIntervals in designDomain. Chain {} is not part of InterfaceBoundary'.format(chain))
            for interval in intervals:
                if len(interval) == 1:
                    designDomainResName.append((*interval, chain))
                elif len(interval) == 2:
                    designDomainResName.extend([(i, chain) for i in range(interval[0], interval[1]+1)])
                else:
                    raise ValueError('bad ResIntervals in designDomain. Chain {} interval {}'.format(chain, interval))

            # Get the res type filters
        ignorResType = designDomainBoundary.get('IgnorResType', '').split('-')
            # apply the res type filter
        self._designDomainBoundary = list()
        for resName in designDomainResName:
            ID, chain = resName
            poseIndex = self.pose.pdb_info().pdb2pose(chain, ID)
            resType = self.pose.residue(poseIndex).name1()
            if resType not in ignorResType:
                self._designDomainBoundary.append(resName)

    def getDesignResidues(self):
        if not self.jobType:
            raise ValueError('JobType is not set. Can not get DesignResidues')
        elif not self.targetAA:
            raise ValueError('TargetAA is not set. Can not get DesignResidues')
        elif not self.neighborDistanceCutoff:
            raise ValueError('NeighborDistanceCutoff is not set. Can not get DesignResidues')
        elif not self.neighborNumberCutoff:
            raise ValueError('NeighborNumberCutoff is not set. Can not get DesignResidues')
        elif not self.interfaceBoundary:
            raise ValueError('InterfaceBoundary is not set. Can not get DesignResidues')
        elif not self.designDomainBoundary:
            raise ValueError('DesignDomainBoundary is not set. Can not get DesignResidues')

        # Make an interface
        if self.jobType in ['AB']:
            interface = InterfaceAsymmetric(self.pose,
                                              self.interfaceBoundary.interfaceChains,
                                              self.interfaceBoundary.A_boundary,
                                              self.interfaceBoundary.B_boundary,
                                              self.targetAA,
                                              self.neighborDistanceCutoff,
                                              self.neighborNumberCutoff)
        elif self.jobType in ['AA_BB', 'AA_CD']:
            interface = InterfaceSymmetric(self.pose,
                                              self.interfaceBoundary.interfaceChains,
                                              self.interfaceBoundary.A_boundary,
                                              self.interfaceBoundary.B_boundary,
                                              self.targetAA,
                                              self.neighborDistanceCutoff,
                                              self.neighborNumberCutoff)
        else:
            raise ValueError('failed initiating design residues. Unknown jobtype {}'.format(self.jobType))


        designResidues = list()
        for res in interface.residues:
                # Get the resnames of the residue and its mirror residue
            poseIndex_1 = res.poseIndex
            resName_1 =  self.pose.pdb_info().pose2pdb(poseIndex_1).split()
            resName_1 = (int(resName_1[0]), resName_1[1])

                # impose symmetry on the design resides
            if self.jobType in ['AA_BB']:
                poseIndex_2 = interface.getMirrorResPoseIndex(poseIndex_1)
                resName_2 = self.pose.pdb_info().pose2pdb(poseIndex_2).split()
                resName_2 = (int(resName_2[0]), resName_2[1])
                if resName_1 in self.designDomainBoundary and resName_2 in self.designDomainBoundary:
                    designResidues.append(res)
                # No symmetry is needed, as chains don't need to be identical
            elif self.jobType in ['AB', 'AA_CD']:
                if resName_1 in self.designDomainBoundary:
                    designResidues.append(res)

        if len(designResidues) == 0:
            raise ValueError('No design residues identified.')

        self.designResidues = designResidues

    def show(self):
        # Write some concepts of the Yaml control file in the console or the output file
        print('------------------------------------------------------------------------------------------------')
        print('DEBUG: {}'.format(Constants.DEBUG))
        print("jobType: {}".format(self.jobType))
        print('PDB: {}'.format(self.pose.pdb_info().name()))
        print('Path to Outputs: {}'.format(self.outPath))
        print('Path to Final Results: {}'.format(self.bestPosesPath))
        print('Path to Scratch: {}'.format(self.scratch))
        print('Simulation Conditions:')
        print('     nPoses: {}\n     writeAll:  {}'.format(self.nPoses, self.writeALL))
        print('     nSteps: {}\n     nIterations: {}'.format(self.nSteps, self.nIterations))
        print('     SimulationTime: {} s\n     SimulationTimeFrequency: {}'
                                                      .format(int(self.simulationTime), self.simulationTimeFrequency))
        if self.anneal:
            print('     Anneal: True\n     kT_high: {}\n     kT_low: {}\n     kT_decay: {}\n     kT_highScale: {}'
                                                 .format(self.kT_high, self.kT_low, self.kT_decay, self.kT_highScale))
        else:
            print('     Anneal: False\n     kT: {}'.format(self.kT))
        print('     RankingMetric: {}'.format(self.rankingMetric))
        print('     SpawningMetric: {}'.format(self.spawningMetric))
        if self.spawningMetricSteps:
            for element in self.spawningMetricSteps:
                print('         Itiration ratio: {}  {}'.format(element[0], element[1]))

        print('Interface parameters:')
        print('NeighborDistanceCutoff: {}'.format(self.neighborDistanceCutoff))
        print('NeighborNumberCutoff: {}'.format(self.neighborNumberCutoff))
        print('InterfaceBoundary:')
        print('     Chain {}: {}'.format(self.interfaceBoundary.interfaceChains[0], self.interfaceBoundary.A_boundary))
        print('     Chain {}: {}'.format(self.interfaceBoundary.interfaceChains[1], self.interfaceBoundary.B_boundary))
        #print('DesignBoundary:')
        #print('     Residue    Neighbors')
        #for res in self.designResidues:
        #    print('     ({} {}) {}'.format(''.join(map(str, res.name)), res.currentAA,
        #                                            ','.join([''.join(map(str, name)) for name in res.neighborsName])))

    def read(self, inputFile):
        self.jobType = inputFile.get('JobType', '')
        self.pose = inputFile.get('PDB', '')

        print('Initiating simulation parameters.')
        self.nPoses = inputFile.get('nPoses', 0)
        self.nIterations = inputFile.get('nIterations', 10)
        self.nSteps = inputFile.get('nSteps', 100)
        self.packerLoops = inputFile.get('PackerLoops', 10)
        self.kT = inputFile.get('kT', 1.0)
        self.anneal = inputFile.get('Anneal', True)
        self.kT_high = inputFile.get('kT_high', 1000)
        self.kT_low = inputFile.get('kT_low', 1)
        self.kT_decay = inputFile.get('kT_decay', True)
        self.kT_highScale = inputFile.get('kT_highScale', True)
        self.writeALL = inputFile.get('WriteALL', False)
        self.simulationTime = inputFile.get('Time', False)

        print('Initiating output paths.')
        self.name = inputFile.get('Name', '')
        self.outPath = inputFile.get('Name', '')
        self.bestPosesPath = inputFile.get('Name', '')
        self.scratch = inputFile.get('Scratch', '')

        #print('Initiating metrics.')
        #self.rankingMetric = inputFile.get('RankingMetric', 'FullAtom')
        #self.spawningMetric = inputFile.get('SpawningMetric', 'FullAtom')
        #self.spawningMetricSteps = inputFile.get('SpawningMetricSteps', False)

        print('Initiating interface parameters.')
        self.interfaceMutationPosition = inputFile.get('InterfaceMutationPosition', 'Local')
        self.interfaceMutationRestype = inputFile.get('InterfaceMutationRestype', 'All')
        self.complexRepacking = inputFile.get('ComplexRepacking', 'Local')
        self.complexMinimization = inputFile.get('ComplexMinimization', 'Global')
        self.energyWeight = str(inputFile.get('EnergyWeight', '')).split()
        self.jump = inputFile.get('JumpFrequency', 0)
        self.neighborDistanceCutoff = inputFile.get('NeighborDistanceCutoff', 5.0)
        self.neighborNumberCutoff = inputFile.get('NeighborNumberCutoff', 3)
        self.targetAA = inputFile.get('TargetAA', 'ALLA-C')
        self.interfaceBoundary = inputFile.get('InterfaceBoundary', dict())
        self.designDomainBoundary = inputFile.get('DesignDomainBoundary', dict())

        print('Initiating design residues.')
        self.getDesignResidues()

        print('Initiating constraints.')
        self.constraints = inputFile.get('Constraints', None)
"""

class ResultBase(object):
    """
    Servers as a dictionary for the results.
    """
    def __init__(self):
        self._pose: Pose = Pose()

    @property
    def pose(self):
        return self._pose

    @pose.setter
    def pose(self, pose: Pose):
        self._pose.assign(pose)

"""
class ResultInterface(ResultBase):
    def __init__(self):
        super().__init__()
        self._score: float = 0.0
        self._domain: InterfaceDomainMover = InterfaceDomainMover()
        self._energy: list = []
        self._sequence = ''

    @property
    def score(self):
        return self._score

    @score.setter
    def score(self, score: float):
        self._score = score

    @property
    def domain(self):
        return self._domain

    @domain.setter
    def domain(self, domain: InterfaceDomainMover):
        self._domain.assign(domain)

    @property
    def energy(self):
        return self._energy

    @energy.setter
    def energy(self, energy: list):
        self._energy = energy

    @property
    def sequence(self):
        return self._sequence

    @sequence.setter
    def sequence(self, sequence: str):
        self._sequence = sequence

    def getState(self):
        return self.pose, self.domain, self.score, self.energy

    def setState(self, pose: Pose, domain: InterfaceDomainMover, score: float, energy: list):
        self.pose = pose
        self.domain.assign(domain)
        self.score = score
        self.energy = energy
        self.sequence = domain.getSequence(pose)
"""