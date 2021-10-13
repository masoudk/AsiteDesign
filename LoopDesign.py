# Global imports
import time
import copy
import re
import yaml
import sys
import os
from itertools import cycle
from numpy import exp, abs
from mpi4py import MPI

# PyRosetta import
import pyrosetta as pr
from pyrosetta.rosetta.utility  import vector1_bool
from pyrosetta.rosetta.core.kinematics import MoveMap
from pyrosetta.rosetta.core.scoring import chainbreak
from pyrosetta.rosetta.core.pack.task import TaskFactory
from pyrosetta.rosetta.protocols.loops import Loop, Loops, loop_rmsd, set_single_loop_fold_tree, add_single_cutpoint_variant
from pyrosetta.rosetta.protocols.loops.loop_mover.refine import LoopMover_Refine_CCD
from pyrosetta.rosetta.protocols.loops.loop_closure.ccd import CCDLoopClosureMover
from pyrosetta.rosetta.protocols.simple_moves import ClassicFragmentMover, ReturnSidechainMover
from pyrosetta.rosetta.protocols.minimization_packing import PackRotamersMover
from pyrosetta.rosetta.core.fragment import ConstantLengthFragSet

pr.init(''' -out:level 0 ''')
#pr.init(''' -out:level 0 -no_his_his_pairE -extrachi_cutoff 1 -multi_cool_annealer 10 -ex1 -ex2 -use_input_sc ''')

# TODO: A new method in Design class for constructing an initial guess to be used as the reference structure
class Inputs(object):
    pass


class Result(object):
    pass


class PoseBin(object):
    def __init__(self, center, pose=None, energy=9999, distance = None, count=0):
        self.center = center
        self.distance = distance
        self.count = count
        self.pose = pr.Pose()
        if pose is not None:
            self.pose.assign(pose)
        self.energy = energy


class ResultClustersByRMSD(object):

    def __init__(self, binSize, maxElements=100):

        self.binSize = binSize
        self.bins = list()

            # Initialize the bins for the results
        for i in range(maxElements):
            center = (i * self.binSize) + (binSize * 0.5)
            bin = PoseBin(center)
            self.bins.append(bin)
            print('{}:{:.2f}'.format(i, center), end=' ')
        print('\n','-'*120)

    def cluster(self, result, replaceBy='Energy'):
        """
        :param result:
        :param replaceBy: str could be 'Energy' or 'Distance'
        :return:
        """

            # Find the bin and how far is the data from center
        index = int(result.rmsd/ self.binSize)
        distance = abs(self.bins[index].center - result.rmsd)
        #print('-------------------->rmsd {:.2f}:,  index {}:'.format(result.rmsd, index))

            # If bin is empty add the pose as the representative
        if self.bins[index].count == 0:
            self.bins[index].count += 1
            self.bins[index].distance = distance
            self.bins[index].pose.assign(result.residue)
            self.bins[index].energy = result.residue.energies().total_energy()


        else:
            self.bins[index].count += 1
            # add the pose only if it is better based on the mode replaceBy='Energy' or Distance'
            if re.match('Energy', replaceBy, re.IGNORECASE):
                if result.energy < self.bins[index].energy:
                    self.bins[index].distance = distance
                    self.bins[index].pose.assign(result.residue)
                    self.bins[index].energy = result.residue.energies().total_energy()

            elif re.match('Distance', replaceBy, re.IGNORECASE):
                if distance < self.bins[index].distance:
                    self.bins[index].distance = distance
                    self.bins[index].pose.assign(result.residue)
                    self.bins[index].energy = result.energy

    def selectRepresentativePoses(self, nElements, selectBy='Energy'):
        """
        :param nElements:
        :param selectBy: Energy or BinCount
        :return:
        """

        selectedPoses = list()
        sortedClusters = list()

            # Make a copy
        for bin in self.bins:
            if bin.count > 0:
                #print('found non empty bin with center at {}.'.format(bin.center))
                binI = PoseBin(center=bin.center, pose=bin.pose, energy=bin.energy, distance=bin.distance, count=bin.count)
                sortedClusters.append(binI)

            # Sort by energy. The lowest energy poses are first
        if re.match('Energy', selectBy):
            sortedClusters.sort(key=lambda x: x.energy)

            # Sort by bin count. The bins with least count are first
        elif re.match('BinCount', selectBy):
            sortedClusters.sort(key=lambda x: x.count)

        print('Clusters:')
        for bin in sortedClusters:
            if bin.count > 0:
                print('center {:.2f},  energy: {:.2f} count: {}'.format(bin.center, bin.energy, bin.count))

        print('-'*100)
        count = 0
        for bin in sortedClusters:
            if bin.count > 0:
                #print('selected bin: {:.2f},  energy: {:.2f},  count: {}'.format(bin.center, bin.energy, bin.count))
                selectedPoses.append(bin.pose)
                count += 1
            if count == nElements:
                break
        print('DONE Selection by {}'.format(selectBy))
        return selectedPoses

    def dumpClusters(self, outPath='', prefix=''):

        if prefix:
            prefix = '{}-'.format(prefix)

        for bin in self.bins:
            if bin.count > 0:
                bin.pose.dump_pdb(os.path.join(outPath, '{}Cluster{:.2f}-E{:.2f}.pdb'.format(prefix, bin.center, bin.energy)))


class LoopInfo(object):
    """
    This class keeps the loop data and can be serialized via pickle. The Loop class in pyrosetta
    is not serializable.
    """
    def __init__(self, startResId, startResChain, endResId, endResChain, cutResId, cutResChain):
        self.startResId = startResId
        self.startResChain = startResChain
        self.startPoseIndex = None

        self.endResId = endResId
        self.endResChain = endResChain
        self.endPoseIndex = None

        self.cutResId = cutResId
        self.cutResChain = cutResChain
        self.cutPoseIndex = None

    def setPoseIndex(self, pose):

            posInfo = pose.pdb_info()
            self.startPoseIndex = posInfo.pdb2pose(self.startResChain, self.startResId)
            self.endPoseIndex = posInfo.pdb2pose(self.endResChain, self.endResId)
            self.cutPoseIndex = posInfo.pdb2pose(self.cutResChain, self.cutResId)


class LoopSamplerByFragments(object):
    def __init__(self, pose, refPose, loopInfo, fragmentsFile, fragmentLength, anneal=True, kT=0.6,
                            kT_highFA=2.0, kT_lowFA=0.6, kT_highCent=2.0, kT_lowCent=0.6, kT_decay= True,
                                modelingOuterCycles=10, modelingInnerCycles=500, refinementOuterCycles=5,
                                                        refinementInnerCycles=10, UseTheInitialPose=False):


            # The pose to be modeled
        self.pose = pr.Pose()
        self.pose.assign(pose)

            # The reference pose for calculation of RSMD.
        self.refPose = pr.Pose()
        self.refPose = refPose


            # Get the loop info
        self.loopStart = loopInfo.startPoseIndex
        self.loopEnd = loopInfo.endPoseIndex
        loopCut = loopInfo.cutPoseIndex

            # Set the loop
        self.loop = Loop(self.loopStart, self.loopEnd, loopCut)
        set_single_loop_fold_tree(self.pose, self.loop)
        add_single_cutpoint_variant(self.pose, self.loop)

            # Set the Loops obj used by refine
        self.sampleLoops = Loops()
        self.sampleLoops.add_loop(self.loop)


            # For recovering the side chains
        self.initialPose = pr.Pose()
        self.initialPose.assign(pose)

            # To pass somthing even if nothing was found
        self.finalPose = pr.Pose()
        self.finalPose.assign(pose)

        self.finalEnergy = 9999.9
        self.finalRMSD = 0.000

            # Set the start and final kT for centroid stage of modeling
        self.anneal = anneal
        self.kT_highCent = kT_highCent
        self.kT_lowCent = kT_lowCent
        self.kT_decay = kT_decay
            # to be used in the case of no anneal
        self.kT = kT
        #print(self.kT_highCent, self.kT_lowCent, self.kT_decay)

            # Set nSeps in centroid stage of modeling
        self.modelingOuterCycles = modelingOuterCycles
        self.modelingInnerCycles = modelingInnerCycles

            # Set the movemap
        self.movemap = MoveMap()
        self.movemap.set_bb_true_range(self.loopStart, self.loopEnd)
        self.movemap.set_chi_true_range(self.loopStart-2, self.loopEnd+2)

            # Set packer task
        self.taskpack = TaskFactory.create_packer_task(self.pose)
        self.taskpack.restrict_to_repacking()

        repackFlag = vector1_bool()
            # Set all to not repack
        repackFlag.extend([0]*self.pose.size())
            # Except for the loop region
        for i in range(self.loopStart, self.loopEnd + 1):
            repackFlag[i] = 1
            # set the repack aa
        self.taskpack.restrict_to_residues(repackFlag)

        try:
                # Fragment obj can not be pickled out of the box
            self.fragSet = ConstantLengthFragSet(fragmentLength, fragmentsFile)

        except:
            killProccesses('problem reading fragment file')


            # Set score functions for centroid model
        self.scorefxnCent = pr.create_score_function('cen_std')
        self.scorefxnCent.set_weight(chainbreak, 1) # Penalize loop break

            # Set the full atom score function
        self.scorefxnFA = pr.get_fa_scorefxn()

            # initialize  movers
        self.fragmentMover = ClassicFragmentMover(self.fragSet, self.movemap)
        self.ccdClosure = CCDLoopClosureMover(self.loop, self.movemap)
        self.packMover = PackRotamersMover(self.scorefxnFA, self.taskpack)

            # This mover is problem, since it chooses the repack domain on its own
        self.refineMover = LoopMover_Refine_CCD(self.sampleLoops)
        self.refineMover.temp_initial(kT_highFA)
        self.refineMover.temp_final(kT_lowFA)
        self.refineMover.outer_cycles(refinementOuterCycles)
        self.refineMover.max_inner_cycles(refinementInnerCycles)

            # Auxiliary functions for conversion between full atom and centroid model
        self.toCentroid = pr.SwitchResidueTypeSetMover('centroid')
        self.toFullAtom = pr.SwitchResidueTypeSetMover('fa_standard')
        self.recoverSideChains = ReturnSidechainMover(self.initialPose)


            # Convert to Centroid model
        self.toCentroid.apply(self.pose)

        if not UseTheInitialPose:
            # Randomize the loop
            self.prepareLoop()

            # Set the MC obj for evaluation of loop modeling for centroid stage. The kT will be updated later
            # based on annealing
        self.mc = pr.MonteCarlo(self.pose, self.scorefxnCent, self.kT)


    def run(self):

            # Make sure pose in centroid form
        if not self.pose.is_centroid():
            self.toCentroid.apply(self.pose)

             # The kT will be updated later based on annealing
        self.mc = pr.MonteCarlo(self.pose, self.scorefxnCent, self.kT)

            # perform the low resolution centroid modeling
        for outerStep in range(self.modelingOuterCycles):

                # Recover the best so far
            self.mc.recover_low(self.pose)

            for innerStep in range(self.modelingInnerCycles):

                #print(innerStep, self.modelingInnerCycles)
                if self.anneal: # Change the kT if annealing
                    self.kT = self.getKT(x=innerStep, Th=self.kT_highCent, Tl=self.kT_lowCent, N=self.modelingInnerCycles, k=self.kT_decay)
                    self.mc.set_temperature(self.kT)

                #print('Loop centroid modeling outerStep: {}, innerStep: {}, at kT {}: '.format(outerStep, innerStep, self.kT))
                    # Do the centroid modeling
                self.fragmentMover.apply(self.pose)
                    # close the loop
                self.ccdClosure.apply(self.pose)
                    # evaluate the loop
                self.mc.boltzmann(self.pose)
                    #print('AAAAACCCCCCCCCCCCCCCCCEEEEEEEEEEEEEEEEEEEEEEEPPPPPPPPPPPPPPPPPTTTTTTTTTTTTTTEEEEEEEEEED')

            # load the lowest energy
        self.mc.recover_low(self.pose)
            # load the last accepted pose. This is redundant, the last accepted pose
            # is already assigned
        #self.pose.assign(self.mc.last_accepted_pose())

            # recover the full atom representation
        #print("recovering full atom.")
        self.toFullAtom.apply(self.pose)
        self.recoverSideChains.apply(self.pose)

            # repack
        self.packMover.apply(self.pose)

            # refine
        self.refineMover.apply(self.pose)

            # Set the final results
        self.finalPose.assign(self.pose)
        self.finalEnergy = self.pose.energies().total_energy()
        self.finalRMSD = loop_rmsd(self.pose, self.refPose, self.sampleLoops, True)

    def prepareLoop(self):

            # Convert to Centroid model
        if not self.pose.is_centroid():
            self.toCentroid.apply(self.pose)

            # Make a temp pose
        poseTemp = pr.Pose()
        poseTemp.assign(self.pose)
        score = self.scorefxnCent(poseTemp)

            # Straight the loop from the cut point
        for i in range(self.loopStart , self.loopEnd + 1):
            poseTemp.set_phi(i , -180)
            poseTemp.set_psi(i , 180)

            # Do some round of fragmentMove minimization
        for i in range(50):
            #print('Prep the loop Step: {}'.format(i))
            self.fragmentMover.apply(poseTemp)
            if self.scorefxnCent(poseTemp) < self.pose.energies().total_energy():
                self.pose.assign(poseTemp)

    def getKT(self, x, Th, Tl, N, k):

            # If only one step, no point calculating
        if N == 1:
            return Tl
        else:
            if not k:
                deltaT = Th-Tl
                dT = deltaT/(N-1)
                return Th - (dT * x)
            else:
                k = 1/(N * 0.1)
                T = Tl + (Th * exp(-k*x))
                if T > Th:
                    return Th
                elif T < Tl or x == (N-1):
                    return Tl
                else:
                    return T


class Design(object):

    def __init__(self, confFile):
        self.comm = MPI.COMM_WORLD
        self.rank = self.comm.Get_rank()
        self.nProccess = self.comm.Get_size()

        self.sampler = None

            # Only used in explorers
        self.result = Result()

            # Only used in master process
        self.finalResults = None
        self.spawning = None

        if self.nProccess == 1:
            self.designLoopSerial(confFile)

        elif self.nProccess > 1:
            self.designLoopMPI(confFile)

    def designLoopMPI(self, confFile):

            # Set up the inputs and initialize the calculations
        inputs = None

        # The master node read the inputs and keep a copy.
        if self.rank == 0:
            inputs = self.initializeInputs(confFile)
            self.finalResults = ResultClustersByRMSD(binSize=inputs.binSize, maxElements=inputs.maxBin)
            #pose, design_residues, catalytic_residues, catalytic_constraints, nIterations, nPoses, nsteps, nresets, kT = inputs

        # Broadcast the inputs
        inputs = self.comm.bcast(inputs, root=0)

        # Initiate the explorers
        if self.rank != 0:
            self.initializeCalculation(inputs)


        for iteration in range(inputs.nIterations):
            if self.rank != 0:
                #print('----------------------------------------------Rank {} start exploring iteration: {}'.format(self.rank, iteration))
                #start = time.time()
                self.sampler.run()
                    # Pack the result
                self.result.pose = self.sampler.finalPose
                self.result.energy = self.sampler.finalEnergy
                self.result.rmsd = self.sampler.finalRMSD
                #print('Started Sending from {} at: {}'.format(time.time(), self.rank))
                self.comm.send(self.result, dest=0, tag=self.rank)
                #end = time.time()
                #print('----------------------------------------------Rank {} done exploring, iteration {} in {} seconds'.format(self.rank, iteration, end-start))
                newPose = self.comm.recv(source=0, tag=self.rank)

                    # Update the pose
                self.sampler.residue.assign(newPose)
                #print('----------------------------------------------Rank {} received new data'.format(self.rank))

            elif self.rank == 0:
                start = time.time()
                for proccess in range(1, self.nProccess):
                    #print('-------------------------------------------Rank {} receiving from process {} .'.format(self.rank, proccess))
                    result = self.comm.recv(source=proccess, tag=proccess)
                    #print('Done receiving in {} at: {}'.format(time.time(), self.rank))

                        # Add the new result to the finalResults for clustering
                    self.finalResults.cluster(result, replaceBy='Energy')
                    #print('-------------------------------------------Rank {} done receiving from process {} process in {} seconds.'.format(self.rank, proccess, end-start))


                    # Select the newData based on spawning method
                self.spawning = inputs.spawning
                # If it is already Energy or BinCount. Just pass. if Combined set it based on spawningSwitch
                if re.match('Combined', inputs.spawning):
                    if iteration < inputs.spawningSwitch:
                        self.spawning = 'BinCount'
                    else:
                        self.spawning = 'Energy'

                newData = self.finalResults.selectRepresentativePoses(nElements=inputs.nPoses, selectBy=self.spawning)

                    # Spawn the nPoses best currectResults
                print('------------------> Spawning {} new data at iteration {}, using {} method'.format(len(newData), iteration, self.spawning))
                replica = cycle(range(len(newData)))
                for proccess in range(1, self.nProccess):
                    self.comm.send(newData[next(replica)], dest=proccess, tag=proccess)


                    # Write PDB of the current iteration
                if inputs.writeALL:
                    self.writeBestResults(newData, inputs.nPoses, iteration, prefix=inputs.prefixName, output=inputs.outPath)

                end = time.time()
                #self.finalResults.dumpClusters(outPath=inputs.outPath, prefix=iteration)
                print('------------------>>> Done iteration {} in {:.2f} s.<<<----------------------'.format(iteration, end-start))

            # Wrap up
        if self.rank == 0:
            self.writeBestResults(newData, inputs.nPoses, iteration, prefix=inputs.prefixName, output=inputs.bestPosesPath)
            self.finalize(inputs)
            return newData

    def designLoopSerial(self, confFile):
        print(" Running the Serial Version. nIterations and nPoses are set to 1.")
            # Get the inputs
        inputs = self.initializeInputs(confFile)

            # Set up the calculations
        self.initializeCalculation(inputs)

            # run the calculations
        self.sampler.run()

            # Pack the result
        self.result.pose = self.sampler.finalPose
        self.result.energy = self.sampler.finalEnergy
        self.result.rmsd = self.sampler.finalRMSD


            # Write the pose
        self.result.pose.dump_pdb(os.path.join(inputs.bestPsesPath, '{}_bestpose'.format(inputs.prefixName)))

            # clean unused folders
        os.rmdir(inputs.scratch)
        os.rmdir(inputs.outPath)

        return self.result

    def initializeCalculation(self, inputs):

        self.sampler = LoopSamplerByFragments(pose=inputs.residue, refPose=inputs.residue, loopInfo=inputs.loopInfo,
                                              fragmentsFile=inputs.fragmentsFile, fragmentLength=inputs.fragmentLength,
                                              anneal=inputs.anneal, kT=inputs.kT, kT_highFA=inputs.kT_highFA,
                                              kT_lowFA=inputs.kT_lowFA, kT_highCent=inputs.kT_highCent,
                                              kT_lowCent=inputs.kT_lowCent, kT_decay=inputs.kT_decay,
                                              modelingOuterCycles=inputs.modelingOuterCycles,
                                              modelingInnerCycles=inputs.modelingInnerCycles,
                                              refinementOuterCycles=inputs.refinementOuterCycles,
                                              refinementInnerCycles=inputs.refinementInnerCycles,
                                              UseTheInitialPose=inputs.UseTheInitialPose)

    def initializeInputs(self, confFile):
        """
        Thi function should deal with input management
        :param args:
        :param kwargs:
        :return:
        """
        with open(confFile, 'r') as f:
            inputs = yaml.safe_load(f)

            # Read compulsory inputs
        try:
            pdbFile = inputs['PDB']
            startID, startChain = inputs['Loop']['start'].split('-')
            endID, endChain = inputs['Loop']['end'].split('-')
            cutID, cutChain = inputs['Loop']['cut'].split('-')

            loopInfo = LoopInfo(int(startID), startChain, int(endID), endChain, int(cutID), cutChain)
            nPoses = inputs['nPoses']

            if re.match('DesignByFragments', inputs['JobType']):
                jobType = 'DesignByFragments'
                fragmentsFile = inputs['FragmentFile']
                fragmentLength = inputs['FragmentLength']

                    # Check the fragment file exist
                if not os.path.isfile(fragmentsFile):
                    raise ValueError

            else:
                jobType = None
                fragmentsFile = None
                fragmentLength = None
                raise NotImplementedError
        except KeyError as e:
            killProccesses('reading input file, not found {}'.format(e))

            # Read the optional inputs
        prefixName = inputs.get('Name', '')
        writeALL = inputs.get('WriteALL', False)
        nIterations = inputs.get('nIterations', 10)
        UseTheInitialPose = inputs.get('UseTheInitialPose', False)
        kT = inputs.get('kT', 1.0)
        modelingOuterCycles = inputs.get('modelingOuterCycles', 10)
        modelingInnerCycles = inputs.get('modelingInnerCycles', 500)
        refinementOuterCycles = inputs.get('refinementOuterCycles', 5)
        refinementInnerCycles = inputs.get('refinementInnerCycles', 10)
        binSize = inputs.get('BinSize', 0.5)
        maxBin = inputs.get('MaxBin', 100)

        spawning = inputs.get('Spawning', 'Combined')
        if spawning not in ('Combined', 'Energy', 'BinCount'):
            killProccesses('wrong spawning keyword. Should be "Combined", "Energy", or "BinCount".')

        if re.match('Combined', spawning):
            spawningSwitch = inputs.get('SpawningSwitch', None)

            if spawningSwitch is None:
                spawningSwitch = int(nIterations * (2 / 3))
                print('Warning, Combined spawning is specified with out defining a SpawningSwitch, Setting it to {}.'.format(spawningSwitch))

            # Set up the annealing input
        anneal = inputs.get('Anneal', True)
        if anneal:
            kT_highFA = inputs.get('kT_highFA', 2.0)
            kT_lowFA = inputs.get('kT_lowFA', 0.6)
            kT_highCent = inputs.get('kT_highCent', 1000)
            kT_lowCent = inputs.get('kT_lowCent', 1.0)
            kT_decay = inputs.get('kT_decay', True)
        else:
            kT_highFA = 2.0
            kT_lowFA = 0.6
            kT_highCent = kT
            kT_lowCent = kT
            kT_decay = False


        try:
                # Make the pose
            pose = pr.pose_from_pdb(pdbFile)

                # Set pose indices of the loop info
            loopInfo.setPoseIndex(pose)

        except Exception as e:
            killProccesses('Problem initiating the input: {}'.format(e))

            # Prepare the output folder
        if prefixName:
            outPath = os.path.join('{}_output'.format(prefixName))
            bestPosesPath = os.path.join('{}_bestposes'.format(prefixName))
        else:
            outPath = os.path.join('output')
            bestPosesPath = os.path.join('bestposes')

        if not os.path.isdir(outPath):
            os.makedirs(outPath)
        else:
            for i in range(100):
                try:
                    newOutPath = os.path.join('{}_{}'.format(outPath, i))
                    os.makedirs(newOutPath)
                    break
                except:
                    pass
            if i == 99:
                killProccesses('Failed creating output folders.')
            else:
                outPath = newOutPath

        if not os.path.isdir(bestPosesPath):
            os.makedirs(bestPosesPath)
        else:
            for i in range(100):
                try:
                    newBestPosesPath = os.path.join('{}_{}'.format(bestPosesPath, i))
                    os.makedirs(newBestPosesPath)
                    break
                except:
                    pass
            if i == 99:
                killProccesses('Failed creating bestPoses folders.')
            else:
                bestPosesPath = newBestPosesPath

            # Prepare the scratch folder
        scratch = os.path.join('.scratch')
        if not os.path.isdir(scratch):
            os.makedirs(scratch)

        inputs = Inputs()

        inputs.jobType = jobType
        inputs.pose = pose
        inputs.loopInfo = loopInfo
        inputs.fragmentsFile = fragmentsFile
        inputs.fragmentLength = fragmentLength

        inputs.UseTheInitialPose = UseTheInitialPose
        inputs.nIterations = nIterations
        inputs.nPoses = nPoses
        inputs.binSize = binSize
        inputs.maxBin = maxBin
        inputs.spawning = spawning
        inputs.spawningSwitch = spawningSwitch

        inputs.modelingOuterCycles = modelingOuterCycles
        inputs.modelingInnerCycles = modelingInnerCycles
        inputs.refinementOuterCycles = refinementOuterCycles
        inputs.refinementInnerCycles = refinementInnerCycles

        inputs.kT = kT
        inputs.anneal = anneal
        inputs.kT_highFA = kT_highFA
        inputs.kT_lowFA = kT_lowFA
        inputs.kT_highCent = kT_highCent
        inputs.kT_lowCent = kT_lowCent
        inputs.kT_decay = kT_decay

        inputs.outPath = outPath
        inputs.scratch = scratch
        inputs.bestPosesPath = bestPosesPath
        inputs.prefixName = prefixName
        inputs.writeALL = writeALL

        return inputs

    def writeBestResults(self, poses, nElements, Iteration, prefix='', output=''):

        for i, pose in enumerate(poses):
            energy = pose.energies().total_energy()
            if prefix:
                name = '{}_I{}_N{}_{:.1f}'.format(prefix, Iteration, i, energy)
            else:
                name = '{}_I{}_N{}_{:.1f}'.format('pose', Iteration, i, energy)

            pdbName = '{}.pdb'.format(name)
            pose.dump_pdb(os.path.join(output, pdbName))

                # Stop if the results have more elements that nElements
            if i == nElements:
                break

    def finalize(self, inputs):
        if os.path.isdir(inputs.scratch):
            os.rmdir(inputs.scratch)


def killProccesses(msg):
    print('Error >>> {}'.format(msg))
    if MPI.COMM_WORLD.size > 1:
        MPI.COMM_WORLD.Abort(1)
    else:
        exit(1)


if __name__ == "__main__":
    try:
        confFile = sys.argv[1]
        if not os.path.isfile(confFile):
            raise ValueError
    except Exception as e:
        killProccesses('No conf file is given'.format(e))

    results = Design(confFile)