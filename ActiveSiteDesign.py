# Global imports
import os, sys
import re
import traceback
import time
import datetime
import shutil
import yaml
import pickle
import argparse
from numpy import exp, sum, array, where, sqrt, any, std, mean, square
from numpy.random import randint, normal, uniform
from random import shuffle
from numpy import mean
from numpy.linalg import norm
from itertools import cycle
from io import StringIO
from mpi4py import MPI

# Bio Python import
from Bio.PDB.Polypeptide import one_to_three

# Biotite import
from biotite.structure.io.pdb import PDBFile

# Local imports
import Constants
from BaseClases import Inputs, Result, Residue, Ligand
from BaseClases import DesignDomainWildCards, GeometricConstraint, SequenceConstraint
from MiscellaneousUtilityFunctions import killProccesses, getKT, deep_getsizeof, availableMemory

from IO import InputActiveSiteDesign

from MoversRosetta import getScoreFunction
from MoversRosetta import DesignPose, ActiveSiteMover, ActiveSiteSampler
from Docking import LigandRigidBodyMover, LigandPackingMover, LigandNeighborsPackingMover, LigandMover

# PyRosetta import
import pyrosetta as pr
from pyrosetta.rosetta.std import ostringstream, istringstream
from pyrosetta.rosetta.utility import vector1_std_shared_ptr_const_core_conformation_Residue_t
from pyrosetta.rosetta.protocols.geometry import centroid_by_residues
from pyrosetta.rosetta.protocols.toolbox.pose_manipulation import superimpose_pose_on_subset_CA
from pyrosetta.rosetta.core.scoring import score_type_from_name

pr.init(''' -out:level 0 -no_his_his_pairE -extrachi_cutoff 1 -multi_cool_annealer 10 -ex1 -ex2 -use_input_sc ''')


class ActiveSiteDesign(object):

    def __init__(self):
        self.comm = MPI.COMM_WORLD
        self.rank = self.comm.Get_rank()
        self.nProccess = self.comm.Get_size()
        self.nProcessOMP = self.getOMPnProcesses()

        #self.scorefxn = None
        # self.domain = None
        # self.mover = None
        self.sampler = None

        self.energyFullAtom = getScoreFunction('fullAtom')
        self.energyFullAtom.set_weight(score_type_from_name('fa_dun'), 0.1)
        #self.energyFullAtom.set_weight(score_type_from_name('rama_prepro'), 0.0)
        self.energyFullAtomWithConstraints = getScoreFunction('fullAtomWithConstraints')
        self.energyFullAtomWithConstraints.set_weight(score_type_from_name('fa_dun'), 0.1)
        #self.energyFullAtomWithConstraints.set_weight(score_type_from_name('rama_prepro'), 0.0)
        self.energyOnlyConstraints = getScoreFunction('onlyConstraints')

    def run(self, confFile):
        if self.nProccess == 1:
            self.designCatalyticResidueSerial(confFile)

        elif self.nProccess > 1:
            try:
                self.designCatalyticResidueMPI(confFile)
            except Exception as e:
                traceback.print_tb(e.__traceback__)
                print('\n')
                killProccesses('Failed at process {}: {}'.format(self.rank, e))

    def designCatalyticResidueSerial(self, confFile):

        startTimeTotal = time.time()
        startDate = datetime.datetime.now()

        # Get the inputs
        inputs = self.initializeInputs(confFile)
        print("Running the Serial Version. nIterations and nPoses are set to 1.")

        # Initialize the sampler
        self.initializeCalculation(inputs)

        # Run Sampler
        self.sampler.run()

        # Calculate energies and put energy, dPose, and sequence in a results object
        result = self.getState(metric=None)

        # write best results
        self.writeResultsSerial(results=[result], nElements=1, iteration='', output=inputs.bestPosesPath,
                                                                                    energyType=inputs.rankingMetric)

        endTimeTotal = time.time()
        endDate = datetime.datetime.now()

        # write final summary
        self.printFinalSummarySerial(results=[result], nElements=1)

        # Write the final timming
        self.printFinalTimming(startDate=startDate, endDate=endDate, timeTotal=endTimeTotal-startTimeTotal)

        # Clean stuff
        self.clean(inputs)

        # return result
        return result

    def designCatalyticResidueMPI(self, confFile):

        # initialize the inputs
        inputs = None
        # The master node read the inputs and keep a copy.
        if self.rank == 0:

            inputs = self.initializeInputs(confFile)
            startTimeTotal = time.time()
            startDate = datetime.datetime.now()
            iterationResultsFileNames = list()

        # Boradcast the inputs
        inputs = self.comm.bcast(inputs, root=0)

        # Initiate the explorers. Run on all cores,
        # Makes it easier to catch errors
        # Used for Ligand data during clustering
        self.initializeCalculation(inputs)

        # Needed for dynamic adjustment of nIterations
        masterNodeLagTime = 0  # Keeps track of the time needed for I/O on master node
        iterationTime = 0  # Keeps track of iteration time
        iteration = 0

        # Main iteration
        while iteration < inputs.nIterations:
            if self.rank != 0:

                # Perform Annealing. Set the kT_high for each iteration linearly scale the kT_high if it is set
                if inputs.anneal and inputs.kT_highScale:
                    self.sampler.kT_high = getKT(x=iteration, Th=inputs.kT_high, Tl=inputs.kT_low,
                                                                                        N=inputs.nIterations, k=False)
                # run the simulation
                self.sampler.run()

                # Update the spawningMetric if needed
                if inputs.spawningMetricSteps is not None:
                    self.updateSpawningMetric(inputs, iteration)

                # Get the current state
                oldState = self.getState(metric=inputs.spawningMetric)

                # Send
                self.comm.send(oldState, dest=0, tag=self.rank)

                # Receive
                newState = self.comm.recv(source=0, tag=self.rank)

                # Update current State.
                self.setState(state=newState)

                # Update nIteration if simulation time is given
                if inputs.simulationTime and (iteration % inputs.simulationTimeFrequency) == 0:
                        inputs.nIterations = self.comm.recv(source=0, tag=self.rank)

            elif self.rank == 0:

                # save the time
                start = time.time()

                # Perform Annealing. Set the kT_high for each iteration linearly scale the kT_high if it is set
                if inputs.anneal and inputs.kT_highScale:
                    kT_high = getKT(x=iteration, Th=inputs.kT_high, Tl=inputs.kT_low, N=inputs.nIterations, k=False)
                    kT_low = inputs.kT_low
                elif inputs.anneal and not inputs.kT_highScale:
                    kT_high = inputs.kT_high
                    kT_low = inputs.kT_low
                else:
                    kT_high = inputs.kT
                    kT_low = inputs.kT


                # Receive from explorers
                currentResults = list()
                for proccess in range(1, self.nProccess):
                    result = self.comm.recv(source=proccess, tag=proccess)
                    currentResults.append(result)

                # Update the spawningMetric if needed
                if inputs.spawningMetricSteps is not None:
                    self.updateSpawningMetric(inputs, iteration)

                # spawn the next iteration.
                if inputs.spawningMethod == 'Adaptive':
                    self.spawnAdaptive(currentResults, inputs.nPoses)

                elif inputs.spawningMethod == 'REM':
                    # Set the kT
                    if inputs.anneal:
                        kT = inputs.kT_low
                    else:
                        kT = inputs.kT
                    # Spawn
                    self.spawnREM(currentResults, kT)

                end = time.time()
                iterationTime += (end - start)

                # Update nIteration if time is given
                if inputs.simulationTime and (iteration % inputs.simulationTimeFrequency) == 0:
                    inputs.nIterations = self.getNIteration(iteration, inputs.nIterations, iterationTime,
                                                           masterNodeLagTime, inputs.simulationTime)
                    # Send the new nIterations to nodes
                    for proccess in range(1, self.nProccess):
                        self.comm.send(inputs.nIterations, dest=proccess, tag=proccess)

                # Compile the iteration results
                iterationResults, sequenceDiffList = self.getIterationResults(results=currentResults, nElement=inputs.nPoses,
                                                                                metric=inputs.rankingMetric)
                                                                                

                self.printIterattionResults(results=iterationResults, sequenceDiffList=sequenceDiffList, iteration=iteration, time=end - start,
                                            nIterations=inputs.nIterations, spawningMetric=inputs.spawningMetric,
                                            kT_high=kT_high, kT_low=kT_low)

                # Write the iteration results
                fileName = os.path.join(inputs.scratch, 'data_Iteration_{}'.format(iteration))
                iterationResultsFileNames.append(fileName)
                with open(fileName, 'wb') as file:
                    pickle.dump(iterationResults, file)

                # Write PDB of the current iteration
                if inputs.writeALL:
                    self.writeResults(results=iterationResults, sequenceDiffList=sequenceDiffList, nElements=inputs.nPoses, iteration=iteration,
                                      prefix=inputs.name, output=inputs.outPath, energyType=inputs.rankingMetric)

                masterNodeLagTime += (time.time() - end)

            # End of the while
            iteration += 1

        # Compiling the Final results and analysis. Only master node
        if self.rank == 0:
            bestResults = list()
            for file in iterationResultsFileNames:
                with open(file, 'rb') as infile:
                    bestResult = pickle.load(infile)
                    bestResults.extend(bestResult)
                    bestResults = self.sortResultsByMetric(results=bestResults, metric=inputs.rankingMetric)
                    bestResults, finalSequenceDiffList = self.selectResultsBySequenceDiff(results=bestResults, nElements=inputs.nPoses)

            self.writeResults(results=bestResults, sequenceDiffList=finalSequenceDiffList, nElements=inputs.nPoses, iteration='', prefix=inputs.name,
                                                        output=inputs.bestPosesPath, energyType=inputs.rankingMetric)

            # Write the final summary
            endTimeTotal = time.time()
            endDate = datetime.datetime.now()
            self.printFinalSummary(results=bestResults, sequenceDiffList=sequenceDiffList, nElements=inputs.nPoses, filename=inputs.name)

            # Write the final timing
            self.printFinalTimming(startDate=startDate, endDate=endDate, timeTotal=endTimeTotal-startTimeTotal)

            # Clean up stuff
            self.clean(inputs)

            return bestResults

    def getIterationResults(self, results, nElement, metric=None):

        iterationResults = self.sortResultsByMetric(results=results, metric=metric)
        iterationResults, SequenceDiffList = self.selectResultsBySequenceDiff(results=results, nElements=nElement)

        return iterationResults, SequenceDiffList

    def getNIteration(self, iteration, nIterations, iterationTime, masterNodeLagTime, simulationTime):
            # Get the MAX nIteration from average iteration time and masterNodeLagTime

            # increment by 1 to account for starting from 0
            iteration += 1

            if iteration == 1:
                averagedIterationTime = (iterationTime / iteration) + (masterNodeLagTime / (iteration))
            else:
                averagedIterationTime = (iterationTime / iteration) + (masterNodeLagTime / (iteration - 1))

            newnIterations = int(simulationTime / averagedIterationTime) - 2

            if newnIterations < nIterations:
                nIterations = newnIterations
            return nIterations

    def spawnAdaptive(self, results, nElements):
        """
        Spawns a new iteration from the previous results by Adaptive sampling.
        The results are first ordered by spawning metric and then the un repeated
        sequences are selected. This results in selecting the lowest energy pose
        for each unique sequence.
        :param results:
        :param metric:
        :return: New states:
        """

        # Sort the best current results.
        currentResults = self.sortResultsByMetric(results)

        # Select the newData from the correctResults based on sequenceDiff.
        newData, SequenceDiffList = self.selectResultsBySequenceDiff(results=currentResults, nElements=nElements)

        # Spawn the nPoses best correctResults
        replica = cycle(range(len(newData)))
        for proccess in range(1, self.nProccess):
            self.comm.send(newData[next(replica)], dest=proccess, tag=proccess)

    def spawnREM(self, results, kT):
        """
        Spawns a new iteration from the previous results by REM scheme.
        The results are first ordered by spawning metric and then the un repeated
        sequences are selected. This results in selecting the lowest energy pose
        for each unique sequence.
        :param results:
        """

        processIndex = [i for i in range(1, self.nProccess)]
        while processIndex:
            # Randomize the indices
            shuffle(processIndex)
            if len(processIndex) > 1:
                i = processIndex.pop(0)
                j = processIndex.pop(1)

                # Test sending replica i to explorer j. The result indeces are from 0
                if self.acceptMove(E_new=results[i-1].spawningEnergy, E_old=results[j-1].spawningEnergy, kT=kT):
                    self.comm.send(results[i-1], dest=j, tag=j)
                else:
                    self.comm.send(results[j-1], dest=j, tag=j)

                # Test sending replica j to explorer i
                if self.acceptMove(E_new=results[j-1].spawningEnergy, E_old=results[i-1].spawningEnergy, kT=kT):
                    self.comm.send(results[j-1], dest=i, tag=i)
                else:
                    self.comm.send(results[i-1], dest=i, tag=i)

            # One replica remain with out partner.
            else:
                i = processIndex.pop(0)
                self.comm.send(results[i-1], dest=i, tag=i)

    def updateSpawningMetric(self, inputs, iteration):
        for element in inputs.spawningMetricSteps:
            iterationRatio, spawningMetricMethod = element
            if iteration < (iterationRatio * inputs.nIterations):
                inputs.spawningMetric = spawningMetricMethod
                break

    def getState(self, metric):
        """
        Calculates different energies and Pack the current Results
        :return: Result
        """
        state = Result()

        # Get the Dsign pose
        dPose = self.sampler.getdPose()
        state.dPose = dPose

        # Get the acceptance ratio
        state.acceptanceRatio = self.sampler.acceptanceRatio
        state.ligandAcceptedRatio = self.sampler.ligandAcceptedRatio
        state.activeSiteAcceptedRatio = self.sampler.activeSiteAcceptedRatio

        # Compute energies
        state.scoreFullAtom = self.energyFullAtom(dPose.pose)
        state.scoreOnlyConstraints = self.energyOnlyConstraints(dPose.pose)
        state.scoreSASA = self.sampler.getSASAConstraint(dPose.pose)
        state.scoreLigand = self.sampler.getLigandEnergy(dPose.pose)
        state.scoreFullAtomWithConstraints = state.scoreFullAtom + state.scoreOnlyConstraints + state.scoreSASA

        # get the spawning energy
        if metric == 'FullAtomWithConstraints':
            state.spawningEnergy = state.scoreFullAtomWithConstraints

        elif metric == 'FullAtom':
            state.spawningEnergy = state.scoreFullAtom

        elif metric == 'OnlyConstraints':
            state.spawningEnergy = state.scoreOnlyConstraints

        elif metric == 'SASA':
            state.spawningEnergy = state.scoreSASA

        elif metric == 'Ligand':
            state.spawningEnergy = state.scoreSASA

        elif metric is None:
            state.spawningEnergy = state.scoreFullAtomWithConstraints

        else:
            raise ValueError(' {} spawningMetric is not defined.'.format(metric))

        # Get the sequence of the design domain and its differance against the original one (initial structure)
        state.sequence = self.sampler.getSequence()
        state.sequenceDiff = self.sampler.getSequenceDiff()

        return state

    def setState(self, state):
        self.sampler.setdPose(state.dPose)

    def sortResultsByMetric(self, results, metric=None):

        # Use the spawning if nothing is given, it used in both Adaptive and REM sampling
        if metric is None:
            results.sort(key=lambda element: element.spawningEnergy)
        elif metric == 'FullAtomWithConstraints':
            results.sort(key=lambda element: element.scoreFullAtomWithConstraints)
        elif metric == 'FullAtom':
            results.sort(key=lambda element: element.scoreFullAtom)
        elif metric == 'OnlyConstraints':
            results.sort(key=lambda element: element.scoreOnlyConstraints)
        elif metric == 'SASA':
            results.sort(key=lambda element: element.scoreSASA)
        elif metric == 'Ligand':
            results.sort(key=lambda element: element.scoreLigand)
        else:
            raise ValueError('Bad sorting metric.')

        return results

    def selectResultsBySequence(self, results, nElements):
        counter = 0
        newData = list()
        sequenceList = list()
        for element in results:
            if element.sequence not in sequenceList:
                # print(element.sequence)
                newData.append(element)
                sequenceList.append(element.sequence)
                counter += 1

            if counter == nElements:
                break

        return newData

    def selectResultsBySequenceDiff(self, results, nElements):
        counter = 0
        newData = list()
        sequenceDiffList = list()
        for element in results:
            if element.sequenceDiff not in sequenceDiffList:
                # print(element.sequence)
                newData.append(element)
                sequenceDiffList.append(element.sequenceDiff)
                counter += 1

            if counter == nElements:
                break

        return newData, sequenceDiffList

    def printIterattionResults(self, results, sequenceDiffList, iteration, nIterations, time, spawningMetric, kT_high, kT_low):
        iteration += 1
        print('------------------------------------------------------------------------------------------------')
        print('Iteration {}/{} done in {:.1f} s, SpawningMetric: {}, kT_high: {:8.1f}, kT_low: {:8.1f}.'.format(iteration, nIterations, time, spawningMetric,kT_high, kT_low))

        print('Rank\tTotal Energy\tFull Atom Energy\tConstraints Energy\tLigand(s) Energy\tSASA Energy\tTotal Acceptance Ratio\tActiveSite Acceptance Ratio\tLigand(s) Acceptance Ratio\tSequence\tMutations')
        lineFormat = '{:4d}\t{:8.1f}\t{:8.1f}\t{:8.1f}\t{:8.1f}\t{:8.1f}\t{:8.1f}\t{:8.1f}\t{:8.1f}\t{}\t{}'
        for i, data in enumerate(results):

            print(lineFormat.format(i, data.scoreFullAtomWithConstraints, data.scoreFullAtom, data.scoreOnlyConstraints,
                                    data.scoreLigand, data.scoreSASA, data.acceptanceRatio,
                                    data.activeSiteAcceptedRatio, data.ligandAcceptedRatio, data.sequence, "/".join(sequenceDiffList[i])))

    def printFinalSummary(self, results, sequenceDiffList, nElements, filename):
        print('\n\n')
        print('------------------------------------------------------------------------------------------------------')
        print('                                           FINAL RESULTS                                              ')
        print('------------------------------------------------------------------------------------------------------')
        
        output_csv = open("{}.csv".format(filename), "wt")

        output_csv.write("Rank,Total Energy,Full Atom Energy,Constraints Energy,Ligand(s) Energy,SASA Energy,Total Acceptance Ratio,ActiveSite Acceptance Ratio,Ligand(s) Acceptance Ratio,Sequence,Mutations\n")
        print('Rank\tTotal Energy\tFull Atom Energy\tConstraints Energy\tLigand(s) Energy\tSASA Energy\tTotal Acceptance Ratio\tActiveSite Acceptance Ratio\tLigand(s) Acceptance Ratio\tSequence\tMutations')
        lineFormat = '{:4d}\t{:8.1f}\t{:8.1f}\t{:8.1f}\t{:8.1f}\t{:8.1f}\t{:8.1f}\t{:8.1f}\t{:8.1f}\t{}\t{}'
        writeFormat = '{},{:.1f},{:.1f},{:.1f},{:.1f},{:.1f},{:.1f},{:.1f},{:.1f},{},{}\n'
        for i, data in enumerate(results):
            if i == nElements:
                break

            output_csv.write(writeFormat.format(i, data.scoreFullAtomWithConstraints, data.scoreFullAtom, data.scoreOnlyConstraints,
                                    data.scoreLigand, data.scoreSASA, data.acceptanceRatio,
                                    data.activeSiteAcceptedRatio, data.ligandAcceptedRatio, data.sequence, "/".join(sequenceDiffList[i])))
            print(lineFormat.format(i, data.scoreFullAtomWithConstraints, data.scoreFullAtom, data.scoreOnlyConstraints,
                                    data.scoreLigand, data.scoreSASA, data.acceptanceRatio,
                                    data.activeSiteAcceptedRatio, data.ligandAcceptedRatio, data.sequence, "/".join(sequenceDiffList[i])))
        output_csv.close()
    
    def printFinalSummarySerial(self, results, nElements):
        print('\n\n')
        print('------------------------------------------------------------------------------------------------------')
        print('                                           FINAL RESULTS                                              ')
        print('------------------------------------------------------------------------------------------------------')

        print('Rank\tTotal Energy\tFull Atom Energy\tConstraints Energy\tLigand(s) Energy\tSASA Energy\tTotal Acceptance Ratio\tActiveSite Acceptance Ratio\tLigand(s) Acceptance Ratio\tSequence\tMutations')
        lineFormat = '{:4d}\t{:8.1f}\t{:8.1f}\t{:8.1f}\t{:8.1f}\t{:8.1f}\t{:8.1f}\t{:8.1f}\t{:8.1f}\t{}'
        for i, data in enumerate(results):
            if i == nElements:
                break

            print(lineFormat.format(i, data.scoreFullAtomWithConstraints, data.scoreFullAtom, data.scoreOnlyConstraints,
                                    data.scoreLigand, data.scoreSASA, data.acceptanceRatio,
                                    data.activeSiteAcceptedRatio, data.ligandAcceptedRatio, data.sequence))

    def printFinalTimming(self, startDate, endDate, timeTotal):
        print('-----------------------------------------NORMAL TERMINATION------------------------------------------')
        print('Start Date: {}'.format(str(startDate).split('.')[0]))
        print('End Date: {}'.format(str(endDate).split('.')[0]))
        print('Elapsed time: {:.1f} s'.format(timeTotal))
        print('------------------------------------------------------------------------------------------------------')
        print('\n\n\n\n')

    def writeResults(self, results, sequenceDiffList, nElements, iteration='', prefix='', output='', energyType='FullAtomWithConstraints'):
        """
        Writes the PDB file for each results
        """
        if iteration != '':
            iteration = str((iteration+1))

        for i, result in enumerate(results):
            if energyType == 'FullAtomWithConstraints':
                energy = result.scoreFullAtomWithConstraints

            elif energyType == 'OnlyConstraints':
                energy = result.scoreOnlyConstraints

            elif energyType == 'SASA':
                energy = result.scoreSASA

            elif energyType == 'Ligand':
                energy = result.scoreLigand

            else:
                energy = result.scoreFullAtom

            dPose = result.dPose
            pose = dPose.pose

            if prefix:
                name = '{}_I{}_N{}_{}_{:.1f}'.format(prefix, iteration, i, "_".join(sequenceDiffList[i]), energy)
            else:
                name = '{}_I{}_N{}_{}_{:.1f}'.format('pose', iteration, i, "_".join(sequenceDiffList[i]), energy)

            pdbName = '{}.pdb'.format(name)
            pose.dump_pdb(os.path.join(output, pdbName))

            # Stop if the results have more elements that nElements
            if i == nElements:
                break
               
    def writeResultsSerial(self, results, nElements, iteration='', prefix='', output='', energyType='FullAtomWithConstraints'):
        """
        Writes the PDB file for each results
        """
        if iteration != '':
            iteration = str((iteration+1))

        for i, result in enumerate(results):
            if energyType == 'FullAtomWithConstraints':
                energy = result.scoreFullAtomWithConstraints

            elif energyType == 'OnlyConstraints':
                energy = result.scoreOnlyConstraints

            elif energyType == 'SASA':
                energy = result.scoreSASA

            elif energyType == 'Ligand':
                energy = result.scoreLigand

            else:
                energy = result.scoreFullAtom

            dPose = result.dPose
            pose = dPose.pose

            if prefix:
                name = '{}_I{}_N{}_{:.1f}'.format(prefix, iteration, i, energy)
            else:
                name = '{}_I{}_N{}_{:.1f}'.format('pose', iteration, i, energy)

            pdbName = '{}.pdb'.format(name)
            pose.dump_pdb(os.path.join(output, pdbName))

            # Stop if the results have more elements that nElements
            if i == nElements:
                break

    def initializeCalculation(self, inputs):
        """
        Initializes the needed Movers and the sampler
        """
        # 1) Initiate the DesignPose
        dPose = DesignPose()
        dPose.initiateDesign(residues=inputs.designResidues, pose=inputs.pose)
        dPose.initiateCatalytic(catalytic=inputs.catalyticResidues)
        dPose.initiateConstraints(constraints=inputs.constraints)
        # 2) Initiate score function
        #self.scorefxn = getScoreFunction(mode='fullAtomWithConstraints')
        #self.scorefxn.set_weight(score_type_from_name('fa_dun'), 0.0)

        # 3 ) Initiate Ligand movers
        ligandMovers = list()
        if inputs.ligands and inputs.ligandsParm:
            for ligand, ligandParm in zip(inputs.ligands, inputs.ligandsParm):

                # 3.1) initiate a rigid body mover
                if ligandParm['RigidBody']:
                    rigidbodyMover = LigandRigidBodyMover(ligand=ligand,
                                                          pose=inputs.pose,
                                                          dockingCenter=ligandParm['DockingCenter'],
                                                          simulationCenter=ligandParm['SimulationCenter'],
                                                          simulationRadius=ligandParm['SimulationRadius'],
                                                          nonStandardNeighbors=None,
                                                          neighborDistCutoff=ligandParm['NeighbourCutoff'],
                                                          translationSTD=ligandParm['TranslationSTD'],
                                                          rotationSTD=ligandParm['RotationSTD'],
                                                          translationLoops=ligandParm['TranslationLoops'],
                                                          rotationLoops=ligandParm['RotationLoops'],
                                                          sidechainCoupling=ligandParm['SideChainCoupling'],
                                                          sidechainCouplingMax=ligandParm['SideChainCouplingMax'],
                                                          sideChainCouplingExcludedPoseIndex=None,
                                                          backboneCoupling=1.0,
                                                          overlap=ligandParm['ClashOverlap'],
                                                          sasaScaling=ligandParm['SasaScaling'],
                                                          sasaCutoff=ligandParm['SasaCutoff'],
                                                          translationScale=ligandParm['TranslationScale'],
                                                          rotationScale=ligandParm['RotationScale'])

                    # Set the nonStandardNeighbors. Add all ligands. Their core atoms are already assigned
                    # even for only rigid body ones
                    for ligand_i in inputs.ligands:
                        # skip itself
                        if ligand_i.name == ligand.name:
                            continue
                        rigidbodyMover.addNonStandardNeighborDict(ligand_i)
                else:
                    rigidbodyMover = None

                # 3.2) Initiate Ligand Packer, regrdless of input, it is used for ligand minimization
                packerMover = LigandPackingMover(ligand=ligand,
                                                 pose=inputs.pose,
                                                 nonStandardNeighbors=None,
                                                 neighborDistCutoff=ligandParm['NeighbourCutoff'],
                                                 mainChainCoupling=1.0,
                                                 sideChainCoupling=ligandParm['SideChainCoupling'],
                                                 sideChainCouplingMax=ligandParm['SideChainCouplingMax'],
                                                 sideChainCouplingExcludedPoseIndex=None,
                                                 sideChainsGridPoints=ligandParm['TorsionsGridPoints'],
                                                 sideChainsGridLimit=ligandParm['SideChainsGridLimit'],
                                                 maxGrid=ligandParm['MaxGrid'],
                                                 minGrid=ligandParm['MinGrid'],
                                                 gridInterval=ligandParm['GridInterval'],
                                                 numberOfGridNeighborhood=ligandParm['NumberOfGridNeighborhoods'],
                                                 packingLoops=ligandParm['PackingLoops'],
                                                 minimization=True,
                                                 fullEnergy=ligandParm['FullEnergy'],
                                                 nRandomTorsionPurturbation=ligandParm['nRandomTorsionPurturbation'],
                                                 nproc=self.nProcessOMP)
                if self.rank == 0:
                    print(packerMover.show())
                # Set the nonStandardNeighbors. Add all ligands. Their core atoms are already assigned
                # even for only rigid body ones
                for ligand_i in inputs.ligands:
                    # skip itself
                    if ligand_i.name == ligand.name:
                        continue
                    packerMover.addNonStandardNeighborDict(ligand_i)


                # 3.3) Initiate the neighbor packer
                neighbourPacker = LigandNeighborsPackingMover(ligand=ligand,
                                                              excludedResiduePoseIndex=None,
                                                              neighborDistCutoff=ligandParm['NeighbourCutoff'],
                                                              scratch=inputs.scratch)

                # 3.4 ) Combine everything in a ligand mover
                ligandMover = LigandMover(ligand=ligand,
                                          ligandRigidBodyMover=rigidbodyMover,
                                          doRigidBody=ligandParm['RigidBody'],
                                          ligandPackingMover=packerMover,
                                          doPacking=ligandParm['Packing'],
                                          ligandNeighborsPackingMover=neighbourPacker,
                                          doNeighborPacking=True,
                                          ligandPurturbationMode=ligandParm['PerturbationMode'],
                                          ligandPurturbationLoops=ligandParm['PerturbationLoops'],
                                          sasaConstraint=ligandParm['SasaConstraint'])

                ligandMovers.append(ligandMover)

        # 4) initiate active site mover
        activeSiteMover = ActiveSiteMover(activeSiteDesignMode=inputs.activeSiteDesignMode,
                                          mimimizeBackbone=inputs.mimimizeBackbone,
                                          activeSiteLoops=inputs.activeSiteLoops,
                                          nNoneCatalytic=inputs.nNoneCatalytic,
                                          scratch=inputs.scratch)

        # 5) Initiate active site sampler
        self.sampler = ActiveSiteSampler(softRepulsion=inputs.softRepulsion,
                                         dynamicSideChainCoupling=inputs.dynamicSideChainCoupling,
                                         activeSiteSampling=inputs.activeSiteSampling,
                                         ligandSampling=inputs.ligandSampling,
                                         kT=inputs.kT,
                                         nSteps=inputs.nSteps,
                                         anneal=inputs.anneal,
                                         kT_high=inputs.kT_high,
                                         kT_low=inputs.kT_low,
                                         kT_decay=inputs.kT_decay)

        if self.rank == 0:
            print(dPose.state())

        # 6) Sets the movers
        self.sampler.setLigandMovers(ligandMovers)
        self.sampler.setActiveSiteMover(activeSiteMover)
        self.sampler.setScoreFunction(self.energyFullAtomWithConstraints)
        # attempt to initialize the pose.
        self.sampler.setInitialdPose(dPose, scratch=inputs.scratch)

    def initializeInputs(self, confFile):
        """
        This function deals with input management
        :param confFile: describing the calculation details
        :return inputs: object that contains all inputs (pose, DesignDomainMover, kt, ...)
        """
        print("\n\nStarting ActiveSiteDesign using {} explorers with {} thread(s)".format(self.nProccess - 1, self.nProcessOMP))
        print("Reading Input file: {}.".format(confFile))

        with open(confFile, 'r') as f:
            inputFile = yaml.safe_load(f)

        try:
            inputs = InputActiveSiteDesign()
            inputs.read(inputFile)
            return inputs
        except Exception as e:
            traceback.print_tb(e.__traceback__)
            print('\n')
            raise ValueError('Error reading input file, {}'.format(e))

    def clean(self, inputs):
        if os.path.isdir(inputs.scratch):
            shutil.rmtree(inputs.scratch, ignore_errors=True)

    def getOMPnProcesses(self):

        nProccessOMP = 1
        threadFound = False

        # Check different queuing systems
        if not threadFound:
            try:
                nProccessOMP = os.environ["SLURM_CPUS_PER_TASK"]
                threadFound = True

            except:
                pass

        if not threadFound:
            try:
                nProccessOMP = os.environ["OMP_NUM_THREADS"]
                threadFound = True
            except:
                pass

        nProccessOMP = int(nProccessOMP)
        if nProccessOMP < 1:
            raise ValueError('Failed initialize calculations. Failed to get number of threads. Set OMP_NUM_THREADS manually.')
        else:
            os.environ["OMP_NUM_THREADS"] = str(nProccessOMP)

        return nProccessOMP

    def acceptMove(self, E_new, E_old, kT):
        dV = (E_new - E_old)
        if dV < 0:
            W = 1
        else:
            W = exp(-dV / kT)

        if W > uniform(0, 1):
            return True
        else:
            return False


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Active Site Design')
    parser.add_argument('conf', type=str, help='Configuration file.')
    parser.add_argument('-debug', action='store_true', help='Debug mode.')

    args = parser.parse_args()
    Constants.DEBUG = args.debug
    confFile = args.conf
    if not os.path.isfile(confFile):
        killProccesses('No conf file is given')

    design = ActiveSiteDesign()
    results = design.run(confFile)
