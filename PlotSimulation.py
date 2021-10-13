import os
import time
import copy
import numpy as np
import matplotlib.pyplot as plt
from Analysis import Analysis
from seaborn import heatmap, barplot
from matplotlib.widgets import Cursor
from sklearn.cluster import AgglomerativeClustering
import random
import pyrosetta as pr
from pyrosetta.rosetta.utility import vector1_unsigned_long
from pyrosetta.rosetta.protocols.toolbox.pose_manipulation import superimpose_pose_on_subset_CA
pr.init(''' -out:level 0 -no_his_his_pairE -extrachi_cutoff 1 -multi_cool_annealer 10 -ex1 -ex2 -use_input_sc ''')

#resultFileName = '/media/masoud/WRKP/EDesign/tests/DesigCatalytic-PEF-WithXS1-M2-AsMIN-Coupled-SCCopu-005/PEF.out'

class PlotResults(object):
    def __init__(self):
        self.exit = False
        self.inputFileName = None
        self.refStructure = None
        self.refStructureFileName = None
        self.pathToFinalResults = None
        self.pathToOutputs = None
        self.prefixName = ''
        self.parmFileNmaes = list()
        self.resultsFull = list()
        self.sequenceFull = list()
        self.resultsFinal = list()
        self.sequenceFinal = list()
        self.residueNames = list()
        self.ligandNames = list()
        self.labels = {1: 'Full Atom Energy', 2:  'Constraints Energy', 3: 'Ligands Energy', 4: 'SASA Energy'}


        self.runUserInterface()

    def runUserInterface(self):
        while not self.exit:
            self.printMain()
            self.getCommand()

    def printMain(self):
        string = ''
        string += 'Commands List:\n'
        string += ' q:   Exit\n'
        string += ' i:   Read Input\n'
        #string += ' s:   Load Structure\n'
        string += ' pi:  Plot Iterations Results\n'
        string += ' pf:  Plot Final Results\n'
        #string += ' ci:  Cluster Iterations Results\n'
        string += ' cf:  Cluster Final Results\n'

        print(string)

    def getCommand(self):
        command = input()
        if command == 'q' or command == 'Q':
            self.exit = True

        elif command == 'i':
            self.readInputs()

        #elif command == 's':
        #    self.readStructure()

        elif command == 'pi' or command == 'pf':

            # Check the results are loaded
            if command == 'pi' and len(self.resultsFull) == 0:
                print('No iteration results is loaded.')
                return

            if command == 'pf' and len(self.resultsFinal) == 0:
                print('No final results is loaded.')
                return

            # Get the plot X, Y, Z
            xyz_input = input('Enter the plotting metrics Number: X  Y  Z(Optional)\n    1:  FullEnergy\n    2:  ConstraintsEnergy\n    3:  LigandsEnergy\n    4:  SASAEnergy\n')
            xyz_input = xyz_input.split()

            # Check the inputs
            try:
                if len(xyz_input) == 2:
                    x, y, z = int(xyz_input[0]), int(xyz_input[1]), 0
                elif len(xyz_input) == 3:
                    x, y, z = int(xyz_input[0]), int(xyz_input[1]), int(xyz_input[2])
                else:
                    raise ValueError
            except:
                print('Bad input for XYZ metrics')
                return

            # print or save
            if command == 'pi':
                self.printResults(x, y, z)
            elif command == 'pf':
                self.printResultsFinal(x, y, z)

        elif command == 'ci' or command == 'cf':

            if command == 'cf' and len(self.resultsFinal) == 0:
                print('No final results is loaded.')
                return

            #if command == 'ci':
            #    self.clusterFull()

            if command == 'cf':
                self.clusterFinal()

        else:
            print('Command {} is not valid'.format(command))

    """
    def clusterFull(self, x, y, rankingMetric):
        cutoff = input('Enter the distance cutoff (>0): ')
        try:
            cutoff = float(cutoff)
        except:
            print('Bad cutoff value')
            return
        if cutoff <= 0:
            print('Bad cutoff value')
            return


        refSequence = list()
        caPositions = list()
        for residueName in self.residueNames:
            id, chain = int(residueName.split('-')[0]), residueName.split('-')[1]
            poseIndex = self.refStructure.pdb_info().pdb2pose(chain, id)
            residue = self.refStructure.residue(poseIndex)

            refSequence.append(residue.name1())
            caPositions.append(residue.xyz('CA'))

        #print(self.residues)
        #print('----------------------------------->>>>>', refSequence)
        caPositions = np.array(caPositions)

        entryClusterIndex = list()
        clusters = list()
        clusterFound = False
        for sequence in self.sequenceFull:

            #print(clusters)
            clusterFound = False
            mutationindice = [i for i in range(len(refSequence)) if refSequence[i] != sequence[i]]
            icentroid = np.zeros(3)
            for index in mutationindice:
                icentroid +=  caPositions[index]
            icentroid /= len(mutationindice)

            # Get the distance to cluster
            for cindex, ccentroid in enumerate(clusters):
                dist = np.linalg.norm(icentroid - ccentroid)
                if dist <= cutoff:
                    entryClusterIndex.append(cindex)
                    clusterFound = True
                    break

            if not clusterFound:
                clusters.append(icentroid)
                entryClusterIndex.append(len(clusters) - 1)

        fig, ax = plt.subplots(nrows=1, ncols=1)
        sc = plt.scatter(self.resultsFull[:, x + 1], self.resultsFull[:, y + 1], c=entryClusterIndex, cmap='rainbow', marker='o', edgecolors="black")
        xLable = self.labels[x]
        yLable = self.labels[y]

        ax.set_xlabel(xLable)
        ax.set_ylabel(yLable)

        #plt.colorbar(sc, ax=ax)

        annot = ax.annotate("", xy=(0, 0), xytext=(20, 20), textcoords="offset points", bbox=dict(boxstyle="round", fc="w"), arrowprops=dict(arrowstyle="->"))
        annot.set_visible(False)


        def update_annot(ind):
            pos = sc.get_offsets()[ind["ind"][0]]
            annot.xy = pos
            text = '\n'.join(['Iteration {:.0f} - Rank {:.0f}'.format(self.resultsFull[i][0], self.resultsFull[i][1]) for i in ind["ind"]])
            #text = "{}, {}".format(" ".join(list(map(str, ind["ind"]))),
            #                       " ".join([names[n] for n in ind["ind"]]))
            annot.set_text(text)
            #annot.get_bbox_patch().set_facecolor(cmap(norm(c[ind["ind"][0]])))
            annot.get_bbox_patch().set_alpha(1)

        def hover(event):
            vis = annot.get_visible()
            if event.inaxes == ax:
                cont, ind = sc.contains(event)
                if cont:
                    update_annot(ind)
                    annot.set_visible(True)
                    fig.canvas.draw_idle()
                else:
                    if vis:
                        annot.set_visible(False)
                        fig.canvas.draw_idle()


        fig.canvas.mpl_connect("motion_notify_event", hover)
        plt.show()

        # print the results
        entryClusters = [[] for i in range(len(clusters))]
        for i, entry in enumerate(self.resultsFull):
            index = entryClusterIndex[i]
            entryClusters[index].append(entry)

        with open('clusters_Full_{}'.format(self.inputFileName), 'w') as f:
            for i, entryCluster in enumerate(entryClusters):
                f.write('Cluster {}: \n'.format(i))
                entryCluster.sort(key=lambda element: element[rankingMetric+1])
                for entry in entryCluster:
                    #print(entry)
                    f.write('Iteration: {:3.0f}    Rank:  {:3.0f}    Full Atom Energy: {:10.1f}    Constraints Energy: {:10.1f}    Ligands Energy: {:10.1f}    SASA Energy: {:10.1f}\n'.format(entry[0], entry[1], entry[2], entry[3], entry[4], entry[5]))
    """

    def clusterFinal(self):

        # load ref structure and get the sequence, active site pose index and ligand pose index
        for i, parm in enumerate(self.parmFileNmaes):
            if not os.path.isfile(parm):
                print('Input defined parameter file {} was not found!'.format(parm))
                self.parmFileNmaes[i] = input('Enter the file name of the parameter file: ')
                if not os.path.isfile(self.parmFileNmaes[i]):
                    print('Parameter File {} was not found! exiting...'.format(self.parmFileNmaes[i]))

        if not os.path.isfile(self.refStructureFileName):
            print('Input defined reference structure {} was not found!'.format(self.refStructureFileName))
            self.refStructureFileName = input('Enter the file name of the reference structure: ')
            if not os.path.isfile(self.refStructureFileName):
                print('Reference structure {} was not found! exiting...'.format(self.refStructureFileName))
                return

        self.refStructure = self.readStructure(self.refStructureFileName, self.parmFileNmaes)

        refSequence = list()
        activesitePoseIndices = list()
        ligandasPoseIndices = list()
        for residueName in self.residueNames:
            id, chain = int(residueName.split('-')[0]), residueName.split('-')[1]
            poseIndex = self.refStructure.pdb_info().pdb2pose(chain, id)
            activesitePoseIndices.append(poseIndex)
            refSequence.append(self.refStructure.residue(poseIndex).name1())

        for ligandName in self.ligandNames:
            id, chain = int(ligandName.split('-')[0]), ligandName.split('-')[1]
            poseIndex = self.refStructure.pdb_info().pdb2pose(chain, id)
            ligandasPoseIndices.append(poseIndex)

        #print(refSequence)
        #print(activesitePoseIndices)
        #print(ligandasPoseIndices)

        if not os.path.isdir(self.pathToFinalResults):
            print('Input defined final results directory {} was not found!'.format(self.pathToFinalResults))
            self.pathToFinalResults = input('Enter the final results directory: ')
            if not os.path.isfile(self.pathToFinalResults):
                print('Final results directory {} was not found! exiting...'.format(self.pathToFinalResults))
                return

        # Get design CA XYZ and ligand(s) XYZ
        ligandsXYZ = dict()
        activesiteGOMXYZ = dict()
        activesiteVec1 = vector1_unsigned_long()
        activesiteVec1.extend(activesitePoseIndices)
        for strucName in os.listdir(self.pathToFinalResults):

            start = time.time()

            # Get the rank
            irank = int(strucName.split('_')[-2].split('N')[-1])

            # read the structures in the final output
            istruct = self.readStructure(os.path.join(self.pathToFinalResults, strucName), self.parmFileNmaes)

            # Superimpose Structures to the reference
            superimpose_pose_on_subset_CA(istruct, self.refStructure, activesiteVec1)

            # Get design GOM XYZ
            caXYZ = np.array([list(istruct.residue(poseIndex).xyz('CA')) for poseIndex in activesitePoseIndices])
            activesiteGOMXYZ[irank] = caXYZ.sum(axis=0)/caXYZ.shape[0]

            # ligand(s) XYZ
            ligsXYZ = list()
            for poseIndex in ligandasPoseIndices:
                ligsXYZ.append(np.array([list(istruct.residue(poseIndex).xyz(i)) for i in range(1, istruct.residue(poseIndex).natoms() + 1)]))
            ligandsXYZ[irank] = ligsXYZ

            end = time.time()
            print('Read {} in {:10.1f}s'.format(strucName, end-start))

        # Compute pairwise GOM dist and ligandRMSD Matrices (Tensor)
        nEntry = len(self.resultsFinal)
        distGOM = np.zeros((nEntry, nEntry))
        distRMSD = np.zeros((nEntry, nEntry))
        for i in range(nEntry):
            for j in range(i+1, nEntry):
                # GOM dist
                iGOM = activesiteGOMXYZ[i]
                jGOM = activesiteGOMXYZ[j]
                distGOM[i, j] = np.linalg.norm(iGOM - jGOM)
                distGOM[j, i] = distGOM[i, j]

                # RMDS dist
                rmsdTotal = 0
                for iligandXYZ, jligandXYZ in zip(ligandsXYZ[i], ligandsXYZ[j]):
                    rmsd = iligandXYZ - jligandXYZ
                    rmsd = np.square(rmsd)
                    rmsd = np.sum(rmsd)
                    rmsd = rmsd / jligandXYZ.shape[0]
                    rmsd = np.sqrt(rmsd)
                    rmsdTotal += rmsd
                distRMSD[i, j] = rmsdTotal
                distRMSD[j, i] = rmsdTotal


        while True:

            nClusters = input('Enter the number of clusters or q to exit: ')
            if nClusters == 'q':
                return
            try:
                nClusters = int(nClusters)
            except:
                print('Bad value. Number of clusters should be > 1.')
                continue
            if nClusters <= 1:
                print('Bad value. Number of clusters should be > 1.')
                continue

            weights = input('Enter the active site and ligand weights: ')
            try:
                activesiteWeight,  ligandWeight= float(weights.split()[0]), float(weights.split()[1])
            except:
                print('Bad values for the active site and ligand weights')
                continue

            # Get the plot X, Y, Z
            xyz_input = input('Enter the plotting metrics Number: X  Y\n    1:  FullEnergy\n    2:  ConstraintsEnergy\n    3:  LigandsEnergy\n    4:  SASAEnergy\n')
            xyz_input = xyz_input.split()

            # Check the inputs
            try:
                x, y = int(xyz_input[0]), int(xyz_input[1])
            except:
                print('Bad input for XY metrics')
                continue

            rankingMetric = input('Enter ranking metrics Number: \n    1:  FullEnergy\n    2:  ConstraintsEnergy\n    3:  LigandsEnergy\n    4:  SASAEnergy\n')
            rankingMetric = rankingMetric.split()
            try:
                rankingMetric = int(rankingMetric[0])
            except:
                print('Bad input for ranking metrics')
                continue

            distGOM = (distGOM - distGOM.min()) / (distGOM.max() - distGOM.min())
            distRMSD = (distRMSD - distRMSD.min()) / (distRMSD.max() - distRMSD.min())

            # Compute tensor element wise norms (norm Matrix)
            distNorm = (activesiteWeight * distGOM) + (ligandWeight * distRMSD)

            clustering = AgglomerativeClustering(n_clusters=nClusters, affinity='precomputed', linkage='complete').fit(distNorm)
            entryClusterIndex = clustering.labels_

            """
            # do a k nearest neighbor
            clusters_p = [[i] for i in random.sample(range(nEntry), nClusters)]
            for k in range(100):
                #print(k, clusters_p)
                # reset the current cluster
                clusters_c = [[] for i in range(nClusters)]
                for i in range(nEntry):
                    clustrIndex = 0
                    dist_p = float('inf')
                    for ci, cluster in enumerate(clusters_p):
                        # skip if the cluster is empty
                        if len(cluster) == 0:
                            continue
                        # Compute the average dist of point to the cluster
                        dist_c = 0
                        for element in cluster:
                            dist_c += distNorm[element, i]
                        dist_c /= len(cluster)
                        # assing if a better distance is found
                        if dist_c < dist_p:
                            clustrIndex = ci
                            dist_p = dist_c

                    # save the results
                    clusters_c[clustrIndex].append(i)
                # evaluate the convergence
                if clusters_c == clusters_p:
                    break
                else:
                    clusters_p = copy.deepcopy(clusters_c)

            for i in clusters_p:
                print(i)
            """

            fig, ax = plt.subplots(nrows=1, ncols=1)
            sc = plt.scatter(self.resultsFinal[:, x + 1], self.resultsFinal[:, y + 1], c=entryClusterIndex, cmap='rainbow', marker='o', edgecolors="black")
            xLable = self.labels[x]
            yLable = self.labels[y]

            ax.set_xlabel(xLable)
            ax.set_ylabel(yLable)

            #plt.colorbar(sc, ax=ax)

            annot = ax.annotate("", xy=(0, 0), xytext=(20, 20), textcoords="offset points", bbox=dict(boxstyle="round", fc="w"), arrowprops=dict(arrowstyle="->"))
            annot.set_visible(False)


            def update_annot(ind):
                pos = sc.get_offsets()[ind["ind"][0]]
                annot.xy = pos
                text = '\n'.join(['Iteration {:.0f} - Rank {:.0f}'.format(self.resultsFinal[i][0], self.resultsFinal[i][1]) for i in ind["ind"]])
                #text = "{}, {}".format(" ".join(list(map(str, ind["ind"]))),
                #                       " ".join([names[n] for n in ind["ind"]]))
                annot.set_text(text)
                #annot.get_bbox_patch().set_facecolor(cmap(norm(c[ind["ind"][0]])))
                annot.get_bbox_patch().set_alpha(1)

            def hover(event):
                vis = annot.get_visible()
                if event.inaxes == ax:
                    cont, ind = sc.contains(event)
                    if cont:
                        update_annot(ind)
                        annot.set_visible(True)
                        fig.canvas.draw_idle()
                    else:
                        if vis:
                            annot.set_visible(False)
                            fig.canvas.draw_idle()


            fig.canvas.mpl_connect("motion_notify_event", hover)
            plt.show()

            # print the results
            entryClusters = [[] for i in range(clustering.n_clusters)]
            for i, entry in enumerate(self.resultsFinal):
                index = entryClusterIndex[i]
                entryClusters[index].append(entry)

            with open('clusters_AS{}_LIG{}_Final_{}'.format(activesiteWeight, ligandWeight, self.inputFileName), 'w') as f:
                for i, entryCluster in enumerate(entryClusters):
                    f.write('Cluster {}: \n'.format(i))
                    entryCluster.sort(key=lambda element: element[rankingMetric+1])
                    for entry in entryCluster:
                        #print(entry)
                        f.write('   Rank:  {:3.0f}    Full Atom Energy: {:10.1f}    Constraints Energy: {:10.1f}    Ligands Energy: {:10.1f}    SASA Energy: {:10.1f}\n'.format(entry[1], entry[2], entry[3], entry[4], entry[5]))

    def printResults(self, x, y, z):

        fig, ax = plt.subplots(nrows=1, ncols=1)
        if z == 0:
            sc = plt.scatter(self.resultsFull[:, x+1], self.resultsFull[:, y+1], marker='o', edgecolors="black")
        else:
            sc = plt.scatter(self.resultsFull[:, x + 1], self.resultsFull[:, y + 1], c=self.resultsFull[:, z + 1], marker='o', edgecolors="black")

        xLable = self.labels[x]
        yLable = self.labels[y]

        ax.set_xlabel(xLable)
        ax.set_ylabel(yLable)

        plt.colorbar(sc, ax=ax)

        annot = ax.annotate("", xy=(0, 0), xytext=(20, 20), textcoords="offset points", bbox=dict(boxstyle="round", fc="w"), arrowprops=dict(arrowstyle="->"))
        annot.set_visible(False)


        def update_annot(ind):
            pos = sc.get_offsets()[ind["ind"][0]]
            annot.xy = pos
            text = '\n'.join(['Iteration {:.0f} - Rank {:.0f}'.format(self.resultsFull[i][0], self.resultsFull[i][1]) for i in ind["ind"]])
            #text = "{}, {}".format(" ".join(list(map(str, ind["ind"]))),
            #                       " ".join([names[n] for n in ind["ind"]]))
            annot.set_text(text)
            #annot.get_bbox_patch().set_facecolor(cmap(norm(c[ind["ind"][0]])))
            annot.get_bbox_patch().set_alpha(1)

        def hover(event):
            vis = annot.get_visible()
            if event.inaxes == ax:
                cont, ind = sc.contains(event)
                if cont:
                    update_annot(ind)
                    annot.set_visible(True)
                    fig.canvas.draw_idle()
                else:
                    if vis:
                        annot.set_visible(False)
                        fig.canvas.draw_idle()


        fig.canvas.mpl_connect("motion_notify_event", hover)

        plt.show()

        # Save Figures
        PMatrix = Analysis.SequencesPMatrix(self.sequenceFull)
        SVector = Analysis.SequencesSVector(self.sequenceFull)
        UMatrix = Analysis.SequencesUMatrix(self.sequenceFull)

        residues = ['{}{}'.format(i.split('-')[0], i.split('-')[1]) for i in self.residueNames]
        fig, ((ax_00, ax_01), (ax_10, ax_11)) = plt.subplots(nrows=2, ncols=2, figsize=(8, 8))
        fig.tight_layout(pad=2.0)

        if z == 0:
            ax_00.scatter(self.resultsFull[:, x+1], self.resultsFull[:, y+1], marker='o', edgecolors="black")
        else:
            ax_00.scatter(self.resultsFull[:, x + 1], self.resultsFull[:, y + 1], c=self.resultsFull[:, z + 1], marker='o', edgecolors="black")

        xLable = self.labels[x]
        yLable = self.labels[y]

        ax_00.set_xlabel(xLable)
        ax_00.set_ylabel(yLable)

        ax_Pmatrix = heatmap(PMatrix, ax=ax_01, xticklabels=Analysis.AminoAcids, yticklabels=residues, cmap="Blues", linewidth=0.5)
        ax_Pmatrix.xaxis.tick_top()  # x axis on top
        ax_Pmatrix.xaxis.set_label_position('top')
        #ax_01_heatMap.tick_params(labelsize=10)

        ax_Pmatrix.set_xlabel('Amino Acids')
        ax_Pmatrix.set_ylabel('Positions')

        ax_SVector = barplot(x=residues, y=SVector, ax=ax_10, color="skyblue")
        ax_SVector.set_xticklabels(rotation=90, labels=residues)

        ax_SVector.set_xlabel('Positions')
        ax_SVector.set_ylabel('Entropy')

        ax_UMatrix = heatmap(UMatrix, ax=ax_11, xticklabels=residues, yticklabels=residues, cmap="Blues_r", linewidth=0.5)
        ax_UMatrix.xaxis.tick_bottom()
        ax_UMatrix.xaxis.set_label_position('bottom')
        ax_UMatrix.set_xticklabels(rotation=90, labels=residues)

        ax_UMatrix.set_xlabel('Positions')
        ax_UMatrix.set_ylabel('Positions')


        plt.show()
        #fig.autofmt_xdate()
        #plt.savefig(os.path.join(self.fileName.split('.')[-1] + '_plot' + '.pdf'), dpi=300)

    def printResultsFinal(self, x, y, z):

        fig, ax = plt.subplots(nrows=1, ncols=1)
        if z == 0:
            sc = plt.scatter(self.resultsFull[:, x+1], self.resultsFull[:, y+1], marker='o', c='gray')
            sc = plt.scatter(self.resultsFinal[:, x + 1], self.resultsFinal[:, y + 1], marker='o', edgecolors="black")
        else:
            sc = plt.scatter(self.resultsFull[:, x + 1], self.resultsFull[:, y + 1], marker='o', c='gray')
            sc = plt.scatter(self.resultsFinal[:, x + 1], self.resultsFinal[:, y + 1], c=self.resultsFinal[:, z + 1], marker='o', edgecolors="black")

        xLable = self.labels[x]
        yLable = self.labels[y]

        ax.set_xlabel(xLable)
        ax.set_ylabel(yLable)

        plt.colorbar(sc, ax=ax)

        annot = ax.annotate("", xy=(0, 0), xytext=(20, 20), textcoords="offset points",
                            bbox=dict(boxstyle="round", fc="w"),
                            arrowprops=dict(arrowstyle="->"))
        annot.set_visible(False)


        def update_annot(ind):
            pos = sc.get_offsets()[ind["ind"][0]]
            annot.xy = pos
            text = '\n'.join(['Rank {:.0f}'.format(self.resultsFinal[i][1]) for i in ind["ind"]])
            #text = "{}, {}".format(" ".join(list(map(str, ind["ind"]))),
            #                       " ".join([names[n] for n in ind["ind"]]))
            annot.set_text(text)
            #annot.get_bbox_patch().set_facecolor(cmap(norm(c[ind["ind"][0]])))
            annot.get_bbox_patch().set_alpha(1)

        def hover(event):
            vis = annot.get_visible()
            if event.inaxes == ax:
                cont, ind = sc.contains(event)
                if cont:
                    update_annot(ind)
                    annot.set_visible(True)
                    fig.canvas.draw_idle()
                else:
                    if vis:
                        annot.set_visible(False)
                        fig.canvas.draw_idle()


        fig.canvas.mpl_connect("motion_notify_event", hover)

        plt.show()

        # Save Figures
        PMatrix = Analysis.SequencesPMatrix(self.sequenceFinal)
        SVector = Analysis.SequencesSVector(self.sequenceFinal)
        UMatrix = Analysis.SequencesUMatrix(self.sequenceFinal)

        residues = ['{}{}'.format(i.split('-')[0], i.split('-')[1]) for i in self.residueNames]
        fig, ((ax_00, ax_01), (ax_10, ax_11)) = plt.subplots(nrows=2, ncols=2, figsize=(8, 8))
        fig.tight_layout(pad=2.0)

        if z == 0:
            ax_00.scatter(self.resultsFinal[:, x+1], self.resultsFinal[:, y+1], marker='o', edgecolors="black")
        else:
            ax_00.scatter(self.resultsFinal[:, x + 1], self.resultsFinal[:, y + 1], c=self.resultsFinal[:, z + 1], marker='o', edgecolors="black")

        xLable = self.labels[x]
        yLable = self.labels[y]

        ax_00.set_xlabel(xLable)
        ax_00.set_ylabel(yLable)

        ax_Pmatrix = heatmap(PMatrix, ax=ax_01, xticklabels=Analysis.AminoAcids, yticklabels=residues, cmap="Blues", linewidth=0.5)
        ax_Pmatrix.xaxis.tick_top()  # x axis on top
        ax_Pmatrix.xaxis.set_label_position('top')
        #ax_01_heatMap.tick_params(labelsize=10)

        ax_Pmatrix.set_xlabel('Amino Acids')
        ax_Pmatrix.set_ylabel('Positions')

        ax_SVector = barplot(x=residues, y=SVector, ax=ax_10, color="skyblue")
        ax_SVector.set_xticklabels(rotation=90, labels=residues)

        ax_SVector.set_xlabel('Positions')
        ax_SVector.set_ylabel('Entropy')

        ax_UMatrix = heatmap(UMatrix, ax=ax_11, xticklabels=residues, yticklabels=residues, cmap="Blues_r", linewidth=0.5)
        ax_UMatrix.xaxis.tick_bottom()
        ax_UMatrix.xaxis.set_label_position('bottom')
        ax_UMatrix.set_xticklabels(rotation=90, labels=residues)

        ax_UMatrix.set_xlabel('Positions')
        ax_UMatrix.set_ylabel('Positions')


        plt.show()

    def readStructure(self, pdbFileName, ParmfileNames):

        structure = pr.Pose()
        res_set = structure.conformation().modifiable_residue_type_set_for_conf()
        res_set.read_files_for_base_residue_types(pr.Vector1(ParmfileNames))
        structure.conformation().reset_residue_type_set_for_conf(res_set)
        pr.pose_from_file(structure, pdbFileName)

        return structure

    def readInputs(self):

        self.inputFileName = input('Enter input file Name: ')

        # Check the input file
        if not os.path.isfile(self.inputFileName):
            print('{} was not found.\n')
            self.inputFileName = None
            return

        # Open the input file:
        f = open(self.inputFileName, 'r')

        # initiate the variables
        readResidues = False
        residuesLoaded = False
        readFinal = False
        readresults = True
        readLigands = False
        ligandsLoaded = False

        self.resultsFull = list()
        self.resultsFinal = list()
        self.sequenceFull = list()
        self.sequenceFinal = list()
        self.residueNames = list()

        # Read the input file
        for line in f:
            line_split = line.split()
            # Skip the empty line
            if len(line_split) == 0:
                continue

            if 'Parameter Files:' in line:
                self.parmFileNmaes = line.split('Parameter Files:')[-1].split()

            if 'PDB:' in line:
                self.refStructureFileName = line.split('PDB:')[-1].split()[0]

            if 'Path to Outputs:' in line:
                self.pathToOutputs = line.split('Path to Outputs:')[-1].split()[0]
                self.prefixName = line.split('_output')[0]

            if 'Path to Final Results:' in line:
                self.pathToFinalResults = line.split('Path to Final Results:')[-1].split()[0]
                self.prefixName = line.split('_final_pose')[0]

            # Load the res names
            if 'Design residues:' in line:
                readResidues = True
                continue

            if readResidues and not residuesLoaded:
                try:
                    id, chain = int(line_split[0].split('-')[0]), line_split[0].split('-')[1]
                    self.residueNames.append('{}-{}'.format(id, chain))
                except:
                    readResidues = False
                    residuesLoaded = True

            if 'Ligands:' in line:
                readLigands = True
                continue

            if readLigands and '   ' != line[:3]:
                readLigands = False
                ligandsLoaded = True

            if readLigands and not ligandsLoaded:
                if 'Ligand (' in line:
                    self.ligandNames.append('-'.join(line.split('Ligand')[-1].split('(')[-1].split("'):")[0].split(", '")))
                    #print('XXX  ', self.ligandNames)

            # set the final results flag
            if '       FINAL RESULTS        ' in line:
                readFinal = True
            # Terminate the reading
            #if '      FINAL RESULTS CLUSTERS      ' in line:
            #    break

            # Read iteration
            if line_split[0] == 'Iteration':
                iteration = line_split[1].split('/')[0]
            # Read iteration results
            if not readFinal and line_split[0] == '#':
                rank, fullEnergy, constraintsEnergy, ligandsEnergy, SASAEnergy, sequence = line_split[1], line_split[3], line_split[4], line_split[5], line_split[6], line_split[10]
                self.resultsFull.append([iteration, rank, fullEnergy, constraintsEnergy, ligandsEnergy, SASAEnergy])
                self.sequenceFull.append(sequence)
            # read final results
            elif readFinal and line_split[0] == '#':
                rank, fullEnergy, constraintsEnergy, ligandsEnergy, SASAEnergy, sequence = line_split[1], line_split[3], line_split[4], line_split[5], line_split[6], line_split[10]
                self.resultsFinal.append([-1, rank, fullEnergy, constraintsEnergy, ligandsEnergy, SASAEnergy])
                self.sequenceFinal.append(sequence)

        self.resultsFull = np.array(self.resultsFull, dtype=np.float)
        self.resultsFinal = np.array(self.resultsFinal, dtype=np.float)

        #print(self.resultsFull)
        #print(self.resultsFinal)
        #print(self.residues)
        # close the file
        f.close()

if __name__ == '__main__':
    PlotResults()