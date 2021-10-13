import os
import numpy as np
import matplotlib.pyplot as plt
from Analysis import Analysis
from seaborn import heatmap, barplot
from matplotlib.widgets import Cursor
import matplotlib.patches as mpatches

#resultFileName = '/media/masoud/WRKP/EDesign/tests/DesigCatalytic-PEF-WithXS1-M2-AsMIN-Coupled-SCCopu-005/PEF.out'

class PlotResults(object):
    def __init__(self):
        self.exit = False
        self.fileName_a = None
        self.resultsFull_a = list()
        self.sequenceFull_a = list()
        self.resultsFinal_a = list()
        self.sequenceFinal_a = list()
        self.residues_a = list()

        self.fileName_b = None
        self.resultsFull_b = list()
        self.sequenceFull_b = list()
        self.resultsFinal_b = list()
        self.sequenceFinal_b = list()
        self.residues_b = list()

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
        string += ' i:   Read Input i\n'
        string += ' j:   Read Input j\n'
        string += ' pi:  Plot Iterations Results\n'
        string += ' pf:  Plot Final Results\n'

        print(string)

    def getCommand(self):
        command = input()
        if command == 'q' or command == 'Q':
            self.exit = True

        elif command == 'i' or command == 'j':
            self.readInputs(command)

        elif command == 'pi' or 'pf':

            # Check the results are loaded
            if command == 'pi' and (len(self.resultsFull_a) == 0 or len(self.resultsFull_b) == 0):
                print('No iteration results is loaded.')
                return

            if command == 'pf' and (len(self.resultsFinal_a) == 0 or len(self.resultsFinal_b) == 0):
                print('No final results is loaded.')
                return

            # Get the plot X, Y
            xyz_input = input('Enter the plotting metrics Number: X  Y\n    1:  FullEnergy\n    2:  ConstraintsEnergy\n    3:  LigandsEnergy\n    4:  SASAEnergy\n')
            xyz_input = xyz_input.split()

            # Check the inputs
            try:
                if len(xyz_input) == 2:
                    x, y = int(xyz_input[0]), int(xyz_input[1])
                else:
                    raise ValueError
            except:
                print('Bad input for XY metrics')
                return

            # print or save
            if command == 'pi':
                self.printResults(x, y)
            elif command == 'pf':
                self.printResultsFinal(x, y)

        else:
            print('Command {} is not valid'.format(command))

    def printResults(self, x, y):

        names = ['Shared', self.fileName_a, self.fileName_b]
        colors = ['gray', 'red', 'blue']
        combinedList = list()
        colorList = list()
        source = list()
        for i, entry in enumerate(self.resultsFull_a):
            combinedList.append(entry)
            source.append('RED')
            if self.sequenceFull_a[i] in self.sequenceFull_b:
                colorList.append('gray')
            else:
                colorList.append('red')

        for i, entry in enumerate(self.resultsFull_b):
            combinedList.append(entry)
            source.append('BLUE')
            if self.sequenceFull_b[i] in self.sequenceFull_a:
                colorList.append('gray')
            else:
                colorList.append('blue')

        recs = list()
        for i in range(0, 3):
            recs.append(mpatches.Rectangle((0, 0), 1, 1, fc=colors[i]))

        combinedList = np.array(combinedList)
        colorList = np.array(colorList)

        fig, ax = plt.subplots(nrows=1, ncols=1)
        sc = plt.scatter(combinedList[:, x + 1], combinedList[:, y + 1], c=colorList, marker='o')

        xLable = self.labels[x]
        yLable = self.labels[y]

        plt.legend(recs,names,loc=1)

        ax.set_xlabel(xLable)
        ax.set_ylabel(yLable)

        #plt.colorbar(sc, ax=ax)

        annot = ax.annotate("", xy=(0, 0), xytext=(20, 20), textcoords="offset points", bbox=dict(boxstyle="round", fc="w"), arrowprops=dict(arrowstyle="->"))
        annot.set_visible(False)


        def update_annot(ind):
            pos = sc.get_offsets()[ind["ind"][0]]
            annot.xy = pos
            text = '\n'.join(['Source: {} - Iteration {:.0f} - Rank {:.0f}'.format(source[i], combinedList[i][0], combinedList[i][1]) for i in ind["ind"]])
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

    def printResultsFinal(self, x, y):

        names = ['Shared', self.fileName_a, self.fileName_b]
        colors = ['gray', 'red', 'blue']
        combinedList = list()
        colorList = list()
        source = list()
        seq_a = list()
        seq_b = list()
        seq_shared = list()

        for i, entry in enumerate(self.resultsFinal_a):
            combinedList.append(entry)
            source.append('RED')
            if self.sequenceFinal_a[i] in self.sequenceFinal_b:
                colorList.append('gray')
                if self.sequenceFinal_a[i] not in seq_shared:
                    seq_shared.append(self.sequenceFinal_a[i])
            else:
                colorList.append('red')
                seq_a.append(self.sequenceFinal_a[i])

        for i, entry in enumerate(self.resultsFinal_b):
            combinedList.append(entry)
            source.append('BLUE')
            if self.sequenceFinal_b[i] in self.sequenceFinal_a:
                colorList.append('gray')
                if self.sequenceFinal_b[i] not in seq_shared:
                    seq_shared.append(self.sequenceFinal_b[i])
            else:
                colorList.append('blue')
                seq_b.append(self.sequenceFinal_b[i])

        recs = list()
        for i in range(0, 3):
            recs.append(mpatches.Rectangle((0, 0), 1, 1, fc=colors[i]))

        combinedList = np.array(combinedList)
        colorList = np.array(colorList)

        fig, ax = plt.subplots(nrows=1, ncols=1)
        sc = plt.scatter(combinedList[:, x + 1], combinedList[:, y + 1], c=colorList, marker='o')

        xLable = self.labels[x]
        yLable = self.labels[y]

        plt.legend(recs,names,loc=1)

        ax.set_xlabel(xLable)
        ax.set_ylabel(yLable)

        #plt.colorbar(sc, ax=ax)

        annot = ax.annotate("", xy=(0, 0), xytext=(20, 20), textcoords="offset points", bbox=dict(boxstyle="round", fc="w"), arrowprops=dict(arrowstyle="->"))
        annot.set_visible(False)


        def update_annot(ind):
            pos = sc.get_offsets()[ind["ind"][0]]
            annot.xy = pos
            text = '\n'.join(['Source: {} - Rank {:.0f}'.format(source[i], combinedList[i][1]) for i in ind["ind"]])
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

        f = open('Difference.out', 'w')
        f.write('Shared Sequences:\n')
        for seq in seq_shared:
            f.write('{}\n'.format(seq))
            index = self.sequenceFinal_a.index(seq)
            entry = self.resultsFinal_a[index]
            name = self.fileName_a
            f.write('   File: {}   Rank:  {:3.0f}    Full Atom Energy: {:10.1f}    Constraints Energy: {:10.1f}    Ligands Energy: {:10.1f}    SASA Energy: {:10.1f}\n'.format(name, entry[1], entry[2], entry[3], entry[4], entry[5]))

            index = self.sequenceFinal_b.index(seq)
            entry = self.resultsFinal_b[index]
            name = self.fileName_b
            f.write('   File: {}   Rank:  {:3.0f}    Full Atom Energy: {:10.1f}    Constraints Energy: {:10.1f}    Ligands Energy: {:10.1f}    SASA Energy: {:10.1f}\n'.format(name, entry[1], entry[2], entry[3], entry[4], entry[5]))

        f.write('------------------------------------------------------------------------------------------------------------------\n')
        f.write('Unique Sequences of {}: \n'.format(self.fileName_a))
        for seq in seq_a:
            f.write('{}\n'.format(seq))
            index = self.sequenceFinal_a.index(seq)
            entry = self.resultsFinal_a[index]
            name = self.fileName_a
            f.write('   File: {}   Rank:  {:3.0f}    Full Atom Energy: {:10.1f}    Constraints Energy: {:10.1f}    Ligands Energy: {:10.1f}    SASA Energy: {:10.1f}\n'.format(name, entry[1], entry[2], entry[3], entry[4], entry[5]))

        f.write('------------------------------------------------------------------------------------------------------------------\n')
        f.write('Unique Sequences of {}: \n'.format(self.fileName_b))
        for seq in seq_b:
            f.write('{}\n'.format(seq))
            index = self.sequenceFinal_b.index(seq)
            entry = self.resultsFinal_b[index]
            name = self.fileName_b
            f.write('   File: {}   Rank:  {:3.0f}    Full Atom Energy: {:10.1f}    Constraints Energy: {:10.1f}    Ligands Energy: {:10.1f}    SASA Energy: {:10.1f}\n'.format(name, entry[1], entry[2], entry[3], entry[4], entry[5]))

        f.close()
    def readInputs(self, inputFlag):

        fileName = input('Enter input file Name: ')

        # Check the input file
        if not os.path.isfile(fileName):
            print('{} was not found.\n')
            return

        if inputFlag == 'i':
            self.fileName_a = fileName
        else:
            self.fileName_b = fileName

        # Open the input file:
        f = open(fileName, 'r')

        # initiate the variables
        readResidues = False
        residuesLoaded = False
        readFinal = False
        readresults = True
        resultsFull = list()
        resultsFinal = list()
        sequenceFull = list()
        sequenceFinal = list()
        residues = list()

        # Read the input file
        for line in f:
            line_split = line.split()
            # Skip the empty line
            if len(line_split) == 0:
                continue

            # Load the res names
            if 'Design residues:' in line:
                readResidues = True
                continue

            if readResidues and not residuesLoaded:
                try:
                    id, chain = int(line_split[0].split('-')[0]), line_split[0].split('-')[1]
                    residues.append('{}{}'.format(id, chain))
                except:
                    readResidues = False
                    residuesLoaded = True


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
                resultsFull.append([iteration, rank, fullEnergy, constraintsEnergy, ligandsEnergy, SASAEnergy])
                sequenceFull.append(sequence)
            # read final results
            elif readFinal and line_split[0] == '#':
                rank, fullEnergy, constraintsEnergy, ligandsEnergy, SASAEnergy, sequence = line_split[1], line_split[3], line_split[4], line_split[5], line_split[6], line_split[10]
                resultsFinal.append([-1, rank, fullEnergy, constraintsEnergy, ligandsEnergy, SASAEnergy])
                sequenceFinal.append(sequence)

        resultsFull = np.array(resultsFull, dtype=np.float)
        resultsFinal = np.array(resultsFinal, dtype=np.float)

        #print(self.resultsFull)
        #print(self.resultsFinal)
        #print(self.residues)
        # close the file
        f.close()
        if inputFlag == 'i':
            self.resultsFull_a = resultsFull
            self.resultsFinal_a = resultsFinal
            self.sequenceFull_a = sequenceFull
            self.sequenceFinal_a = sequenceFinal
            self.residues_a = residues
        else:
            self.resultsFull_b = resultsFull
            self.resultsFinal_b = resultsFinal
            self.sequenceFull_b = sequenceFull
            self.sequenceFinal_b = sequenceFinal
            self.residues_b = residues


if __name__ == '__main__':
    PlotResults()