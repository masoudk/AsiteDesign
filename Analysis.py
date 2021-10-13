import numpy as np
from scipy import linalg as sclinalg
from numpy import array, zeros, ones, nonzero, dot, diag
import math
import time
from collections import Counter

class Analysis(object):
    AminoAcids = sorted(list('ARNDCEQGHILKMFPSTWYV'))
    """
    Methods with name starting with CAPITL NAME return a vectors/matrix (distributions) encoded as dict
    """
    # reff: https://www.kaggle.com/shakedzy/alone-in-the-woods-using-theil-s-u-for-survival
    # reff: https://en.wikipedia.org/wiki/Uncertainty_coefficient

    @classmethod
    def P(cls, x: list) -> dict:
        x_counter = Counter(x)
        x_total = sum(x_counter.values())
        p_x = dict()
        for x, count in x_counter.items():
            p_x[x] = count / x_total
        return p_x

    @classmethod
    def P_joint(cls, x: list, y: list) -> dict:
        xy_counter = Counter(list(zip(x, y)))
        xy_total = sum(xy_counter.values())
        p_xy = dict()
        for xy in xy_counter.keys():
            p_xy[xy] = xy_counter[xy] / xy_total
        return p_xy

    @classmethod
    def C_joint(cls, x: list, y: list) -> dict:
        P_y = cls.P(y)
        P_xy = cls.P_joint(x, y)
        Cond_xy = dict()
        for xy in P_xy:
            Cond_xy[xy] = P_xy[xy] / P_y[xy[1]]
        return Cond_xy

    @classmethod
    def s(cls, x: list) -> float:
        p_x = cls.P(x)
        s_x = 0.0
        for x in p_x.keys():
            s_x += -1 * p_x[x] * math.log(p_x[x], math.e)
        return s_x

    @classmethod
    def s_cond(cls, x: list, y: list) -> float:
        P_xy = cls.P_joint(x, y)
        C_xy = cls.C_joint(x, y)
        s_xy = 0.0
        for xy in C_xy.keys():
            s_xy += -1 * P_xy[xy] * math.log(C_xy[xy], math.e)
        return s_xy

    @classmethod
    def S(cls, X: list) -> list:
        S = list()
        for x in X:
            S.append(cls.s(x))
        return S

    @classmethod
    def S_cond(cls, X: list) -> list:
        index = len(X)
        S_XY = list()
        for i in range(index):
            S_xY = list()
            for j in range(index):
                S_xY.append(cls.s_cond(X[i], X[j]))
            S_XY.append(S_xY)
        return S_XY

    @classmethod
    def U(cls, X: list) -> list:
        index = len(X)
        S_X = cls.S(X)
        S_XY = cls.S_cond(X)
        U = list()
        for i in range(index):
            U_xY = list()
            for j in range(index):
                if S_X[i] == 0:
                    U_xy = 1.0
                else:
                    U_xy = (S_X[i] - S_XY[i][j]) / S_X[i]
                U_xY.append(U_xy)
            U.append(U_xY)
        return U

    @classmethod
    def SequencesPMatrix(cls, sequences: list):
        sequences = [list(i) for i in sequences ]
        positions = list(zip(*sequences))
        sequencesPMF = list()
        for position in positions:
            PMF_dict = cls.P(position)
            PMF_list = list()
            for aa in Analysis.AminoAcids:
                PMF_list.append(PMF_dict.get(aa, 0.0))
            sequencesPMF.append(PMF_list)
        return sequencesPMF

    @classmethod
    def SequencesSVector(cls, sequences: list):
        sequences = [list(i) for i in sequences ]
        positions = list(zip(*sequences))
        return cls.S(positions)

    @classmethod
    def SequencesUMatrix(cls, sequences: list):
        sequences = [list(i) for i in sequences ]
        positions = list(zip(*sequences))
        return cls.U(positions)


class NMF(object):
    def __init__(self):
        np.set_printoptions(formatter={'float': lambda x: "{0:0.6f}".format(x)})
        self.aminoAcidList = list()
        self.aminoAcidDict = dict()
        self.aminoAcidDim = 0
        self.setAminoAcid()

        self.iteration = 2
        self.latentDim = 5
        self.rate = 0.1
        self.X = None
        self.A = None
        self.lambdas = None

    def setAminoAcid(self):
        self.aminoAcidList = sorted(list('ARNDCEQGHILKMFPSTWYV'))
        self.aminoAcidDim = len(self.aminoAcidList)
        for i, AA in enumerate(self.aminoAcidList):
            self.aminoAcidDict[AA] = i

    def train(self, seq: list):

        # Get the dimenstions
        self.sample_dim, self.position_dim  = len(seq), len(seq[0])

        # transform the seq list to position lists
        positions = self.transformSequenceToPositionList(seq)

        # get the marginals
        self.estimatePairwiseMarginals(positions)

        # initialize the A matrices and lambdas
        self.initialize_snpa()

        for iteration in range(self.iteration):
            print('-------------------------------------------------')
            print('iteration: ', iteration)
            #for i in range(self.A.shape[1]):
            #    print('{}'.format(['{:.1f}'.format(x) for x in self.A[1, i, :]]))
            print('L--------------\n', '{}'.format(['{:.1f}'.format(x) for x in self.lambdas]))

            self.update_A()
            self.update_lambdas()
            #print('A -------------\n', self.A)
            #print('L--------------\n', self.lambdas)

    def gradient_Ak(self, X, Al, lambdas, Ak):
        L = diag(lambdas)
        J = ones(shape=(self.aminoAcidDim, self.aminoAcidDim))
        AlL = dot(Al, L)
        AkLAl = dot(Ak, dot(L.transpose(), Al.transpose()))
        print(AkLAl)
        G = - dot(X.transpose()/AkLAl, AlL) + dot(J, dot(Al, L))
        return G

    def gradient_L(self, X, Al, lambdas, Ak):
        L = diag(lambdas)
        J = ones(shape=(self.aminoAcidDim, self.aminoAcidDim))
        AlJAk = dot(Al.transpose(), dot(J, Ak))
        AlLAk = dot(Al, dot(L, Ak.transpose()))
        G = - dot(Al.transpose(), dot(X.transpose()/AlLAk, Ak)) + AlJAk
        return G

    def KL_divergence(self, X, Al, lambdas, Ak):
        L = diag(lambdas)
        AlLAk = dot(Al, dot(L, Ak.transpose()))
        D_kl = 0.0
        for m in range(self.aminoAcidDim):
            for n in range(self.aminoAcidDim):
                D_kl += (X[m, n] * np.log(X[m, n] / AlLAk[m, n])) - X[m, n] + AlLAk[m, n]
        return D_kl

    def update_A(self):
        # Optimize Ak while keeping all Al constants
        Ak_new = zeros(self.A.shape)
        for k in range(self.A.shape[0]):
            for l in range(self.A.shape[0]):
                if l == k: continue
                G = self.gradient_Ak(X=self.X[l, k, :, :], Al=self.A[l, :, :], lambdas=self.lambdas, Ak=self.A[k, :, :])
                #print('GAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA  \n', G)
                for i in range(G.shape[0]):
                    for j in range(G.shape[1]):
                        # compute the update denominator
                        row_sum = 0
                        for s in range(G.shape[1]):
                            row_sum += self.A[k, i, s] * np.exp(-self.rate * G[i, s])
                        # compute update
                        Ak_new[k, i, j] += (self.A[k, i, j] * np.exp(-self.rate * G[i, j])) / row_sum
                # normalize column
                Ak_new[k, :, :] = Ak_new[k, :, :] / np.sum(Ak_new[k, :, :], 0)
        # replace

        self.A[:, :, :] = Ak_new[:, :, :]

    def update_lambdas(self):
        lambdas_new = zeros(self.lambdas.shape)
        for k in range(self.A.shape[0]):
            for l in range(self.A.shape[0]):
                if l == k: continue
                G = self.gradient_L(X=self.X[l, k, :, :], Al=self.A[l, :, :], lambdas=self.lambdas, Ak=self.A[k, :, :])
                #print('GLLLLLLLLLLLLLLLLLLLLLLLLLLLL  \n', G)
                for j in range(G.shape[1]):
                    # compute the update denominator
                    row_sum = 0
                    for s in range(G.shape[1]):
                        row_sum += self.lambdas[s] * np.exp(-self.rate * G[s, s])
                    # compute update
                    lambdas_new[j] += (self.lambdas[j] * np.exp(-self.rate * G[j, j])) / row_sum
        # replace
        self.lambdas[:] = lambdas_new[:] / np.sum(lambdas_new)

    def initialize_random(self):
        #self.A = ones(shape=(self.position_dim, self.aminoAcidDim, self.latentDim)) * 0.5
        self.A = np.random.rand(self.position_dim, self.aminoAcidDim, self.latentDim)
        #self.lambdas = ones(shape=self.latentDim) * 0.5
        self.lambdas = np.random.rand(self.latentDim)

    def initialize_snpa(self):

        # Divided the position index into two set
        d = int(np.ceil(self.position_dim/2))

        # Compute X_bar
        X_bar = list()
        for i in range(d):
            blockRow = list()
            for j in range(d, self.position_dim):
                blockRow.append(self.X[i, j, :, :])
            X_bar.append(blockRow)

        X_bar = np.block(X_bar)
        w, H = self.snpa(M=X_bar, r=self.latentDim)
        print(w)
        print('----------------------------')
        for i in H:
            print('H {}'.format(['{:.1f}'.format(j) for j in i]))
        #print(H)
        print('----------------------------')
        print(w.shape)
        print('----------------------------')
        print(H.shape)

        # Update the dimension of latent (latent alphabet)
        self.latentDim = w.shape[0]

        # extract W_bar
        W_bar = X_bar[:, w]
        print('W_bar\n', W_bar.shape, W_bar)
        H_bar = H.transpose()
        print('H_bar\n', H_bar.shape, H_bar)

        # Extract A matrix
        self.A = np.zeros((self.position_dim, self.aminoAcidDim, self.latentDim))

        # A s that belong to the first set
        for l in range(d):
            self.A[l, :, :] = W_bar[l * self.aminoAcidDim : (l + 1) * self.aminoAcidDim, :]
            self.A[l, :, :] /= np.sum(self.A[l, :, :], 0)


        # A s that belong to the second set
        for l in range(d, self.position_dim):
            print('H_bar', l, d, (l - d) * self.aminoAcidDim, '->', ((l + 1) - d) * self.aminoAcidDim, self.position_dim)
            self.A[l, :, :] = H_bar[(l - d) * self.aminoAcidDim: ((l + 1) - d) * self.aminoAcidDim, :]
            self.A[l, :, :] /= np.sum(self.A[l, :, :], 0)

        #for i in range(self.position_dim):
        #    print('i\n', self.A[i, :, :])

        # Update W_bar, H_bar
        W_bar = np.vstack([self.A[l, :, :] for l in range(d)])
        H_bar = np.vstack([self.A[l, :, :] for l in range(d, self.position_dim)])

        print('X_bar', X_bar.shape)
        print('W_bar', W_bar.shape)
        print('H_bar', H_bar.shape)

        print(np.sum(X_bar - np.dot(W_bar, H_bar.transpose())))
        print(np.dot(W_bar, H_bar.transpose()))

        WH = sclinalg.khatri_rao(H_bar, W_bar)
        WH_inv = np.linalg.pinv(WH)
        X_vector = np.ndarray.flatten(X_bar, 'F')[:, np.newaxis]
        print(WH_inv.shape)
        print(X_vector.shape)
        lambdas = np.dot(WH_inv, X_vector)
        self.lambdas = np.ndarray.flatten(lambdas)
        print(self.lambdas)
        for i in self.A:
            print(i, ' <<<<<')

    def estimateGradient(self, Ai):
        G = zeros(shape=Ai.shape)
        return G

    def estimatePairwiseMarginals(self, positions):

        pairwiseMarginalsTensor = zeros((self.position_dim, self.position_dim, self.aminoAcidDim, self.aminoAcidDim))

        # for all pairs of RVs
        for i in range(self.position_dim):
            for j in range(self.position_dim):
                # for each position pair (or RVs) compute the marginal matrix with 20 alphabet (aa)
                #Xij = zeros(shape=(self.aminoAcidDim, self.aminoAcidDim))

                # adding some noise to avoid division by zero
                Xij = np.random.rand(self.aminoAcidDim, self.aminoAcidDim) * 1e-1

                # iterate over samples
                for s in range(self.sample_dim):
                    i_aa_count = Counter(positions[i])
                    j_aa_count = Counter(positions[j])

                    # For all alphabet combinations
                    for l in range(self.aminoAcidDim):
                        for k in range(self.aminoAcidDim):
                            if i_aa_count.get(self.aminoAcidList[l], 0) and j_aa_count.get(self.aminoAcidList[k], 0):
                                Xij[l, k] += 1
                            else:
                                Xij[l, k] += 0
                # normalize
                Xij /= Xij.sum()
                # save
                print(Xij)
                pairwiseMarginalsTensor[i, j, :, :] = Xij[:, :]

        #print(pairwiseMarginalsTensor)
        self.X =  pairwiseMarginalsTensor

    def transformSequenceToPositionList(self, seq: list):
        # Convert the list of string to list of list
        sequenceTemp = [list(i) for i in seq ]
        positionList = list(zip(*sequenceTemp))

        return positionList

    def simplexProj(self, y):
        """
        Reference:
            "Successive Nonnegative Projection Algorithm for Robust Nonnegative Blind Source Separation"
            by Gillis. (2014), doi : 10.1137/130946782
            https://github.com/lwchen6309/successive-nonnegative-projection-algorithm/blob/master/snpa.py

        Given y,  computes its projection x* onto the simplex

              Delta = { x | x >= 0 and sum(x) <= 1 },

        that is, x* = argmin_x ||x-y||_2  such that  x in Delta.


        See Appendix A.1 in N. Gillis, Successive Nonnegative Projection
        Algorithm for Robust Nonnegative Blind Source Separation, arXiv, 2013.


        x = SimplexProj(y)

        ****** Input ******
        y    : input vector.

        ****** Output ******
        x    : projection of y onto Delta.
        """

        if len(y.shape) == 1:  # Reshape to (1,-1) if y is a vector.
            y = y.reshape(1, -1)

        x = y.copy()
        x[x < 0] = 0
        K = np.flatnonzero(np.sum(x, 0) > 1)
        x[:, K] = self.blockSimplexProj(y[:, K])
        return x

    def blockSimplexProj(self, y):
        """
        Reference:
            "Successive Nonnegative Projection Algorithm for Robust Nonnegative Blind Source Separation"
            by Gillis. (2014), doi : 10.1137/130946782
            https://github.com/lwchen6309/successive-nonnegative-projection-algorithm/blob/master/snpa.py

        Same as function SimplexProj except that sum(max(Y,0)) > 1. """

        r, m = y.shape
        ys = -np.sort(-y, axis=0)
        mu = np.zeros(m, dtype=float)
        S = np.zeros((r, m), dtype=float)

        for i in range(1, r):
            S[i, :] = np.sum(ys[:i, :] - ys[i, :], 0)
            colInd_ge1 = np.flatnonzero(S[i, :] >= 1)
            colInd_lt1 = np.flatnonzero(S[i, :] < 1)
            if len(colInd_ge1) > 0:
                mu[colInd_ge1] = (1 - S[i - 1, colInd_ge1]) / i - ys[i - 1, colInd_ge1]
            if i == r:
                mu[colInd_lt1] = (1 - S[r, colInd_lt1]) / (r + 1) - ys[r, colInd_lt1]
        x = y + mu
        x[x < 0] = 0
        return x

    def fastGrad_simplexProj(self, M, U, V=None, maxiter=500):
        """
        Reference:
            "Successive Nonnegative Projection Algorithm for Robust Nonnegative Blind Source Separation"
            by Gillis. (2014), doi : 10.1137/130946782
            https://github.com/lwchen6309/successive-nonnegative-projection-algorithm/blob/master/snpa.py

        Fast gradient method to solve least squares on the unit simplex.
        See Nesterov, Introductory Lectures on Convex Optimization: A Basic
        Course, Kluwer Academic Publisher, 2004.
        This code solves:
                    min_{V(:,j) in Delta, forall j}  ||M-UV||_F^2,
        where Delta = { x | sum x_i <= 1, x_i >= 0 for all i }.

        See also Appendix A in N. Gillis, Successive Nonnegative Projection
        Algorithm for Robust Nonnegative Blind Source Separation, arXiv, 2013.
        [V,e] = FGMfcnls(M,U,V,maxiter)
        ****** Input ******
        M      : m-by-n data matrix
        U      : m-by-r basis matrix
        V      : initialization for the fast gradient method
                 (optional, use [] if none)
        maxiter: maximum numbre of iterations (default = 500).
        ****** Output ******
        V      : V(:,j) = argmin_{x in Delta}  ||M-Ux||_F^2 forall j.
        e      : e(i) = error at the ith iteration
        """

        m, n = M.shape
        m, r = U.shape

        # Initialization of V
        if V is None:
            V = np.zeros((r, n), dtype=float)
            for col_M in range(n):
                # Distance between ith column of M and columns of U
                disti = np.sum((U - M[:, col_M].reshape(-1, 1)) ** 2, 0)
                min_col_U = np.argmin(disti)
                V[min_col_U, col_M] = 1

        # Hessian and Lipschitz constant
        UtU = U.T.dot(U)
        L = np.linalg.norm(UtU, ord=2)  # 2-norm
        # Linear term
        UtM = U.T.dot(M)
        nM = np.linalg.norm(M) ** 2  # Frobenius norm
        # Projection
        alpha = [0.05, 0]  # Parameter, can be tuned.
        err = [0, 0]
        V = self.simplexProj(V)  # Project initialization onto the simplex
        Y = V  # second sequence

        delta = 1e-6
        # Stop if ||V^{k}-V^{k+1}||_F <= delta * ||V^{0}-V^{1}||_F
        for i in range(maxiter):
            # Previous iterate
            Vprev = V
            # FGM Coefficients
            alpha[1] = (np.sqrt(alpha[0] ** 4 + 4 * alpha[0] ** 2) - alpha[0] ** 2) / 2
            beta = alpha[0] * (1 - alpha[0]) / (alpha[0] ** 2 + alpha[1])
            # Projected gradient step from Y
            V = self.simplexProj(Y - (UtU.dot(Y) - UtM) / L)
            # `Optimal' linear combination of iterates
            Y = V + beta * (V - Vprev)
            # Error
            err[1] = nM - 2 * np.sum(np.ravel(V * UtM)) + np.sum(np.ravel(UtU * V.dot(V.T)))

            # Restart: fast gradient methods do not guarantee the objective
            # function to decrease, a good heursitic seems to restart whenever it
            # increases although the global convergence rate is lost! This could
            # be commented out.
            if i > 0 and err[1] > err[0]:
                Y = V
            if i is 0:
                eps0 = np.linalg.norm(V - Vprev)
            eps = np.linalg.norm(V - Vprev)
            if eps < delta * eps0:
                break
            # Update
            alpha[0] = alpha[1]
            err[0] = err[1]

        return V, err[1]

    def snpa(self, M, r, normalize=False, maxitn=100):
        """
        Reference:
            "Successive Nonnegative Projection Algorithm for Robust Nonnegative Blind Source Separation"
            by Gillis. (2014), doi : 10.1137/130946782
            https://github.com/lwchen6309/successive-nonnegative-projection-algorithm/blob/master/snpa.py

        Successive Nonnegative Projection Algorithm (variant with f(.) = ||.||^2)

        *** Description ***
        At each step of the algorithm, the column of M maximizing ||.||_2 is
        extracted, and M is updated with the residual of the projection of its
        columns onto the convex hull of the columns extracted so far.

        See N. Gillis, Successive Nonnegative Projection Algorithm for Robust
        Nonnegative Blind Source Separation, arXiv, 2013.


        [J,H] = SNPA(M,r,normalize)

        ****** Input ******
        M = WH + N : a (normalized) noisy separable matrix, that is, W is full rank,
                     H = [I,H']P where I is the identity matrix, H'>= 0 and its
                     columns sum to at most one, P is a permutation matrix, and
                     N is sufficiently small.
        r          : number of columns to be extracted.
        normalize  : normalize=1 will scale the columns of M so that they sum to one,
                     hence matrix H will satisfy the assumption above for any
                     nonnegative separable matrix M.
                     normalize=0 is the default value for which no scaling is
                     performed. For example, in hyperspectral imaging, this
                     assumption is already satisfied and normalization is not
                     necessary.

        ****** Output ******
        J        : index set of the extracted columns.
        H        : optimal weights, that is, H argmin_{X >= 0} ||M-M(:,K)X||_F
        """

        m, n = M.shape

        if normalize:
            # Normalization of the columns of M so that they sum to one
            M /= (np.sum(M, 0) + 1e-15)

        normM = np.sum(M ** 2, 0)
        nM = np.max(normM)
        J = np.array([], dtype=int)
        # Perform r recursion steps (unless the relative approximation error is
        # smaller than 10^-9)
        for i in range(r):
            if np.max(normM) / nM <= 1e-12:
                break

            # Select the column of M with largest l2-norm
            b = np.argmax(normM)
            a = normM[b]
            #print('normM')
            #print(normM)
            # Norms of the columns of the input matrix M
            if i is 0:
                normM1 = normM.copy()

            # Check ties up to 1e-6 precision
            b = np.flatnonzero((a - normM) / a <= 1e-12)
            print('b after tie')
            print(b)
            # In case of a tie, select column with largest norm of the input matrix M
            if len(b) > 1:
                d = np.argmax(normM1[b])
                b = b[d]
            # Update the index set, and extracted column
            J = np.append(J, int(b))
            print('J')
            print(J)

            # Update residual
            if i is 0:
                # Initialization using 10 iterations of coordinate descent
                # H = nnlsHALSupdt(M,M(:,J),[],10);
                # Fast gradient method for min_{y in Delta} ||M(:,i)-M(:,J)y||
                H, _ = self.fastGrad_simplexProj(M, M[:, J], None, maxitn)
            else:
                H[:, J[i]] = 0
                h = np.zeros((1, n), dtype=float)
                h[0, J[i]] = 1
                H = np.vstack([H, h])
                H, _ = self.fastGrad_simplexProj(M, M[:, J], H, maxitn)

            # Update norms
            R = M - M[:, J].dot(H)
            normM = np.sum(R ** 2, 0)

        return J, H






