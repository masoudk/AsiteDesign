from pyrosetta import Pose
from pyrosetta import get_fa_scorefxn
from pyrosetta.rosetta.core.scoring import ScoreType
cimport cython
cimport openmp
import numpy as np
import random
from libc.stdio cimport printf
from libc.math cimport cos, sin, acos, sqrt, M_PI, acos, atan2, fabs, pow, exp
from libc.float cimport DBL_MAX
from cython.parallel cimport prange
from cython.parallel cimport parallel
from pyrosetta.rosetta.utility import vector1_numeric_xyzVector_double_t as vec1

"""Contains following functions
self_energy_fast         -> ligand_ligand_reduced
pair_energy_fast         -> ligand_environment_reduced
                         -> bond
                         -> angle
self_matrices            -> ligand_ligand_matrices
pair_matrices            -> ligand_environment_matrices
pair_collision           -> ligand_environment_collision
                         -> switch
                         -> col_epsi
selectTorsionsByDistance -> select_torsions_by_distance
                         -> angular_distance_squared
"""
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(True)
def reduced_energy(double [:, :] ligand_xyz,
                   double[:, :] environment_xyz,

                   double [:] ligand_qij,
                   double [:] ligand_dljr,
                   double [:] ligand_dljep,
                   double [:, :] ligand_sol_prefactor,
                   double [:] ligand_lklam,
                   int [:, :] ligand_nonbonded,
                   double [:] ligand_nonbondedWeights,
                   double ligand_coupling,
                   double [:] ligand_weights,

                   double[:] ligand_mask,
                   double[:, :] ligand_environment_qij,
                   double[:, :] ligand_environment_dljr,
                   double[:, :] ligand_environment_dljep,
                   double[:, :, :] ligand_environment_sol_prefactor,
                   double [:] environmen_lklam,
                   int [:] environment_sideChainFalgs,
                   int [:] environment_mainChainFalgs,
                   double sideChainCoupling,
                   double mainChainCoupling,
                   double [:] ligand_environment_weights):

    cdef double energy = 0.0
    energy += ligand_ligand_reduced(ligand_xyz,
                                  ligand_qij,
                                  ligand_dljr,
                                  ligand_dljep,
                                  ligand_sol_prefactor,
                                  ligand_lklam,
                                  ligand_nonbonded,
                                  ligand_nonbondedWeights,
                                  ligand_coupling,
                                  ligand_weights)

    # Compute ligand_environment Energy
    energy += ligand_environment_reduced(ligand_xyz,
                                       environment_xyz,
                                       ligand_mask,
                                       ligand_environment_qij,
                                       ligand_environment_dljr,
                                       ligand_environment_dljep,
                                       ligand_environment_sol_prefactor,
                                       ligand_lklam,
                                       environmen_lklam,
                                       environment_sideChainFalgs,
                                       environment_mainChainFalgs,
                                       sideChainCoupling,
                                       mainChainCoupling,
                                       ligand_environment_weights)
    return energy

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(True)
cdef double ligand_ligand_reduced(double [:, :] ligand_xyz,
                          double [:] ligand_qij,
                          double [:] ligand_dljr,
                          double [:] ligand_dljep,
                          double [:, :] ligand_sol_prefactor,
                          double [:] ligand_lklam,
                          int [:, :] ligand_nonbonded,
                          double [:] ligand_nonbondedWeights,
                          double coupling,
                          double [:] weights) nogil:
    """
    Computes the reduced ligand ligand potential energy from precomputed matrices
    """
    cdef int N = ligand_nonbonded.shape[0]
    cdef int i, j
    cdef double dx, dy, dz, dij, dlj, dlj_dij, dlj_dij3, dlj_dij6, dlj_dij12, epsilon, s
    cdef double sol_expfactor_i, sol_expfactor_j
    cdef double rep, atr, col, sol, C

    C = 0.0
    for pair in range(N):
            #i = ligand_nonbonded[pair, 0]
            #j = ligand_nonbonded[pair, 1]
            dx = ligand_xyz[ligand_nonbonded[pair, 0], 0] - ligand_xyz[ligand_nonbonded[pair, 1], 0]
            dy = ligand_xyz[ligand_nonbonded[pair, 0], 1] - ligand_xyz[ligand_nonbonded[pair, 1], 1]
            dz = ligand_xyz[ligand_nonbonded[pair, 0], 2] - ligand_xyz[ligand_nonbonded[pair, 1], 2]
            dij = dx * dx + dy * dy + dz * dz
            dij = sqrt(dij)

            #dlj = ligand_dljr[pair]
            #eij = ligand_dljep[pair]

            # Compute rep
            if dij <= 0.6 * ligand_dljr[pair]:
                dlj_dij = 1/0.6
                dlj_dij3 = dlj_dij * dlj_dij * dlj_dij
                dlj_dij6 = dlj_dij3 * dlj_dij3
                dlj_dij12 = dlj_dij6 * dlj_dij6
                rep = (ligand_dljep[pair] * (dlj_dij12 - (2 * dlj_dij6))) #- (dlj_dij*dij)
            elif 0.6 * ligand_dljr[pair] < dij and dij <= ligand_dljr[pair]:
                dlj_dij = ligand_dljr[pair]/dij
                dlj_dij3 = dlj_dij * dlj_dij * dlj_dij
                dlj_dij6 = dlj_dij3 * dlj_dij3
                dlj_dij12 = dlj_dij6 * dlj_dij6
                rep = ligand_dljep[pair] * (dlj_dij12 - (2 * dlj_dij6))
            elif ligand_dljr[pair] < dij:
                rep =  0

            # Compute atr
            if dij <= ligand_dljr[pair]:
                atr = -ligand_dljep[pair]
            elif ligand_dljr[pair] < dij and dij <= 4.5:
                dlj_dij = ligand_dljr[pair]/dij
                dlj_dij3 = dlj_dij * dlj_dij * dlj_dij
                dlj_dij6 = dlj_dij3 * dlj_dij3
                dlj_dij12 = dlj_dij6 * dlj_dij6
                atr = ligand_dljep[pair] * (dlj_dij12 - (2 * dlj_dij6))
            elif 4.5 < dij and dij <= 6.0:
                dlj_dij = ligand_dljr[pair]/dij
                dlj_dij3 = dlj_dij * dlj_dij * dlj_dij
                dlj_dij6 = dlj_dij3 * dlj_dij3
                dlj_dij12 = dlj_dij6 * dlj_dij6
                atr = ligand_dljep[pair] * (dlj_dij12 - (2 * dlj_dij6)) * switch(dij, 4.5, 6.0)
            elif 6.0 < dij:
                atr = 0


            # compute electrostatic
            #qij = ligand_qij[pair]
            #epsilon = col_epsi(dij)
            epsilon = 6
            if dij < 1.45:
                col = (ligand_qij[pair]/ (1.45 * epsilon))
            elif 1.45 <= dij and dij < 1.85:
                s = switch(dij, 1.45, 1.85)
                col = ((ligand_qij[pair]/ (1.45 * epsilon)) * s) + ((ligand_qij[pair]/epsilon) * ((1 / (dij*dij)) - (1 / (5.5*5.5))) * (1 - s))
            elif 1.85 <= dij and dij < 4.5:
                col = (ligand_qij[pair]/epsilon) * ((1 / (dij*dij)) - (1 / (5.5*5.5)))
            elif 4.5 <= dij and dij < 5.5:
                col = (ligand_qij[pair]/epsilon) * ((1 / (dij*dij)) - (1 / (5.5*5.5))) * switch(dij, 4.5, 5.5)
            else:
                col = 0


            # compute solvation
            sol = 0
            if weights[3] != 0:
                if dij <= ligand_dljr[pair] - 0.3:
                    sol += ligand_sol_prefactor[pair, 0]
                    sol += ligand_sol_prefactor[pair, 1]

                elif ligand_dljr[pair] - 0.3 < dij and dij <= ligand_dljr[pair] + 0.2:
                    s = switch(dij, ligand_dljr[pair] - 0.3, ligand_dljr[pair] + 0.2)
                    sol_expfactor_i = (dij - ligand_dljr[pair]) / ligand_lklam[ligand_nonbonded[pair, 0]]
                    sol_expfactor_j = (dij - ligand_dljr[pair]) / ligand_lklam[ligand_nonbonded[pair, 1]]
                    sol += (ligand_sol_prefactor[pair, 0] * s) + ((ligand_sol_prefactor[pair, 0] * exp(-1 * (sol_expfactor_i * sol_expfactor_i))) * (1-s))
                    sol += (ligand_sol_prefactor[pair, 1] * s) + ((ligand_sol_prefactor[pair, 1] * exp(-1 * (sol_expfactor_j * sol_expfactor_j))) * (1-s))

                elif ligand_dljr[pair] + 0.2 < dij and dij <= 4.5:
                    sol_expfactor_i = (dij - ligand_dljr[pair]) / ligand_lklam[ligand_nonbonded[pair, 0]]
                    sol_expfactor_j = (dij - ligand_dljr[pair]) / ligand_lklam[ligand_nonbonded[pair, 1]]
                    sol += ligand_sol_prefactor[pair, 0] * exp(-1 * (sol_expfactor_i * sol_expfactor_i))
                    sol += ligand_sol_prefactor[pair, 1] * exp(-1 * (sol_expfactor_j * sol_expfactor_j))

                elif 4.5 < dij and dij <= 6.0:
                    s = switch(dij, 4.5, 6.0)
                    sol_expfactor_i = (dij - ligand_dljr[pair]) / ligand_lklam[ligand_nonbonded[pair, 0]]
                    sol_expfactor_j = (dij - ligand_dljr[pair]) / ligand_lklam[ligand_nonbonded[pair, 1]]
                    sol += ligand_sol_prefactor[pair, 0] * exp(-1 * (sol_expfactor_i * sol_expfactor_i)) * s
                    sol += ligand_sol_prefactor[pair, 1] * exp(-1 * (sol_expfactor_j * sol_expfactor_j)) * s

                else:
                    sol = 0
            #print(rep, atr, col, sol)
            #print((weights[0] * rep, weights[1] * atr, weights[2] * col, weights[3] * sol))
            #print(coupling)
            C += coupling * ligand_nonbondedWeights[pair] * (weights[0] * rep + weights[1] * atr + weights[2] * col + weights[3] * sol)

    return C


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(True)
cdef double ligand_environment_reduced(double[:, :] ligand_xyz,
                               double[:, :] environment_xyz,
                               double[:] ligand_mask,
                               double[:, :] ligand_environment_qij,
                               double[:, :] ligand_environment_dljr,
                               double[:, :] ligand_environment_dljep,
                               double[:, :, :] ligand_environment_sol_prefactor,
                               double [:] ligand_lklam,
                               double [:] environmen_lklam,
                               int [:] environment_sideChainFalgs,
                               int [:] environment_mainChainFalgs,
                               double sideChainCoupling,
                               double mainChainCoupling,
                               double [:] weights) nogil:

    cdef int N = ligand_xyz.shape[0]
    cdef int M = environment_xyz.shape[0]
    cdef int i, j
    cdef double dx, dy, dz, dij, dlj, dlj_dij, dlj_dij3, dlj_dij6, dlj_dij12, epsilon, s
    cdef double sol_expfactor_i, sol_expfactor_j, coupling
    cdef double rep, atr, col, sol, C = 0.0


    for i in range(N):

        if ligand_mask[i] == 0:
            continue

        for j in range(M):
            coupling = (environment_sideChainFalgs[j] * sideChainCoupling) + (environment_mainChainFalgs[j] * mainChainCoupling)
            if coupling == 0:
                continue

            #print('couplinfs', coupling, overlap)
            dx = ligand_xyz[i, 0] - environment_xyz[j, 0]
            dy = ligand_xyz[i, 1] - environment_xyz[j, 1]
            dz = ligand_xyz[i, 2] - environment_xyz[j, 2]

            dij = dx * dx + dy * dy + dz * dz
            dij = sqrt(dij)


            #dlj = ligand_environment_dljr[i, j]
            #eij = ligand_environment_dljep[i, j]
            # Compute rep
            if dij <= 0.6 * ligand_environment_dljr[i, j]:
                dlj_dij = 1/0.6
                dlj_dij3 = dlj_dij * dlj_dij * dlj_dij
                dlj_dij6 = dlj_dij3 * dlj_dij3
                dlj_dij12 = dlj_dij6 * dlj_dij6
                rep = (ligand_environment_dljep[i, j] * (dlj_dij12 - (2 * dlj_dij6))) - (dlj_dij*dij)
            elif 0.6 * ligand_environment_dljr[i, j] < dij and dij <= ligand_environment_dljr[i, j]:
                dlj_dij = ligand_environment_dljr[i, j]/dij
                dlj_dij3 = dlj_dij * dlj_dij * dlj_dij
                dlj_dij6 = dlj_dij3 * dlj_dij3
                dlj_dij12 = dlj_dij6 * dlj_dij6
                rep = ligand_environment_dljep[i, j] * (dlj_dij12 - (2 * dlj_dij6))
            elif ligand_environment_dljr[i, j] < dij:
                rep =  0

            # Compute atr
            if dij <= ligand_environment_dljr[i, j]:
                atr = -ligand_environment_dljep[i, j]
            elif ligand_environment_dljr[i, j] < dij and dij <= 4.5:
                dlj_dij = ligand_environment_dljr[i, j]/dij
                dlj_dij3 = dlj_dij * dlj_dij * dlj_dij
                dlj_dij6 = dlj_dij3 * dlj_dij3
                dlj_dij12 = dlj_dij6 * dlj_dij6
                atr = ligand_environment_dljep[i, j] * (dlj_dij12 - (2 * dlj_dij6))
            elif 4.5 < dij and dij <= 6.0:
                dlj_dij = ligand_environment_dljr[i, j]/dij
                dlj_dij3 = dlj_dij * dlj_dij * dlj_dij
                dlj_dij6 = dlj_dij3 * dlj_dij3
                dlj_dij12 = dlj_dij6 * dlj_dij6
                atr = ligand_environment_dljep[i, j] * (dlj_dij12 - (2 * dlj_dij6)) * switch(dij, 4.5, 6.0)
            elif 6.0 < dij:
                atr = 0
            #if atr > 0:
            #    print(i, j, atr, dij, dlj, eij)

            # compute electrostatic
            #epsilon = col_epsi(dij)
            epsilon = 6
            if dij < 1.45:
                col = (ligand_environment_qij[i, j]/ (1.45 * epsilon))
            elif 1.45 <= dij and dij < 1.85:
                s = switch(dij, 1.45, 1.85)
                col = ((ligand_environment_qij[i, j]/ (1.45 * epsilon)) * s) + ((ligand_environment_qij[i, j]/epsilon) * ((1 / (dij*dij)) - (1 / (5.5*5.5))) * (1 - s))
            elif 1.85 <= dij and dij < 4.5:
                col = (ligand_environment_qij[i, j]/epsilon) * ((1 / (dij*dij)) - (1 / (5.5*5.5)))
            elif 4.5 <= dij and dij < 5.5:
                col = (ligand_environment_qij[i, j]/epsilon) * ((1 / (dij*dij)) - (1 / (5.5*5.5))) * switch(dij, 4.5, 5.5)
            else:
                col = 0


            # compute solvation
            sol = 0
            if weights[3] != 0:
                #print('sol')
                if dij <= ligand_environment_dljr[i, j] - 0.3:
                    sol += ligand_environment_sol_prefactor[i, j, 0]
                    sol += ligand_environment_sol_prefactor[i, j, 1]

                elif ligand_environment_dljr[i, j] - 0.3 < dij and dij <= ligand_environment_dljr[i, j] + 0.2:
                    s = switch(dij, ligand_environment_dljr[i, j] - 0.3, ligand_environment_dljr[i, j] + 0.2)
                    sol_expfactor_i = (dij - ligand_environment_dljr[i, j]) / ligand_lklam[i]
                    sol_expfactor_j = (dij - ligand_environment_dljr[i, j]) / environmen_lklam[j]
                    sol += (ligand_environment_sol_prefactor[i, j, 0] * s) + ((ligand_environment_sol_prefactor[i, j, 0] * exp(-1 * (sol_expfactor_i * sol_expfactor_i))) * (1-s))
                    sol += (ligand_environment_sol_prefactor[i, j, 1] * s) + ((ligand_environment_sol_prefactor[i, j, 1] * exp(-1 * (sol_expfactor_j * sol_expfactor_j))) * (1-s))

                elif ligand_environment_dljr[i, j] + 0.2 < dij and dij <= 4.5:
                    sol_expfactor_i = (dij - ligand_environment_dljr[i, j]) / ligand_lklam[i]
                    sol_expfactor_j = (dij - ligand_environment_dljr[i, j]) / environmen_lklam[j]
                    sol += ligand_environment_sol_prefactor[i, j, 0] * exp(-1 * (sol_expfactor_i * sol_expfactor_i))
                    sol += ligand_environment_sol_prefactor[i, j, 1] * exp(-1 * (sol_expfactor_j * sol_expfactor_j))

                elif 4.5 < dij and dij <= 6.0:
                    s = switch(dij, 4.5, 6.0)
                    sol_expfactor_i = (dij - ligand_environment_dljr[i, j]) / ligand_lklam[i]
                    sol_expfactor_j = (dij - ligand_environment_dljr[i, j]) / environmen_lklam[j]
                    sol += ligand_environment_sol_prefactor[i, j, 0] * exp(-1 * (sol_expfactor_i * sol_expfactor_i)) * s
                    sol += ligand_environment_sol_prefactor[i, j, 1] * exp(-1 * (sol_expfactor_j * sol_expfactor_j)) * s

            C += coupling * ( weights[0] * rep +  weights[1] * atr + weights[2] * col + weights[3] * sol)

    return C

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(True)
def bond(double [:, :] ligand_xyz,
         int [:, :] bond_atomIndex,
         double [:] equilibrium_distance,
         double [:] force_constance):
    """
    Computes bond potential
    """
    cdef int i, j, ibond, N = bond_atomIndex.shape[0]
    cdef double dx, dy, dz, dji, C

    C = 0.0
    for ibond in range(N):
            i = bond_atomIndex[ibond, 0]
            j = bond_atomIndex[ibond, 1]

            dx = ligand_xyz[i, 0] - ligand_xyz[j, 0]
            dy = ligand_xyz[i, 1] - ligand_xyz[j, 1]
            dz = ligand_xyz[i, 2] - ligand_xyz[j, 2]

            dji = dx * dx + dy * dy + dz * dz
            dji = sqrt(dji)

            d = dji - equilibrium_distance[ibond]
            C += force_constance[ibond] * d * d

    return C


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(True)
def angle(double [:, :] ligand_xyz,
          int [:, :] angle_atomIndex,
          double [:] equilibrium_angle,
          double [:] force_constance):
    """
    Computes bond potential
    """
    cdef int i, j, iangle, N = angle_atomIndex.shape[0]
    cdef double dx, dy, dz, dji, djk, djidjk, teta, dteta, C

    C = 0.0
    for iangle in range(N):
            i = angle_atomIndex[iangle, 0]
            j = angle_atomIndex[iangle, 1]
            k = angle_atomIndex[iangle, 2]

            dxji = ligand_xyz[i, 0] - ligand_xyz[j, 0]
            dyji = ligand_xyz[i, 1] - ligand_xyz[j, 1]
            dzji = ligand_xyz[i, 2] - ligand_xyz[j, 2]

            dxjk = ligand_xyz[k, 0] - ligand_xyz[j, 0]
            dyjk = ligand_xyz[k, 1] - ligand_xyz[j, 1]
            dzjk = ligand_xyz[k, 2] - ligand_xyz[j, 2]

            dji = dxji * dxji + dyji * dyji + dzji * dzji
            dji = sqrt(dji)

            djk = dxjk * dxjk + dyjk * dyjk + dzjk * dzjk
            djk = sqrt(djk)

            djidjk = dxji * dxjk + dyji * dyjk + dzji * dzjk
            djidjk = djidjk / (dji * djk)

            if djidjk > 1.0: djidjk = 1
            if djidjk < -1.0: djidjk = -1

            teta = acos(djidjk)
            dteta = teta - equilibrium_angle[iangle]
            C += force_constance[iangle] * dteta * dteta

    return C





@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(True)
def ligand_ligand_matrices(double [:] ligand_q,
                           double [:] ligand_qij,
                           double [:] ligand_ljr,
                           double [:] ligand_dljr,
                           double [:] ligand_ljep,
                           double [:] ligand_dljep,
                           double [:] ligand_lkdg,
                           double [:] ligand_lklam,
                           double [:] ligand_lkvol,
                           double [:, :] ligand_sol_prefactor,
                           int [:, :] ligand_nonbonded):

    cdef int N = ligand_nonbonded.shape[0]
    cdef int i, j
    #cdef double dx, dy, dz, dij, dij_v, dij_col, dij_sol, dlj, epij
    #cdef double dlj_dij, dlj_dij_3, dlj_dij_6, dlj_dij_12
    #cdef double sol_prefactor_i, sol_prefactor_j, sol_expfactor_i, sol_expfactor_j
    #cdef double vdw, coupling, col, sol, C = 0.0
    cdef double pi_3_2 = pow(M_PI, 3/2)

    for pair in range(N):

        i = ligand_nonbonded[pair, 0]
        j = ligand_nonbonded[pair, 1]

        ligand_qij[pair] = 332 * ligand_q[i] * ligand_q[j]
        ligand_dljr[pair] = ligand_ljr[i] + ligand_ljr[j]
        ligand_dljep[pair] = sqrt((ligand_ljep[i] * ligand_ljep[j]))


        ligand_sol_prefactor[pair, 0] = (-ligand_lkvol[j] * ligand_lkdg[i]) / (ligand_lklam[i] * ligand_ljr[i] * ligand_ljr[i] * 2 * pi_3_2)
        ligand_sol_prefactor[pair, 1] = (-ligand_lkvol[i] * ligand_lkdg[j]) / (ligand_lklam[j] * ligand_ljr[j] * ligand_ljr[j] * 2 * pi_3_2)


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(True)
def ligand_environment_matrices(double [:] ligand_q,
                                double [:] environment_q,
                                double [:, :] ligand_environment_qij,
                                double [:] ligand_ljr,
                                double [:] environment_ljr,
                                double [:, :]  ligand_environment_dljr,
                                double [:] ligand_ljep,
                                double [:] environment_ljep,
                                double [:, :] ligand_environment_ljep,
                                double [:] ligand_lkdg,
                                double [:] environment_lkdg,
                                double [:] ligand_lklam,
                                double [:] environment_lklam,
                                double [:] ligand_lkvol,
                                double [:] environment_lkvol,
                                double [:, :, :] ligand_environment_sol_prefactor):

    cdef int N = ligand_q.shape[0]
    cdef int M = environment_q.shape[0]
    cdef int i, j
   # cdef double dx, dy, dz, dij, dij_v, dij_col, dij_sol, dlj, epij
   # cdef double dlj_dij, dlj_dij_3, dlj_dij_6, dlj_dij_12
    #cdef double sol_prefactor_i, sol_prefactor_j, sol_expfactor_i, sol_expfactor_j
    #cdef double vdw, coupling, col, sol, C = 0.0
    cdef double pi_3_2 = pow(M_PI, 3 / 2)

    for i in range(N):
        for j in range(M):

            ligand_environment_dljr[i, j] = ligand_ljr[i] + environment_ljr[j]
            ligand_environment_ljep[i, j] = sqrt((ligand_ljep[i] * environment_ljep[j]))
            ligand_environment_qij[i, j] = 332 * ligand_q[i] * environment_q[j]

            ligand_environment_sol_prefactor[i, j, 0] = (-environment_lkvol[j] * ligand_lkdg[i]) / (ligand_lklam[i] * ligand_ljr[i] * ligand_ljr[i] * 2 * pi_3_2)
            ligand_environment_sol_prefactor[i, j, 1] = (-ligand_lkvol[i] * environment_lkdg[j]) / (environment_lklam[j] * environment_ljr[j] * environment_ljr[j] * 2 * pi_3_2)

            #print(i, j, ligand_environment_sol_prefactor[i, j, 0], ligand_lkdg[i])
            #print(i, j, ligand_environment_sol_prefactor[i, j, 1], Blkdg[j])


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(True)
def ligand_environment_collision(double[:, :] ligand_xyz,
                                 double[:, :] environment_xyz,
                                 double[:] ligand_lj,
                                 double[:] environment_lj,
                                 int [:] environment_sideChainFalgs,
                                 int [:] environment_mainChainFalgs,
                                 double sideChainCoupling,
                                 double mainChainCoupling,
                                 double overlap):

    cdef int N = ligand_xyz.shape[0]
    cdef int M = environment_xyz.shape[0]
    cdef int i, j, C = 0
    cdef double dx, dy, dz, dij, dlj
    cdef double coupling
    #cdef double rep, atr, col, sol


    for i in range(N):
        for j in range(M):
            coupling = (environment_sideChainFalgs[j] * sideChainCoupling) + (environment_mainChainFalgs[j] * mainChainCoupling)
            if coupling == 0:
                continue
            #print('couplinfs', coupling, overlap)
            dx = ligand_xyz[i, 0] - environment_xyz[j, 0]
            dy = ligand_xyz[i, 1] - environment_xyz[j, 1]
            dz = ligand_xyz[i, 2] - environment_xyz[j, 2]

            dij = dx * dx + dy * dy + dz * dz
            dij = sqrt(dij)

            dlj = ligand_lj[i] + environment_lj[j]

            # Terminate if collision found
            if dij < dlj * overlap:
                C = 1
                #print('BBBB colli: i, j', i, j, dij, dlj, coupling, environment_sideChainFalgs[j], environment_mainChainFalgs[j] )
                #print(ligand_xyz[i, 0], ligand_xyz[i, 1], ligand_xyz[i, 2])
                #print(environment_xyz[j, 0], environment_xyz[j, 1], environment_xyz[j, 2])
                return C
    return C


cdef inline double switch(double x, double min, double max) nogil:
    cdef double r2_a2, b2_a2, r2a2_b2a2
    if x < min:
        return 1
    elif x > max:
        return 0
    else:
        r2_a2 = (x * x) - (min * min)
        b2_a2 = (max * max) - (min * min)
        r2a2_b2a2 = r2_a2/b2_a2
        return (1 + r2a2_b2a2**2 * ((2 * r2a2_b2a2) -3))


cdef inline double col_epsi(double x):
    cdef double d_4, g
    d_4 = x/0.4
    g = (1 + d_4 + (d_4**2)/2) * exp(-d_4)
    return g*6.0 + (1-g)*80


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(True)
def select_torsions_by_distance(double[:, :] T, int K, double cutoff_percent):
    cdef int N = T.shape[0]
    cdef int M = T.shape[1]
    cdef int i, j, counter, k
    cdef float [:, :] D
    cdef double [:, :] C
    cdef float distance, distanceMax, cutoff_value
    cdef int [:] C_index

    # Compute distance matrix
    D = np.zeros((N, N), dtype=np.float32)
    distanceMax = 0.0
    for i in range(N):
        for j in range(i+1, N):
            distance =  angular_distance_squared(T[i, :-1], T[j, :-1])
            D[i, j] = distance
            D[j, i] = distance
            if distanceMax < distance:
                distanceMax = distance

    # compute the cut off value for the torsion vector
    cutoff_value = distanceMax * cutoff_percent

    # initiate the C
    C = np.zeros((K, M), dtype=np.float64)
    C[0, :] = T[0, :]
    counter = 1
    C_index = np.zeros(K, dtype=np.int32)


    # Assign
    for i in range(N):
        if counter >= K:
            break
        for j in range(counter):
            distance = D[i, C_index[j]]
            #print(distance, cutoff_value)
            if distance >= cutoff_value:
                C[counter, :] = T[i, :]
                C_index[counter] = i
                #print(i, j, distance, cutoff_value, counter, ['{:.1f}'.format(k) for k in np.array(T[i, :])])
                counter += 1
                break
    return np.array(C[0:counter, :])



cdef inline double angular_distance_squared(double [:] A, double [:] B):
    cdef int N = A.shape[0]
    cdef int i
    cdef double dx, D = 0.0

    for i in range(N):
        dx = A[i] - B[i]
        dx = fabs(dx)
        if dx <= 180:
            D += dx * dx
        else:
            D += (360 - dx) * (360 - dx)
    return D

"""                     Contains following functions
                         -> ligand_steep_decent
                         -> ligand_nonbonded_gradient
                         - ligand_enviroment_gradient
                         -> bond
                         -> angle
                         -> switch
"""
"""                     Vector definition conventions
The displacement vector between two particles i, and j is defined as:
                                uij = rj - ri
where the ri and rj are position vectors. Thus, force vectors for pair
potentials are defined as
                                fi = -g * uji
                                fj = -g * uij
where g is the gradient and uij = -uij.

"""

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(True)
def ligand_steep_decent(double [:, :] ligand_xyz,
                        double[:, :] environment_xyz,

                        double [:] ligand_qij,
                        double [:] ligand_dljr,
                        double [:] ligand_dljep,
                        double [:, :] ligand_sol_prefactor,
                        double [:] ligand_lklam,
                        int [:, :] ligand_nonbonded,
                        double [:] ligand_nonbondedWeights,
                        double ligand_coupling,
                        double [:] ligand_weights,

                        int[:, :] bond_atomIndex,
                        double[:] equilibrium_distance,
                        double[:] bond_force_constance,

                        int [:, :] angle_atomIndex,
                        double [:] equilibrium_angle,
                        double [:] angle_force_constance,

                        #double[:, :] ligand_xyz,
                        double[:] ligand_mask,
                        double[:, :] ligand_environment_qij,
                        double[:, :] ligand_environment_dljr,
                        double[:, :] ligand_environment_dljep,
                        double[:, :, :] ligand_environment_sol_prefactor,
                        #double [:] ligand_lklam,
                        double [:] environmen_lklam,
                        int [:] environment_sideChainFalgs,
                        int [:] environment_mainChainFalgs,
                        double sideChainCoupling,
                        double mainChainCoupling,
                        double [:] ligand_environment_weights,
                        double convergence_cutoff,
                        int max_cycle,
                        double max_line_search):

    cdef int N = ligand_xyz.shape[0]
    cdef int M = ligand_xyz.shape[1]
    cdef double [:, :] gradient_vector, xyz_tm
    cdef double E_new, E_old, g_Max, g_norm2, step
    cdef int i, j, k

    gradient_vector = np.zeros((N, 3), dtype=np.float64)
    xyz_tm = np.zeros((N, M), dtype=np.float64)

    #print('BBBB initiate')
    # Compute initial energy E_old
    E_old = 0.0
    E_old += ligand_ligand_reduced(ligand_xyz,
                                   ligand_qij,
                                   ligand_dljr,
                                   ligand_dljep,
                                   ligand_sol_prefactor,
                                   ligand_lklam,
                                   ligand_nonbonded,
                                   ligand_nonbondedWeights,
                                   ligand_coupling,
                                   ligand_weights)

    # Compute ligand_environment Energy
    E_old += ligand_environment_reduced(ligand_xyz,
                                        environment_xyz,
                                        ligand_mask,
                                        ligand_environment_qij,
                                        ligand_environment_dljr,
                                        ligand_environment_dljep,
                                        ligand_environment_sol_prefactor,
                                        ligand_lklam,
                                        environmen_lklam,
                                        environment_sideChainFalgs,
                                        environment_mainChainFalgs,
                                        sideChainCoupling,
                                        mainChainCoupling,
                                        ligand_environment_weights)

    # Compute bond Energy
    E_old += bond(ligand_xyz,
                  bond_atomIndex,
                  equilibrium_distance,
                  bond_force_constance)

    # Compute Angles Energy
    E_old += angle(ligand_xyz,
                   angle_atomIndex,
                   equilibrium_angle,
                   angle_force_constance)
    #print('BBB initial energy', E_old)
    # Minimize
    for i in range(max_cycle):
        #print('BBBB cycle', i)

        # rest stuff
        g_Max = -DBL_MAX
        g_norm2 = 0.0
        E_new = 0.0
        # Reset the gradient_vector
        for j in range(N):
            gradient_vector[j, 0] = 0.0
            gradient_vector[j, 1] = 0.0
            gradient_vector[j, 2] = 0.0

        # Compute ligand ligand non-bonded gradient
        ligand_ligand_gradient(ligand_xyz,
                               ligand_qij,
                               ligand_dljr,
                               ligand_dljep,
                               ligand_sol_prefactor,
                               ligand_lklam,
                               ligand_nonbonded,
                               ligand_nonbondedWeights,
                               ligand_coupling,
                               ligand_weights,
                               gradient_vector)
        #for j in range(N):
        #    print('BBB grad ligand_ligand_gradient: ', gradient_vector[j, 0],  gradient_vector[j, 1], gradient_vector[j, 2])

        # Compute ligand environment non-bonded gradient

        ligand_environment_gradient(ligand_xyz,
                                    environment_xyz,
                                    ligand_mask,
                                    ligand_environment_qij,
                                    ligand_environment_dljr,
                                    ligand_environment_dljep,
                                    ligand_environment_sol_prefactor,
                                    ligand_lklam,
                                    environmen_lklam,
                                    environment_sideChainFalgs,
                                    environment_mainChainFalgs,
                                    sideChainCoupling,
                                    mainChainCoupling,
                                    ligand_environment_weights,
                                    gradient_vector)
        #for j in range(N):
        #    print('BBB grad ligand_environment_gradient: ', gradient_vector[j, 0],  gradient_vector[j, 1], gradient_vector[j, 2])


        # Compute ligand bond gradient
        bond_gradient(ligand_xyz,
                      bond_atomIndex,
                      equilibrium_distance,
                      bond_force_constance,
                      gradient_vector)

        #for j in range(N):
        #    print('BBB grad bond_gradient: ', gradient_vector[j, 0],  gradient_vector[j, 1], gradient_vector[j, 2])


        # Compute ligand angle gradient
        angle_gradient(ligand_xyz,
                      angle_atomIndex,
                      equilibrium_angle,
                      angle_force_constance,
                      gradient_vector)

        #for j in range(N):
        #    print('BBB grad angle_gradient: ', gradient_vector[j, 0],  gradient_vector[j, 1], gradient_vector[j, 2])

        # Go over the grad vector
        for j in range(N):
            # Compute the dot product of grad vector. Treat it as a flat matrix
            g_norm2 += gradient_vector[j, 0] * gradient_vector[j, 0]
            g_norm2 += gradient_vector[j, 1] * gradient_vector[j, 1]
            g_norm2 += gradient_vector[j, 2] * gradient_vector[j, 2]
            #print('BBB grad vector: ', gradient_vector[j, 0],  gradient_vector[j, 1], gradient_vector[j, 2])
            # get g_Max
            if gradient_vector[j, 0] > g_Max:
                g_Max = gradient_vector[j, 0]
            if gradient_vector[j, 1] > g_Max:
                g_Max = gradient_vector[j, 1]
            if gradient_vector[j, 2] > g_Max:
                g_Max = gradient_vector[j, 2]

        # Choose a Step size such that g_Max*Step = 0.5
        if g_Max == 0.0:
            step = 0.0
        else:
            step = max_line_search/g_Max
        #step = 0.0
        #print('BBB g_max', g_Max, 'max_line_search', max_line_search, 'step', step)

        for j in range(10000):
            #print('initiate line search')
            #return 0
            # Get the new coordinate
            for k in range(N):
                xyz_tm[k, 0] = ligand_xyz[k, 0] + (-1 * gradient_vector[k, 0] * step)
                xyz_tm[k, 1] = ligand_xyz[k, 1] + (-1 * gradient_vector[k, 1] * step)
                xyz_tm[k, 2] = ligand_xyz[k, 2] + (-1 * gradient_vector[k, 2] * step)

            # compute E_new
            # Compute ligand_ligand Energy
            E_new = 0
            E_new += ligand_ligand_reduced(xyz_tm,
                                           ligand_qij,
                                           ligand_dljr,
                                           ligand_dljep,
                                           ligand_sol_prefactor,
                                           ligand_lklam,
                                           ligand_nonbonded,
                                           ligand_nonbondedWeights,
                                           ligand_coupling,
                                           ligand_weights)

            # Compute ligand_environment Energy
            E_new += ligand_environment_reduced(xyz_tm,
                                                environment_xyz,
                                                ligand_mask,
                                                ligand_environment_qij,
                                                ligand_environment_dljr,
                                                ligand_environment_dljep,
                                                ligand_environment_sol_prefactor,
                                                ligand_lklam,
                                                environmen_lklam,
                                                environment_sideChainFalgs,
                                                environment_mainChainFalgs,
                                                sideChainCoupling,
                                                mainChainCoupling,
                                                ligand_environment_weights)

            # Compute bond Energy
            E_new += bond(xyz_tm,
                          bond_atomIndex,
                          equilibrium_distance,
                          bond_force_constance)

            # Compute Angles Energy
            E_new += angle(xyz_tm,
                           angle_atomIndex,
                           equilibrium_angle,
                           angle_force_constance)

            #print('     line search step ', j, 'BBB E_new', E_new, 'step', step, 'condition', E_old)
            if E_new > E_old:
                step *= 0.5
                continue
            else:  # accept the move
                break

        #update xyz
        for k in range(N):
            ligand_xyz[k, 0] = xyz_tm[k, 0]
            ligand_xyz[k, 1] = xyz_tm[k, 1]
            ligand_xyz[k, 2] = xyz_tm[k, 2]

        #print('Final energy', E_old)

        # Quite if energy converges
        if fabs(E_new - E_old) < convergence_cutoff:
            return 0

        else:
            E_old = E_new



@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(True)
cdef inline void ligand_ligand_gradient(double [:, :] ligand_xyz,
                                        double [:] ligand_qij,
                                        double [:] ligand_dljr,
                                        double [:] ligand_dljep,
                                        double [:, :] ligand_sol_prefactor,
                                        double [:] ligand_lklam,
                                        int [:, :] ligand_nonbonded,
                                        double [:] ligand_nonbondedWeights,
                                        double ligand_coupling,
                                        double [:] ligand_weights,
                                        double [:, :] gradient_vector):

    cdef int N = ligand_nonbonded.shape[0]
    cdef int i, j
    cdef double dxij, dyij, dzij, dij, dlj, dlj_dij, dlj_dij3, dlj_dij6, dlj_dij12, epsilon, s, s_p
    cdef double sol_expfactor_i, sol_expfactor_j
    cdef double rep, atr, col, sol, grad_magnititude

    for pair in range(N):
            # Assuming fist element in pair is i and the other is j
            i = ligand_nonbonded[pair, 0]
            j = ligand_nonbonded[pair, 1]
            dxij = ligand_xyz[j, 0] - ligand_xyz[i, 0]
            dyij = ligand_xyz[j, 1] - ligand_xyz[i, 1]
            dzij = ligand_xyz[j, 2] - ligand_xyz[i, 2]
            dij = dxij * dxij + dyij * dyij + dzij * dzij
            dij = sqrt(dij)

            #dlj = ligand_dljr[pair]
            #eij = ligand_dljep[pair]

            # Compute rep gradient magnitude
            if dij <= 0.6 * ligand_dljr[pair]:
                # constant, replacing dij = 0.6*dljr
                dlj_dij = 1/0.6
                dlj_dij3 = dlj_dij * dlj_dij * dlj_dij
                dlj_dij6 = dlj_dij3 * dlj_dij3
                dlj_dij12 = dlj_dij6 * dlj_dij6
                rep = (-12 * ligand_dljep[pair]/(0.6 * ligand_dljr[pair])) * (dlj_dij12 - dlj_dij6) #- (dlj_dij*dij)

            elif 0.6 * ligand_dljr[pair] < dij and dij <= ligand_dljr[pair]:
                dlj_dij = ligand_dljr[pair]/dij
                dlj_dij3 = dlj_dij * dlj_dij * dlj_dij
                dlj_dij6 = dlj_dij3 * dlj_dij3
                dlj_dij12 = dlj_dij6 * dlj_dij6
                rep = (-12 * ligand_dljep[pair]/dij) * (dlj_dij12 - dlj_dij6)

            elif ligand_dljr[pair] < dij:
                rep =  0

            # Compute atr
            if dij <= ligand_dljr[pair]:
                #atr = ligand_dljep[pair]
                atr = 0

            elif ligand_dljr[pair] < dij and dij <= 4.5:
                dlj_dij = ligand_dljr[pair]/dij
                dlj_dij3 = dlj_dij * dlj_dij * dlj_dij
                dlj_dij6 = dlj_dij3 * dlj_dij3
                dlj_dij12 = dlj_dij6 * dlj_dij6

                atr =  (-12 * ligand_dljep[pair]/dij) * (dlj_dij12 - dlj_dij6)

            elif 4.5 < dij and dij <= 6.0:
                dlj_dij = ligand_dljr[pair]/dij
                dlj_dij3 = dlj_dij * dlj_dij * dlj_dij
                dlj_dij6 = dlj_dij3 * dlj_dij3
                dlj_dij12 = dlj_dij6 * dlj_dij6

                atr = ( (-12 * ligand_dljep[pair]/dij) * (dlj_dij12 - dlj_dij6) * switch(dij, 4.5, 6.0)) + \
                             ( ligand_dljep[pair] * (dlj_dij12 - (2 * dlj_dij6)) * switch_gradient(dij, 4.5, 6.0))

            elif 6.0 < dij:
                atr = 0


            # compute electrostatic
            epsilon = 6
            if dij < 1.45:
                col = -ligand_qij[pair] / (1.45 * 1.45 * epsilon)

            elif 1.45 <= dij and dij < 1.85:
                s = switch(dij, 1.45, 1.85)
                s_p = switch_gradient(dij, 1.45, 1.85)

                col = ((-ligand_qij[pair]/ (1.45 * 1.45 * epsilon)) * s) + \
                      ((ligand_qij[pair]/ (1.45 * epsilon)) * s_p) +       \
                      ((-2 * ligand_qij[pair]/epsilon) * (1 / (dij*dij*dij)) * (1 - s)) + \
                      ((ligand_qij[pair]/epsilon) * ((1 / (dij*dij)) - (1/(5.5 * 5.5))) * (-s_p))

            elif 1.85 <= dij and dij < 4.5:
                col = ( (-2 * ligand_qij[pair]/epsilon) * (1 / (dij*dij*dij)) )

            elif 4.5 <= dij and dij < 5.5:
                col = ( (-2 * ligand_qij[pair]/epsilon) * (1 / (dij*dij*dij)) * s ) +  \
                      ( (ligand_qij[pair]/epsilon) * ((1/(dij*dij)) - (1/(5.5*5.5))) * switch_gradient(dij, 4.5, 5.5) )

            else:
                col = 0




            # compute solvation
            sol = 0
            if ligand_weights[3] != 0:
                if dij <= ligand_dljr[pair] - 0.3:

                    #sol += -ligand_sol_prefactor[pair, 0]
                    #sol += -ligand_sol_prefactor[pair, 1]
                    sol +=0

                elif ligand_dljr[pair] - 0.3 < dij and dij <= ligand_dljr[pair] + 0.2:

                    s = switch(dij, ligand_dljr[pair] - 0.3, ligand_dljr[pair] + 0.2)
                    s_p = switch_gradient(dij, ligand_dljr[pair] - 0.3, ligand_dljr[pair] + 0.2)

                    sol_expfactor_i = (dij - ligand_dljr[pair]) / ligand_lklam[i]
                    sol_expfactor_j = (dij - ligand_dljr[pair]) / ligand_lklam[j]

                    sol += (ligand_sol_prefactor[pair, 0] * s_p) + \
                           ((ligand_sol_prefactor[pair, 0] * exp(-1 * (sol_expfactor_i * sol_expfactor_i)) * (-2 * (dij - ligand_dljr[pair]) / (ligand_lklam[i] * ligand_lklam[i]))) * (1 - s)) + \
                           (ligand_sol_prefactor[pair, 0] *  exp(-1 * (sol_expfactor_i * sol_expfactor_i)) * (-s_p))

                    sol += (ligand_sol_prefactor[pair, 1] * s_p) + \
                           ((ligand_sol_prefactor[pair, 1] * exp(-1 * (sol_expfactor_j * sol_expfactor_j)) * (-2 * (dij - ligand_dljr[pair]) / (ligand_lklam[j] * ligand_lklam[j])) ) * (1 -s)) + \
                           (ligand_sol_prefactor[pair, 1] *  exp(-1 * (sol_expfactor_j * sol_expfactor_j)) * (-s_p))


                elif ligand_dljr[pair] + 0.2 < dij and dij <= 4.5:
                    sol_expfactor_i = (dij - ligand_dljr[pair]) / ligand_lklam[i]
                    sol_expfactor_j = (dij - ligand_dljr[pair]) / ligand_lklam[j]

                    sol += ( ligand_sol_prefactor[pair, 0] * exp(-1 * (sol_expfactor_i * sol_expfactor_i)) * (-2 * (dij - ligand_dljr[pair]) / (ligand_lklam[i] * ligand_lklam[i])) )
                    sol += ( ligand_sol_prefactor[pair, 1] * exp(-1 * (sol_expfactor_j * sol_expfactor_j)) * (-2 * (dij - ligand_dljr[pair]) / (ligand_lklam[j] * ligand_lklam[j])) )


                elif 4.5 < dij and dij <= 6.0:
                    s = switch(dij, 4.5, 6.0)
                    s_p = switch_gradient(dij, 4.5, 6.0)

                    sol_expfactor_i = (dij - ligand_dljr[pair]) / ligand_lklam[i]
                    sol_expfactor_j = (dij - ligand_dljr[pair]) / ligand_lklam[j]

                    sol += ( ligand_sol_prefactor[pair, 0] * exp(-1 * (sol_expfactor_i * sol_expfactor_i)) * (-2 * (dij - ligand_dljr[pair]) / (ligand_lklam[i] * ligand_lklam[i])) * s) + \
                           ( ligand_sol_prefactor[pair, 0] * exp(-1 * (sol_expfactor_i * sol_expfactor_i)) * s_p )


                    sol += ( ligand_sol_prefactor[pair, 1] * exp(-1 * (sol_expfactor_j * sol_expfactor_j)) * (-2 * (dij - ligand_dljr[pair]) / (ligand_lklam[j] * ligand_lklam[j])) * s) + \
                           (ligand_sol_prefactor[pair, 1] * exp(-1 * (sol_expfactor_j * sol_expfactor_j)) * s_p)

                else:
                    sol = 0


            grad_magnititude = ligand_coupling * ligand_nonbondedWeights[pair] * (ligand_weights[0] * rep + ligand_weights[1] * atr + ligand_weights[2] * col + ligand_weights[3] * sol)

            # Set forces for atom j
            gradient_vector[j, 0] = gradient_vector[j, 0] + ((dxij * grad_magnititude)/dij)
            gradient_vector[j, 1] = gradient_vector[j, 1] + ((dyij * grad_magnititude)/dij)
            gradient_vector[j, 2] = gradient_vector[j, 2] + ((dzij * grad_magnititude)/dij)

            # For atom i we use -dxij
            gradient_vector[i, 0] = gradient_vector[i, 0] + ((-dxij  * grad_magnititude)/dij)
            gradient_vector[i, 1] = gradient_vector[i, 1] + ((-dyij  * grad_magnititude)/dij)
            gradient_vector[i, 2] = gradient_vector[i, 2] + ((-dzij  * grad_magnititude)/dij)

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(True)
def ligand_environment_gradient(double[:, :] ligand_xyz,
                               double[:, :] environment_xyz,
                               double[:] ligand_mask,
                               double[:, :] ligand_environment_qij,
                               double[:, :] ligand_environment_dljr,
                               double[:, :] ligand_environment_dljep,
                               double[:, :, :] ligand_environment_sol_prefactor,
                               double [:] ligand_lklam,
                               double [:] environmen_lklam,
                               int [:] environment_sideChainFalgs,
                               int [:] environment_mainChainFalgs,
                               double sideChainCoupling,
                               double mainChainCoupling,
                               double [:] ligand_environment_weights,
                               double [:, :] gradient_vector):

    cdef int N = ligand_xyz.shape[0]
    cdef int M = environment_xyz.shape[0]
    cdef int i, j
    cdef double dxji, dyji, dzji, dji, dlj, dlj_dji, dlj_dji3, dlj_dji6, dlj_dji12, epsilon, s
    cdef double sol_expfactor_i, sol_expfactor_j, coupling
    cdef double rep, atr, col, sol, grad_magnititude

    for i in range(N):
        if ligand_mask[i] == 0:
            continue

        for j in range(M):
            coupling = (environment_sideChainFalgs[j] * sideChainCoupling) + (environment_mainChainFalgs[j] * mainChainCoupling)
            if coupling == 0:
                continue

            # The vector element are computed wrt ligand
            dxji = ligand_xyz[i, 0] - environment_xyz[j, 0]
            dyji = ligand_xyz[i, 1] - environment_xyz[j, 1]
            dzji = ligand_xyz[i, 2] - environment_xyz[j, 2]
            #if j == 1591 and i == 26:
            #    print('BBB_ coord lig, env', np.array(ligand_xyz[i]), np.array(environment_xyz[j]))
            dji = dxji * dxji + dyji * dyji + dzji * dzji
            dji = sqrt(dji)

            # Compute rep gradient magnitude
            if dji <= 0.6 * ligand_environment_dljr[i, j]:
                # constant, replacing dji = 0.6*dljr
                dlj_dji = 1 / 0.6
                dlj_dji3 = dlj_dji * dlj_dji * dlj_dji
                dlj_dji6 = dlj_dji3 * dlj_dji3
                dlj_dji12 = dlj_dji6 * dlj_dji6

                rep = ((-12 * ligand_environment_dljep[i, j]) / (0.6 * ligand_environment_dljr[i, j])) * (dlj_dji12 - dlj_dji6)  # - (dlj_dji*dji)

            elif 0.6 * ligand_environment_dljr[i, j] < dji and dji <= ligand_environment_dljr[i, j]:
                dlj_dji = ligand_environment_dljr[i, j] / dji
                dlj_dji3 = dlj_dji * dlj_dji * dlj_dji
                dlj_dji6 = dlj_dji3 * dlj_dji3
                dlj_dji12 = dlj_dji6 * dlj_dji6

                rep = (-12 * ligand_environment_dljep[i, j] / dji) * (dlj_dji12 - dlj_dji6)

            elif ligand_environment_dljr[i, j] < dji:
                rep = 0


            # Compute atr
            if dji <= ligand_environment_dljr[i, j]:
                #atr = ligand_environment_dljr[i, j]
                atr = 0.0
            elif ligand_environment_dljr[i, j] < dji and dji <= 4.5:
                dlj_dji = ligand_environment_dljr[i, j] / dji
                dlj_dji3 = dlj_dji * dlj_dji * dlj_dji
                dlj_dji6 = dlj_dji3 * dlj_dji3
                dlj_dji12 = dlj_dji6 * dlj_dji6

                atr = (-12 * ligand_environment_dljep[i, j]/ dji) * (dlj_dji12 - dlj_dji6)

            elif 4.5 < dji and dji <= 6.0:
                dlj_dji = ligand_environment_dljr[i, j] / dji
                dlj_dji3 = dlj_dji * dlj_dji * dlj_dji
                dlj_dji6 = dlj_dji3 * dlj_dji3
                dlj_dji12 = dlj_dji6 * dlj_dji6

                atr = (((-12 * ligand_environment_dljep[i, j] / dji)) * (dlj_dji12 - dlj_dji6) * switch(dji, 4.5, 6.0)) + \
                        (ligand_environment_dljep[i, j] * (dlj_dji12 - (2 * dlj_dji6)) * switch_gradient(dji, 4.5, 6.0))

            elif 6.0 < dji:
                atr = 0


            # compute electrostatic
            epsilon = 6
            if dji < 1.45:
                col = -ligand_environment_qij[i, j] / (1.45 * 1.45 * epsilon)

            elif 1.45 <= dji and dji < 1.85:
                s = switch(dji, 1.45, 1.85)
                s_p = switch_gradient(dji, 1.45, 1.85)

                col = ((-ligand_environment_qij[i, j] / (1.45 * 1.45 * epsilon)) * s) + \
                      ((ligand_environment_qij[i, j] / (1.45 * epsilon)) * s_p) + \
                      ((-2 * ligand_environment_qij[i, j] / epsilon) * (1 / (dji * dji * dji)) * (1 - s)) + \
                      ((ligand_environment_qij[i, j] / epsilon) * ((1 / (dji * dji)) - (1 / (5.5 * 5.5))) * (-s_p))

            elif 1.85 <= dji and dji < 4.5:
                col = ((-2 * ligand_environment_qij[i, j] / epsilon) * (1 / (dji * dji * dji)))

            elif 4.5 <= dji and dji < 5.5:
                col = ((-2 * ligand_environment_qij[i, j] / epsilon) * (1 / (dji * dji * dji)) * s) + \
                      ((ligand_environment_qij[i, j] / epsilon) * ((1 / (dji * dji)) - (1 / (5.5 * 5.5))) * switch_gradient(dji, 4.5, 5.5))

            else:
                col = 0


            # compute solvation
            sol = 0
            if ligand_environment_weights[3] != 0:
                if dji <= ligand_environment_dljr[i, j] - 0.3:

                    #sol += -ligand_environment_sol_prefactor[i, j, 0]
                    #sol += -ligand_environment_sol_prefactor[i, j, 1]
                    sol += 0

                elif ligand_environment_dljr[i, j] - 0.3 < dji and dji <= ligand_environment_dljr[i, j] + 0.2:

                    s = switch(dji, ligand_environment_dljr[i, j] - 0.3, ligand_environment_dljr[i, j] + 0.2)
                    s_p = switch_gradient(dji, ligand_environment_dljr[i, j] - 0.3, ligand_environment_dljr[i, j] + 0.2)

                    sol_expfactor_i = (dji - ligand_environment_dljr[i, j]) / ligand_lklam[i]
                    sol_expfactor_j = (dji - ligand_environment_dljr[i, j]) / environmen_lklam[j]

                    sol += (ligand_environment_sol_prefactor[i, j, 0] * s_p) + \
                           ((ligand_environment_sol_prefactor[i, j, 0] * exp(-1 * (sol_expfactor_i * sol_expfactor_i)) * (-2 * (dji - ligand_environment_dljr[i, j]) / (ligand_lklam[i] * ligand_lklam[i]))) * (1 - s)) + \
                           (ligand_environment_sol_prefactor[i, j, 0] * exp(-1 * (sol_expfactor_i * sol_expfactor_i)) * (-s_p))

                    sol += (ligand_environment_sol_prefactor[i, j, 1] * s_p) + \
                           ((ligand_environment_sol_prefactor[i, j, 1] * exp(-1 * (sol_expfactor_j * sol_expfactor_j)) * (-2 * (dji - ligand_environment_dljr[i, j]) / (environmen_lklam[j] * environmen_lklam[j]))) * (1 - s)) + \
                           (ligand_environment_sol_prefactor[i, j, 1] * exp(-1 * (sol_expfactor_j * sol_expfactor_j)) * (-s_p))


                elif ligand_environment_dljr[i, j] + 0.2 < dji and dji <= 4.5:
                    sol_expfactor_i = (dji - ligand_environment_dljr[i, j]) / ligand_lklam[i]
                    sol_expfactor_j = (dji - ligand_environment_dljr[i, j]) / environmen_lklam[j]

                    sol += (ligand_environment_sol_prefactor[i, j, 0] * exp(-1 * (sol_expfactor_i * sol_expfactor_i)) * (-2 * (dji - ligand_environment_dljr[i, j]) / (ligand_lklam[i] * ligand_lklam[i])))
                    sol += (ligand_environment_sol_prefactor[i, j, 1] * exp(-1 * (sol_expfactor_j * sol_expfactor_j)) * (-2 * (dji - ligand_environment_dljr[i, j]) / (environmen_lklam[j] * environmen_lklam[j])))

                elif 4.5 < dji and dji <= 6.0:
                    s = switch(dji, 4.5, 6.0)
                    s_p = switch_gradient(dji, 4.5, 6.0)

                    sol_expfactor_i = (dji - ligand_environment_dljr[i, j]) / ligand_lklam[i]
                    sol_expfactor_j = (dji - ligand_environment_dljr[i, j]) / environmen_lklam[j]

                    sol += (ligand_environment_sol_prefactor[i, j, 0] * exp(-1 * (sol_expfactor_i * sol_expfactor_i)) * (-2 * (dji - ligand_environment_dljr[i, j]) / (ligand_lklam[i] * ligand_lklam[i])) * s) + \
                           (ligand_environment_sol_prefactor[i, j, 0] * exp(-1 * (sol_expfactor_i * sol_expfactor_i)) * s_p)

                    sol += (ligand_environment_sol_prefactor[i, j, 1] * exp(-1 * (sol_expfactor_j * sol_expfactor_j)) * (-2 * (dji - ligand_environment_dljr[i, j]) / environmen_lklam[j] * environmen_lklam[j]) * s) + \
                           (ligand_environment_sol_prefactor[i, j, 1] * exp(-1 * (sol_expfactor_j * sol_expfactor_j)) * s_p)

                else:
                    sol = 0


            #if rep > 1000 or atr > 1000 or col > 1000 or sol > 1000 or dji ==0:
            #    print('BBB ligand_environment_gradient componenet', rep, atr, col, sol, dxji, dyji, dzji, dji)
            #    print('BBB found 0 distance', i, j, dji)
            #    return

            grad_magnititude = coupling * (ligand_environment_weights[0] * rep + ligand_environment_weights[1] * atr + ligand_environment_weights[2] * col + ligand_environment_weights[3] * sol)


            # Set forces for atom i (ligand)
            gradient_vector[i, 0] = gradient_vector[i, 0] + ((dxji * grad_magnititude) / dji)
            gradient_vector[i, 1] = gradient_vector[i, 1] + ((dyji * grad_magnititude) / dji)
            gradient_vector[i, 2] = gradient_vector[i, 2] + ((dzji * grad_magnititude) / dji)


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(True)
def bond_gradient(double [:, :] ligand_xyz,
                  int [:, :] bond_atomIndex,
                  double [:] equilibrium_distance,
                  double [:] force_constance,
                  double [:, :] gradient_vector):
    """
    Computes the gradient of bond potential
    """
    cdef int i, j, ibond, N = bond_atomIndex.shape[0]
    cdef double dxji, dyji, dzji, dji, g

    for ibond in range(N):
            i = bond_atomIndex[ibond, 0]
            j = bond_atomIndex[ibond, 1]

            dxji = ligand_xyz[i, 0] - ligand_xyz[j, 0]
            dyji = ligand_xyz[i, 1] - ligand_xyz[j, 1]
            dzji = ligand_xyz[i, 2] - ligand_xyz[j, 2]

            dji = dxji * dxji + dyji * dyji + dzji * dzji
            dji = sqrt(dji)
            #if i == 16 and j == 39:
            #    print('BBB O5 H16 dji', dji)
            d = dji - equilibrium_distance[ibond]
            g = 2 * force_constance[ibond] * d

            gradient_vector[i, 0] = gradient_vector[i, 0] + ((dxji * g) / dji)
            gradient_vector[i, 1] = gradient_vector[i, 1] + ((dxji * g) / dji)
            gradient_vector[i, 2] = gradient_vector[i, 2] + ((dxji * g) / dji)

            # For atom j we use -dxji
            gradient_vector[j, 0] = gradient_vector[j, 0] + ((-dxji * g) / dji)
            gradient_vector[j, 1] = gradient_vector[j, 1] + ((-dxji * g) / dji)
            gradient_vector[j, 2] = gradient_vector[j, 2] + ((-dxji * g) / dji)


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(True)
def angle_gradient(double [:, :] ligand_xyz,
                  int [:, :] angle_atomIndex,
                  double [:] equilibrium_angle,
                  double [:] force_constance,
                  double [:, :] gradient_vector):
    """
    Computes bond potential
    """
    cdef int i, j, iangle, N = angle_atomIndex.shape[0]
    cdef double dx, dy, dz, dji, djk, djidjk, djidjk_n, teta, dteta, sn, g
    cdef double dji2inv, djk2inv, gxi, gyi, gzi, gxk, gyk, gzk

    for iangle in range(N):
            i = angle_atomIndex[iangle, 0]
            j = angle_atomIndex[iangle, 1]
            k = angle_atomIndex[iangle, 2]

            dxji = ligand_xyz[i, 0] - ligand_xyz[j, 0]
            dyji = ligand_xyz[i, 1] - ligand_xyz[j, 1]
            dzji = ligand_xyz[i, 2] - ligand_xyz[j, 2]

            dxjk = ligand_xyz[k, 0] - ligand_xyz[j, 0]
            dyjk = ligand_xyz[k, 1] - ligand_xyz[j, 1]
            dzjk = ligand_xyz[k, 2] - ligand_xyz[j, 2]

            dji = dxji * dxji + dyji * dyji + dzji * dzji
            dji = sqrt(dji)
            dji2inv = 1/(dji*dji)

            djk = dxjk * dxjk + dyjk * dyjk + dzjk * dzjk
            djk = sqrt(djk)
            djk2inv = 1/(djk*djk)

            djidjk = dxji * dxjk + dyji * dyjk + dzji * dzjk
            djidjk_n = djidjk / (dji * djk)

            if djidjk_n > 1.0: djidjk_n = 1
            if djidjk_n < -1.0: djdjidjk_nidjk = -1

            teta = acos(djidjk_n)
            dteta = teta - equilibrium_angle[iangle]

            sn = sin(teta)
            # To avoid zero division
            if sn == 0: sn = 0.000000001

            g = (-2 * force_constance[iangle] * dteta) / (sn * dji * djk)

            gxi = dxjk - (djidjk * dji2inv * dxji)
            gyi = dyjk - (djidjk * dji2inv * dyji)
            gzi = dzjk - (djidjk * dji2inv * dzji)

            gxk = dxji - (djidjk * dji2inv * dxjk)
            gyk = dyji - (djidjk * dji2inv * dyjk)
            gzk = dzji - (djidjk * dji2inv * dzjk)

            gradient_vector[i, 0] = gradient_vector[i, 0] + (g * gxi)
            gradient_vector[i, 1] = gradient_vector[i, 1] + (g * gyi)
            gradient_vector[i, 2] = gradient_vector[i, 2] + (g * gzi)

            gradient_vector[k, 0] = gradient_vector[k, 0] + (g * gxk)
            gradient_vector[k, 1] = gradient_vector[k, 1] + (g * gyk)
            gradient_vector[k, 2] = gradient_vector[k, 2] + (g * gzk)

            gradient_vector[j, 0] = gradient_vector[j, 0] - (g * (gxi + gxk))
            gradient_vector[j, 1] = gradient_vector[j, 1] - (g * (gxi + gxk))
            gradient_vector[j, 2] = gradient_vector[j, 2] - (g * (gxi + gxk))


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(True)
cdef inline double switch_gradient(double x, double min, double max):
    cdef double r2, a2, b2_a2
    if x < min:
        return 1
    elif x > max:
        return 0
    else:
        r2 = x * x
        a2 =  min * min
        b2 = max * max
        b2_a2 = b2 - a2
        return (12*x*(a2 - r2)*(b2 - r2))/(b2_a2*b2_a2*b2_a2)


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(True)
def grid_search(double [:, :] grid,
                double [:, :] ligand_xyz,
                double[:, :] environment_xyz,

                double [:] ligand_qij,
                double [:] ligand_dljr,
                double [:] ligand_dljep,
                double [:, :] ligand_sol_prefactor,
                double [:] ligand_lklam,
                int [:, :] ligand_nonbonded,
                double [:] ligand_nonbondedWeights,
                double ligand_coupling,
                double [:] ligand_weights,

                double[:] ligand_mask,
                int [:, :] sideChainsChiXYZIndices,
                int [:, :] sideChainSubTreeMasks,

                double[:, :] ligand_environment_qij,
                double[:, :] ligand_environment_dljr,
                double[:, :] ligand_environment_dljep,
                double[:, :, :] ligand_environment_sol_prefactor,
                double [:] environmen_lklam,
                int [:] environment_sideChainFalgs,
                int [:] environment_mainChainFalgs,
                double sideChainCoupling,
                double mainChainCoupling,
                double [:] ligand_environment_weights):

    cdef int grid_N = grid.shape[0]
    cdef int grid_M = grid.shape[1]
    cdef int i, j, k
    #print('BB ----- ', ligand_xyz.shape)
    cdef double [:, :] xyz_tm = np.zeros((ligand_xyz.shape[0], ligand_xyz.shape[1]))
    #cdef double [:, :] rotation_vectors = np.zeros((sideChainsChiXYZIndices.shape[0], 3))
    cdef double [3] u_ji, u_jk, u_lk, n_ijk, n_jkl, nn_cross, p, rotation_vector
    cdef double [4] q
    cdef double n_ijk_nrom, n_jkl_norm, nn_dot, sign, rotation_vector_norm
    cdef double [:] torsion_current = np.zeros(sideChainsChiXYZIndices.shape[0])
    cdef double rot_angle

    # Get the current torsion vector
    for i in range(sideChainsChiXYZIndices.shape[0]):
        #print('BBB Cord i:', ligand_xyz[sideChainsChiXYZIndices[i, 0], 0], ligand_xyz[sideChainsChiXYZIndices[i, 0], 1], ligand_xyz[sideChainsChiXYZIndices[i, 0], 2])
        #print('BBB Cord j:', ligand_xyz[sideChainsChiXYZIndices[i, 1], 0], ligand_xyz[sideChainsChiXYZIndices[i, 1], 1], ligand_xyz[sideChainsChiXYZIndices[i, 1], 2])
        #print('BBB Cord k:', ligand_xyz[sideChainsChiXYZIndices[i, 2], 0], ligand_xyz[sideChainsChiXYZIndices[i, 2], 1], ligand_xyz[sideChainsChiXYZIndices[i, 2], 2])
        #print('BBB Cord l:', ligand_xyz[sideChainsChiXYZIndices[i, 3], 0], ligand_xyz[sideChainsChiXYZIndices[i, 3], 1], ligand_xyz[sideChainsChiXYZIndices[i, 3], 2])

        u_ji[0] = ligand_xyz[sideChainsChiXYZIndices[i, 0], 0] - ligand_xyz[sideChainsChiXYZIndices[i, 1], 0]
        u_ji[1] = ligand_xyz[sideChainsChiXYZIndices[i, 0], 1] - ligand_xyz[sideChainsChiXYZIndices[i, 1], 1]
        u_ji[2] = ligand_xyz[sideChainsChiXYZIndices[i, 0], 2] - ligand_xyz[sideChainsChiXYZIndices[i, 1], 2]

        u_jk[0] = ligand_xyz[sideChainsChiXYZIndices[i, 2], 0] - ligand_xyz[sideChainsChiXYZIndices[i, 1], 0]
        u_jk[1] = ligand_xyz[sideChainsChiXYZIndices[i, 2], 1] - ligand_xyz[sideChainsChiXYZIndices[i, 1], 1]
        u_jk[2] = ligand_xyz[sideChainsChiXYZIndices[i, 2], 2] - ligand_xyz[sideChainsChiXYZIndices[i, 1], 2]

        u_lk[0] = ligand_xyz[sideChainsChiXYZIndices[i, 2], 0] - ligand_xyz[sideChainsChiXYZIndices[i, 3], 0]
        u_lk[1] = ligand_xyz[sideChainsChiXYZIndices[i, 2], 1] - ligand_xyz[sideChainsChiXYZIndices[i, 3], 1]
        u_lk[2] = ligand_xyz[sideChainsChiXYZIndices[i, 2], 2] - ligand_xyz[sideChainsChiXYZIndices[i, 3], 2]

        #print('BBB', u_ji)
        #print('BBB', u_jk)
        #print('BBB', u_lk)

        n_ijk[0] = u_ji[1] * u_jk[2] - u_ji[2] * u_jk[1]
        n_ijk[1] = u_ji[2] * u_jk[0] - u_ji[0] * u_jk[2]
        n_ijk[2] = u_ji[0] * u_jk[1] - u_ji[1] * u_jk[0]

        n_jkl[0] = u_jk[1] * u_lk[2] - u_jk[2] * u_lk[1]
        n_jkl[1] = u_jk[2] * u_lk[0] - u_jk[0] * u_lk[2]
        n_jkl[2] = u_jk[0] * u_lk[1] - u_jk[1] * u_lk[0]

        n_ijk_nrom = sqrt(n_ijk[0] * n_ijk[0] + n_ijk[1] * n_ijk[1] + n_ijk[2] * n_ijk[2])
        n_jkl_norm = sqrt(n_jkl[0] * n_jkl[0] + n_jkl[1] * n_jkl[1] + n_jkl[2] * n_jkl[2])

        n_ijk[0] = n_ijk[0] / n_ijk_nrom
        n_ijk[1] = n_ijk[1] / n_ijk_nrom
        n_ijk[2] = n_ijk[2] / n_ijk_nrom

        n_jkl[0] = n_jkl[0] / n_jkl_norm
        n_jkl[1] = n_jkl[1] / n_jkl_norm
        n_jkl[2] = n_jkl[2] / n_jkl_norm

        #print('BBB', n_ijk)
        #print('BBB', n_jkl)

        nn_dot = n_ijk[0] * n_jkl[0] + n_ijk[1] * n_jkl[1] + n_ijk[2] * n_jkl[2]

        nn_cross[0] = n_ijk[1] * n_jkl[2] - n_ijk[2] * n_jkl[1]
        nn_cross[1] = n_ijk[2] * n_jkl[0] - n_ijk[0] * n_jkl[2]
        nn_cross[2] = n_ijk[0] * n_jkl[1] - n_ijk[1] * n_jkl[0]

        # Correct for precision error
        if nn_dot > 1: nn_dot = 1.0
        if nn_dot < -1: nn_dot = -1.0

        # get the angle sign
        sign = u_jk[0] * nn_cross[0] + u_jk[1] * nn_cross[1] + u_jk[2] * nn_cross[2]

        # Save current torsion angles
        if sign < 0:
            torsion_current[i] = acos(nn_dot) * (180.0 / M_PI) * -1
        else:
            torsion_current[i] = acos(nn_dot) * (180.0 / M_PI)


        #print('BBBB ----!!!', np.array(sideChainsChiXYZIndices[i, :]), np.array(torsion_current[i]))

    for i in range(grid_N):

        # Fil up the tem matrix
        for k in range(ligand_xyz.shape[0]):
            xyz_tm[k, 0] = ligand_xyz[k, 0]
            xyz_tm[k, 1] = ligand_xyz[k, 1]
            xyz_tm[k, 2] = ligand_xyz[k, 2]

        # iterate over torsion vector
        for j in range(grid_M - 1):

            # Compute rotation angle for chi j
            rot_angle = grid[i, j] - torsion_current[j]
            if rot_angle < -180.0:
                rot_angle += 360.0
            elif rot_angle >= 180.0:
                rot_angle -= 360.0
            # Convert to radian
            rot_angle *= (M_PI / 180)

            # Compute rotation vectors
            rotation_vector[0] = xyz_tm[sideChainsChiXYZIndices[j, 2], 0] - xyz_tm[sideChainsChiXYZIndices[j, 1], 0]
            rotation_vector[1] = xyz_tm[sideChainsChiXYZIndices[j, 2], 1] - xyz_tm[sideChainsChiXYZIndices[j, 1], 1]
            rotation_vector[2] = xyz_tm[sideChainsChiXYZIndices[j, 2], 2] - xyz_tm[sideChainsChiXYZIndices[j, 1], 2]

            rotation_vector_norm = sqrt(rotation_vector[0] * rotation_vector[0] + rotation_vector[1] * rotation_vector[1] + rotation_vector[2] * rotation_vector[2])

            rotation_vector[0] = rotation_vector[0] / rotation_vector_norm
            rotation_vector[1] = rotation_vector[1] / rotation_vector_norm
            rotation_vector[2] = rotation_vector[2] / rotation_vector_norm

            # compute the rotation quaternion
            rot_angle *= 0.5
            q[0] = cos(rot_angle)
            q[1] = sin(rot_angle) * rotation_vector[0]
            q[2] = sin(rot_angle) * rotation_vector[1]
            q[3] = sin(rot_angle) * rotation_vector[2]


            # Rotate the subtree
            #print(j, np.array(sideChainSubTreeMasks[j, :]))
            for k in range(xyz_tm.shape[0]):
                # Operate only on the subtree atoms for chi j
                if sideChainSubTreeMasks[j, k] == 0:
                    continue
                else:
                    #Translate point to new coordinate system (rotation vector)
                    p[0] = xyz_tm[k, 0] - xyz_tm[sideChainsChiXYZIndices[j, 1], 0]
                    p[1] = xyz_tm[k, 1] - xyz_tm[sideChainsChiXYZIndices[j, 1], 1]
                    p[2] = xyz_tm[k, 2] - xyz_tm[sideChainsChiXYZIndices[j, 1], 2]

                    # rotate
                    xyz_tm[k, 0] = p[0] * (q[0] * q[0] + q[1] * q[1] - q[2] * q[2] - q[3] * q[3]) + 2 * p[1] * (q[1] * q[2] - q[0] * q[3]) + 2 * p[2] * (q[0] * q[2] + q[1] * q[3])
                    xyz_tm[k, 1] = 2 * p[0] * (q[0] * q[3] + q[1] * q[2]) +  p[1] * (q[0] * q[0] - q[1] * q[1] + q[2] * q[2] - q[3] * q[3]) + 2 * p[2] * (q[2] * q[3] - q[0] * q[1])
                    xyz_tm[k, 2] = 2 * p[0] * (q[1] * q[3] - q[0] * q[2]) + 2 * p[1] * (q[0] * q[1] + q[2] * q[3]) + p[2] * (q[0] * q[0] - q[1] * q[1] - q[2] * q[2] + q[3] * q[3])

                    # translate back to the original coordinate system
                    xyz_tm[k, 0] = xyz_tm[k, 0] + xyz_tm[sideChainsChiXYZIndices[j, 1], 0]
                    xyz_tm[k, 1] = xyz_tm[k, 1] + xyz_tm[sideChainsChiXYZIndices[j, 1], 1]
                    xyz_tm[k, 2] = xyz_tm[k, 2] + xyz_tm[sideChainsChiXYZIndices[j, 1], 2]

            """
            #-----------------------------------------------------------------------------------------------------------
            #-----------------------------------------------------------------------------------------------------------
            #-----------------------------------------------------------------------------------------------------------
            print('ABBB Cord i:', xyz_tm[sideChainsChiXYZIndices[j, 0], 0], xyz_tm[sideChainsChiXYZIndices[j, 0], 1], xyz_tm[sideChainsChiXYZIndices[j, 0], 2])
            print('ABBB Cord j:', xyz_tm[sideChainsChiXYZIndices[j, 1], 0], xyz_tm[sideChainsChiXYZIndices[j, 1], 1], xyz_tm[sideChainsChiXYZIndices[j, 1], 2])
            print('ABBB Cord k:', xyz_tm[sideChainsChiXYZIndices[j, 2], 0], xyz_tm[sideChainsChiXYZIndices[j, 2], 1], xyz_tm[sideChainsChiXYZIndices[j, 2], 2])
            print('ABBB Cord l:', xyz_tm[sideChainsChiXYZIndices[j, 3], 0], xyz_tm[sideChainsChiXYZIndices[j, 3], 1], xyz_tm[sideChainsChiXYZIndices[j, 3], 2])

            u_ji[0] = xyz_tm[sideChainsChiXYZIndices[j, 0], 0] - xyz_tm[sideChainsChiXYZIndices[j, 1], 0]
            u_ji[1] = xyz_tm[sideChainsChiXYZIndices[j, 0], 1] - xyz_tm[sideChainsChiXYZIndices[j, 1], 1]
            u_ji[2] = xyz_tm[sideChainsChiXYZIndices[j, 0], 2] - xyz_tm[sideChainsChiXYZIndices[j, 1], 2]

            u_jk[0] = xyz_tm[sideChainsChiXYZIndices[j, 2], 0] - xyz_tm[sideChainsChiXYZIndices[j, 1], 0]
            u_jk[1] = xyz_tm[sideChainsChiXYZIndices[j, 2], 1] - xyz_tm[sideChainsChiXYZIndices[j, 1], 1]
            u_jk[2] = xyz_tm[sideChainsChiXYZIndices[j, 2], 2] - xyz_tm[sideChainsChiXYZIndices[j, 1], 2]

            u_lk[0] = xyz_tm[sideChainsChiXYZIndices[j, 2], 0] - xyz_tm[sideChainsChiXYZIndices[j, 3], 0]
            u_lk[1] = xyz_tm[sideChainsChiXYZIndices[j, 2], 1] - xyz_tm[sideChainsChiXYZIndices[j, 3], 1]
            u_lk[2] = xyz_tm[sideChainsChiXYZIndices[j, 2], 2] - xyz_tm[sideChainsChiXYZIndices[j, 3], 2]

            n_ijk[0] = u_ji[1] * u_jk[2] - u_ji[2] * u_jk[1]
            n_ijk[1] = u_ji[2] * u_jk[0] - u_ji[0] * u_jk[2]
            n_ijk[2] = u_ji[0] * u_jk[1] - u_ji[1] * u_jk[0]

            n_jkl[0] = u_jk[1] * u_lk[2] - u_jk[2] * u_lk[1]
            n_jkl[1] = u_jk[2] * u_lk[0] - u_jk[0] * u_lk[2]
            n_jkl[2] = u_jk[0] * u_lk[1] - u_jk[1] * u_lk[0]

            n_ijk_nrom = sqrt(n_ijk[0] * n_ijk[0] + n_ijk[1] * n_ijk[1] + n_ijk[2] * n_ijk[2])
            n_jkl_norm = sqrt(n_jkl[0] * n_jkl[0] + n_jkl[1] * n_jkl[1] + n_jkl[2] * n_jkl[2])

            n_ijk[0] = n_ijk[0] / n_ijk_nrom
            n_ijk[1] = n_ijk[1] / n_ijk_nrom
            n_ijk[2] = n_ijk[2] / n_ijk_nrom

            n_jkl[0] = n_jkl[0] / n_jkl_norm
            n_jkl[1] = n_jkl[1] / n_jkl_norm
            n_jkl[2] = n_jkl[2] / n_jkl_norm

            nn_dot = n_ijk[0] * n_jkl[0] + n_ijk[1] * n_jkl[1] + n_ijk[2] * n_jkl[2]

            nn_cross[0] = n_ijk[1] * n_jkl[2] - n_ijk[2] * n_jkl[1]
            nn_cross[1] = n_ijk[2] * n_jkl[0] - n_ijk[0] * n_jkl[2]
            nn_cross[2] = n_ijk[0] * n_jkl[1] - n_ijk[1] * n_jkl[0]

            # Correct for precision error
            if nn_dot > 1: nn_dot = 1.0
            if nn_dot < -1: nn_dot = -1.0

            # get the angle sign
            sign = u_jk[0] * nn_cross[0] + u_jk[1] * nn_cross[1] + u_jk[2] * nn_cross[2]

            # Save current torsion angles
            if sign < 0:
                print(i, j, 'current', acos(nn_dot) * (180.0 / M_PI) * -1, 'target', grid[i, j])
            else:
                print(i, j, 'current', acos(nn_dot) * (180.0 / M_PI), 'target', grid[i, j])
            # -----------------------------------------------------------------------------------------------------------
            # -----------------------------------------------------------------------------------------------------------
            # -----------------------------------------------------------------------------------------------------------
            """
        #print(np.array(grid[i, :]), np.array(torsion_current))

        #print(np.array(xyz_tm))
        #exit()

        # Compute energy
        grid[i, grid_M - 1] = 0.0
        grid[i, grid_M - 1] = grid[i, grid_M - 1] + ligand_ligand_reduced(xyz_tm,
                                                     ligand_qij,
                                                     ligand_dljr,
                                                     ligand_dljep,
                                                     ligand_sol_prefactor,
                                                     ligand_lklam,
                                                     ligand_nonbonded,
                                                     ligand_nonbondedWeights,
                                                     ligand_coupling,
                                                     ligand_weights)

        # Compute ligand_environment Energy
        grid[i, grid_M - 1] = grid[i, grid_M - 1] + ligand_environment_reduced(xyz_tm,
                                                          environment_xyz,
                                                          ligand_mask,
                                                          ligand_environment_qij,
                                                          ligand_environment_dljr,
                                                          ligand_environment_dljep,
                                                          ligand_environment_sol_prefactor,
                                                          ligand_lklam,
                                                          environmen_lklam,
                                                          environment_sideChainFalgs,
                                                          environment_mainChainFalgs,
                                                          sideChainCoupling,
                                                          mainChainCoupling,
                                                          ligand_environment_weights)
        #print('BBB En: ', np.array(grid[i, :]))

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(True)
def grid_search_parallel(double [:, :] grid,
                         double [:, :] ligand_xyz,
                         double[:, :] environment_xyz,

                         double [:] ligand_qij,
                         double [:] ligand_dljr,
                         double [:] ligand_dljep,
                         double [:, :] ligand_sol_prefactor,
                         double [:] ligand_lklam,
                         int [:, :] ligand_nonbonded,
                         double [:] ligand_nonbondedWeights,
                         double ligand_coupling,
                         double [:] ligand_weights,

                         double[:] ligand_mask,
                         int [:, :] sideChainsChiXYZIndices,
                         int [:, :] sideChainSubTreeMasks,

                         double[:, :] ligand_environment_qij,
                         double[:, :] ligand_environment_dljr,
                         double[:, :] ligand_environment_dljep,
                         double[:, :, :] ligand_environment_sol_prefactor,
                         double [:] environmen_lklam,
                         int [:] environment_sideChainFalgs,
                         int [:] environment_mainChainFalgs,
                         double sideChainCoupling,
                         double mainChainCoupling,
                         double [:] ligand_environment_weights,
                         int number_of_threads):

    cdef int grid_N = grid.shape[0]
    cdef int grid_M = grid.shape[1]
    cdef int ligand_xyz_N = ligand_xyz.shape[0]
    cdef int ligand_xyz_M = grid.shape[1]
    cdef int i, j, k, tid
    #print('BB ----- ', ligand_xyz.shape)

    cdef double [:, :, :] xyz_tm = np.zeros((number_of_threads, ligand_xyz_N, ligand_xyz_M), dtype=np.float64)
    #cdef double [:, :] rotation_vectors = np.zeros((sideChainsChiXYZIndices.shape[0], 3))
    cdef double u_ji_0, u_ji_1, u_ji_2
    cdef double u_jk_0, u_jk_1, u_jk_2
    cdef double u_lk_0, u_lk_1, u_lk_2
    cdef double n_ijk_0, n_ijk_1, n_ijk_2
    cdef double n_jkl_0, n_jkl_1, n_jkl_2
    cdef double nn_cross_0, nn_cross_1, nn_cross_2
    cdef double p0, p1, p2
    cdef double rotation_vector_0, rotation_vector_1, rotation_vector_2
    cdef double q0, q1, q2, q3
    cdef double n_ijk_nrom, n_jkl_norm, nn_dot, sign, rotation_vector_norm, energy
    cdef double [:] torsion_current = np.zeros(sideChainsChiXYZIndices.shape[0])
    cdef double rot_angle

    #print(xyz_tm.shape)
    #print(np.array(xyz_tm))
    # Get the current torsion vector
    for i in range(sideChainsChiXYZIndices.shape[0]):
        #print('BBB Cord i:', ligand_xyz[sideChainsChiXYZIndices[i, 0], 0], ligand_xyz[sideChainsChiXYZIndices[i, 0], 1], ligand_xyz[sideChainsChiXYZIndices[i, 0], 2])
        #print('BBB Cord j:', ligand_xyz[sideChainsChiXYZIndices[i, 1], 0], ligand_xyz[sideChainsChiXYZIndices[i, 1], 1], ligand_xyz[sideChainsChiXYZIndices[i, 1], 2])
        #print('BBB Cord k:', ligand_xyz[sideChainsChiXYZIndices[i, 2], 0], ligand_xyz[sideChainsChiXYZIndices[i, 2], 1], ligand_xyz[sideChainsChiXYZIndices[i, 2], 2])
        #print('BBB Cord l:', ligand_xyz[sideChainsChiXYZIndices[i, 3], 0], ligand_xyz[sideChainsChiXYZIndices[i, 3], 1], ligand_xyz[sideChainsChiXYZIndices[i, 3], 2])

        u_ji_0 = ligand_xyz[sideChainsChiXYZIndices[i, 0], 0] - ligand_xyz[sideChainsChiXYZIndices[i, 1], 0]
        u_ji_1 = ligand_xyz[sideChainsChiXYZIndices[i, 0], 1] - ligand_xyz[sideChainsChiXYZIndices[i, 1], 1]
        u_ji_2 = ligand_xyz[sideChainsChiXYZIndices[i, 0], 2] - ligand_xyz[sideChainsChiXYZIndices[i, 1], 2]

        u_jk_0 = ligand_xyz[sideChainsChiXYZIndices[i, 2], 0] - ligand_xyz[sideChainsChiXYZIndices[i, 1], 0]
        u_jk_1 = ligand_xyz[sideChainsChiXYZIndices[i, 2], 1] - ligand_xyz[sideChainsChiXYZIndices[i, 1], 1]
        u_jk_2 = ligand_xyz[sideChainsChiXYZIndices[i, 2], 2] - ligand_xyz[sideChainsChiXYZIndices[i, 1], 2]

        u_lk_0 = ligand_xyz[sideChainsChiXYZIndices[i, 2], 0] - ligand_xyz[sideChainsChiXYZIndices[i, 3], 0]
        u_lk_1 = ligand_xyz[sideChainsChiXYZIndices[i, 2], 1] - ligand_xyz[sideChainsChiXYZIndices[i, 3], 1]
        u_lk_2 = ligand_xyz[sideChainsChiXYZIndices[i, 2], 2] - ligand_xyz[sideChainsChiXYZIndices[i, 3], 2]

        #print('BBB', u_ji)
        #print('BBB', u_jk)
        #print('BBB', u_lk)

        n_ijk_0 = u_ji_1 * u_jk_2 - u_ji_2 * u_jk_1
        n_ijk_1 = u_ji_2 * u_jk_0 - u_ji_0 * u_jk_2
        n_ijk_2 = u_ji_0 * u_jk_1 - u_ji_1 * u_jk_0

        n_jkl_0 = u_jk_1 * u_lk_2 - u_jk_2 * u_lk_1
        n_jkl_1 = u_jk_2 * u_lk_0 - u_jk_0 * u_lk_2
        n_jkl_2 = u_jk_0 * u_lk_1 - u_jk_1 * u_lk_0

        n_ijk_nrom = sqrt(n_ijk_0 * n_ijk_0 + n_ijk_1 * n_ijk_1 + n_ijk_2 * n_ijk_2)
        n_jkl_norm = sqrt(n_jkl_0 * n_jkl_0 + n_jkl_1 * n_jkl_1 + n_jkl_2 * n_jkl_2)

        n_ijk_0 = n_ijk_0 / n_ijk_nrom
        n_ijk_1 = n_ijk_1 / n_ijk_nrom
        n_ijk_2 = n_ijk_2 / n_ijk_nrom

        n_jkl_0 = n_jkl_0 / n_jkl_norm
        n_jkl_1 = n_jkl_1 / n_jkl_norm
        n_jkl_2 = n_jkl_2 / n_jkl_norm

        #print('BBB', n_ijk)
        #print('BBB', n_jkl)

        nn_dot = n_ijk_0 * n_jkl_0 + n_ijk_1 * n_jkl_1 + n_ijk_2 * n_jkl_2

        nn_cross_0 = n_ijk_1 * n_jkl_2 - n_ijk_2 * n_jkl_1
        nn_cross_1 = n_ijk_2 * n_jkl_0 - n_ijk_0 * n_jkl_2
        nn_cross_2 = n_ijk_0 * n_jkl_1 - n_ijk_1 * n_jkl_0

        # Correct for precision error
        if nn_dot > 1: nn_dot = 1.0
        if nn_dot < -1: nn_dot = -1.0

        # get the angle sign
        sign = u_jk_0 * nn_cross_0 + u_jk_1 * nn_cross_1 + u_jk_2 * nn_cross_2

        # Save current torsion angles
        if sign < 0:
            torsion_current[i] = acos(nn_dot) * (180.0 / M_PI) * -1
        else:
            torsion_current[i] = acos(nn_dot) * (180.0 / M_PI)


        #print('BBBB ----!!!', np.array(sideChainsChiXYZIndices[i, :]), np.array(torsion_current[i]))

    with nogil, parallel():
        tid = openmp.omp_get_thread_num()
        #printf("------------------->>> %i  %i  %i  %i \n", openmp.omp_get_num_threads(), tid, ligand_xyz_N, ligand_xyz_M)

        for i in prange(grid_N, schedule='static'):
            #printf('%i  %i \n', tid, i)

            # Fil up the tem matrix
            for k in range(ligand_xyz_N):
                #printf('%i  %f  %f  %f \n', tid, ligand_xyz[k, 0], ligand_xyz[k, 1], ligand_xyz[k, 2])
                #printf('%i  %f  %f  %f \n', tid, xyz_tm[tid, k, 0], xyz_tm[tid, k, 1], xyz_tm[tid, k, 2])
                xyz_tm[tid, k, 0] = ligand_xyz[k, 0]
                xyz_tm[tid, k, 1] = ligand_xyz[k, 1]
                xyz_tm[tid, k, 2] = ligand_xyz[k, 2]
                #printf('%i  %f  %f  %f \n', tid, xyz_tm[tid, k, 0], xyz_tm[tid, k, 1], xyz_tm[tid, k, 2])

            # iterate over torsion vector
            for j in range(grid_M - 1):

                #if i == 0:
                #    printf('%i %i %i %i\n', i, j, openmp.omp_get_thread_num(), openmp.omp_get_num_threads())
                #    printf("%i BBB Cord i: %f %f %f \n", openmp.omp_get_thread_num(), xyz_tm[tid, sideChainsChiXYZIndices[j, 0], 0], xyz_tm[tid, sideChainsChiXYZIndices[j, 0], 1], xyz_tm[tid, sideChainsChiXYZIndices[j, 0], 2])
                #    printf("%i BBB Cord j: %f %f %f \n", openmp.omp_get_thread_num(), xyz_tm[tid, sideChainsChiXYZIndices[j, 1], 0], xyz_tm[tid, sideChainsChiXYZIndices[j, 1], 1], xyz_tm[tid, sideChainsChiXYZIndices[j, 1], 2])
                #    printf("%i BBB Cord k: %f %f %f \n", openmp.omp_get_thread_num(), xyz_tm[tid, sideChainsChiXYZIndices[j, 2], 0], xyz_tm[tid, sideChainsChiXYZIndices[j, 2], 1], xyz_tm[tid, sideChainsChiXYZIndices[j, 2], 2])
                #    printf("%i BBB Cord l: %f %f %f \n", openmp.omp_get_thread_num(), xyz_tm[tid, sideChainsChiXYZIndices[j, 3], 0], xyz_tm[tid, sideChainsChiXYZIndices[j, 3], 1], xyz_tm[tid, sideChainsChiXYZIndices[j, 3], 2])

                # Compute rotation angle for chi j
                rot_angle = grid[i, j] - torsion_current[j]
                if rot_angle < -180.0:
                    rot_angle = rot_angle + 360.0
                elif rot_angle >= 180.0:
                    rot_angle = rot_angle - 360.0

                # Convert to radian
                rot_angle = rot_angle * (M_PI / 180)

                # Compute rotation vectors
                rotation_vector_0 = xyz_tm[tid, sideChainsChiXYZIndices[j, 2], 0] - xyz_tm[tid, sideChainsChiXYZIndices[j, 1], 0]
                rotation_vector_1 = xyz_tm[tid, sideChainsChiXYZIndices[j, 2], 1] - xyz_tm[tid, sideChainsChiXYZIndices[j, 1], 1]
                rotation_vector_2 = xyz_tm[tid, sideChainsChiXYZIndices[j, 2], 2] - xyz_tm[tid, sideChainsChiXYZIndices[j, 1], 2]

                rotation_vector_norm = sqrt(rotation_vector_0 * rotation_vector_0 + rotation_vector_1 * rotation_vector_1 + rotation_vector_2 * rotation_vector_2)

                rotation_vector_0 = rotation_vector_0 / rotation_vector_norm
                rotation_vector_1 = rotation_vector_1 / rotation_vector_norm
                rotation_vector_2 = rotation_vector_2 / rotation_vector_norm

                # compute the rotation quaternion
                rot_angle = rot_angle * 0.5
                q0 = cos(rot_angle)
                q1 = sin(rot_angle) * rotation_vector_0
                q2 = sin(rot_angle) * rotation_vector_1
                q3 = sin(rot_angle) * rotation_vector_2


                # Rotate the subtree
                #print(j, np.array(sideChainSubTreeMasks[j, :]))
                for k in range(ligand_xyz_N):
                    # Operate only on the subtree atoms for chi j
                    if sideChainSubTreeMasks[j, k] == 0:
                        continue
                    else:
                        #Translate point to new coordinate system (rotation vector)
                        p0 = xyz_tm[tid, k, 0] - xyz_tm[tid, sideChainsChiXYZIndices[j, 1], 0]
                        p1 = xyz_tm[tid, k, 1] - xyz_tm[tid, sideChainsChiXYZIndices[j, 1], 1]
                        p2 = xyz_tm[tid, k, 2] - xyz_tm[tid, sideChainsChiXYZIndices[j, 1], 2]

                        # rotate
                        xyz_tm[tid, k, 0] = p0 * (q0 * q0 + q1 * q1 - q2 * q2 - q3 * q3) + 2 * p1 * (q1 * q2 - q0 * q3) + 2 * p2 * (q0 * q2 + q1 * q3)
                        xyz_tm[tid, k, 1] = 2 * p0 * (q0 * q3 + q1 * q2) +  p1 * (q0 * q0 - q1 * q1 + q2 * q2 - q3 * q3) + 2 * p2 * (q2 * q3 - q0 * q1)
                        xyz_tm[tid, k, 2] = 2 * p0 * (q1 * q3 - q0 * q2) + 2 * p1 * (q0 * q1 + q2 * q3) + p2 * (q0 * q0 - q1 * q1 - q2 * q2 + q3 * q3)

                        # translate back to the original coordinate system
                        xyz_tm[tid, k, 0] = xyz_tm[tid, k, 0] + xyz_tm[tid, sideChainsChiXYZIndices[j, 1], 0]
                        xyz_tm[tid, k, 1] = xyz_tm[tid, k, 1] + xyz_tm[tid, sideChainsChiXYZIndices[j, 1], 1]
                        xyz_tm[tid, k, 2] = xyz_tm[tid, k, 2] + xyz_tm[tid, sideChainsChiXYZIndices[j, 1], 2]


                """
                #-----------------------------------------------------------------------------------------------------------
                #-----------------------------------------------------------------------------------------------------------
                #-----------------------------------------------------------------------------------------------------------
                
                if i == 0:
                    printf("%i ABBB Cord i: %f %f %f \n", openmp.omp_get_thread_num(), xyz_tm[tid, sideChainsChiXYZIndices[j, 0], 0], xyz_tm[tid, sideChainsChiXYZIndices[j, 0], 1], xyz_tm[tid, sideChainsChiXYZIndices[j, 0], 2])
                    printf("%i ABBB Cord j: %f %f %f \n", openmp.omp_get_thread_num(), xyz_tm[tid, sideChainsChiXYZIndices[j, 1], 0], xyz_tm[tid, sideChainsChiXYZIndices[j, 1], 1], xyz_tm[tid, sideChainsChiXYZIndices[j, 1], 2])
                    printf("%i ABBB Cord k: %f %f %f \n", openmp.omp_get_thread_num(), xyz_tm[tid, sideChainsChiXYZIndices[j, 2], 0], xyz_tm[tid, sideChainsChiXYZIndices[j, 2], 1], xyz_tm[tid, sideChainsChiXYZIndices[j, 2], 2])
                    printf("%i ABBB Cord l: %f %f %f \n", openmp.omp_get_thread_num(), xyz_tm[tid, sideChainsChiXYZIndices[j, 3], 0], xyz_tm[tid, sideChainsChiXYZIndices[j, 3], 1], xyz_tm[tid, sideChainsChiXYZIndices[j, 3], 2])

                u_ji_0 = xyz_tm[tid, sideChainsChiXYZIndices[j, 0], 0] - xyz_tm[tid, sideChainsChiXYZIndices[j, 1], 0]
                u_ji_1 = xyz_tm[tid, sideChainsChiXYZIndices[j, 0], 1] - xyz_tm[tid, sideChainsChiXYZIndices[j, 1], 1]
                u_ji_2 = xyz_tm[tid, sideChainsChiXYZIndices[j, 0], 2] - xyz_tm[tid, sideChainsChiXYZIndices[j, 1], 2]

                u_jk_0 = xyz_tm[tid, sideChainsChiXYZIndices[j, 2], 0] - xyz_tm[tid, sideChainsChiXYZIndices[j, 1], 0]
                u_jk_1 = xyz_tm[tid, sideChainsChiXYZIndices[j, 2], 1] - xyz_tm[tid, sideChainsChiXYZIndices[j, 1], 1]
                u_jk_2 = xyz_tm[tid, sideChainsChiXYZIndices[j, 2], 2] - xyz_tm[tid, sideChainsChiXYZIndices[j, 1], 2]

                u_lk_0 = xyz_tm[tid, sideChainsChiXYZIndices[j, 2], 0] - xyz_tm[tid, sideChainsChiXYZIndices[j, 3], 0]
                u_lk_1 = xyz_tm[tid, sideChainsChiXYZIndices[j, 2], 1] - xyz_tm[tid, sideChainsChiXYZIndices[j, 3], 1]
                u_lk_2 = xyz_tm[tid, sideChainsChiXYZIndices[j, 2], 2] - xyz_tm[tid, sideChainsChiXYZIndices[j, 3], 2]

                # print('BBB', u_ji)
                # print('BBB', u_jk)
                # print('BBB', u_lk)

                n_ijk_0 = u_ji_1 * u_jk_2 - u_ji_2 * u_jk_1
                n_ijk_1 = u_ji_2 * u_jk_0 - u_ji_0 * u_jk_2
                n_ijk_2 = u_ji_0 * u_jk_1 - u_ji_1 * u_jk_0

                n_jkl_0 = u_jk_1 * u_lk_2 - u_jk_2 * u_lk_1
                n_jkl_1 = u_jk_2 * u_lk_0 - u_jk_0 * u_lk_2
                n_jkl_2 = u_jk_0 * u_lk_1 - u_jk_1 * u_lk_0

                n_ijk_nrom = sqrt(n_ijk_0 * n_ijk_0 + n_ijk_1 * n_ijk_1 + n_ijk_2 * n_ijk_2)
                n_jkl_norm = sqrt(n_jkl_0 * n_jkl_0 + n_jkl_1 * n_jkl_1 + n_jkl_2 * n_jkl_2)

                n_ijk_0 = n_ijk_0 / n_ijk_nrom
                n_ijk_1 = n_ijk_1 / n_ijk_nrom
                n_ijk_2 = n_ijk_2 / n_ijk_nrom

                n_jkl_0 = n_jkl_0 / n_jkl_norm
                n_jkl_1 = n_jkl_1 / n_jkl_norm
                n_jkl_2 = n_jkl_2 / n_jkl_norm

                # print('BBB', n_ijk)
                # print('BBB', n_jkl)

                nn_dot = n_ijk_0 * n_jkl_0 + n_ijk_1 * n_jkl_1 + n_ijk_2 * n_jkl_2

                nn_cross_0 = n_ijk_1 * n_jkl_2 - n_ijk_2 * n_jkl_1
                nn_cross_1 = n_ijk_2 * n_jkl_0 - n_ijk_0 * n_jkl_2
                nn_cross_2 = n_ijk_0 * n_jkl_1 - n_ijk_1 * n_jkl_0

                # Correct for precision error
                if nn_dot > 1: nn_dot = 1.0
                if nn_dot < -1: nn_dot = -1.0

                # get the angle sign
                sign = u_jk_0 * nn_cross_0 + u_jk_1 * nn_cross_1 + u_jk_2 * nn_cross_2

                # Save current torsion angles
                if i == 0:
                    if sign < 0:
                        printf("%i %i %i current %f target %f \n", openmp.omp_get_thread_num(), i, j, acos(nn_dot) * (180.0 / M_PI) * -1, grid[i, j])
                    else:
                        printf("%i %i %i current %f target %f \n", openmp.omp_get_thread_num(), i, j, acos(nn_dot) * (180.0 / M_PI), grid[i, j])
                # -----------------------------------------------------------------------------------------------------------
                # -----------------------------------------------------------------------------------------------------------
                # -----------------------------------------------------------------------------------------------------------
                """
            
            #print(np.array(xyz_tm))
            #exit()
            # Compute energy
            grid[i, grid_M - 1] = 0.0
            grid[i, grid_M - 1] = grid[i, grid_M - 1] + ligand_ligand_reduced(xyz_tm[tid, :, :],
                                          ligand_qij,
                                          ligand_dljr,
                                          ligand_dljep,
                                          ligand_sol_prefactor,
                                          ligand_lklam,
                                          ligand_nonbonded,
                                          ligand_nonbondedWeights,
                                          ligand_coupling,
                                          ligand_weights)

            # TODO ??????????????????????????????????
            grid[i, grid_M - 1] = grid[i, grid_M - 1]

            # Compute ligand_environment Energy
            grid[i, grid_M - 1] = grid[i, grid_M - 1] + ligand_environment_reduced(xyz_tm[tid, :, :],
                                                                       environment_xyz,
                                                                       ligand_mask,
                                                                       ligand_environment_qij,
                                                                       ligand_environment_dljr,
                                                                       ligand_environment_dljep,
                                                                       ligand_environment_sol_prefactor,
                                                                       ligand_lklam,
                                                                       environmen_lklam,
                                                                       environment_sideChainFalgs,
                                                                       environment_mainChainFalgs,
                                                                       sideChainCoupling,
                                                                       mainChainCoupling,
                                                                       ligand_environment_weights)
