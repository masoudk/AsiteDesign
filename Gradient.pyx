from pyrosetta import Pose
from pyrosetta import get_fa_scorefxn
from pyrosetta.rosetta.core.scoring import ScoreType
cimport cython
cimport openmp
import numpy as np
import random
from libc.math cimport cos, sin, acos, sqrt, M_PI, acos, atan2, fabs, pow, exp
from libc.float cimport DBL_MAX
from cython.parallel cimport prange
from cython.parallel cimport parallel
from pyrosetta.rosetta.utility import vector1_numeric_xyzVector_double_t as vec1
##from Energy cimport switch
from Energy import ligand_ligand_reduced, ligand_environment_reduced, bond, angle

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
                        int max_line_search):

    cdef int N = ligand_xyz.shape[0]
    cdef int M = ligand_xyz.shape[1]
    cdef double [:, :] gradient_vector, xyz_tm
    cdef double E_new, E_old, g_Max, g_norm2, step
    cdef int i, j, k

    gradient_vector = np.zeros((N, 3), dtype=np.float64)
    xyz_tm = np.zeros((N, M), dtype=np.float64)


    # Compute initial energy E_old
    E_old = 0.0
    E_old += ligand_ligand_reduced(xyz_tm,
                                   ligand_qij,
                                   ligand_dljr,
                                   ligand_dljep,
                                   ligand_sol_prefactor,
                                   ligand_lklam,
                                   ligand_nonbonded,
                                   ligand_nonbondedWeights,
                                   coupling,
                                   weights)

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
                                        weights)

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

# Minimize
    for i in range(max_cycle):

        # rest stuff
        g_Max = -DBL_MAX
        g_norm2 = 0.0
        E_new = 0.0

        # Reset the gradient_vector
        for j in range(N):
            gradient_vector[j, 0] = 0.0
            gradient_vector[j, 1] = 0.0
            gradient_vector[j, 2] = 0.0

        # Compute gradient by xyz
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

        # Compute ligand bond gradient
        bond_gradient(ligand_xyz,
                      bond_bond_atomIndex,
                      equilibrium_distance,
                      force_constance,
                      gradient_vector)

        # Compute ligand angle gradient
        angle_gradient(ligand_xyz,
                      angle_atomIndex,
                      equilibrium_angle,
                      angle_force_constance,
                      gradient_vector)


        # Go over the grad vector
        for j in range(N):
            # Compute the dot product of grad vector. Treat it as a flat matrix
            g_norm2 += gradient_vector[j, 0] * gradient_vector[j, 0]
            g_norm2 += gradient_vector[j, 1] * gradient_vector[j, 1]
            g_norm2 += gradient_vector[j, 2] * gradient_vector[j, 2]

            # get g_Max
            if gradient_vector[j, 0] > g_Max:
                g_Max = gradient_vector[j, 0]
            if gradient_vector[j, 1] > g_Max:
                g_Max = gradient_vector[j, 1]
            if gradient_vector[j, 2] > g_Max:
                g_Max = gradient_vector[j, 2]

        # Choose a Step size such that g_Max*Step = 0.5
        step = 0.5/g_Max

        for j in range(max_line_search):

            # Get the new coordinate
            for k in range(N):
                xyz_tm[k, 0] = ligand_xyz[k, 0] * (-1 * gradient_vector[k, 0] * step)
                xyz_tm[k, 1] = ligand_xyz[k, 1] * (-1 * gradient_vector[k, 1] * step)
                xyz_tm[k, 2] = ligand_xyz[k, 2] * (-1 * gradient_vector[k, 2] * step)

            # compute E_new
            # Compute ligand_ligand Energy
            E_new += ligand_ligand_reduced(xyz_tm,
                                           ligand_qij,
                                           ligand_dljr,
                                           ligand_dljep,
                                           ligand_sol_prefactor,
                                           ligand_lklam,
                                           ligand_nonbonded,
                                           ligand_nonbondedWeights,
                                           coupling,
                                           weights)

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
                                                weights)

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

            if E_new > E_old:
                step *= 0.5
                continue
            else:  # accept the move
                #update xyz
                for k in range(N):
                    ligand_xyz[k, 0] = xyz_tm[k, 0]
                    ligand_xyz[k, 1] = xyz_tm[k, 1]
                    ligand_xyz[k, 2] = xyz_tm[k, 2]
                break

        # Quite if energy converges
        if fabs(E_new - E_old) < 0.1:
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
                atr = ligand_dljep[pair]

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

                atr = ( ((-12 * ligand_dljep[pair]/dij)) * (dlj_dij12 - dlj_dij6) * switch(dij, 4.5, 6.0)) + \
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
                      ((ligand_qij[pair]/epsilon) * ( (1 / (dij*dij)) - (1 / (5.5 * 5.5)) ) * (-s_p))

            elif 1.85 <= dij and dij < 4.5:
                col = ( (-2 * ligand_qij[pair]/epsilon) * (1 / (dij*dij*dij)) )

            elif 4.5 <= dij and dij < 5.5:
                col = ( (-2 * ligand_qij[pair]/epsilon) * (1 / (dij*dij*dij)) * s ) +  \
                      ( (ligand_qij[pair]/epsilon) * ( (1 / (dij*dij)) - (1 / (5.5*5.5)) ) * switch_gradient(dij, 4.5, 5.5) )

            else:
                col = 0




            # compute solvation
            sol = 0
            if ligand_weights[3] != 0:
                if dij <= ligand_dljr[pair] - 0.3:

                    sol += -ligand_sol_prefactor[pair, 0]
                    sol += -ligand_sol_prefactor[pair, 1]

                elif ligand_dljr[pair] - 0.3 < dij and dij <= ligand_dljr[pair] + 0.2:

                    s = switch(dij, ligand_dljr[pair] - 0.3, ligand_dljr[pair] + 0.2)
                    s_p = switch_gradient(dij, ligand_dljr[pair] - 0.3, ligand_dljr[pair] + 0.2)

                    sol_expfactor_i = (dij - ligand_dljr[pair]) / ligand_lklam[i]
                    sol_expfactor_j = (dij - ligand_dljr[pair]) / ligand_lklam[j]

                    sol += (-ligand_sol_prefactor[pair, 0] * s) + \
                           (ligand_sol_prefactor[pair, 0] * s_p) + \
                           ((ligand_sol_prefactor[pair, 0] * exp(-1 * (sol_expfactor_i * sol_expfactor_i)) * (-2 * (dij - ligand_dljr[pair]) / (ligand_lklam[i] * ligand_lklam[i]))) * (1 - s)) + \
                           (ligand_sol_prefactor[pair, 0] *  exp(-1 * (sol_expfactor_i * sol_expfactor_i)) * (-s_p))

                    sol += (-ligand_sol_prefactor[pair, 1] * s) + \
                           (ligand_sol_prefactor[pair, 1] * s_p) + \
                           ((ligand_sol_prefactor[pair, 1] * exp(-1 * (sol_expfactor_j * sol_expfactor_j)) * (-2 * (dij - ligand_dljr[pair]) / (ligand_lklam[j] * ligand_lklam[j])) )* (1 -s)) + \
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
    cdef double dxji, dyji, dzji, dij, dlj, dlj_dij, dlj_dij3, dlj_dij6, dlj_dij12, epsilon, s
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

            dij = dxji * dxji + dxji * dxji + dxji * dxji
            dij = sqrt(dij)

            # Compute rep gradient magnitude
            if dij <= 0.6 * ligand_environment_dljr[i, j]:
                # constant, replacing dij = 0.6*dljr
                dlj_dij = 1 / 0.6
                dlj_dij3 = dlj_dij * dlj_dij * dlj_dij
                dlj_dij6 = dlj_dij3 * dlj_dij3
                dlj_dij12 = dlj_dij6 * dlj_dij6

                rep = (-12 * ligand_environment_dljep[i, j] / (0.6 * ligand_environment_dljr[i, j])) * (dlj_dij12 - dlj_dij6)  # - (dlj_dij*dij)

            elif 0.6 * ligand_environment_dljr[i, j] < dij and dij <= ligand_environment_dljr[i, j]:
                dlj_dij = ligand_environment_dljr[i, j] / dij
                dlj_dij3 = dlj_dij * dlj_dij * dlj_dij
                dlj_dij6 = dlj_dij3 * dlj_dij3
                dlj_dij12 = dlj_dij6 * dlj_dij6

                rep = (-12 * ligand_environment_dljep[i, j] / dij) * (dlj_dij12 - dlj_dij6)

            elif ligand_environment_dljr[i, j] < dij:
                rep = 0


            # Compute atr
            if dij <= ligand_environment_dljr[i, j]:
                atr = ligand_environment_dljr[i, j]

            elif ligand_environment_dljr[i, j] < dij and dij <= 4.5:
                dlj_dij = ligand_environment_dljr[i, j] / dij
                dlj_dij3 = dlj_dij * dlj_dij * dlj_dij
                dlj_dij6 = dlj_dij3 * dlj_dij3
                dlj_dij12 = dlj_dij6 * dlj_dij6

                atr = (-12 * ligand_environment_dljep[i, j]/ dij) * (dlj_dij12 - dlj_dij6)

            elif 4.5 < dij and dij <= 6.0:
                dlj_dij = ligand_environment_dljr[i, j] / dij
                dlj_dij3 = dlj_dij * dlj_dij * dlj_dij
                dlj_dij6 = dlj_dij3 * dlj_dij3
                dlj_dij12 = dlj_dij6 * dlj_dij6

                atr = (((-12 * ligand_environment_dljep[i, j] / dij)) * (dlj_dij12 - dlj_dij6) * switch(dij, 4.5, 6.0)) + \
                        (ligand_environment_dljep[i, j] * (dlj_dij12 - (2 * dlj_dij6)) * switch_gradient(dij, 4.5, 6.0))

            elif 6.0 < dij:
                atr = 0


            # compute electrostatic
            epsilon = 6
            if dij < 1.45:
                col = -ligand_environment_qij[i, j] / (1.45 * 1.45 * epsilon)

            elif 1.45 <= dij and dij < 1.85:
                s = switch(dij, 1.45, 1.85)
                s_p = switch_gradient(dij, 1.45, 1.85)

                col = ((-ligand_environment_qij[i, j] / (1.45 * 1.45 * epsilon)) * s) + \
                      ((ligand_environment_qij[i, j] / (1.45 * epsilon)) * s_p) + \
                      ((-2 * ligand_environment_qij[i, j] / epsilon) * (1 / (dij * dij * dij)) * (1 - s)) + \
                      ((ligand_environment_qij[i, j] / epsilon) * ((1 / (dij * dij)) - (1 / (5.5 * 5.5))) * (-s_p))

            elif 1.85 <= dij and dij < 4.5:
                col = ((-2 * ligand_environment_qij[i, j] / epsilon) * (1 / (dij * dij * dij)))

            elif 4.5 <= dij and dij < 5.5:
                col = ((-2 * ligand_environment_qij[i, j] / epsilon) * (1 / (dij * dij * dij)) * s) + \
                      ((ligand_environment_qij[i, j] / epsilon) * ((1 / (dij * dij)) - (1 / (5.5 * 5.5))) * switch_gradient(dij, 4.5, 5.5))

            else:
                col = 0


            # compute solvation
            sol = 0
            if ligand_environment_weights[3] != 0:
                if dij <= ligand_environment_dljr[i, j] - 0.3:

                    sol += -ligand_environment_sol_prefactor[i, j, 0]
                    sol += -ligand_environment_sol_prefactor[i, j, 1]

                elif ligand_environment_dljr[i, j] - 0.3 < dij and dij <= ligand_environment_dljr[i, j] + 0.2:

                    s = switch(dij, ligand_environment_dljr[i, j] - 0.3, ligand_environment_dljr[i, j] + 0.2)
                    s_p = switch_gradient(dij, ligand_environment_dljr[i, j] - 0.3, ligand_environment_dljr[i, j] + 0.2)

                    sol_expfactor_i = (dij - ligand_environment_dljr[i, j]) / ligand_lklam[i]
                    sol_expfactor_j = (dij - ligand_environment_dljr[i, j]) / environmen_lklam[j]

                    sol += (-ligand_environment_sol_prefactor[i, j, 0] * s) + \
                           (ligand_environment_sol_prefactor[i, j, 0] * s_p) + \
                           ((ligand_environment_sol_prefactor[i, j, 0] * exp(-1 * (sol_expfactor_i * sol_expfactor_i)) * (-2 * (dij - ligand_environment_dljr[i, j]) / (ligand_lklam[i] * ligand_lklam[i]))) * (1 - s)) + \
                           (ligand_environment_sol_prefactor[i, j, 0] * exp(-1 * (sol_expfactor_i * sol_expfactor_i)) * (-s_p))

                    sol += (-ligand_environment_sol_prefactor[i, j, 1] * s) + \
                           (ligand_environment_sol_prefactor[i, j, 1] * s_p) + \
                           ((ligand_environment_sol_prefactor[i, j, 1] * exp(-1 * (sol_expfactor_j * sol_expfactor_j)) * (-2 * (dij - ligand_environment_dljr[i, j]) / (environmen_lklam[j] * environmen_lklam[j]))) * (1 - s)) + \
                           (ligand_environment_sol_prefactor[i, j, 1] * exp(-1 * (sol_expfactor_j * sol_expfactor_j)) * (-s_p))


                elif ligand_environment_dljr[i, j] + 0.2 < dij and dij <= 4.5:
                    sol_expfactor_i = (dij - ligand_environment_dljr[i, j]) / ligand_lklam[i]
                    sol_expfactor_j = (dij - ligand_environment_dljr[i, j]) / environmen_lklam[j]

                    sol += (ligand_environment_sol_prefactor[i, j, 0] * exp(-1 * (sol_expfactor_i * sol_expfactor_i)) * (-2 * (dij - ligand_environment_dljr[i, j]) / (ligand_lklam[i] * ligand_lklam[i])))
                    sol += (ligand_environment_sol_prefactor[i, j, 1] * exp(-1 * (sol_expfactor_j * sol_expfactor_j)) * (-2 * (dij - ligand_environment_dljr[i, j]) / (environmen_lklam[j] * environmen_lklam[j])))

                elif 4.5 < dij and dij <= 6.0:
                    s = switch(dij, 4.5, 6.0)
                    s_p = switch_gradient(dij, 4.5, 6.0)

                    sol_expfactor_i = (dij - ligand_environment_dljr[i, j]) / ligand_lklam[i]
                    sol_expfactor_j = (dij - ligand_environment_dljr[i, j]) / environmen_lklam[j]

                    sol += (ligand_environment_sol_prefactor[i, j, 0] * exp(-1 * (sol_expfactor_i * sol_expfactor_i)) * (-2 * (dij - ligand_environment_dljr[i, j]) / (ligand_lklam[i] * ligand_lklam[i])) * s) + \
                           (ligand_environment_sol_prefactor[i, j, 0] * exp(-1 * (sol_expfactor_i * sol_expfactor_i)) * s_p)

                    sol += (ligand_environment_sol_prefactor[i, j, 1] * exp(-1 * (sol_expfactor_j * sol_expfactor_j)) * (-2 * (dij - ligand_environment_dljr[i, j]) / environmen_lklam[j] * environmen_lklam[j]) * s) + \
                           (ligand_environment_sol_prefactor[i, j, 1] * exp(-1 * (sol_expfactor_j * sol_expfactor_j)) * s_p)

                else:
                    sol = 0


            grad_magnititude = coupling * (ligand_environment_weights[0] * rep + ligand_environment_weights[1] * atr + ligand_environment_weights[2] * col + ligand_environment_weights[3] * sol)


            # Set forces for atom i (ligand)
            gradient_vector[i, 0] = gradient_vector[i, 0] + ((dxji * grad_magnititude) / dij)
            gradient_vector[i, 1] = gradient_vector[i, 1] + ((dyji * grad_magnititude) / dij)
            gradient_vector[i, 2] = gradient_vector[i, 2] + ((dzji * grad_magnititude) / dij)


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
            if sn == 0: sn = 0.00000001

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


