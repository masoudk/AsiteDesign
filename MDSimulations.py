# Global imports
import argparse
import re
import time
import simtk.openmm as mm
import simtk.unit as unit
import simtk.openmm.app as app

"""
A module for running various MD simulations. For now it only runs simpleMD. Later when more options are added a new flag is needed to specify the simulation type.
usage:

python3.5 Simulations.py -jn test -tf PL1-system.prmtop -cf PL1-system.inpcrd -ff AMBER -ns 10000 -st 300 -pt CPU  -ms 10 -es 100 -lo 10 -tr 10

"""

def getArguments():

    parser = argparse.ArgumentParser(description='Run MD Simulations')
    parser.add_argument('-ty', type=str,   metavar='type',                   required=True,    help='Simulation type: "MD", "REMD"')
    parser.add_argument('-tf', type=str,   metavar='topology file',          required=True,    help='Topology file in Amber format')
    parser.add_argument('-cf', type=str,   metavar='coordinate file',        required=True,    help='Coordinate file in Amber format')
    parser.add_argument('-ff', type=str,   metavar='file format',            default='Amber',  help='The format of input files. Currently only Amber is supported, which is the default value.')
    parser.add_argument('-jn', type=str,   metavar='job name',               default='Output', help='Jobe name will be used as a stem for naming all files.')
    parser.add_argument('-ns', type=int,   metavar='number of steps',        required=True,    help='Number of MD steps, where each step is 2fs.')
    parser.add_argument('-st', type=float, metavar='simulation temperature', default=298.15,   help='The target temperature of simulation. The default is 298.15.')
    parser.add_argument('-pt', type=str,   metavar='platform type',          default='CPU',    help='Platform type can be CPU, CUDA, and OpenCL')
    parser.add_argument('-ng', type=int,   metavar='number of gpu',          default=None,     help='Number of gpus should be specifiec if CUDA is the platform')
    parser.add_argument('-ms', type=int,   metavar='minimization steps',     default=2000,     help='Number of minimization stpes. Default is 2000.')
    parser.add_argument('-es', type=int,   metavar='equilibration steps',    default=10000,    help='Number of equilibration stpes. Default is 10000*2fs')
    parser.add_argument('-lo', type=int,   metavar='log frequency',          default=1000,     help='The frequency of writing lof file. Default is 1 per 1000')
    parser.add_argument('-tr', type=int,   metavar='traj frequency',         default=1000,     help='The frequency of writing dcd file. Default is 1 per 1000')

    return parser.parse_args()


def getInputFiles(topologyFile, coordinateFile, fileType):

    if re.match('Amber', fileType, re.IGNORECASE):
        return app.AmberPrmtopFile(topologyFile),  app.AmberInpcrdFile(coordinateFile).positions
    else:
        raise ValueError("Error >>> can not read inputs. Only Amber is implemented")


def makeSystem(topology, inputType='Amber'):
    if re.match('Amber', inputType, re.IGNORECASE):
        return topology.createSystem(nonbondedMethod=app.PME, nonbondedCutoff=1*unit.nanometer, constraints=app.HBonds)
    else:
        raise ValueError("Error >>> can not make system from %s files." % inputType)


def getPeriodicPositionRestraintsOnHeavyAtoms(topology, coordinate, k):

        force = mm.CustomExternalForce("k*periodicdistance(x, y, z, x0, y0, z0)^2")
        force.addGlobalParameter("k", k * unit.kilocalories_per_mole / unit.angstroms ** 2)
        force.addPerParticleParameter("x0")
        force.addPerParticleParameter("y0")
        force.addPerParticleParameter("z0")

        for i, atom in enumerate(topology.topology.atoms()):
            if atom.element.symbol != "H":
                force.addParticle(i, coordinate[i].value_in_unit(unit.nanometers))

        return force


def getPeriodicPositionRestraintsOnAllAtoms(topology, coordinate, k):

        force = mm.CustomExternalForce("k*periodicdistance(x, y, z, x0, y0, z0)^2")
        force.addGlobalParameter("k", k * unit.kilocalories_per_mole / unit.angstroms ** 2)
        force.addPerParticleParameter("x0")
        force.addPerParticleParameter("y0")
        force.addPerParticleParameter("z0")

        for i, atom in enumerate(topology.topology.atoms()):
            force.addParticle(i, coordinate[i].value_in_unit(unit.nanometers))

        return force


def runMinimize(topology, coordinate, temperature, inputType, platform , platformProperties, jobName, maxItiration,
                                                                                                    heavyAtomrestraint):

    # Build the system
    system = makeSystem(topology, inputType)

    if heavyAtomrestraint:
        positionRestraints = getPeriodicPositionRestraintsOnHeavyAtoms(topology, coordinate, heavyAtomrestraint)
        system.addForce(positionRestraints)

    # Build simulation object
    integrator = mm.LangevinIntegrator(temperature*unit.kelvin, 1/unit.picosecond, 0.002*unit.picoseconds)
    platform = mm.Platform.getPlatformByName(platform)

    simulation = app.Simulation(topology.topology, system, integrator, platform, platformProperties)
    simulation.context.setPositions(coordinate)

    #if re.match('Amber', inputType, re.IGNORECASE):
    #    if coordinate.boxVectors is not None: simulation.context.setPeriodicBoxVectors(*coordinate.boxVectors)

    simulation.minimizeEnergy(maxItiration)

    # Write the final structure of the output.
    with open('{}-min-000.pdb'.format(jobName), 'w') as f:
        app.PDBFile.writeFile(topology.topology, simulation.context.getState(getPositions=True).getPositions(), f)

    return simulation.context.getState(getPositions=True, getVelocities=True)


def runNVT(topology, coordinate, velocity, temperature, inputType, platform , platformProperties, steps, jobname,
                                                                    logFrequency, trajFrequency, heavyAtomrestraint, timeStep):

    # Build the system
    system = makeSystem(topology, inputType)

    if heavyAtomrestraint:
        positionRestraints = getPeriodicPositionRestraintsOnAllAtoms(topology, coordinate, heavyAtomrestraint)
        system.addForce(positionRestraints)

    thermostat = mm.AndersenThermostat(temperature*unit.kelvin, 1/unit.picosecond)
    system.addForce(thermostat)

    # Build simulation object
    integrator = mm.LangevinIntegrator(temperature*unit.kelvin, 1/unit.picosecond, timeStep*unit.picoseconds)
    platform = mm.Platform.getPlatformByName(platform)

    simulation = app.Simulation(topology.topology, system, integrator, platform, platformProperties=platformProperties)
    simulation.context.setPositions(coordinate)
    if velocity:
        simulation.context.setVelocities(velocity)
    else:
        simulation.context.setVelocitiesToTemperature(temperature * unit.kelvin, 1)

    # Add reporters
    fileName = "{}-NVT-{}.log".format(jobname, int(temperature))
    reporterLog = app.StateDataReporter(fileName, logFrequency, step=True, temperature=True, kineticEnergy=True,
                                                                    potentialEnergy=True, volume=True, separator=' ')
    simulation.reporters.append(reporterLog)

    fileName = "{}-NVT-{}.dcd".format(jobname, int(temperature))
    simulation.reporters.append(app.DCDReporter(fileName, trajFrequency))

    # Run simulations
    simulation.step(steps)

    return simulation.context.getState(getPositions=True, getVelocities=True)


def runNPT(topology, coordinate, velocity, temperature, inputType, platform , platformProperties, steps, jobname,
                                                            logFrequency, trajFrequency, heavyAtomrestraint, timeStep):

    # Build the system
    system = makeSystem(topology, inputType)

    if heavyAtomrestraint:
        positionRestraints = getPeriodicPositionRestraintsOnHeavyAtoms(topology, coordinate, heavyAtomrestraint)
        system.addForce(positionRestraints)

    thermostat = mm.AndersenThermostat(temperature*unit.kelvin, 1/unit.picosecond)
    system.addForce(thermostat)

    barostat = mm.MonteCarloBarostat(1 * unit.bar, temperature * unit.kelvin)
    system.addForce(barostat)


    # Build simulation object
    integrator = mm.LangevinIntegrator(temperature*unit.kelvin, 1/unit.picosecond, timeStep*unit.picoseconds)
    platform = mm.Platform.getPlatformByName(platform)

    simulation = app.Simulation(topology.topology, system, integrator, platform, platformProperties=platformProperties)
    simulation.context.setPositions(coordinate)
    if velocity:
        simulation.context.setVelocities(velocity)
    else:
        simulation.context.setVelocitiesToTemperature(temperature * unit.kelvin, 1)

    # Add reporters
    fileName = "{}-NPT-{}.log".format(jobname, int(temperature))
    reporterLog = app.StateDataReporter(fileName, logFrequency, step=True, temperature=True, kineticEnergy=True,
                                                                                potentialEnergy=True, separator=' ')
    simulation.reporters.append(reporterLog)

    fileName = "{}-NPT-{}.dcd".format(jobname, int(temperature))
    simulation.reporters.append(app.DCDReporter(fileName, trajFrequency))

    # Run simulations
    simulation.step(steps)

    return simulation.context.getState(getPositions=True, getVelocities=True)


def equilibrationSlow(topology, coordinate, temperature, inputType, platform, platformProperties, jobName,
                                                    minimizationSteps, equilibrationSteps, logFrequency, trajFrequency):
    dtm = temperature/3
    currentTemperature = 0.0

    print("Starting minimization.", flush=True)
    # Initial minimization
    start = time.time()
    state = runMinimize(topology, coordinate, temperature, inputType, platform, platformProperties, jobName,
                                                                    minimizationSteps, heavyAtomrestraint=0)
    end = time.time()
    print("Minimization is finished in {:3.1f}".format(end - start), flush=True)

    currentCoordinate = state.getPositions()
    currentVelocity = state.getVelocities()
    currentTemperature += dtm

    print("Starting NVT MD at {} K".format(int(currentTemperature)), flush=True)
    start = time.time()
    # NVT MD at 100 K for 10000 steps (20 ps)
    state = runNVT(topology, currentCoordinate, currentVelocity, currentTemperature, inputType, platform,
                            platformProperties, equilibrationSteps, jobName, logFrequency, trajFrequency,
                                                                    heavyAtomrestraint=5.0, timeStep=0.0005)
    end = time.time()
    print("NVT MD at {} K is finished in {:3.1f}".format(currentTemperature, end - start), flush=True)

    currentCoordinate = state.getPositions()
    currentVelocity = state.getVelocities()
    currentTemperature += dtm

    print("Starting NVT MD at {} K".format(int(currentTemperature)), flush=True)
    start = time.time()
    # NVT MD at 200 K for 10000 steps (20 ps) restraint on heavyAtoms 1
    state = runNVT(topology, currentCoordinate, currentVelocity, currentTemperature, inputType, platform,
                            platformProperties, equilibrationSteps, jobName, logFrequency, trajFrequency,
                                                                     heavyAtomrestraint=1.0, timeStep=0.001)

    end = time.time()
    print("NVT MD at {} K is finished in {:3.1f}".format(currentTemperature, end - start), flush=True)

    currentCoordinate = state.getPositions()
    currentVelocity = state.getVelocities()
    currentTemperature += dtm

    print("Starting NVT MD at {} K".format(int(currentTemperature)), flush=True)
    start = time.time()
    # NVT MD at 300 K for 10000 steps (20 ps) restraint on heavyAtoms 0
    state = runNVT(topology, currentCoordinate, currentVelocity, currentTemperature, inputType, platform,
                            platformProperties, equilibrationSteps, jobName, logFrequency, trajFrequency,
                                                                     heavyAtomrestraint=1.0, timeStep=0.002)

    end = time.time()
    print("NVT MD at {} K is finished in {:3.1f}".format(currentTemperature, end - start), flush=True)

    currentCoordinate = state.getPositions()
    currentVelocity = state.getVelocities()

    print("Starting NPT at {} K".format(int(currentTemperature)), flush=True)
    start = time.time()
    # NPT MD at 300 K for 10000 steps (20 ps) restraint on heavyAtoms 0
    state = runNPT(topology, currentCoordinate, currentVelocity, currentTemperature, inputType, platform,
                            platformProperties, equilibrationSteps, jobName, logFrequency, trajFrequency,
                                                                     heavyAtomrestraint=0.0, timeStep=0.002)
    end = time.time()
    print("NPT MD at {} K is finished in {:3.1f}".format(currentTemperature, end - start), flush=True)
    return state


def simpleMD(topology, coordinate, temperature, mdSteps, inputType, platform, platformProperties, jobName,
                                                    minimizationSteps, equilibrationSteps, logFrequency, trajFrequency):

    start = time.time()
    state = equilibrationSlow(topology, coordinate, temperature, inputType, platform, platformProperties, jobName,
                                                    minimizationSteps, equilibrationSteps, logFrequency, trajFrequency)
    end = time.time()
    print('Equilibrtion is finished in: {:3.1f}'.format(end-start), flush=True)
    currentCoordinate = state.getPositions()
    currentVelocity = state.getVelocities()

    print("Starting production NPT MD at {} K".format(int(temperature)), flush=True)
    start = time.time()
    jobName = "{}-MD".format(jobName)
    # Production - NPT MD at 300 K for mdSteps
    state = runNPT(topology, currentCoordinate, currentVelocity, temperature, inputType, platform,
                                platformProperties, mdSteps, jobName, logFrequency, trajFrequency,
                                                                      heavyAtomrestraint=0.0, timeStep=0.002)
    end = time.time()
    print('Production is finished in: {:3.1f}'.format(end-start), flush=True)


def reMD(topology, coordinate, targetTemperature, mdSteps, inputType, platform, platformProperties, jobName):
    pass


def main():

    # Print Version
    print("Using OpenMM version: {}".format(str(mm.Platform.getOpenMMVersion())), flush=True)

    # Get the input files
    arg = getArguments()

    # Read the topology and coordinate files and set the platform
    topology, coordinate = getInputFiles(topologyFile=arg.tf, coordinateFile=arg.cf,  fileType=arg.ff)
    simulationType = arg.ty
    jobName = arg.jn
    mdSteps = arg.ns
    targetTemperature = arg.st
    inputType = arg.ff
    platform = arg.pt
    numberOfGPUs = arg.ng
    minimizationSteps = arg.ms
    equilibrationSteps = arg.es
    logFrequency = arg.lo
    trajFrequency = arg.tr

    if re.match(platform, 'CUDA', re.IGNORECASE) and numberOfGPUs:
        platformProperties = {"Precision": "mixed", "DeviceIndex": ",".join([str(x) for x in range(numberOfGPUs)])}

    elif re.match(platform, 'CUDA', re.IGNORECASE) and numberOfGPUs is None:
        raise ValueError("Error >>> CUDA option is chosen without specifying number of GPUs. Set the '-ng' flag")

    else:
        platformProperties = {}

    if re.match("MD", simulationType, re.IGNORECASE):
        # Run simulation
        simpleMD(topology, coordinate, targetTemperature, mdSteps, inputType, platform, platformProperties, jobName,
                                                    minimizationSteps, equilibrationSteps, logFrequency, trajFrequency)

    elif re.match("REMD", simulationType, re.IGNORECASE):
        # Run Replica exchange
        reMD(topology, coordinate, targetTemperature, mdSteps, inputType, platform, platformProperties, jobName)


if __name__ == '__main__':
    main()


