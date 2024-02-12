#!/usr/bin/env python3

from pathlib import Path
from typing import Optional
import random
import numpy as np
import concurrent.futures
import itertools

import acts
import acts.examples

u = acts.UnitConstants


def runTruthTrackingKalman(
    trackingGeometry: acts.TrackingGeometry,
    trackingGeometry_misal: acts.TrackingGeometry,
    field: acts.MagneticFieldProvider,
    outputDir: Path,
    digiConfigFile: Path,
    directNavigation=False,
    reverseFilteringMomThreshold=0 * u.GeV,
    s: acts.examples.Sequencer = None,
    inputParticlePath: Optional[Path] = None,
):
    from acts.examples.simulation import (
        addParticleGun,
        EtaConfig,
        MomentumConfig,
        PhiConfig,
        ParticleConfig,
        addFatras,
        addDigitization,
    )
    from acts.examples.reconstruction import (
        addSeeding,
        SeedingAlgorithm,
        TruthSeedRanges,
        addKalmanTracks,
        addVertexFitting,
        VertexFinder,
    )
#changed acts.logging.INFO to acts.logging.FATAL
    s = s or acts.examples.Sequencer(
        events=200000, numThreads=-1, logLevel=acts.logging.FATAL
    )

    rnd = acts.examples.RandomNumbers()
    rand_ = random.randint(0, 50000)
    outputDir = Path(outputDir)


    if inputParticlePath is None:
        addParticleGun(
            s,
            MomentumConfig(0 * u.GeV, 2500.0 * u.GeV),
            EtaConfig(-0.01, 0.01, uniform=True),
            PhiConfig(-0.06, 0.06 * u.degree),
            ParticleConfig(1, acts.PdgParticle.eMuon, True),
            vtxGen=acts.examples.GaussianVertexGenerator(
                stddev=acts.Vector4(
                    1300 * u.mm, 500 * u.mm, 200 * u.mm, 0 * u.ns
                ),
                mean=acts.Vector4(3750, 0, 0, 0),
            ),
            multiplicity=1,
            rnd=rnd,
            outputDirRoot=outputDir,
        )

    else:
        acts.logging.getLogger("Truth tracking example").info(
            "Reading particles from %s", inputParticlePath.resolve()
        )
        assert inputParticlePath.exists()
        s.addReader(
            acts.examples.RootParticleReader(
                level=acts.logging.INFO,
                filePath=str(inputParticlePath.resolve()),
                particleCollection="particles_input",
                orderedEvents=False,
            )
        )

    addFatras(
        s,
        trackingGeometry,
        field,
        rnd=rnd,
        # enableInteractions=True
    )

    addDigitization(
        s,
        trackingGeometry,
        field,
        digiConfigFile=digiConfigFile,
        rnd=rnd,
        #outputDirRoot=str(outputDir / "digitisation")
    )

    #s.addWriter(
    #    acts.examples.RootSimHitWriter(
    #        level=acts.logging.INFO,
    #        inputSimHits="simhits",
    #        filePath=str(outputDir / "simhits.root"),
    #        
    #    )
    #)

    #added to try and read in partway through
    #s.addReader(
    #        acts.examples.RootSimHitReader(
    #            filePath=str(outputDir / "simhits.root"),
    #            treeName="hits",
    #            simHitCollection="simhits",
    #        )
    #    )

    addSeeding(
        s,
        trackingGeometry_misal,
        field,
        seedingAlgorithm=SeedingAlgorithm.TruthSmeared,
        rnd=rnd,
        truthSeedRanges=TruthSeedRanges(
            pt=(0 * u.MeV, None),
            nHits=(5, None),
            eta=(-0.5,0.5)
        ),
        #outputDirRoot=str(outputDir / "seeding")
    )

    addKalmanTracks(
        s,
        trackingGeometry_misal,
        field,
        directNavigation,
        reverseFilteringMomThreshold,
        energyLoss= True,
        multipleScattering= True,
    )

    #addVertexFitting(
     #   s,
     #   field,
     #   vertexFinder=VertexFinder.Truth,
     #   outputDirRoot=outputDir,
    #)


    #Changed all acts.logging.INFO to acts.logging.FATAL
    #Output
    s.addWriter(
        acts.examples.RootTrajectoryStatesWriter(
            level=acts.logging.FATAL,
            inputTrajectories="trajectories",
            inputParticles="truth_seeds_selected",
            inputSimHits="simhits",
            inputMeasurementParticlesMap="measurement_particles_map",
            inputMeasurementSimHitsMap="measurement_simhits_map",
            filePath=str(outputDir / "trackstates_fitter.root"),
        )
    )

    s.addWriter(
        acts.examples.RootTrajectorySummaryWriter(
            level=acts.logging.FATAL,
            inputTrajectories="trajectories",
            inputParticles="truth_seeds_selected",
            inputMeasurementParticlesMap="measurement_particles_map",
            filePath=str(outputDir / "tracksummary_fitter.root"),
        )
    )

    #s.addWriter(
    #    acts.examples.TrackFinderPerformanceWriter(
    #        level=acts.logging.FATAL,
    #        inputProtoTracks="sorted_truth_particle_tracks"
    #        if directNavigation
    #        else "truth_particle_tracks",
    #        inputParticles="truth_seeds_selected",
    #        inputMeasurementParticlesMap="measurement_particles_map",
    #        filePath=str(outputDir / "performance_track_finder.root"),
    #    )
    #)
#
    #s.addWriter(
    #    acts.examples.TrackFitterPerformanceWriter(
    #        level=acts.logging.FATAL,
    #        inputTrajectories="trajectories",
    #        inputParticles="truth_seeds_selected",
    #        inputMeasurementParticlesMap="measurement_particles_map",
    #        filePath=str(outputDir / "performance_track_fitter.root"),
    #    )
    #)
#
    return s

def runACTS(args):

    offsetIndex, inputIndex, offset_x, offset_y, offset_z, rotation_x, rotation_y, rotation_z, field, digiConfigFile, trackingGeometry, inputPath, rand_, outputPath = args

    detector_misal, trackingGeometry_misal, decorators_misal = acts.examples.AlignedTelescopeDetector.create(
            bounds=[500, 1500], positions=[10000, 10500, 11000, 19500, 20000, 20500], binValue=0,
            offsets=[offset_z, offset_y, offset_x], rotations=[rotation_z, rotation_y, rotation_x],
            thickness=4,rnd=rand_, 
            sigmaInPlane=0.0 , sigmaOutPlane=0.0 , sigmaOutRot=0.0 , sigmaInRot=0.00 
        )

    inputPath = Path(inputPath + str(inputIndex) + ".root")

    runTruthTrackingKalman(
        trackingGeometry = trackingGeometry_misal,
        trackingGeometry_misal = trackingGeometry,
        field = field,
        digiConfigFile = digiConfigFile,
        outputDir = outputPath + str(offsetIndex) + "/" + str(inputIndex),
        inputParticlePath = inputPath,
    ).run()

def runACTSreprop(args):

    offsetIndex, field, digiConfigFile, trackingGeometry, rand_, outputPath = args
    
    inputPath = Path(outputPath + str(offsetIndex) + "/repropagation.root")
    
    runTruthTrackingKalman(
        trackingGeometry = trackingGeometry,
        trackingGeometry_misal = trackingGeometry,
        field = field,
        digiConfigFile = digiConfigFile,
        outputDir = outputPath + str(offsetIndex) + "/",
        inputParticlePath = inputPath,
    ).run()

def main():

    srcdir = Path(__file__).resolve().parent.parent.parent.parent

    detector, trackingGeometry, decorators = acts.examples.TelescopeDetector.create(
        bounds=[500, 1500], positions=[10000, 10500, 11000, 19500, 20000, 20500], binValue=0,thickness=4,
    )

    rand_ = random.randint(0, 50000)

    #inputParticlePath = Path("/home/chri6112/Downloads/nmuon_acts_sample_200k_0.root")
    outputPath = "output/100224_10_200k@200k_yz_13_6/"
    
    digiConfigFile= "/data/atlassmallfiles/users/chri6112/Acts/acts/Examples/Algorithms/Digitization/share/default-smearing-config-telescope.json"

    offsets_x = np.loadtxt(outputPath + "offsets_x.csv", delimiter = ",")
    offsets_y = np.loadtxt(outputPath + "offsets_y.csv", delimiter = ",")
    offsets_z = np.loadtxt(outputPath + "offsets_z.csv", delimiter = ",")

    rotations_x = np.loadtxt(outputPath + "rotations_x.csv", delimiter=",")
    rotations_y = np.loadtxt(outputPath + "rotations_y.csv", delimiter=",")
    rotations_z = np.loadtxt(outputPath + "rotations_z.csv", delimiter=",")
  
    #outputPath = outputPath + "0/reprop/"
    #inputParticlePath = Path("output/040224_1_500k@200k/0/reprop/repropagation.root")

    field = acts.RestrictedBField(acts.Vector3(0* u.T, 0, 1.0 * u.T))# 0,0,1

    n_offsets = 10
    n_inputs = 13

    inputParticlePath = "/data/atlassmallfiles/users/chri6112/Muon Flux Data/Run Data/muon_filter_200k_"

    offsetIndex = np.repeat(np.arange(0, n_offsets), n_inputs)
    inputIndex = np.tile(np.arange(0,n_inputs), n_offsets)

    reprop = False

    if reprop == False:

        input_args = list(zip(offsetIndex, inputIndex, np.repeat(offsets_x, n_inputs, axis = 0), np.repeat(offsets_y, n_inputs, axis = 0), np.repeat(-offsets_z, n_inputs, axis = 0), np.repeat(rotations_x, n_inputs, axis = 0), np.repeat(rotations_y, n_inputs, axis = 0), np.repeat(-rotations_z, n_inputs, axis = 0), itertools.repeat(field), itertools.repeat(digiConfigFile), itertools.repeat(trackingGeometry), itertools.repeat(inputParticlePath), itertools.repeat(rand_), itertools.repeat(outputPath)))
        print("Y")
        with concurrent.futures.ThreadPoolExecutor(max_workers = 100) as executor:
            executor.map(runACTS, input_args)

    else:
        increment = 40
        input_args = list(zip(np.arange(0 + increment, n_offsets + increment), itertools.repeat(field), itertools.repeat(digiConfigFile), itertools.repeat(trackingGeometry), itertools.repeat(rand_), itertools.repeat(outputPath)))
        with concurrent.futures.ThreadPoolExecutor(max_workers = 50) as executor:
            executor.map(runACTSreprop, input_args)


if "__main__" == __name__:

    main()