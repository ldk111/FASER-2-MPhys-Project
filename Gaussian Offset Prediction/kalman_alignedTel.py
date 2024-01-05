#!/usr/bin/env python3

from pathlib import Path
from typing import Optional
import random
import numpy as np
import concurrent.futures()
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
    )

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

def runACTS(index, offset_z, offset_y, field, digiConfigFile, trackingGeometry, inputParticlePath, rand_):

    detector_misal, trackingGeometry_misal, decorators_misal = acts.examples.AlignedTelescopeDetector.create(
            bounds=[500, 1500], positions=[10000, 10500, 11000, 19500, 20000, 20500], binValue=0, offsets=[offset_z, offset_y], thickness=4,rnd=rand_, 
            sigmaInPlane=0.0 , sigmaOutPlane=0.0 , sigmaOutRot=0.0 , sigmaInRot=0.00 
        )

    runTruthTrackingKalman(
        trackingGeometry=trackingGeometry,
        trackingGeometry_misal=trackingGeometry_misal,
        field=field,
        digiConfigFile=digiConfigFile,
        outputDir="output/gaussian_offset_" + index,
        # outputDir="./Output_ttk/Out9",
        inputParticlePath=inputParticlePath,
    ).run()

def main():

    srcdir = Path(__file__).resolve().parent.parent.parent.parent

    detector, trackingGeometry, decorators = acts.examples.TelescopeDetector.create(
        bounds=[500, 1500], positions=[10000, 10500, 11000, 19500, 20000, 20500], binValue=0,thickness=4,
    )

    rand_ = random.randint(0, 50000)

    offsets_y = np.loadtxt("offsets_y.csv", delimiter = ",")
    offsets_z = np.loadtxt("offsets_z.csv", delimiter = ",")

    digiConfigFile= "/data/atlassmallfiles/users/chri6112/Acts/acts/Examples/Algorithms/Digitization/share/default-smearing-config-telescope.json"

    inputParticlePath = Path("/home/chri6112/Downloads/nmuon_acts_sample_200k.root")

    field = acts.RestrictedBField(acts.Vector3(0* u.T, 0, 1.0 * u.T))

    input_args = list(zip(np.arange(0, len(offsets_y)), offsets_z, offsets_y, itertools.repeat(field), itertools.repeat(digiConfigFile), itertools.repeat(trackingGeometry), itertools.repeat(inputParticlePath), itertools.repeat(rand_)))
    
    with concurrent.futures.ProcessPoolExecutor(max_workers = 8) as executor:
        executor.map(runACTS, input_args)

if "__main__" == __name__:

    main()