import bilby 

import sys
import os

# Get the current file's directory
current_dir = os.path.dirname(os.path.abspath(__file__))

# Construct the path to the `code` directory
code_dir = os.path.join(current_dir, '..', 'mutov')

# Add the `code` directory to sys.path
sys.path.append(code_dir)

from source import mutov_lal_binary_neutron_star_H_0

"""
Tutorial to demonstrate running parameter estimation on a binary neutron star
system, sampling over pressure vs density space rather than tidal deformability
"""

# Specify the output directory and the name of the simulation.
outdir = "outdir"
label = "bns_example_muTOV"

# We are going to inject a binary neutron star waveform.
# Note that here we inject EOS parameters directly, but it is also 
# possible to inject tidal deformabilities
# We inject the Piecewise Polytrope representation of the SLy EOS (Read+,2009)
injection_parameters = dict(
    mass_1=1.5,
    mass_2=1.3,
    chi_1=0.02,
    chi_2=0.02,
    luminosity_distance=250.0,
    H_0=67.4,
    theta_jn=0.4,
    psi=2.659,
    phase=1.3,
    geocent_time=1126259642.413,
    ra=1.375,
    dec=-1.2108,
    log_P1 =34.384,
    Gamma_1 = 3.005,
    Gamma_2=2.988,
    Gamma_3=2.851
)

# Set the duration and sampling frequency of the data segment that we're going
duration = 32
sampling_frequency = 2048
start_time = injection_parameters["geocent_time"] + 2 - duration

# Fixed arguments passed into the source model. The analysis starts at 40 Hz.
waveform_arguments = dict(
    waveform_approximant="IMRPhenomPv2_NRTidal",
    reference_frequency=50.0,
    minimum_frequency=40.0,
)

# Create the waveform_generator using the muTOV frequency domain source model
waveform_generator = bilby.gw.WaveformGenerator(
    duration=duration,
    sampling_frequency=sampling_frequency,
    frequency_domain_source_model=mutov_lal_binary_neutron_star_H_0,
    parameter_conversion=bilby.gw.conversion.convert_to_lal_binary_neutron_star_parameters,
    waveform_arguments=waveform_arguments,
)

# Set up interferometers.  In this case we'll use three interferometers
# (LIGO-Hanford (H1), LIGO-Livingston (L1), and Virgo (V1)).
# These default to their design sensitivity and start at 40 Hz.
interferometers = bilby.gw.detector.InterferometerList(["H1", "L1", "V1"])
for interferometer in interferometers:
    interferometer.minimum_frequency = 40
interferometers.set_strain_data_from_power_spectral_densities(
    sampling_frequency=sampling_frequency, duration=duration, start_time=start_time
)
interferometers.inject_signal(
    parameters=injection_parameters, waveform_generator=waveform_generator
)


# Load the default prior for binary neutron stars.
# We are chosing to fix H_0 here to make the sampling more efficent
priors = bilby.gw.prior.BNSPriorDict()
for key in [
    "psi",
    "geocent_time",
    "ra",
    "dec",
    "chi_1",
    "chi_2",
    "theta_jn",
    "luminosity_distance",
    "phase",
    "H_0",
]:
    priors[key] = injection_parameters[key]
del priors["mass_ratio"], priors['lambda_1'], priors['lambda_2']
priors["chirp_mass"] = bilby.core.prior.Gaussian(
    1.215, 0.1, name="chirp_mass", unit="$M_{\\odot}$"
)
priors["symmetric_mass_ratio"] = bilby.core.prior.Uniform(
    0.1, 0.25, name="symmetric_mass_ratio"
)

# Set EOS priors
priors["Gamma_1"] = bilby.core.prior.Uniform(2.0,4.5,name="gamma_1")
priors["Gamma_2"] = bilby.core.prior.Uniform(1.5,4.5,name="gamma_2")
priors["Gamma_3"] = bilby.core.prior.Uniform(1.5,4.5,name="gamma_3")
priors["log_P1"] = bilby.core.prior.Uniform(33.5,34.8,name="log_P1") 

# Initialise the likelihood by passing in the interferometer data (IFOs)
# and the waveform generator
likelihood = bilby.gw.GravitationalWaveTransient(
    interferometers=interferometers,
    waveform_generator=waveform_generator,
)

# Run sampler.  In this case we're going to use the `nestle` sampler
result = bilby.run_sampler(
    likelihood=likelihood,
    priors=priors,
    sampler="nestle",
    npoints=100,
    injection_parameters=injection_parameters,
    outdir=outdir,
    label=label,
    conversion_function=bilby.gw.conversion.generate_all_bns_parameters,
)

result.plot_corner()