import lalsimulation as lalsim
import lal
import json
import numpy as np
from tensorflow import keras
from bilby.gw.source import lal_binary_neutron_star


def convert_mass_to_geo(mass_SI):
    
    mass_geo = mass_SI/((lal.C_SI**2.)/lal.G_SI)
    #print(mass_geo)
    
    return mass_geo

def get_lambda_dimensionless(k2,R,M):
    """
    R and M should both be in units of km?
    """
    big_lambda = 2./3.*k2*((R/M)**5.)
    return big_lambda

def maximum_mass(
        log_p, Gamma_1, Gamma_2, Gamma_3):
    '''
    Parameters
    ----------
    log_p, Gamma_1, Gamma_2, Gamma_3 = float
        piecewise polytrope hyper-parameters in SI units
    
    Returns
    -------
    Maximum mass in solar masses
    '''
    polytrope = lalsim.SimNeutronStarEOS4ParameterPiecewisePolytrope(
        log_p-1, Gamma_1, Gamma_2, Gamma_3)
    polytrope_family = lalsim.CreateSimNeutronStarFamily(polytrope)
    max_mass = lalsim.SimNeutronStarMaximumMass(polytrope_family)/lal.MSUN_SI
    
    return max_mass

def maximum_speed_of_sound(
        log_p, Gamma_1, Gamma_2, Gamma_3):
    '''
    Parameters
    ----------
    log_p, Gamma_1, Gamma_2, Gamma_3 = piecewise polytrope
    hyper parameters
    
    Returns
    -------
    Maximum speed of sound divided by the speed of light
    '''
    polytrope = lalsim.SimNeutronStarEOS4ParameterPiecewisePolytrope(
        log_p-1, Gamma_1, Gamma_2, Gamma_3)
    max_enthalpy = lalsim.SimNeutronStarEOSMaxPseudoEnthalpy(polytrope)
    max_speed_of_sound = lalsim.SimNeutronStarEOSSpeedOfSound(
        max_enthalpy,polytrope)
    
    return max_speed_of_sound/lal.C_SI

def get_lambda(mass,eosfam):

    mass = mass * lal.MSUN_SI
    radius = lalsim.SimNeutronStarRadius(mass,eosfam)
    k2 = lalsim.SimNeutronStarLoveNumberK2(mass,eosfam)
    # Convert to geometrised units for compactness calculation
    mass_geo = convert_mass_to_geo(mass)
    # Calculate lambda 
    the_lambda = get_lambda_dimensionless(k2,radius,mass_geo)

    return the_lambda

def get_redshift(H_0,luminosity_distance):
    c_kms = 3.e5 
    return (H_0*luminosity_distance)/(c_kms)

def get_mass_source(mass,redshift):
    return mass/(1.+redshift)

# Scallings for the keras model 
def scale_val(val,mean,std):
    scaled_val = (val-mean)/std
    return scaled_val

# Read in json 
model_json ="../muTOV/u-TOV_high_mass.json"
json_file = open(model_json)
json_data = json.load(json_file)
model_file_path = json_data["model_filepath"]
mean_mass = json_data["mass_mean"]
std_mass = json_data["mass_std"]
mean_log_P1 = json_data["log_P1_mean"]
std_log_P1 = json_data["log_P1_std"]
mean_Gamma_1 = json_data["Gamma_1_mean"]
std_Gamma_1 = json_data["Gamma_1_std"]
mean_Gamma_2 = json_data["Gamma_2_mean"]
std_Gamma_2 = json_data["Gamma_2_std"]
mean_Gamma_3 = json_data["Gamma_3_mean"]
std_Gamma_3 = json_data["Gamma_3_std"]

mutov = keras.models.load_model(model_file_path) 

def mutov_lal_binary_neutron_star_H_0(
        frequency_array, mass_1, mass_2, luminosity_distance, a_1, tilt_1,
        phi_12, a_2, tilt_2, phi_jl, theta_jn, phase, lambda_1, lambda_2,log_P1,
        Gamma_1,Gamma_2,Gamma_3,H_0,
        **kwargs):

    """ A Binary Neutron Star waveform model using lalsimulation with EOS sampling
        peformed using muTOV  

    Parameters
    ==========
    frequency_array: array_like
        The frequencies at which we want to calculate the strain
    mass_1: float
        The mass of the heavier object in solar masses
    mass_2: float
        The mass of the lighter object in solar masses
    luminosity_distance: float
        The luminosity distance in megaparsec
    a_1: float
        Dimensionless primary spin magnitude
    tilt_1: float
        Primary tilt angle
    phi_12: float
        Azimuthal angle between the two component spins
    a_2: float
        Dimensionless secondary spin magnitude
    tilt_2: float
        Secondary tilt angle
    phi_jl: float
        Azimuthal angle between the total binary angular momentum and the
        orbital angular momentum
    theta_jn: float
        Orbital inclination
    phase: float
        The phase at reference frequency or peak amplitude (depends on waveform)
    lambda_1: float
        Dimensionless tidal deformability of mass_1
    lambda_2: float
        Dimensionless tidal deformability of mass_2
    log_P1: float
        The pressure at which the transition between Gamma_1 and Gamma_2 occurs

    Gamma_1: float
        Dimensionless 1st adibatic index of the polytropic equation of state
    Gamma_2: float
        Dimensionless 2nd adiabatic index of the polytropic equation of state
    Gamma_3: float 
        Dimensionless 3rd adiabatic index of the polytropic equation of state 
    H_0: float 
        Hubble's constant
    kwargs: dict
        Optional keyword arguments
        Supported arguments:

        - waveform_approximant
        - reference_frequency
        - minimum_frequency
        - maximum_frequency
        - catch_waveform_errors
        - pn_spin_order
        - pn_tidal_order
        - pn_phase_order
        - pn_amplitude_order
        - mode_array:
          Activate a specific mode array and evaluate the model using those
          modes only.  e.g. waveform_arguments =
          dict(waveform_approximant='IMRPhenomHM', mode_array=[[2,2],[2,-2])
          returns the 22 and 2-2 modes only of IMRPhenomHM.  You can only
          specify modes that are included in that particular model.  e.g.
          waveform_arguments = dict(waveform_approximant='IMRPhenomHM',
          mode_array=[[2,2],[2,-2],[5,5],[5,-5]]) is not allowed because the
          55 modes are not included in this model.  Be aware that some models
          only take positive modes and return the positive and the negative
          mode together, while others need to call both.  e.g.
          waveform_arguments = dict(waveform_approximant='IMRPhenomHM',
          mode_array=[[2,2],[4,-4]]) returns the 22 and 2-2 of IMRPhenomHM.
          However, waveform_arguments =
          dict(waveform_approximant='IMRPhenomXHM', mode_array=[[2,2],[4,-4]])
          returns the 22 and 4-4 of IMRPhenomXHM.

    Returns
    =======
    dict: A dictionary with the plus and cross polarisation strain modes
    """

    z = get_redshift(H_0,luminosity_distance)

    # Get the source frame masses 
    mass_1_source = get_mass_source(mass_1,z)
    mass_2_source  = get_mass_source(mass_2,z) 

    # Hack fix to inject the tidal deformability from the true, tabulated EOS
    # This is needed because pbilby doesn't allow a different fdsm for injections/sampling  
    # Should only happen during injections, or very, very rarely during sampling, if at all
    # Change these values to your injected EOS
    if (log_P1 == 34.384 and Gamma_1 == 3.005 and Gamma_2==2.988 and Gamma_3==2.851):
        # Create Sly EOSfam 
        # eos_tab = lalsim.SimNeutronStarEOSByName('SLY')
        # eosfam = lalsim.CreateSimNeutronStarFamily(eos_tab)
        eos = lalsim.SimNeutronStarEOS4ParameterPiecewisePolytrope(log_P1-1.,Gamma_1,Gamma_2,Gamma_3)
        eosfam = lalsim.CreateSimNeutronStarFamily(eos)
        # Call lalsim to solve TOV 
        lambda_1 = get_lambda(mass_1_source,eosfam)
        lambda_2 = get_lambda(mass_2_source,eosfam)

    # During regular sampling get TOV with u-TOV
    else:     
        # # Apply scalling to the values to be used in the NN
        mass_1_source_scaled = scale_val(mass_1_source,mean_mass,std_mass)
        mass_2_source_scaled = scale_val(mass_2_source,mean_mass,std_mass)
        log_P1_scaled = scale_val(log_P1,mean_log_P1,std_log_P1)
        Gamma_1_scaled = scale_val(Gamma_1,mean_Gamma_1,std_Gamma_1)
        Gamma_2_scaled = scale_val(Gamma_2,mean_Gamma_2,std_Gamma_2)
        Gamma_3_scaled = scale_val(Gamma_3,mean_Gamma_3,std_Gamma_3)
        # Call molotov to get lambda_1 from gamma's and P1 
        x = np.array([mass_1_source_scaled,log_P1_scaled,Gamma_1_scaled,Gamma_2_scaled,Gamma_3_scaled])
        x = np.expand_dims(x,0)
        lambda_1 = mutov(x)

        # #print(mass_1_source)
        # #print(mass_2_source)

        # The same for lambda_2 
        x = np.array([mass_2_source_scaled,log_P1_scaled,Gamma_1_scaled,Gamma_2_scaled,Gamma_3_scaled])
        x = np.expand_dims(x,0)
        lambda_2 = mutov(x)

        # Temporary bad fix
        if lambda_1 < 0.0:
            lambda_1 = 0.
        if lambda_2 < 0.0:
            lambda_2 = 0.


    return lal_binary_neutron_star(
        frequency_array, mass_1, mass_2, luminosity_distance, a_1, tilt_1,
        phi_12, a_2, tilt_2, phi_jl, theta_jn, phase, lambda_1, lambda_2,
        **kwargs
    )
