from netsquid_simulationtools.parameter_set import Parameter
from netsquid_nv.nv_parameter_set import NVParameterSet
import numpy as np

class SnVParameterSet2026COMPUTAEX(NVParameterSet):
    """
    Parámetros para centros SnV recogidos de distintos artículos científicos por Fundación COMPUTAEX
    """
    _REQUIRED_PARAMETERS = NVParameterSet._REQUIRED_PARAMETERS + [
        Parameter(name="magical_swap_gate_depolar_prob",
                  units=None,
                  perfect_value=0.,
                  type=float),
        Parameter(name="carbon_init_duration",
                  units="ns",
                  perfect_value=0.,
                  type=float),
        Parameter(name="carbon_z_rot_duration",
                  units="ns",
                  perfect_value=0.,
                  type=float),
        Parameter(name="carbon_single_qubit_duration",
                  units="ns",
                  perfect_value=0.,
                  type=float),
        Parameter(name="electron_init_duration",
                  units="ns",
                  perfect_value=0.,
                  type=float),
        Parameter(name="electron_single_qubit_duration",
                  units="ns",
                  perfect_value=0.,
                  type=float),
        Parameter(name="ec_two_qubit_gate_duration",
                  units="ns",
                  perfect_value=0.,
                  type=float),
        Parameter(name="measure_duration",
                  units="ns",
                  perfect_value=0.,
                  type=float),
        Parameter(name="magical_swap_gate_duration",
                  units="ns",
                  perfect_value=0.,
                  type=float),
        Parameter(name="c",
                  units="km/s",
                  perfect_value=300000.,
                  type=float),
        Parameter(name="total_detection_eff",
                  units=None,
                  perfect_value=1.0,
                  type=float),
        Parameter(name="emission_fidelity",
                  units=None,
                  perfect_value=1.0,
                  type=float)
    ]
    c = 206753.41931034482

    time_window = 7. #Iuliano et al. (2026)

    visibility = 0.9 #El usual

    dark_count_rate = 1. #[Hz] Iuliano et al. (2026)
    prob_dark_count = 1. - np.exp(-1. * time_window * 10 ** (-9) * dark_count_rate)

    avg_electron_electron_phase_drift = 0.
    std_electron_electron_phase_drift = 5. * np.pi / 180.  # Menor que en centro NV por ausencia de dipolo [radians]

    p_double_exc = 0.03

    photon_emission_delay = 8.392*10**3 #Iuliano et al. (2026)
    emission_fidelity = 0.9 #Iuliano et al. (2026)

    tau_emissions = 6.67

    real_detection_eff = 0.8
    p_loss_lengths_with_conversion = 0.5
    p_loss_lengths_no_conversion = 5.
    p_zero_phonon = lambda cavity: 0.80 if cavity else 0.57 
    zero_phonon_prob = p_zero_phonon(cavity=False)
    length = 0.001  
    p_photon_not_lost = 10 ** (-1. * p_loss_lengths_no_conversion * length / 10.)
    collection_eff_in_case_no_cavity_no_conversion = 0.0497
    collection_eff_with_conversion = collection_eff_in_case_no_cavity_no_conversion * 0.3
    total_detection_eff = p_photon_not_lost * p_zero_phonon(cavity=False) * collection_eff_in_case_no_cavity_no_conversion * real_detection_eff
    
    prob_detect_excl_transmission_with_conversion_with_cavities = \
        p_zero_phonon(cavity=True) * collection_eff_with_conversion * real_detection_eff
    prob_detect_excl_transmission_no_conversion_no_cavities = \
        p_zero_phonon(cavity=False) * \
        collection_eff_in_case_no_cavity_no_conversion * \
        real_detection_eff
    
    delta_w_list = [377, 62, 77]  # in kHz / (2pi)
    tau_decay_list = [263, 837, 640]  # in ns

    coherent_phase = 0.
    p_fail_class_corr = 0.
    initial_nuclear_phase = 0.

    delta_w = 2 * np.pi * 77000. * 10 ** -9  # Original from QuTech, in rad/ns
    tau_decay = 163.  # in ns, see figure (2)
    product_tau_decay_delta_w = tau_decay * delta_w 

    electron_T1 = 200.0*10**9 # Rosenthal et al. (2023)
    electron_T2 = 10.0*10**6 #Rosenthal et al.(2023)
    carbon_T1 = 10.0**9
    carbon_T2 = 17.2*10**6 #Beukers, H (2025)

    prob_error_0 = 0.001
    prob_error_1 = 0.01  
    electron_init_depolar_prob = 0.
    electron_single_qubit_depolar_prob = (4./3.) * (1. - 0.9951)
    carbon_init_depolar_prob = 0.
    carbon_z_rot_depolar_prob = (4./3.) * (1. - 1.)
    ec_gate_depolar_prob = (4./3.) * (1. - 0.8740)
    magical_swap_gate_depolar_prob = (4./3.) * (1. - 0.987)

    carbon_init_duration = 310E3
    carbon_z_rot_duration = 100.0*10**3
    carbon_single_qubit_duration = 3.0*10**3
    electron_init_duration = 200.0*10**6
    electron_single_qubit_duration = 48.4
    ec_two_qubit_gate_duration = 700.0*10**3
    measure_duration = 30.0*10**6
    magical_swap_gate_duration = 1.000010E6