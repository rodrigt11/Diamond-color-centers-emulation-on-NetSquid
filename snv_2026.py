from netsquid_simulationtools.parameter_set import Parameter
from netsquid_nv.nv_parameter_set import NVParameterSet
import numpy as np
from pathlib import Path
from parameter_loader import load_parameter_set

class SnVParameterSet2026COMPUTAEX(NVParameterSet):
    """
    Experimental values for parameters from recent papers
    """

    EXCEL_PATH = Path(__file__).resolve().parent / "parameters_NV_SnV_code_ready.xlsx"
    CENTRE = "SnV"
    
    REQUIRED_EXCEL_KEYS = (
            'avg_electron_electron_phase_drift',
            'c',
            'carbon_T1',
            'carbon_T2',
            'carbon_init_depolar_prob',
            'carbon_init_duration',
            'carbon_single_qubit_duration',
            'carbon_z_rot_depolar_prob',
            'carbon_z_rot_duration',
            'coherent_phase',
            'collection_eff_in_case_no_cavity_no_conversion',
            'collection_eff_with_conversion',
            'dark_count_rate',
            'delta_w',
            'delta_w_list',
            'ec_gate_depolar_prob',
            'ec_two_qubit_gate_duration',
            'electron_T1',
            'electron_T2',
            'electron_init_depolar_prob',
            'electron_init_duration',
            'electron_single_qubit_depolar_prob',
            'electron_single_qubit_duration',
            'emission_fidelity',
            'initial_nuclear_phase',
            'length',
            'magical_swap_gate_depolar_prob',
            'magical_swap_gate_duration',
            'measure_duration',
            'p_double_exc',
            'p_fail_class_corr',
            'p_loss_lengths_no_conversion',
            'p_loss_lengths_with_conversion',
            'p_zero_phonon',
            'photon_emission_delay',
            'prob_error_0',
            'prob_error_1',
            'real_detection_eff',
            'std_electron_electron_phase_drift',
            'tau_decay',
            'tau_decay_list',
            'tau_emissions',
            'time_window',
            'visibility',
            )

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
    excel = load_parameter_set(EXCEL_PATH, CENTRE, required_keys=REQUIRED_EXCEL_KEYS)
        
    c = float(excel["c"])
    time_window = float(excel["time_window"])
    visibility = float(excel["visibility"])
    dark_count_rate = float(excel["dark_count_rate"])
    prob_dark_count = 1. - np.exp(-1. * time_window * 10 ** (-9) * dark_count_rate)
    
    avg_electron_electron_phase_drift = float(excel["avg_electron_electron_phase_drift"])
    std_electron_electron_phase_drift = float(excel["std_electron_electron_phase_drift"])
    p_double_exc = float(excel["p_double_exc"])
    photon_emission_delay = float(excel["photon_emission_delay"])        
    emission_fidelity = float(excel["emission_fidelity"])
    
    tau_emissions = staticmethod(lambda cavity, values=excel["tau_emissions"]: values[0] if cavity else values[1])
    
    real_detection_eff = float(excel["real_detection_eff"])
    p_loss_lengths_with_conversion = float(excel["p_loss_lengths_with_conversion"])
    p_loss_lengths_no_conversion = float(excel["p_loss_lengths_no_conversion"])
    p_zero_phonon = staticmethod(lambda cavity, values=excel["p_zero_phonon"]: values[0] if cavity else values[1])
    zero_phonon_prob = p_zero_phonon(cavity=False)
    length = float(excel["length"])
    p_photon_not_lost = 10 ** (-1. * p_loss_lengths_no_conversion * length / 10.)
    collection_eff_in_case_no_cavity_no_conversion = float(excel["collection_eff_in_case_no_cavity_no_conversion"])
    collection_eff_with_conversion = collection_eff_in_case_no_cavity_no_conversion * 0.3
    total_detection_eff = p_photon_not_lost * p_zero_phonon(cavity=False) * real_detection_eff * collection_eff_in_case_no_cavity_no_conversion
        
    prob_detect_excl_transmission_with_conversion_with_cavities = \
        p_zero_phonon(cavity=True) * collection_eff_with_conversion * real_detection_eff
    prob_detect_excl_transmission_no_conversion_no_cavities = \
        p_zero_phonon(cavity=False) * \
        collection_eff_in_case_no_cavity_no_conversion * \
        real_detection_eff
                
    delta_w_list = excel["delta_w_list"]
    tau_decay_list = excel["tau_decay_list"]
    coherent_phase = float(excel["coherent_phase"])
    p_fail_class_corr = float(excel["p_fail_class_corr"])
    initial_nuclear_phase = float(excel["initial_nuclear_phase"])
    delta_w = float(excel["delta_w"])
    tau_decay = float(excel["tau_decay"])
    product_tau_decay_delta_w = tau_decay * delta_w 
    
    electron_T1 = float(excel["electron_T1"])
    electron_T2 = float(excel["electron_T2"])
    carbon_T1 = float(excel["carbon_T1"])
    carbon_T2 = float(excel["carbon_T2"])
    
    prob_error_0 = float(excel["prob_error_0"])
    prob_error_1 = float(excel["prob_error_1"]) 
    electron_init_depolar_prob = (4.0 / 3.0) * (1.0 - float(excel["electron_init_depolar_prob"]))
    electron_single_qubit_depolar_prob = (4.0 / 3.0) * (1.0 - float(excel["electron_single_qubit_depolar_prob"]))
    carbon_init_depolar_prob = (4.0 / 3.0) * (1.0 - float(excel["carbon_init_depolar_prob"]))
    carbon_z_rot_depolar_prob = (4.0 / 3.0) * (1.0 - float(excel["carbon_z_rot_depolar_prob"]))
    ec_gate_depolar_prob = (4.0 / 3.0) * (1.0 - float(excel["ec_gate_depolar_prob"]))
    magical_swap_gate_depolar_prob = (4.0 / 3.0) * (1.0 - float(excel["magical_swap_gate_depolar_prob"]))
    
    carbon_init_duration = float(excel["carbon_init_duration"])
    carbon_z_rot_duration = float(excel["carbon_z_rot_duration"])
    carbon_single_qubit_duration = float(excel["carbon_single_qubit_duration"])
    electron_init_duration = float(excel["electron_init_duration"])
    electron_single_qubit_duration = float(excel["electron_single_qubit_duration"])
    ec_two_qubit_gate_duration = float(excel["ec_two_qubit_gate_duration"])
    measure_duration = float(excel["measure_duration"])
    magical_swap_gate_duration = float(excel["magical_swap_gate_duration"])