'''
This module defines the NVProcessor2026 and SnVProcessor2026 classes, which are quantum processors for NV centers and SnV centers, respectively, 
based on parameters collected by Fundación COMPUTAEX from various scientific articles.
'''

import numpy as np
from netsquid.components.qprocessor import QuantumProcessor, PhysicalInstruction
from netsquid.components.models import DepolarNoiseModel, T1T2NoiseModel
from netsquid.components.instructions import INSTR_X, INSTR_Y, INSTR_Z, INSTR_ROT_X, INSTR_ROT_Y, INSTR_ROT_Z, INSTR_H,\
    INSTR_MEASURE, INSTR_SWAP, INSTR_INIT, INSTR_CXDIR, INSTR_EMIT
from netsquid.qubits.operators import Operator
from nv_2026 import NVParameterSet2026COMPUTAEX
from snv_2026 import SnVParameterSet2026COMPUTAEX

class NVProcessor2026(QuantumProcessor):
    
    def __init__(self, num_positions, noiseless=False, **properties):
        default_parameter_set = NVParameterSet2026COMPUTAEX()
        params = default_parameter_set.to_dict() if not noiseless else default_parameter_set.to_perfect_dict()
        params["use_magical_swap"] = True
        params.update(properties)
        for property, value in params.items():
            self.add_property(name=property, value=value, mutable=False)
        self.emission_duration = self.properties["photon_emission_delay"]
        self.electron_position = 0
        super().__init__(name="nv_center_quantum_processor",
                         num_positions=num_positions + 1,
                         mem_noise_models=self._define_memory_noise_models(num_positions))
        self._set_physical_instructions()

    def _define_memory_noise_models(self, num_positions):
        """Defines noise models for the quantum processor's memory positions. We add no memory noise model to the
        emission position.
        """
        electron_qubit_noise = T1T2NoiseModel(T1=self.properties["electron_T1"], T2=self.properties["electron_T2"])
        carbon_qubit_noise = T1T2NoiseModel(T1=self.properties["carbon_T1"], T2=self.properties["carbon_T2"])

        self.electron_position = 0  
        self.carbon_positions = [pos + 1 for pos in range(num_positions - 1)]
        self.emission_position = num_positions
        mem_noise_models = [electron_qubit_noise] + \
                           [carbon_qubit_noise] * len(self.carbon_positions) + [None]
        return mem_noise_models
    
    def _define_instruction_noise_models(self):
        """Defines noise models for the quantum processor's instructions.
        """
        electron_init_noise = \
            DepolarNoiseModel(depolar_rate=self.properties["electron_init_depolar_prob"],
                              time_independent=True)

        electron_single_qubit_noise = \
            DepolarNoiseModel(depolar_rate=self.properties["electron_single_qubit_depolar_prob"],
                              time_independent=True)

        carbon_init_noise = \
            DepolarNoiseModel(depolar_rate=self.properties["carbon_init_depolar_prob"],
                              time_independent=True)

        carbon_z_rot_noise = \
            DepolarNoiseModel(depolar_rate=self.properties["carbon_z_rot_depolar_prob"],
                              time_independent=True)

        ec_noise = \
            DepolarNoiseModel(depolar_rate=self.properties["ec_gate_depolar_prob"],
                              time_independent=True)

        magic_swap_noise = DepolarNoiseModel(depolar_rate=self.properties["magical_swap_gate_depolar_prob"],
                                             time_independent=True)

        self.models["electron_init_noise"] = electron_init_noise
        self.models["electron_single_qubit_noise"] = electron_single_qubit_noise
        self.models["carbon_init_noise"] = carbon_init_noise
        self.models["carbon_z_rot_noise"] = carbon_z_rot_noise
        self.models["ec_noise"] = ec_noise
        self.models["magic_swap_noise"] = magic_swap_noise

    def _set_physical_instructions(self):
        """Initializes the quantum processor's instructions.
        """
        self._define_instruction_noise_models()
        phys_instructions = []

        phys_instructions.append(
            PhysicalInstruction(INSTR_INIT,
                                parallel=False,
                                topology=self.carbon_positions,
                                q_noise_model=self.models["carbon_init_noise"],
                                apply_q_noise_after=True,
                                duration=self.properties["carbon_init_duration"]))

        phys_instructions.append(
            PhysicalInstruction(INSTR_ROT_Z,
                                parallel=False,
                                topology=self.carbon_positions,
                                q_noise_model=self.models["carbon_z_rot_noise"],
                                apply_q_noise_after=True,
                                duration=self.properties["carbon_z_rot_duration"]))
        
        phys_instructions.append(
            PhysicalInstruction(INSTR_H,
                                parallel=False,
                                topology=self.carbon_positions,
                                q_noise_model=self.models["carbon_z_rot_noise"],
                                apply_q_noise_after=True,
                                duration=self.properties["carbon_single_qubit_duration"]))
        
        phys_instructions.append(
            PhysicalInstruction(INSTR_X,
                                parallel=False,
                                topology=self.carbon_positions,
                                q_noise_model=self.models["carbon_z_rot_noise"],
                                apply_q_noise_after=True,
                                duration=self.properties["carbon_single_qubit_duration"]))
        
        phys_instructions.append(
            PhysicalInstruction(INSTR_ROT_X,                                                               # <-------------------------------
                                parallel=False,
                                topology=self.carbon_positions,
                                q_noise_model=self.models["carbon_z_rot_noise"],
                                apply_q_noise_after=True,
                                duration=self.properties["carbon_single_qubit_duration"]))

        phys_instructions.append(
            PhysicalInstruction(INSTR_INIT,
                                parallel=False,
                                topology=[self.electron_position],
                                q_noise_model=self.models["electron_init_noise"],
                                apply_q_noise_after=True,
                                duration=self.properties["electron_init_duration"]))

        for instr in [INSTR_X, INSTR_Y, INSTR_Z, INSTR_ROT_X, INSTR_ROT_Y, INSTR_ROT_Z, INSTR_H]:
            phys_instructions.append(
                PhysicalInstruction(instr,
                                    parallel=False,
                                    topology=[self.electron_position],
                                    q_noise_model=self.models["electron_single_qubit_noise"],
                                    duration=self.properties["electron_single_qubit_duration"]))
        emit_topology = [(self.electron_position, self.emission_position)]

        phys_instructions.append(
            PhysicalInstruction(INSTR_EMIT,
                                parallel=False,
                                topology=emit_topology,
                                duration=self.emission_duration))

        electron_carbon_topologies = \
            [(self.electron_position, carbon_pos) for carbon_pos in self.carbon_positions]
        phys_instructions.append(
            PhysicalInstruction(INSTR_CXDIR,
                                parallel=False,
                                topology=electron_carbon_topologies,
                                q_noise_model=self.models["ec_noise"],
                                apply_q_noise_after=True,            
                                duration=self.properties["ec_two_qubit_gate_duration"]))

        M0 = Operator("M0",
                      np.diag([np.sqrt(1 - self.properties["prob_error_0"]), np.sqrt(self.properties["prob_error_1"])]))
        M1 = Operator("M1",
                      np.diag([np.sqrt(self.properties["prob_error_0"]), np.sqrt(1 - self.properties["prob_error_1"])]))

        phys_instr_measure = PhysicalInstruction(INSTR_MEASURE,
                                                 parallel=False,
                                                 topology=[self.electron_position],
                                                 q_noise_model=None,
                                                 duration=self.properties["measure_duration"],
                                                 meas_operators=[M0, M1])

        phys_instructions.append(phys_instr_measure)

        if self.properties["use_magical_swap"]:
            phys_instructions.append(
                PhysicalInstruction(INSTR_SWAP,
                                    parallel=False,
                                    topology=electron_carbon_topologies,
                                    q_noise_model=self.models["magic_swap_noise"],
                                    apply_q_noise_after=True,
                                    duration=self.properties["magical_swap_gate_duration"]))

        for instruction in phys_instructions:
            self.add_physical_instruction(instruction)

    

class SnVProcessor2026(QuantumProcessor):
    
    def __init__(self, num_positions, noiseless=False, **properties):
        default_parameter_set = SnVParameterSet2026COMPUTAEX()
        params = default_parameter_set.to_dict() if not noiseless else default_parameter_set.to_perfect_dict()
        params["use_magical_swap"] = True
        params.update(properties)
        for property, value in params.items():
            self.add_property(name=property, value=value, mutable=False)
        self.emission_duration = self.properties["photon_emission_delay"]
        self.electron_position = 0
        super().__init__(name="snv_center_quantum_processor",
                         num_positions=num_positions + 1,
                         mem_noise_models=self._define_memory_noise_models(num_positions))
        self._set_physical_instructions()

    def _define_memory_noise_models(self, num_positions):
        """Defines noise models for the quantum processor's memory positions. We add no memory noise model to the
        emission position.
        """
        electron_qubit_noise = T1T2NoiseModel(T1=self.properties["electron_T1"], T2=self.properties["electron_T2"])
        carbon_qubit_noise = T1T2NoiseModel(T1=self.properties["carbon_T1"], T2=self.properties["carbon_T2"])

        self.electron_position = 0  
        self.carbon_positions = [pos + 1 for pos in range(num_positions - 1)]
        self.emission_position = num_positions
        mem_noise_models = [electron_qubit_noise] + \
                           [carbon_qubit_noise] * len(self.carbon_positions) + [None]
        return mem_noise_models
    
    def _define_instruction_noise_models(self):
        """Defines noise models for the quantum processor's instructions.
        """
        electron_init_noise = \
            DepolarNoiseModel(depolar_rate=self.properties["electron_init_depolar_prob"],
                              time_independent=True)

        electron_single_qubit_noise = \
            DepolarNoiseModel(depolar_rate=self.properties["electron_single_qubit_depolar_prob"],
                              time_independent=True)

        carbon_init_noise = \
            DepolarNoiseModel(depolar_rate=self.properties["carbon_init_depolar_prob"],
                              time_independent=True)

        carbon_z_rot_noise = \
            DepolarNoiseModel(depolar_rate=self.properties["carbon_z_rot_depolar_prob"],
                              time_independent=True)

        ec_noise = \
            DepolarNoiseModel(depolar_rate=self.properties["ec_gate_depolar_prob"],
                              time_independent=True)

        magic_swap_noise = DepolarNoiseModel(depolar_rate=self.properties["magical_swap_gate_depolar_prob"],
                                             time_independent=True)

        self.models["electron_init_noise"] = electron_init_noise
        self.models["electron_single_qubit_noise"] = electron_single_qubit_noise
        self.models["carbon_init_noise"] = carbon_init_noise
        self.models["carbon_z_rot_noise"] = carbon_z_rot_noise
        self.models["ec_noise"] = ec_noise
        self.models["magic_swap_noise"] = magic_swap_noise

    def _set_physical_instructions(self):
        """Initializes the quantum processor's instructions.
        """
        self._define_instruction_noise_models()
        phys_instructions = []

        phys_instructions.append(
            PhysicalInstruction(INSTR_INIT,
                                parallel=False,
                                topology=self.carbon_positions,
                                q_noise_model=self.models["carbon_init_noise"],
                                apply_q_noise_after=True,
                                duration=self.properties["carbon_init_duration"]))

        phys_instructions.append(
            PhysicalInstruction(INSTR_ROT_Z,
                                parallel=False,
                                topology=self.carbon_positions,
                                q_noise_model=self.models["carbon_z_rot_noise"],
                                apply_q_noise_after=True,
                                duration=self.properties["carbon_z_rot_duration"]))

        phys_instructions.append(                                                                      # ADDED BY COMPUTAEX AND BASED IN ZHANG ET AL. (2025). 
            PhysicalInstruction(INSTR_H,                                                               # <------------------------------
                                parallel=False,
                                topology=self.carbon_positions,
                                q_noise_model=self.models["carbon_z_rot_noise"],
                                apply_q_noise_after=True,
                                duration=self.properties["carbon_single_qubit_duration"]))
        
        phys_instructions.append(
            PhysicalInstruction(INSTR_X,                                                               # <-------------------------------
                                parallel=False,
                                topology=self.carbon_positions,
                                q_noise_model=self.models["carbon_z_rot_noise"],
                                apply_q_noise_after=True,
                                duration=self.properties["carbon_single_qubit_duration"]))
        
        phys_instructions.append(
            PhysicalInstruction(INSTR_ROT_X,                                                               # <-------------------------------
                                parallel=False,
                                topology=self.carbon_positions,
                                q_noise_model=self.models["carbon_z_rot_noise"],
                                apply_q_noise_after=True,
                                duration=self.properties["carbon_single_qubit_duration"]))

        phys_instructions.append(
            PhysicalInstruction(INSTR_INIT,
                                parallel=False,
                                topology=[self.electron_position],
                                q_noise_model=self.models["electron_init_noise"],
                                apply_q_noise_after=True,
                                duration=self.properties["electron_init_duration"]))

        for instr in [INSTR_X, INSTR_Y, INSTR_Z, INSTR_ROT_X, INSTR_ROT_Y, INSTR_ROT_Z, INSTR_H]:
            phys_instructions.append(
                PhysicalInstruction(instr,
                                    parallel=False,
                                    topology=[self.electron_position],
                                    q_noise_model=self.models["electron_single_qubit_noise"],
                                    duration=self.properties["electron_single_qubit_duration"]))
        emit_topology = [(self.electron_position, self.emission_position)]

        phys_instructions.append(
            PhysicalInstruction(INSTR_EMIT,
                                parallel=False,
                                topology=emit_topology,
                                duration=self.emission_duration))

        electron_carbon_topologies = \
            [(self.electron_position, carbon_pos) for carbon_pos in self.carbon_positions]
        phys_instructions.append(
            PhysicalInstruction(INSTR_CXDIR,
                                parallel=False,
                                topology=electron_carbon_topologies,
                                q_noise_model=self.models["ec_noise"],
                                apply_q_noise_after=True,
                                duration=self.properties["ec_two_qubit_gate_duration"]))

        M0 = Operator("M0",
                      np.diag([np.sqrt(1 - self.properties["prob_error_0"]), np.sqrt(self.properties["prob_error_1"])]))
        M1 = Operator("M1",
                      np.diag([np.sqrt(self.properties["prob_error_0"]), np.sqrt(1 - self.properties["prob_error_1"])]))

        phys_instr_measure = PhysicalInstruction(INSTR_MEASURE,
                                                 parallel=False,
                                                 topology=[self.electron_position],
                                                 q_noise_model=None,
                                                 duration=self.properties["measure_duration"],
                                                 meas_operators=[M0, M1])

        phys_instructions.append(phys_instr_measure)

        if self.properties["use_magical_swap"]:
            phys_instructions.append(
                PhysicalInstruction(INSTR_SWAP,
                                    parallel=False,
                                    topology=electron_carbon_topologies,
                                    q_noise_model=self.models["magic_swap_noise"],
                                    apply_q_noise_after=True,
                                    duration=self.properties["magical_swap_gate_duration"]))

        for instruction in phys_instructions:
            self.add_physical_instruction(instruction)

    

