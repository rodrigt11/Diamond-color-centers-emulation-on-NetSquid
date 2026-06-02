'''
Given two nodes A and B, this protocol creates an entangled pair between them using a heralded connection and a magic distributor. 
It then measures the resulting state and prints the results, including the Bell state, the time taken for entanglement, and the number of attempts needed.
'''

import netsquid as ns
from netsquid.protocols import Protocol, Signals
from netsquid.nodes import Node
from netsquid.components import INSTR_X
import numpy as np
from netsquid_physlayer.heralded_connection import MiddleHeraldedConnection
from nv_2026 import NVParameterSet2026COMPUTAEX 
from netsquid_nv.magic_distributor import NVDoubleClickMagicDistributor
import netsquid.qubits.qubitapi as qapi
from nv_2026 import NVParameterSet2026COMPUTAEX
from snv_2026 import SnVParameterSet2026COMPUTAEX

class Entangler(Protocol):
    def __init__(self, A, B, params):
        if not isinstance(A,Node) or not isinstance(B,Node):
            raise TypeError('A y B deben ser objetos Node')
        if not isinstance(params,NVParameterSet2026COMPUTAEX) and not isinstance(params,SnVParameterSet2026COMPUTAEX):
            raise TypeError('Se debe dar un conjunto de parámetros')
        self.A = A
        self.B = B
        self.params = params
        self.add_signal('Entrelazado')
        self.add_signal(Signals.FINISHED)

    def _on_delivery(self, event):
        self.send_signal('Entrelazado')

    def run(self):
        print(f'[{ns.sim_time():.1f} ns ] Iniciando entrelazamiento')

        p = self.params.to_dict()
        conexion = MiddleHeraldedConnection(
            name="Conexion",
            length=1,
            p_loss_length=p["p_loss_lengths_with_conversion"],
            speed_of_light=p["c"],
            dark_count_probability=p["prob_dark_count"],
            detector_efficiency=p["total_detection_eff"],
            visibility=p["visibility"]
        )

        md = NVDoubleClickMagicDistributor(
            nodes=[self.A, self.B],
            heralded_connection=conexion,
            length_A=0.5,
            length_B=0.5,
            speed_of_light_A=p["c"],
            speed_of_light_B=p["c"],
            emission_duration_A=p["photon_emission_delay"],
            emission_duration_B=p["photon_emission_delay"],
            emission_fidelity_A=p["emission_fidelity"],
            emission_fidelity_B=p["emission_fidelity"],
            p_loss_length_A=p["p_loss_lengths_with_conversion"],
            p_loss_length_B=p["p_loss_lengths_with_conversion"],
            detector_efficiency=p["total_detection_eff"],
            dark_count_probability=p["prob_dark_count"],
            visibility=p["visibility"],
            tau_decay=NVParameterSet2026COMPUTAEX.tau_decay,
            delta_w=NVParameterSet2026COMPUTAEX.delta_w
        )

        md.add_callback(self._on_delivery)
        
        event = md.add_delivery(memory_positions={self.A.ID: 0, self.B.ID: 0}, coin_prob_ph_ph=1.0, coin_prob_ph_dc=1.0, coin_prob_dc_dc=1.0)

        yield self.await_signal(self,'Entrelazado')

        delivery = md.peek_delivery(event, allow_archive=True)
        cycle_time = p["photon_emission_delay"]+(0.5/p["c"])*1e9
        num_intentos = int(delivery.sample.delivery_duration/cycle_time)

        qubit_A = self.A.qmemory.peek(positions=[0])[0]
        qubit_B = self.B.qmemory.peek(positions=[0])[0]
        self.A.qmemory.execute_instruction(INSTR_X,[0])

        yield self.await_program(self.A.qmemory)

        print(qapi.reduced_dm([qubit_A, qubit_B]))

        print(f'Estado de Bell: {delivery.sample.label}')
        print(f'Tiempo de entrelazamiento: {delivery.sample.delivery_duration:.2f} ns')
        print(f'Número de intentos: {num_intentos}')




            