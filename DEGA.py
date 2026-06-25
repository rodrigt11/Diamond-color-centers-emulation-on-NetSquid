# We follow the steps of Zhou, X., Xu, X., Zheng, S., & Luo, L. (2026). Distributed exact generalized Grover’s algorithm. Frontiers of Computer Science, 20(7), 2007905.
# We implement DEGA for 4 qubits, across 2 NV-nodes with 1 communication qubit and 
from netsquid.nodes import Node
from entrelazamiento import Entangler
from nv_2026 import NVParameterSet2026COMPUTAEX
from procesadores import NVProcessor2026, SnVProcessor2026
from netsquid.components import INSTR_SWAP, QuantumProgram, INSTR_H, INSTR_INIT, INSTR_X, INSTR_CXDIR, INSTR_ROT_X
from primitives import Initialization, FlipBit, Hadamards, MultiCZ, Measure
import netsquid as ns
import numpy as np

ns.qubits.qformalism.set_qstate_formalism(ns.qubits.QFormalism.DM)


class Oracle(QuantumProgram):
    def __init__(self, marked_state):
        if not isinstance(marked_state, str):
            raise ValueError("marked_state must be a string")
        if len(marked_state) not in (2,3):
            raise ValueError("Local target string must be 2 or 3 bits long")
        
        self.marked_state = marked_state
        super().__init__()

    def _build(self):
        n = len(self.marked_state)

        data_qubits = list(range(1, n+1))

        flip_qubits = [i+1 for i, bit in enumerate(self.marked_state) if bit=="0"]

        prog = None

        if flip_qubits:
            prog = FlipBit(flip_qubits)

        if n == 2:
            phase_gate = MultiCZ(data_qubits[:-1], data_qubits[-1])
        elif n == 3:
            pass

        prog = phase_gate if prog is None else prog + phase_gate

        if flip_qubits:
            prog = prog + FlipBit(flip_qubits)

        return prog
    
    def program(self):
        yield from self.load(self._build())

class Diffusion(QuantumProgram):
    def __init__(self, marked_state_length):
        self.marked_state_length = marked_state_length
        super().__init__()

    def _build(self):
        data_qubits = list(range(1, self.marked_state_length + 1))

        h_layer_1 = Hadamards(data_qubits)

        # R_0: marca |0...0>, reutilizando Oracle
        r0 = Oracle(marked_state="0" * self.marked_state_length)

        # H^{⊗n} final
        h_layer_2 = Hadamards(data_qubits)

        return h_layer_1 + r0 + h_layer_2

    def program(self):
        yield from self.load(self._build())

class Grover(QuantumProgram):

    def __init__(self, marked_state):
        self.marked_state = marked_state
        super().__init__()

    def _build(self):
        return Oracle(self.marked_state) + Diffusion(len(self.marked_state))
    
    def program(self):
        yield from self.load(self._build())

class Long(QuantumProgram):

    def __init__(self, marked_state):
        self.marked_state = marked_state
        super().__init__()

    def _build(self):
        return Oracle(self.marked_state) + Diffusion(len(self.marked_state))
    
    def program(self):
        yield from self.load(self._build())

class DEGA(QuantumProgram):
    
    def __init__(self, marked_state):
        self.marked_state = marked_state
        super().__init__()

    def _build(self):
        measures = None
        for i in range(1, len(self.marked_state)+1):
            m = Measure(i,f"tau_{i}")
            measures = m if measures is None else measures + m
        return Initialization(len(self.marked_state) + 1) + Hadamards(list(range(1, len(self.marked_state) + 1))) + Grover(self.marked_state) + measures

    def program(self):
        yield from self.load(self._build())

class DEGAMasterProtocol(ns.protocols.protocol.Protocol):
    def __init__(self, marked_state, processor_type):
        if not isinstance(marked_state, str):
            raise ValueError("Marked state must be a string")
        if not processor_type == NVProcessor2026 and not processor_type == SnVProcessor2026:
            raise ValueError("Processor type must be a colour center")
        self.marked_state = marked_state
        self.processor_type = processor_type
        super().__init__()

    def run(self):

        n_nodes = len(self.marked_state) // 2

        node_list = []
        divided_state = []
        if len(self.marked_state) % 2 == 0:
            for i in range(n_nodes):
                node_list.append(Node(name=f"Node {i}", qmemory=self.processor_type(3)))

            for i in range(0, len(self.marked_state, 2)):
                divided_state.append(self.marked_state[i:i+2])
        
        else:
            for i in range(n_nodes-1):
                node_list.append(Node(name=f"Node {i}", qmemory=self.processor_type(3)))
            node_list.append(Node(name=f"Node {n_nodes-1}", qmemory=self.processor_type(4)))
            
            for i in range(0, len(self.marked_state)-3, 2):
                divided_state.append(self.marked_state[i:i+2])
            divided_state.append(self.marked_state[-3:])
        
        local_protocols = []
        for node, target in zip(node_list, divided_state):
            local_protocols.append(DEGAProtocol(node=node, marked_state_node=target))

        for protocol in local_protocols:
            protocol.start()
            yield self.await_signal(sender=protocol, signal_label="FINISHED")


class DEGAProtocol(ns.protocols.NodeProtocol):
    
    def __init__(self, node, marked_state_node):
        if not isinstance(marked_state_node, str):
            raise ValueError("State must be string")
        self.marked_state_node = marked_state_node
        super().__init__(node=node)

    def run(self):
        dega = DEGA(self.marked_state_node)
        self.node.qmemory.execute_program(dega)
        yield self.await_program(self.node.qmemory)

if __name__ == "__main__":
    
    marked_state = str(input("¿Qué estado desea marcar?"))

    dega_master = DEGAMasterProtocol(marked_state, NVProcessor2026)
    dega_master.start()
    
    ns.sim_run()



