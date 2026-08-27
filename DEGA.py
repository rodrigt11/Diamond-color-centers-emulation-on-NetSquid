
# We implement DEGA for 4 qubits, across 2 NV-nodes with 1 communication qubit and 
from netsquid.nodes import Node
from entanglement import Entangler
from nv_2026 import NVParameterSet2026COMPUTAEX
from processors import NVProcessor2026, SnVProcessor2026
from netsquid.components import INSTR_SWAP, QuantumProgram, INSTR_H, INSTR_INIT, INSTR_X, INSTR_CXDIR, INSTR_ROT_X
from primitives import Initialization, FlipBit, Hadamards, MultiCZ, Measure, MultiCPhase
from collections import Counter
import netsquid as ns
import numpy as np

ns.qubits.qformalism.set_qstate_formalism(ns.qubits.QFormalism.DM)

def long_phase(n_qubits):
    """
    Function to calculate theta, J and phi
    """
    theta = np.arcsin(np.sqrt(1.0/2**n_qubits))
    j = (np.pi/2 - theta) // (2 * theta)
    phi = 2 * np.arcsin((np.sin(np.pi / (4*j + 6)))/(np.sin(theta)))

    return phi, j

def run_dega_once(marked_state, processor_type=NVProcessor2026, noiseless=False, verbose=False):
    """
    Function to run DEGA once
    """
    ns.sim_reset()

    dega_master = DEGAMasterProtocol(
        marked_state=marked_state,
        processor_type=processor_type,
        processor_kwargs={"noiseless": noiseless},
        verbose=verbose
    )

    dega_master.start()
    ns.sim_run()

    return {
        "marked_state": marked_state,
        "result": dega_master.result,
        "local_results": dega_master.local_results,
        "divided_state": dega_master.divided_state,
        "success": dega_master.success
    }

def sample_dega(marked_state, shots=1000, processor_type=NVProcessor2026, noiseless=False):
    """
    Function to run DEGA an arbitrary number of times.
    """
    global_counts = Counter()
    success_count = 0

    local_counts = None
    divided_state_ref = None

    for shot in range(shots):
        shot = run_dega_once(
            marked_state=marked_state,
            processor_type=processor_type,
            noiseless=noiseless,
            verbose=False
        )

        result = shot["result"]
        local_results = shot["local_results"]
        divided_state = shot["divided_state"]

        if divided_state_ref is None:
            divided_state_ref = divided_state
            local_counts = [Counter() for _ in divided_state]

        global_counts[result] += 1

        if result == marked_state:
            success_count += 1

        for i, local_result in enumerate(local_results):
            local_counts[i][local_result] += 1

    p_success = success_count / shots

    print()
    print("====================================")
    print("DEGA RESULTS")
    print("====================================")
    print(f"Marked state: {marked_state}")
    print(f"Local states: {divided_state_ref}")
    print(f"Processor: {processor_type.__name__}")
    print(f"Noiseless: {noiseless}")
    print(f"Shots: {shots}")
    print(f"Successes: {success_count}")
    print(f"Success probability: {p_success:.6f}")

    print()
    print("Local distributions:")
    for i, counter in enumerate(local_counts):
        print(f"  Node {i}, local target {divided_state_ref[i]}:")
        for state, count in counter.most_common():
            print(f"    {state}: {count} ({count/shots:.6f})")

    return {
        "marked_state": marked_state,
        "shots": shots,
        "success_count": success_count,
        "p_success": p_success,
        "global_counts": global_counts,
        "local_counts": local_counts,
        "divided_state": divided_state_ref
    }

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
            phi, j = long_phase(len(self.marked_state))
            phase_gate = MultiCPhase(data_qubits[0], data_qubits[1], data_qubits[2], phi)

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

        # R_0: marks |0...0>, reusing Oracle
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
        return Oracle(self.marked_state) + Diffusion(len(self.marked_state)) + Oracle(self.marked_state) + Diffusion(len(self.marked_state))
    
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
        if len(self.marked_state) == 2:
            return Initialization('0'*(len(self.marked_state) + 1)) + Hadamards(list(range(1, len(self.marked_state) + 1))) + Grover(self.marked_state) + measures
        elif len(self.marked_state) == 3:
            return Initialization('0'*(len(self.marked_state) + 1)) + Hadamards(list(range(1, len(self.marked_state) + 1))) + Long(self.marked_state) + measures
        
    def program(self):
        yield from self.load(self._build())



class DEGAMasterProtocol(ns.protocols.protocol.Protocol):
    def __init__(self, marked_state, processor_type, verbose, processor_kwargs=None):
        if not isinstance(marked_state, str):
            raise ValueError("Marked state must be a string")
        if not processor_type == NVProcessor2026 and not processor_type == SnVProcessor2026:
            raise ValueError("Processor type must be a colour center")
        if any(bit not in ("0", "1") for bit in marked_state):
            raise ValueError("Marked state must be in binary code")
        self.marked_state = marked_state
        self.processor_type = processor_type
        self.verbose = verbose
        self.processor_kwargs = processor_kwargs or {}

        self.result = None
        self.local_results = None
        self.divided_state = None
        self.success = None

        super().__init__()

    def run(self):

        n_nodes = len(self.marked_state) // 2

        node_list = []
        divided_state = []
        if len(self.marked_state) % 2 == 0:
            for i in range(n_nodes):
                node_list.append(Node(name=f"Node {i}", qmemory=self.processor_type(3, **self.processor_kwargs)))

            for i in range(0, len(self.marked_state), 2):
                divided_state.append(self.marked_state[i:i+2])
        
        else:
            for i in range(n_nodes-1):
                node_list.append(Node(name=f"Node {i}", qmemory=self.processor_type(3, **self.processor_kwargs)))
            node_list.append(Node(name=f"Node {n_nodes-1}", qmemory=self.processor_type(4, **self.processor_kwargs)))
            
            for i in range(0, len(self.marked_state)-3, 2):
                divided_state.append(self.marked_state[i:i+2])
            divided_state.append(self.marked_state[-3:])
        
        local_protocols = []
        for node, target in zip(node_list, divided_state):
            local_protocols.append(DEGAProtocol(node=node, marked_state_node=target, verbose=self.verbose))

        global_result = ""
        local_results = []
        for protocol in local_protocols:
            protocol.start()
            yield self.await_signal(sender=protocol, signal_label="FINISHED")
            global_result += protocol.result
            local_results.append(protocol.result)


        self.result = global_result
        self.local_results = local_results
        self.divided_state = divided_state
        self.success = global_result == self.marked_state
        
        if self.verbose:
            print(f"DEGA Result: {self.result}")
            print(f"Marked state: {self.marked_state}")

            if global_result == self.marked_state:
                print("SUCCESS")
            else:
                print("FAILURE")


class DEGAProtocol(ns.protocols.NodeProtocol):
    
    def __init__(self, node, marked_state_node, verbose):
        if not isinstance(marked_state_node, str):
            raise ValueError("State must be string")
        self.marked_state_node = marked_state_node
        self.result = None
        self.verbose = verbose
        super().__init__(node=node)
        self.add_signal("FINISHED")

    def run(self):
        dega = DEGA(self.marked_state_node)

        self.node.qmemory.execute_program(dega)
        yield self.await_program(self.node.qmemory)

        self.result = "".join(
            str(dega.output[f"tau_{i}"][0]) 
            for i in range(1, len(self.marked_state_node) + 1)
        )

        if self.verbose:
            print(f"{self.node.name}: {self.result}")

        self.send_signal("FINISHED")


if __name__ == "__main__":
    
    #marked_state = str(input("¿Qué estado desea marcar?"))

    #dega_master = DEGAMasterProtocol(marked_state, NVProcessor2026, processor_kwargs={"noiseless": False})
    #dega_master.start()

    sample_dega("10100",100,processor_type=NVProcessor2026,noiseless=False)
    
    
    




