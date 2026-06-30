'''
This file contains useful primitives for computing with color centers in NetSquid. Some primitives are used in the DEGA algorithm, but they can also be used in other algorithms that require similar operations.
'''

from netsquid.nodes import Node
from entrelazamiento import Entangler
from nv_2026 import NVParameterSet2026COMPUTAEX
from procesadores import NVProcessor2026, SnVProcessor2026
from netsquid.components import INSTR_MEASURE, INSTR_SWAP, QuantumProgram, INSTR_H, INSTR_INIT, INSTR_X, INSTR_CXDIR, INSTR_ROT_X, INSTR_ROT_Z
import netsquid as ns
import numpy as np

ns.qubits.qformalism.set_qstate_formalism(ns.qubits.QFormalism.DM)

class Initialization(QuantumProgram):
    """
    Class for initializing an arbitrary number of qubits, including
    electronic and nuclear spins, by specifying any of the computational basis states. First bit indicates electronic spin state.
    """
    def __init__(self, state=None):
        self.state = str(state)
        super().__init__()

    def program(self):
        for i in range(len(self.state)):
            self.apply(INSTR_INIT, qubit_indices=[i], physical=True)
        
        for i, char in enumerate(self.state):
            if char == "1":
                self.apply(INSTR_X, qubit_indices=[i], physical=True)

        yield self.run()

class FlipBit(QuantumProgram):
    """
    Class used to apply X gate on some qubits. Includes electronic and nuclear spins.
    """
    def __init__(self, qubits):
        if not isinstance(qubits, list) or not all(isinstance(q, int) for q in qubits):
            raise ValueError("Qubits must be a list of integers")
        self.qubits = qubits
        super().__init__()

    def program(self):
        for i in self.qubits:
            self.apply(INSTR_X, qubit_indices=[i], physical=True)

        yield self.run()

class Hadamards(QuantumProgram):
    """
    Class used to apply H gate on some qubits. We must specify them by their indexes. Includes electronic and nuclear spins.
    """
    def __init__(self, qubits):
        if not isinstance(qubits, list) or not all(isinstance(q, int) for q in qubits):
            raise ValueError("Qubits must be a list of integers")
        self.qubits = qubits
        super().__init__()

    def program(self):
        for i in self.qubits:
            self.apply(INSTR_H, qubit_indices=[i], physical=True)

        yield self.run()
        
class RotationZ(QuantumProgram):
    """
    Class used to apply Z rotation on a qubit 
    """
    def __init__(self, qubit_index, angle):
        if not isinstance(qubit_index, int):
            raise ValueError("Qubit index must be an integer")
        if not isinstance(angle, (int, float)):
            raise ValueError("Angle must be a number")
        self.qubit_index = qubit_index
        self.angle = angle
        super().__init__()

    def program(self):
        self.apply(INSTR_ROT_Z, qubit_indices=[self.qubit_index], angle=self.angle, physical=True)
        yield self.run()

class CNOT(QuantumProgram):
    """
    Applyes CNOT gate between two arbitrary qubits through INSTR_CXDIR and one qubit instructions. It assumes that electronic spin must always be the control
    """

    def __init__(self, control, target):
        if not isinstance(control, int) or not isinstance(target, int):
            raise ValueError("Control and target qubits must be integers")
        if control == target:
            raise ValueError("Control and target qubits must be different")
        if target == 0:
            raise ValueError("Target qubit cannot be the communication qubit (index 0)")
        self.control = control
        self.target = target
        super().__init__()
    
    def program(self):
        if self.control != 0:
            self.apply(INSTR_SWAP, qubit_indices=[0, self.control], physical=True)
            self.apply(INSTR_ROT_X, qubit_indices=[self.target], angle=-np.pi/2, physical=True)
            self.apply(INSTR_CXDIR, qubit_indices=[0, self.target], angle=np.pi/2, physical=True)
            self.apply(INSTR_SWAP, qubit_indices=[0, self.control], physical=True)
            self.apply(INSTR_ROT_Z, qubit_indices=[self.control], angle=-np.pi/2, physical=True)
        else:
            self.apply(INSTR_ROT_X, qubit_indices=[self.target], angle=-np.pi/2, physical=True)
            self.apply(INSTR_CXDIR, qubit_indices=[0, self.target], angle=np.pi/2, physical=True)
            self.apply(INSTR_ROT_Z, qubit_indices=[0], angle=-np.pi/2, physical=True)
        
        yield self.run()

class Measure(QuantumProgram):
    def __init__(self, qubit_index, output_key):
        if not isinstance(qubit_index, int):
            raise ValueError("Qubits must be an integer")
        if not isinstance(output_key, str):
            raise ValueError("Output key must be a string")
        self.q = qubit_index
        self.output_key = output_key
        super().__init__()

    def program(self):
        if self.q == 0:
            self.apply(INSTR_MEASURE, qubit_indices=[self.q], physical=True, output_key=self.output_key)
        else:
            self.apply(INSTR_SWAP, qubit_indices=[0, self.q], physical=True)
            self.apply(INSTR_MEASURE, qubit_indices=[0], physical=True, output_key=self.output_key)
            self.apply(INSTR_SWAP, qubit_indices=[0, self.q], physical=True)

        yield self.run()

class MultiCX(QuantumProgram):
    """
    Applies an X gate on the target conditioned to the states of an arbitrary number of control qubits (for 3 controls and beyond is still developing)
    """
    def __init__(self, control_qubits, target_qubit):
        if 0 in control_qubits:
            raise ValueError("Control qubits cannot include the communication qubit (index 0)")
        if not isinstance(control_qubits, list) or not all(isinstance(q, int) for q in control_qubits):
            raise ValueError("Control qubits must be a list of integers")
        if not isinstance(target_qubit, int):
            raise ValueError("Target qubit must be an integer")
        if target_qubit in control_qubits:
            raise ValueError("Target qubit cannot be one of the control qubits")
        self.control_qubits = control_qubits
        self.target_qubit = target_qubit
        super().__init__()

    def _build(self):
        
        if len(self.control_qubits) == 1:
            return CNOT(self.control_qubits[0], self.target_qubit)

        elif len(self.control_qubits) == 2:
            return (
                Hadamards([self.target_qubit]) + 
                CNOT(self.control_qubits[1], self.target_qubit) +
                RotationZ(self.target_qubit, -np.pi/4) +
                CNOT(self.control_qubits[0], self.target_qubit) +
                RotationZ(self.target_qubit, np.pi/4) +
                CNOT(self.control_qubits[1], self.target_qubit) +
                RotationZ(self.target_qubit, -np.pi/4) +
                CNOT(self.control_qubits[0], self.target_qubit) +
                RotationZ(self.target_qubit, np.pi/4) +
                RotationZ(self.control_qubits[1], -np.pi/4) +
                CNOT(self.control_qubits[0], self.control_qubits[1]) +
                Hadamards([self.target_qubit]) +
                RotationZ(self.control_qubits[1], -np.pi/4) +
                CNOT(self.control_qubits[0], self.control_qubits[1]) +
                RotationZ(self.control_qubits[0], np.pi/4) +
                RotationZ(self.control_qubits[1], np.pi/2)
            )

    def program(self):
        yield from self.load(self._build())

class MultiCZ(QuantumProgram):
    """
    Applies a Z gate conditioned to the states of an arbitrary number of control spins. Based in MultiCX class.
    """
    def __init__(self, control_qubits, target_qubit):
        if 0 in control_qubits:
            raise ValueError("Control qubits cannot include the communication qubit (index 0)")
        if not isinstance(control_qubits, list) or not all(isinstance(q, int) for q in control_qubits):
            raise ValueError("Control qubits must be a list of integers")
        if not isinstance(target_qubit, int):
            raise ValueError("Target qubit must be an integer")
        if target_qubit in control_qubits:
            raise ValueError("Target qubit cannot be one of the control qubits")
        self.control_qubits = control_qubits
        self.target_qubit = target_qubit
        super().__init__()
    
    def _build(self):
        return (
            Hadamards([self.target_qubit]) +
            MultiCX(self.control_qubits, self.target_qubit) +
            Hadamards([self.target_qubit])
        )
    
    def program(self):
        yield from self.load(self._build())

class MultiCPhase(QuantumProgram):
    """
    Applies a phase gate conditioned to the state of two control qubits.
    """
    def __init__(self, c1, c2, t, phase):
        if t == 0:
            raise ValueError("Control qubits cannot include the communication qubit (index 0)")
        if not isinstance(c1, int) or not isinstance(c2, int) or not isinstance(t, int):
            raise ValueError("must be integers")
        if c1 == t or c2 == t or c1 == c2:
            raise ValueError("Must be different qubits")
        self.c1 = c1
        self.c2 = c2
        self.t = t
        self.phase = phase
        super().__init__()

    def _build(self):
        c1, c2, t = self.c1, self.c2, self.t
        phi = self.phase

        return (
            RotationZ(c1,  phi/4) +
            RotationZ(c2,  phi/4) +
            RotationZ(t,   phi/4) +

            CNOT(c1, c2) +
            RotationZ(c2, -phi/4) +
            CNOT(c1, c2) +

            CNOT(c1, t) +
            RotationZ(t, -phi/4) +
            CNOT(c1, t) +

            CNOT(c2, t) +
            RotationZ(t, -phi/4) +
            CNOT(c2, t) +

            CNOT(c1, t) +
            CNOT(c2, t) +
            RotationZ(t, phi/4) +
            CNOT(c2, t) +
            CNOT(c1, t)
        )

    def program(self):
        yield from self.load(self._build())



# Code block only used as testbench for the classes
if __name__ == "__main__":
    
    #pr = Initialization(n_qubits=4, state="0110") + CNOT(control=1, target=2) + Measure(qubit_index=2, output_key="result2")
    class pr(QuantumProgram):
        def _build(self):
            return (
                Initialization(n_qubits=3, state="000") +
                Hadamards([1]) +
                CNOT(1,2)
            )
        
        def program(self):
            yield from self.load(self._build())

    # Example usage
    
    class protocol2(ns.protocols.NodeProtocol):

        def run(self):
            prog = pr()
            self.node.qmemory.execute_program(prog)
            yield self.await_program(self.node.qmemory)

            q1, q2 = self.node.qmemory.peek([1, 2])
            dm = ns.qubits.qubitapi.reduced_dm([q1, q2])

            print(dm)

            psi = np.zeros((4, 1), dtype=complex)
            psi[0, 0] = 1 / np.sqrt(2)  # |00>
            psi[3, 0] = 1 / np.sqrt(2)  # |11>

            fidelity = ns.qubits.qubitapi.fidelity([q1,q2],psi)

            print("Fidelidad con Bell canónico:", fidelity)
            print("rho[0,3] =", dm[0, 3])
            
    

    processor = NVProcessor2026(num_positions=3, noiseless=True)
    nodo = Node("nodo", qmemory=processor)
    protocolo = protocol2(node=nodo)
    protocolo.start()
    ns.sim_run()  # Run the simulation to completion
