
from netsquid.nodes import Node
from netsquid.protocols import NodeProtocol
from netsquid.components import ClassicalChannel, FixedDelayModel, QuantumProgram, INSTR_INIT, INSTR_CXDIR, INSTR_SWAP, INSTR_MEASURE, INSTR_ROT_X
from entanglement import Entangler
import netsquid as ns
import numpy as np
import netsquid.qubits.qubitapi as qapi
from nv_2026 import NVParameterSet2026COMPUTAEX
from processors import NVProcessor2026
from netsquid.qubits.qformalism import set_qstate_formalism
from primitives import Exchange, Initialization, CNOT, Measure, FlipBit

set_qstate_formalism(ns.qubits.QFormalism.DM)

class Alice(QuantumProgram):
            def __init__(self, mem_A, anc_A):
                self.mem_A = mem_A
                self.anc_A = anc_A
                super().__init__()

            def _build(self):
                
                return(
                     Initialization(0) +
                     CNOT(self.mem_A, self.anc_A) +
                     Measure(self.anc_A, output_key="bit_alice")
                )

            def program(self):
                yield from self.load(self._build())

class Bob(QuantumProgram):
            def __init__(self, applyX, mem_B, anc_B):
                self.applyX = applyX
                self.mem_B = mem_B
                self.anc_B = anc_B
                super().__init__()
            
            def _build(self):   
                if self.applyX:
                    return (
                        Initialization(0) +
                        FlipBit([self.anc_B]) +
                        CNOT(self.anc_B, self.mem_B)
                    )
                else:
                    return (
                        Initialization(0) +
                        CNOT(self.anc_B, self.mem_B)
                    )

            def program(self):
                yield from self.load(self._build())

class RemoteCNOT(NodeProtocol):
    def __init__(self,node_A, node_B, mem_A, mem_B, anc_A, anc_B, entangler):
        super().__init__(node=node_A)
        if not isinstance(node_A, Node) or not isinstance(node_B, Node):
            raise ValueError("node_A and node_B must be instances of Node.")
        if not isinstance(entangler, Entangler):
            raise ValueError("entangler must be an instance of Entangler.")
        self.node_A = node_A
        self.node_B = node_B
        self.mem_A = mem_A
        self.mem_B = mem_B
        self.anc_A = anc_A
        self.anc_B = anc_B
        self.entangler = entangler

    def run(self):
        print(f'[{ns.sim_time():.1f} ns] Starting remote CNOT...')

        self.entangler.start()
        
        # Wait for the entanglement to be established
        yield self.await_signal(self.entangler, "Entangled")
        print(f"[{ns.sim_time():.1f} ns] Remote CNOT: ¡Succesful entanglement!")

        # If either node's quantum processor is busy, wait for it to finish
        if self.node_A.qmemory.busy:
            yield self.await_program(self.node_A.qmemory)
        if self.node_B.qmemory.busy:
            yield self.await_program(self.node_B.qmemory)

        print(f"[{ns.sim_time():.1f} ns] Remote CNOT: Exchanging qubits states between electronic and nuclear spins...")
        prog_swap_A = Exchange(pos=self.anc_A)
        prog_swap_B = Exchange(pos=self.anc_B)
        
        self.node_A.qmemory.execute_program(prog_swap_A)
        self.node_B.qmemory.execute_program(prog_swap_B)
        
        # Waiting for the swap operations to complete
        yield self.await_program(self.node_A.qmemory)
        yield self.await_program(self.node_B.qmemory)

        qa1 = self.node_A.qmemory.peek(self.anc_A)[0]
        qb1 = self.node_B.qmemory.peek(self.anc_B)[0]
        
        print("Nuclear DM:")
        print(qapi.reduced_dm([qa1, qb1]))
        print(f'Fidelity with |Φ+⟩ : {qapi.fidelity([qa1, qb1], reference_state=ns.b00):.4f}')


        print(f"[{ns.sim_time():.1f} ns] Remote CNOT: Executing Alice's part of the protocol...")
        prog_A = Alice(mem_A=self.mem_A, anc_A=self.anc_A)
        self.node_A.qmemory.execute_program(prog_A)
        yield self.await_program(self.node_A.qmemory)

        # We obtain the classical bit result from Alice's measurement and send it to Bob through the classical channel
        bit_result = prog_A.output['bit_alice'][0]
        print(f"[{ns.sim_time():.1f} ns] Remote CNOT: Alice measured the classical bit = {bit_result}")

        print(f"[{ns.sim_time():.1f} ns] Remote CNOT: Sending bit to Bob through the classical channel...")
        self.node_A.ports['c_out_A'].tx_output(bit_result) # Sends the bit physically
        yield self.await_port_input(self.node_B.ports["c_in_B"])

        message = self.node_B.ports["c_in_B"].rx_input()
        bit_result = message.items[0]
        applyX = (bit_result == 1)

        print(f"[{ns.sim_time():.1f} ns] Remote CNOT: Bob receives the bit. Executing his part of the protocol (applyX={applyX})...")
        prog_B = Bob(applyX=applyX, mem_B=self.mem_B, anc_B=self.anc_B)
        self.node_B.qmemory.execute_program(prog_B)
        yield self.await_program(self.node_B.qmemory)

        print(f"[{ns.sim_time():.1f} ns] Remote CNOT: Protocol completed. Final states of the nuclear spins:")

if __name__ == "__main__":
    model = FixedDelayModel(250.)
    # Hardware
    A = Node('Alice', 
            qmemory=NVProcessor2026(3), 
            port_names=['c_out_A','in_A'])
    B = Node('Bob', 
            qmemory=NVProcessor2026(3),
            port_names=['c_in_B','out_B'])

    channel = ClassicalChannel("ClasicalChannel", length=0.001, models={"delay_model": model})

    A.ports["c_out_A"].connect(channel.ports["send"])
    channel.ports["recv"].connect(B.ports["c_in_B"])

    # Qubits' initial states
    proga = QuantumProgram()
    progb = QuantumProgram()

    proga.apply(INSTR_INIT, qubit_indices=0, physical=True)
    proga.apply(INSTR_INIT, qubit_indices=1, physical=True)
    proga.apply(INSTR_ROT_X, qubit_indices=1, angle=np.pi, physical=True)

    progb.apply(INSTR_INIT, qubit_indices=0, physical=True)
    progb.apply(INSTR_INIT, qubit_indices=1, physical=True)
    #progb.apply(INSTR_ROT_X, qubit_indices=1, angle=np.pi, physical=True)

    A.qmemory.execute_program(proga)
    B.qmemory.execute_program(progb)

    # Emulation
    entanglement = Entangler(A, B, NVParameterSet2026COMPUTAEX())

    master = RemoteCNOT(A, B, 1, 1, 2, 2, entangler=entanglement)
    master.start()

    ns.sim_run()

    print("\n================ RESULTS ================")
    qf_A = A.qmemory.peek(1)[0]
    qf_B = B.qmemory.peek(1)[0]

    if qf_B is not None:
        print("SUCCESS! Bob's nuclear spin is not empty (not None).")
        print("Final density matrix for control and target qubits (nuclear spins):")
        print(qapi.reduced_dm([qf_A, qf_B]))
    else:
        print("🚨 ERROR: Bob's nuclear spin is still empty (None).")
    print("==========================================================")
        