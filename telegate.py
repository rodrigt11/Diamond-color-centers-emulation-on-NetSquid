
from netsquid.nodes import Node
from netsquid.protocols import NodeProtocol
from netsquid.components import ClassicalChannel, FixedDelayModel, QuantumProgram, INSTR_INIT, INSTR_CXDIR, INSTR_SWAP, INSTR_MEASURE, INSTR_ROT_X
from entrelazamiento import Entangler, EntanglingExchange
import netsquid as ns
import numpy as np
import netsquid.qubits.qubitapi as qapi
from nv_2026 import NVParameterSet2026COMPUTAEX
from procesadores import NVProcessor2026
from netsquid.qubits.qformalism import set_qstate_formalism

set_qstate_formalism(ns.qubits.QFormalism.DM)

class Alice(QuantumProgram):
            def __init__(self, mem_A, ancilla_A):
                self.mem_A = mem_A
                self.ancilla_A = ancilla_A
                super().__init__()

            def program(self):
                self.apply(INSTR_INIT, qubit_indices=0, physical=True)
                self.apply(INSTR_SWAP, qubit_indices=[0,self.mem_A], physical=True)

                self.apply(INSTR_ROT_X, qubit_indices=self.ancilla_A, angle=-np.pi/2, physical=True)
                self.apply(INSTR_CXDIR, qubit_indices=[0,self.ancilla_A], angle=-np.pi/2, physical=True)

                self.apply(INSTR_SWAP, qubit_indices=[0,self.mem_A], physical=True)

                self.apply(INSTR_SWAP, qubit_indices=[0,self.ancilla_A], physical=True)

                self.apply(INSTR_MEASURE, qubit_indices=0, physical=True, output_key="bit_alice")

                yield self.run()

class Bob(QuantumProgram):
            def __init__(self, aplicarX, mem_B, ancilla_B):
                self.aplicarX = aplicarX
                self.mem_B = mem_B
                self.ancilla_B = ancilla_B
                super().__init__()
            
            def program(self):

                self.apply(INSTR_INIT, qubit_indices=0, physical=True)

                if self.aplicarX:
                    self.apply(INSTR_ROT_X, qubit_indices=self.ancilla_B, angle=np.pi, physical=True)

                self.apply(INSTR_SWAP, qubit_indices=[0,self.ancilla_B], physical=True)

                self.apply(INSTR_ROT_X, qubit_indices=self.mem_B, angle=-np.pi/2, physical=True)
                self.apply(INSTR_CXDIR, qubit_indices=[0,self.mem_B], angle=-np.pi/2, physical=True)

                self.apply(INSTR_SWAP, qubit_indices=[0,self.ancilla_B], physical=True)

                yield self.run()

class RemoteCNOT(NodeProtocol):
    def __init__(self,nodo_A, nodo_B, mem_A, mem_B, ancilla_A, ancilla_B, entrelazador):
        super().__init__(node=nodo_A)
        if not isinstance(nodo_A, Node) or not isinstance(nodo_B, Node):
            raise ValueError("nodo_A y nodo_B deben ser instancias de Node.")
        if not isinstance(entrelazador, Entangler):
            raise ValueError("entrelazador debe ser una instancia de Entangler.")
        self.nodo_A = nodo_A
        self.nodo_B = nodo_B
        self.mem_A = mem_A
        self.mem_B = mem_B
        self.ancilla_A = ancilla_A
        self.ancilla_B = ancilla_B
        self.entrelazador = entrelazador

    def run(self):
        print(f'[{ns.sim_time():.1f} ns] Iniciando CNOT remoto...')

        self.entrelazador.start()
        
        # Esperamos a que el Entangler termine y emita su señal nativa de éxito
        yield self.await_signal(self.entrelazador, "Entrelazado")
        print(f"[{ns.sim_time():.1f} ns] Telegate: ¡Entrelazamiento completado con éxito por el hardware!")

        # Si los procesadores físicos quedaron ocupados con la heráldica, esperamos a que se liberen
        if self.nodo_A.qmemory.busy:
            yield self.await_program(self.nodo_A.qmemory)
        if self.nodo_B.qmemory.busy:
            yield self.await_program(self.nodo_B.qmemory)

        print(f"[{ns.sim_time():.1f} ns] Telegate: Transfiriendo qubits a los espines nucleares de ancilla...")
        prog_swap_A = EntanglingExchange(pos=self.ancilla_A)
        prog_swap_B = EntanglingExchange(pos=self.ancilla_B)
        
        self.nodo_A.qmemory.execute_program(prog_swap_A)
        self.nodo_B.qmemory.execute_program(prog_swap_B)
        
        # Esperamos firmemente a que AMBOS procesadores terminen el SWAP físico
        yield self.await_program(self.nodo_A.qmemory)
        yield self.await_program(self.nodo_B.qmemory)

        qa1 = self.nodo_A.qmemory.peek(self.ancilla_A)[0]
        qb1 = self.nodo_B.qmemory.peek(self.ancilla_B)[0]
        
        print("DM nucleares:")
        print(qapi.reduced_dm([qa1, qb1]))
        print(f'La fidelidad con el estado |Φ+⟩ es: {qapi.fidelity([qa1, qb1], reference_state=ns.b00):.4f}')


        print(f"[{ns.sim_time():.1f} ns] Telegate: Ejecutando circuito local de Alice...")
        prog_A = Alice(mem_A=self.mem_A, ancilla_A=self.ancilla_A)
        self.nodo_A.qmemory.execute_program(prog_A)
        yield self.await_program(self.nodo_A.qmemory)

        # Leemos el resultado clásico directamente del hardware de Alice
        bit_resultado = prog_A.output['bit_alice'][0]
        print(f"[{ns.sim_time():.1f} ns] Telegate: Alice midió el bit clásico = {bit_resultado}")

        print(f"[{ns.sim_time():.1f} ns] Telegate: Enviando bit a Bob a través del canal clásico...")
        self.nodo_A.ports['c_out_A'].tx_output(bit_resultado) # Envía el bit físicamente
        yield self.await_port_input(self.nodo_B.ports["c_in_B"])

        mensaje = self.nodo_B.ports["c_in_B"].rx_input()
        bit_resultado = mensaje.items[0]
        aplicarX = (bit_resultado == 1)

        print(f"[{ns.sim_time():.1f} ns] Telegate: Bob recibe el bit. Ejecutando su Telegate (aplicarX={aplicarX})...")
        prog_B = Bob(aplicarX=aplicarX, mem_B=self.mem_B, ancilla_B=self.ancilla_B)
        self.nodo_B.qmemory.execute_program(prog_B)
        yield self.await_program(self.nodo_B.qmemory)

        print(f"[{ns.sim_time():.1f} ns] Telegate: ¡Proceso Telegate finalizado por completo!")

if __name__ == "__main__":
    modelo = FixedDelayModel(250.)
    # Hardware
    A = Node('Alice', 
            qmemory=NVProcessor2026(3), 
            port_names=['c_out_A','in_A'])
    B = Node('Bob', 
            qmemory=NVProcessor2026(3),
            port_names=['c_in_B','out_B'])

    canal = ClassicalChannel("CanalClasico", length=0.001, models={"delay_model": modelo})

    A.ports["c_out_A"].connect(canal.ports["send"])
    canal.ports["recv"].connect(B.ports["c_in_B"])

    # Preparación de qubits
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

    # Simulación
    entrelazamiento = Entangler(A, B, NVParameterSet2026COMPUTAEX())

    # Instanciamos ÚNICAMENTE el protocolo maestro encargado de la coreografía
    protocolo_maestro = RemoteCNOT(A, B, 1, 1, 2, 2, entrelazador=entrelazamiento)
    protocolo_maestro.start()

    ns.sim_run()

    print("\n================ INSPECCIÓN DE RESULTADOS ================")
    qubit_final_A = A.qmemory.peek(1)[0]
    qubit_final_B = B.qmemory.peek(1)[0]

    if qubit_final_B is not None:
        print("¡ÉXITO! El núcleo de Bob contiene el qubit procesado.")
        print("Matriz de densidad final entre espines nucleares:")
        print(qapi.reduced_dm([qubit_final_A, qubit_final_B]))
    else:
        print("🚨 ERROR: El núcleo de Bob sigue estando vacío (None).")
    print("==========================================================")
        