# We follow the steps of Zhou, X., Xu, X., Zheng, S., & Luo, L. (2026). Distributed exact generalized Grover’s algorithm. Frontiers of Computer Science, 20(7), 2007905.

from netsquid.nodes import Node
from netsquid.protocols import NodeProtocol, Protocol, Signals
from netsquid.components.cchannel import ClassicalChannel
from entanglement import Entangler
from snv_2026 import SnVParameterSet2026COMPUTAEX
from nv_2026 import NVParameterSet2026COMPUTAEX
from processors import NVProcessor2026, SnVProcessor2026
from netsquid_nv.nv_parameter_set import NVParameterSet
from netsquid.components import INSTR_SWAP, QuantumProgram, INSTR_H, INSTR_INIT, INSTR_X, INSTR_CXDIR, INSTR_ROT_X
from primitives import Initialization, FlipBit, Hadamards, MultiCZ, Measure, MultiCPhase2, Exchange
from collections import Counter
import netsquid as ns
import numpy as np
import time

ns.qubits.qformalism.set_qstate_formalism(ns.qubits.QFormalism.DM)

def split_target(string, partition):
    chunks = []
    start = 0

    for size in partition:
        end = start + size
        chunks.append(string[start:end])
        start = end

    return chunks

def build_local_targets(targets, partition):
    if not targets:
        raise ValueError("Targets cannot be empty")
    
    if not partition:
        raise ValueError("Partition cannot be empty")
    
    if any(not isinstance(size, int) or size <= 0
           for size in partition):
        raise ValueError(
            "Partition sizes must be positive integers"
        )
    
    if any(not isinstance(target, str) for target in targets):
        raise TypeError("Every target must be a string")
    
    n = len(targets[0])

    if any(len(t) != n for t in targets):
        raise ValueError("All targets must have the same length")
    
    if any(set(target) - {"0", "1"} for target in targets):
        raise ValueError(
            "Targets must contain only '0' and '1'"
        )

    if sum(partition) != n:
        raise ValueError("Partition does not match target length")
    
    local_sets = [[] for _ in partition]

    for target in targets:
        chunks = split_target(target, partition)

        for j,chunk in enumerate(chunks):
            if chunk not in local_sets[j]:
                local_sets[j].append(chunk)

    return local_sets

def long_parameters(n_j, n_local_targets):
    if n_j <= 0:
        raise ValueError("n_j must be positive")

    if n_local_targets <= 0:
        raise ValueError("n_local_targets must be positive")

    if n_local_targets > 2**n_j:
        raise ValueError("n_local_targets cannot be larger than 2**n_j")
    
    theta_nj = np.arcsin(np.sqrt(n_local_targets / 2**n_j))
    J_nj = int(np.floor((np.pi/2 - theta_nj)/(2*theta_nj)))
    phase_nj = 2*np.arcsin(np.sin(np.pi/(4*J_nj + 6))/np.sin(theta_nj))

    return theta_nj, J_nj, phase_nj

def global_parameters(a, list_a_j):
    if a <= 0:
        raise ValueError("a must be positive")
    if not list_a_j:
        raise ValueError("Local targets hav not been built")

    prod = 1
    for a_j in list_a_j:
        if a_j <= 0:
            raise ValueError("Local target counts must be positive")
        prod *= a_j

    ratio = a / prod

    if ratio > 1: 
        raise ValueError(f"Invalid global parameters: a={a}, "
            f"list_a_j={list_a_j}, product={prod}, "
            f"a/product={ratio}")
    
    theta_f = np.arcsin(np.sqrt(ratio))
    if not np.isfinite(theta_f):
        raise ValueError("theta_f is not finite")
    
    J_f = int(np.floor((np.pi/2 - theta_f)/(2*theta_f)))
    phase_f = 2*np.arcsin(np.sin(np.pi/(4*J_f + 6))/np.sin(theta_f))

    return theta_f, J_f, phase_f

class OracleDEGGA(QuantumProgram):
    def __init__(self, marked_state, phi):
        if not isinstance(marked_state, str):
            raise ValueError("marked_state must be a string")

        if not marked_state:
            raise ValueError("marked_state cannot be empty")

        if set(marked_state) - {"0", "1"}:
            raise ValueError("marked_state must contain only '0' and '1'")
            
        self.marked_state = marked_state
        self.phi = float(phi)
        super().__init__()
    
    def _build(self):
        n = len(self.marked_state)

        # The data qubits will only be nuclear spins
        data_qubits = list(range(1, n+1))
    
        flip_qubits = [i+1 for i, bit in enumerate(self.marked_state) if bit=="0"]
    
        prog = None
    
        if flip_qubits:
            prog = FlipBit(flip_qubits)
    
        phase_gate = MultiCPhase2(data_qubits, self.phi)
    
        prog = phase_gate if prog is None else prog + phase_gate
    
        if flip_qubits:
            prog = prog + FlipBit(flip_qubits)
    
        return prog
        
    def program(self):
        yield from self.load(self._build())

class DiffusionDEGGA(QuantumProgram):
    def __init__(self, marked_state_length, phi):
        if not isinstance(marked_state_length, int) or marked_state_length <= 0:
            raise ValueError("marked_state_length must be a positive integer")
        self.marked_state_length = marked_state_length
        self.phi = float(phi)
        super().__init__()

    def _build(self):
        data_qubits = list(range(1, self.marked_state_length + 1))

        h_layer_1 = Hadamards(data_qubits)

        # R_0: marks |0...0>, reusing Oracle
        r0 = OracleDEGGA("0" * self.marked_state_length, self.phi)

        # H^{⊗n} final
        h_layer_2 = Hadamards(data_qubits)

        return h_layer_1 + r0 + h_layer_2

    def program(self):
        yield from self.load(self._build())

class DEGGAMasterProtocol(Protocol):

    """
    Master protocol for the Distributed Exact Generalized
    Grover's Algorithm (DEGGA).

    The protocol:

    1. Validates the global target states and the partition.
    2. Creates one colour-centre node per partition element and an additional one as router, needed to execute multi-controlled distributed gates.
    3. Builds the local target sets.
    4. Creates and launches one LocalDEGGAProtocol per node and one GlobalDEGGAProtocol.
    5. Collects and combines the local measurement results.
    """
    
    def __init__(self, targets, partition, processor_type, params, processor_kwargs=None, verbose=True):

        if processor_type not in (NVProcessor2026, SnVProcessor2026):
            raise TypeError("processor_type must be an instance of NVProcessor2026 or SnVProcessor2026")

        if not targets:
            raise ValueError("Targets cannot be empty")

        if not isinstance(targets, (list, tuple)):
            raise TypeError("targets must be a list or tuple")

        if any(not isinstance(target, str) for target in targets):
            raise TypeError("Every target must be a string")

        super().__init__(name="master")

        self.targets = list(targets)
        self.partition = list(partition)
        self.processor_type = processor_type
        self.params = params
        self.verbose = verbose
        self.processor_kwargs = processor_kwargs or {}
        self.n = len(targets[0])
        self.t = len(partition)

        self.nodes = []
        self.local_targets = []
        self.target_chunks = []
        self.local_protocols = []

        self.result = None
        self.local_results = []
        self.success = None

        self.router = None
        self.global_protocol = None

        self.add_signal("LOCAL_COMMAND")
        self.add_signal("GLOBAL_COMMAND")

    def run(self):

        # Building of the local target sets for each node based on the global targets and the partition.
        self.local_targets = build_local_targets(self.targets, self.partition)

        list_a_j = [len(local_targets) for local_targets in self.local_targets]
        theta_f, J_f, phase_f = global_parameters(len(self.targets), list_a_j)

        for target in self.targets:
            self.target_chunks.append(split_target(target, self.partition))

        # Create nodes with quantum memory based on the partition sizes.
        for j, n_j in enumerate(self.partition):
            node = Node(f"Node_{j}", qmemory=self.processor_type(num_positions=n_j+1, **self.processor_kwargs))
            self.nodes.append(node)
        self.router = Node("Router", qmemory=self.processor_type(num_positions=self.t+1, **self.processor_kwargs))

        # Router initialization
        router_initialization = Initialization("0" * (self.t + 1))
        self.router.qmemory.execute_program(router_initialization)
        yield self.await_program(self.router.qmemory)

        # Creation of classical channels between nodes and the router
        for node in self.nodes:
            node.connect_to(remote_node=self.router, 
                            connection=ClassicalChannel(f"CChannel_{node.name}_to_router"),
                            local_port_name=f"cport_{node.name}_to_router",
                            remote_port_name=f"cport_router_from_{node.name}")
            self.router.connect_to(remote_node=node,
                                   connection=ClassicalChannel(f"CChannel_router_to_{node.name}"),
                                   local_port_name=f"cport_router_to_{node.name}",
                                   remote_port_name=f"cport_{node.name}_from_router")

            
        # Create and start a LocalDEGGAProtocol for each node, passing the local targets and partition.
        for j, (node, n_j, local_target) in enumerate(zip(self.nodes, self.partition, self.local_targets)):
            protocol = LocalDEGGAProtocol(node, self, j, local_target, verbose=self.verbose)
            self.local_protocols.append(protocol)

        #Creation of global protocol
        self.global_protocol = GlobalDEGGAProtocol(self.nodes, self, self.targets, self.router, self.params, verbose=self.verbose)

        print(f"DEGGA starts (Zhou et al. (2024)). We search the elements {self.targets} inside the space of size {2**len(self.targets[0])}. Will be initialized {len(self.partition)} {self.processor_type} nodes to search and an additional one as router")

        # Start all local protocols and the global protocol.
        for protocol in self.local_protocols:
            protocol.start()
        self.global_protocol.start()

        # First local step
        first_local_step_finished = self.await_signal(sender=self.local_protocols[0], signal_label="LOCAL_STEP_FINISHED")
        for protocol in self.local_protocols[1:]:
            first_local_step_finished &= self.await_signal(sender=protocol, signal_label="LOCAL_STEP_FINISHED")

        self.send_signal(signal_label="LOCAL_COMMAND", result={"phase": "FIRST_LOCAL_STEP"})

        yield first_local_step_finished

        for global_iteration in range(J_f + 1):

            if self.verbose:
                print(f"Global iteration {global_iteration + 1}/{J_f + 1}")

            # First global step
            self.send_signal(signal_label="GLOBAL_COMMAND", result={"phase": "FIRST_GLOBAL_STEP"})
            yield self.await_signal(sender=self.global_protocol, signal_label="GLOBAL_STEP_FINISHED")

            # Second local step
            second_local_step_finished = self.await_signal(sender=self.local_protocols[0], signal_label="LOCAL_STEP_FINISHED")
            for protocol in self.local_protocols[1:]:
                second_local_step_finished &= self.await_signal(sender=protocol, signal_label="LOCAL_STEP_FINISHED")

            self.send_signal(signal_label="LOCAL_COMMAND", result={"phase": "SECOND_LOCAL_STEP"})
            
            yield second_local_step_finished

            # Second global step
            self.send_signal(signal_label="GLOBAL_COMMAND", result={"phase": "SECOND_GLOBAL_STEP"})
            yield self.await_signal(sender=self.global_protocol, signal_label="GLOBAL_STEP_FINISHED")

            # Third local step
            third_local_step_finished = self.await_signal(sender=self.local_protocols[0], signal_label="LOCAL_STEP_FINISHED")
            for protocol in self.local_protocols[1:]:
                third_local_step_finished &= self.await_signal(sender=protocol, signal_label="LOCAL_STEP_FINISHED")

            self.send_signal(signal_label="LOCAL_COMMAND", result={"phase": "THIRD_LOCAL_STEP"})
            
            yield third_local_step_finished

        # Local measurement step
        measurement_step_finished = self.await_signal(sender=self.local_protocols[0], signal_label="LOCAL_STEP_FINISHED")
        for protocol in self.local_protocols[1:]:
            measurement_step_finished &= self.await_signal(sender=protocol, signal_label="LOCAL_STEP_FINISHED")

        self.send_signal(signal_label="LOCAL_COMMAND", result={"phase": "MEASURE"})

        yield measurement_step_finished

        self.local_results.clear()

        for protocol in self.local_protocols:

            response = (protocol.get_signal_result("LOCAL_STEP_FINISHED", receiver=self))

            if response["phase"] != "MEASURE":
                raise RuntimeError(f"Unexpected response from node {response['node_index']}")

            j = response["node_index"]
            local_result = response["result"]

            if local_result is None:
                raise RuntimeError(
                    f"Node {j} finished MEASURE without returning a result"
                )

            if len(local_result) != self.partition[j]:
                raise RuntimeError(
                    f"Node {j} returned {len(local_result)} bits; "
                    f"expected {self.partition[j]}"
                )

            self.local_results.append(local_result)

        self.result = "".join(self.local_results)

        # Detention of all protocols and collection of results
        local_finished = self.await_signal(sender=self.local_protocols[0], signal_label=Signals.FINISHED)
        for protocol in self.local_protocols[1:]:
            local_finished &= self.await_signal(sender=protocol, signal_label=Signals.FINISHED)
        self.await_signal(sender=self.global_protocol, signal_label=Signals.FINISHED)

        self.send_signal(signal_label="GLOBAL_COMMAND", result={"phase": "STOP"})
        self.send_signal(signal_label="LOCAL_COMMAND", result={"phase": "STOP"})

        yield (local_finished & self.await_signal(sender=self.global_protocol, signal_label=Signals.FINISHED))

        self.result = "".join(self.local_results)
        self.success = self.result in self.targets

        if self.verbose:
            
            print(f"DEGGA result: {self.result}")
            print(f"Global targets: {self.targets}")
            print(f"Partition: {self.partition}")
            print(f"Local targets: {self.local_targets}")

            if self.success:
                print("SUCCESS")
            else:
                print("FAILURE")        
        
class GlobalDEGGAProtocol(Protocol):

    """
    Protocol governing the phases in which distributed gates are applied across nodes (Llovo et al. 2025).
    """

    def __init__(self, nodes, master, global_targets, router, params, verbose=False):

        if not isinstance(nodes, list) or not all(isinstance(node, Node) for node in nodes):
            raise TypeError("nodes must be a list of netsquid.nodes.Node instances")
        if not isinstance(master, Protocol):
            raise TypeError("master must be an instance of netsquid.protocols.Protocol")
        if not isinstance(global_targets, list) or not all(isinstance(t, str) for t in global_targets):
            raise TypeError("global_targets must be a list of strings")
        if not isinstance(router, Node):
            raise TypeError("router must be an instance of netsquid.nodes.Node")
        if not isinstance(params, NVParameterSet2026COMPUTAEX) and not isinstance(params, SnVParameterSet2026COMPUTAEX):
            raise TypeError("params must be an instance of NVParameterSet2026COMPUTAEX or SnVParameterSet2026COMPUTAEX")

        super().__init__(name="global_protocol")

        self.nodes = list(nodes)
        self.master = master
        self.global_targets = global_targets
        self.router = router
        self.verbose = verbose
        self.params = params
        self.result = None
        self.add_signal("GLOBAL_STEP_FINISHED")

    def run(self):

        list_a_j = [len(local_targets) for local_targets in self.master.local_targets]
        theta_f, J_f, phase_f = global_parameters(len(self.global_targets), list_a_j)
        target_chunks = self.master.target_chunks

        while True:

            yield self.await_signal(sender=self.master, signal_label="GLOBAL_COMMAND")
            command = self.master.get_signal_result("GLOBAL_COMMAND", receiver=self)
            phase = command["phase"]

            if phase == "STOP":
                return {"result": self.result}
            
            #First global gate
            if phase == "FIRST_GLOBAL_STEP":
                print(f"First global step. We apply a global Oracle for each target ")
                for target in target_chunks:
    
                    flip_positions_by_node = []

                    # ORACLE'S FIRST PART: flip qubits in each node based on the target chunk
                    print(f"Flip qubits in each node based on the local target chunk")
                    for node, local_chunk in zip(self.nodes, target):
                        data_positions = list(range(1, node.qmemory.num_positions-1))
                        flip_positions = [position for position, bit in zip(data_positions, local_chunk) if bit == "0"]
                        flip_positions_by_node.append(flip_positions)
                        if flip_positions:
                            x_program = FlipBit(flip_positions)
                            node.qmemory.execute_program(x_program)
                            yield self.await_program(node.qmemory)

                    # ORACLE'S SECOND PART: distributed multi-controlled phase gate across nodes
                    # Entanglement generation between nodes and the router
                    print(f"Multi-Controlled Gate: Entanglement between each node's electronic spin and router's nuclear ones")
                    for k, node in enumerate(self.nodes):
                        print(f"Starting entanglement between Router and Node {k}")
                        entangler = Entangler(node, self.router, self.params, False)
                        entanglement_finished = self.await_signal(entangler, signal_label="Pair ready")
                        entangler.start()
                        yield entanglement_finished
                        print(f"Node {k} entangled")
                        exchange = Exchange(k + 1)
                        self.router.qmemory.execute_program(exchange)
                        yield self.await_program(self.router.qmemory)
                        print(f"Node {k}'s electronic spin free")

                    # Node measurement sending to the router
                    print(f"Multi-Controlled Gate: Local operations on each node and electronic spin measurement. The result is sent to the router")
                    node_measurements_received = []
                    for node in self.nodes:

                        print(f"Starting local node operations")
                        phase_program = MultiCPhase2(list(range(node.qmemory.num_positions-1)), np.pi)
                        node.qmemory.execute_program(phase_program)
                        yield self.await_program(node.qmemory)

                        h_program = Hadamards([0])
                        node.qmemory.execute_program(h_program)
                        yield self.await_program(node.qmemory)

                        meas_program = Measure(0, f"q0_{node.name}")
                        node.qmemory.execute_program(meas_program)
                        yield self.await_program(node.qmemory)
                        print(f"Local block completed on {node.name}")

                        node_measurement = meas_program.output[f"q0_{node.name}"][0]
                        node.ports[f"cport_{node.name}_to_router"].tx_output(node_measurement)
                        router_port = self.router.ports[f"cport_router_from_{node.name}"]
                        reception = self.await_port_input(router_port)
                        yield reception
                        print("Router received message")
                        message = router_port.rx_input().items[0]
                        node_measurements_received.append(message)

                    # Operations on central node
                    print(f"Multi-Controlled Gate: Router operations")
                    router_operations = Hadamards([i for i in range(1, len(self.nodes) + 1)])
                    for j, m in enumerate(node_measurements_received):
                        if m == 1:
                            router_operations += FlipBit([j + 1])
                    router_operations += MultiCPhase2([i for i in range(1, len(self.nodes) + 1)], phase_f)
                    self.router.qmemory.execute_program(router_operations)
                    yield self.await_program(self.router.qmemory)

                    # Router measurements and local corrections
                    print("Multi-Controlled Gate: Router's nuclear spins measurement and results sending to each node")
                    router_measurements = []
                    for router_position in range(1, len(self.nodes) + 1):
                        h_program = Hadamards([router_position])
                        self.router.qmemory.execute_program(h_program)
                        yield self.await_program(self.router.qmemory)
                        measurement_program = Measure(router_position, f"R_{router_position}")
                        self.router.qmemory.execute_program(measurement_program)
                        yield self.await_program(self.router.qmemory)
                        router_measurement = measurement_program.output[f"R_{router_position}"][0]
                        router_measurements.append(router_measurement)

                    print(f"Multi-Controlled Gate: Correction on each node")
                    for node_index, node in enumerate(self.nodes):
                        router_measurement = router_measurements[node_index]
                        node_port = node.ports[f"cport_{node.name}_from_router"]
                        reception = self.await_port_input(node_port)
                        self.router.ports[f"cport_router_to_{node.name}"].tx_output(router_measurement)
                        yield reception
                        received_measurement = node_port.rx_input().items[0]
                        if received_measurement == 1:
                            correction = MultiCPhase2(list(range(1, node.qmemory.num_positions-1)), np.pi)
                            node.qmemory.execute_program(correction)
                            yield self.await_program(node.qmemory)

                    #ORACLE'S THIRD PART
                    print(f"Flip qubits in each node based on the local target chunk")
                    for node_index, node in enumerate(self.nodes):
                        flip_positions = (flip_positions_by_node[node_index])
                        if flip_positions:
                            undo_x_program = FlipBit(flip_positions)
                            node.qmemory.execute_program(undo_x_program)
                            yield self.await_program(node.qmemory)

                self.result = {"phase": phase, "implemented": True} 

            elif phase == "SECOND_GLOBAL_STEP":

                print(f"Second global step. We apply a global Oracle to the all-zero global state")
                zero_chunks = ["0" * n_j for n_j in self.master.partition]
                chunks_to_mark = [zero_chunks]

                for target in chunks_to_mark:

                    flip_positions_by_node = []

                    # ORACLE'S FIRST PART: flip qubits in each node based on the target chunk
                    for node, local_chunk in zip(self.nodes, target):
                        data_positions = list(range(1, node.qmemory.num_positions-1))
                        flip_positions = [position for position, bit in zip(data_positions, local_chunk) if bit == "0"]
                        flip_positions_by_node.append(flip_positions)
                        if flip_positions:
                            x_program = FlipBit(flip_positions)
                            node.qmemory.execute_program(x_program)
                            yield self.await_program(node.qmemory)

                    # ORACLE'S SECOND PART: distributed multi-controlled phase gate across nodes
                    # Entanglement generation between nodes and the router
                    for k, node in enumerate(self.nodes):
                        print(f"Starting entangler {k}")
                        entangler = Entangler(node, self.router, self.params, True)
                        entanglement_finished = self.await_signal(entangler, signal_label="Pair ready")
                        entangler.start()
                        yield entanglement_finished
                        print(f"{k} entangled")
                        exchange = Exchange(k + 1)
                        self.router.qmemory.execute_program(exchange)
                        yield self.await_program(self.router.qmemory)
                        print(f"{k} exchanged")

                    # Node measurement sending to the router
                    node_measurements_received = []
                    for node in self.nodes:
                    
                        print(f"Starting local node operations")
                        phase_program = MultiCPhase2(list(range(node.qmemory.num_positions-1)), np.pi)
                        node.qmemory.execute_program(phase_program)
                        yield self.await_program(node.qmemory)
                    
                        h_program = Hadamards([0])
                        node.qmemory.execute_program(h_program)
                        yield self.await_program(node.qmemory)
                    
                        meas_program = Measure(0, f"q0_{node.name}")
                        node.qmemory.execute_program(meas_program)
                        yield self.await_program(node.qmemory)
                        print(f"Local block completed on {node.name}")
                    
                        node_measurement = meas_program.output[f"q0_{node.name}"][0]
                        node.ports[f"cport_{node.name}_to_router"].tx_output(node_measurement)
                        router_port = self.router.ports[f"cport_router_from_{node.name}"]
                        reception = self.await_port_input(router_port)
                        yield reception
                        print("Router received message")
                        message = router_port.rx_input().items[0]
                        node_measurements_received.append(message)

                    # Operations on central node
                    router_operations = Hadamards([i for i in range(1, len(self.nodes) + 1)])
                    for j, m in enumerate(node_measurements_received):
                        if m == 1:
                            router_operations += FlipBit([j + 1])
                    router_operations += MultiCPhase2([i for i in range(1, len(self.nodes) + 1)], phase_f)
                    self.router.qmemory.execute_program(router_operations)
                    yield self.await_program(self.router.qmemory)

                    router_measurements = []
                    for router_position in range(1, len(self.nodes) + 1):
                        h_program = Hadamards([router_position])
                        self.router.qmemory.execute_program(h_program)
                        yield self.await_program(self.router.qmemory)
                        measurement_program = Measure(router_position, f"R_{router_position}")
                        self.router.qmemory.execute_program(measurement_program)
                        yield self.await_program(self.router.qmemory)
                        router_measurement = measurement_program.output[f"R_{router_position}"][0]
                        router_measurements.append(router_measurement)
                    
                    for node_index, node in enumerate(self.nodes):
                        router_measurement = router_measurements[node_index]
                        node_port = node.ports[f"cport_{node.name}_from_router"]
                        reception = self.await_port_input(node_port)
                        self.router.ports[f"cport_router_to_{node.name}"].tx_output(router_measurement)
                        yield reception
                        received_measurement = node_port.rx_input().items[0]
                        if received_measurement == 1:
                            correction = MultiCPhase2(list(range(1, node.qmemory.num_positions-1)), np.pi)
                            node.qmemory.execute_program(correction)
                            yield self.await_program(node.qmemory)

                    #ORACLE'S THIRD PART
                    for node_index, node in enumerate(self.nodes):
                        flip_positions = (flip_positions_by_node[node_index])
                        if flip_positions:
                            undo_x_program = FlipBit(flip_positions)
                            node.qmemory.execute_program(undo_x_program)
                            yield self.await_program(node.qmemory)                

                
                self.result = {"phase": phase, "implemented": True}

            else: 
                raise ValueError(f"Unknown phase: {phase}")

            if self.verbose:
                print(f"Global protocol completed phase: {phase}")

            self.send_signal(signal_label="GLOBAL_STEP_FINISHED", result={"phase": phase, "result": self.result})

class LocalDEGGAProtocol(NodeProtocol):

    """
    Protocol governing the phases in which non-distributed gates are applied in each node
    """

    def __init__(self, node, master, node_index, local_targets, verbose=False):

        if not isinstance(local_targets, list) or not all(isinstance(t, str) for t in local_targets):
            raise TypeError("local_targets must be a list of strings")
        if not isinstance(node, Node):
            raise TypeError("node must be an instance of netsquid.nodes.Node")
        if not isinstance(master, Protocol):
            raise TypeError("master must be an instance of netsquid.protocols.Protocol")

        super().__init__(node=node)

        self.master = master
        self.node_index = node_index
        self.local_targets = local_targets
        self.n_j = len(local_targets[0])  
        self.verbose = verbose
        self.result = None
        self.add_signal("LOCAL_STEP_FINISHED")

    def run(self):

        n_local_targets = len(self.local_targets)
        theta_nj, J_nj, phase_nj = long_parameters(self.n_j, n_local_targets)

        while True:

            yield self.await_signal(sender=self.master, signal_label="LOCAL_COMMAND")
            command = self.master.get_signal_result("LOCAL_COMMAND", receiver=self)
            phase = command["phase"]

            if phase == "STOP":
                return {"node_index": self.node_index, "result": self.result}
                
            if phase == "FIRST_LOCAL_STEP":

                oracle = None
                print(f"First local step in {self.node.name} : Initialization + Hadamards + (OracleDEGGA + DiffusionDEGGA)*(J_nj+1)")
                for target in self.local_targets:
                    target_oracle = OracleDEGGA(target, phase_nj)
                    if oracle is None:
                        oracle = target_oracle
                    else:
                        oracle += target_oracle

                program = Initialization("0" * (self.n_j + 1)) + Hadamards([i for i in range(1, self.n_j + 1)]) + (oracle + DiffusionDEGGA(self.n_j, phase_nj)) * (J_nj + 1)
                self.node.qmemory.execute_program(program)
                yield self.await_program(self.node.qmemory)


            elif phase == "SECOND_LOCAL_STEP":

                print(f"Second local step in {self.node.name} : (DiffusionDEGGA + OracleDEGGA) * (J_nj + 1) + Hadamards")
                oracle = None
                for target in self.local_targets:
                    target_oracle = OracleDEGGA(target, -phase_nj)
                    if oracle is None:
                        oracle = target_oracle
                    else:
                        oracle += target_oracle
                
                program = (DiffusionDEGGA(self.n_j, -phase_nj) + oracle)* (J_nj + 1) + Hadamards([i for i in range(1, self.n_j + 1)])
                self.node.qmemory.execute_program(program)
                yield self.await_program(self.node.qmemory)
                

            elif phase == "THIRD_LOCAL_STEP":

                oracle = None
                print(f"Third local step in {self.node.name} : Hadamards + (OracleDEGGA + DiffusionDEGGA)*(J_nj+1)")
                for target in self.local_targets:
                    target_oracle = OracleDEGGA(target, phase_nj)
                    if oracle is None:
                        oracle = target_oracle
                    else:
                        oracle += target_oracle
                            
                program = Hadamards([i for i in range(1, self.n_j + 1)]) + (oracle + DiffusionDEGGA(self.n_j, phase_nj)) * (J_nj + 1)
                self.node.qmemory.execute_program(program)
                yield self.await_program(self.node.qmemory)

            elif phase == "MEASURE":

                program = None

                measured_bits = []
                print(f"Qubit measurements in {self.node.name}")
                for i in range(1, self.n_j + 1):
                    measurement = Measure(i, output_key=f"q{i}")
                    if program is None:
                        program = measurement
                    else:
                        program += measurement

                self.node.qmemory.execute_program(program)
                yield self.await_program(self.node.qmemory)

                measured_bits = ""

                for i in range(1, self.n_j + 1):
                    measured_bits += str(program.output[f"q{i}"][0])
                
                self.result = measured_bits

            else:
                raise ValueError(f"Unknown phase: {phase}")

            if self.verbose:
                print(f"[{ns.sim_time():.2f} ns] Node {self.node_index} completed phase: {phase}")

            self.send_signal(signal_label="LOCAL_STEP_FINISHED", result={"node_index": self.node_index, "phase": phase, "result": self.result})

if __name__ == "__main__":

    targets_j = ["0000", "1111"]
    partition = [2, 2]

    processor_type = SnVProcessor2026
    parameter_set = NVParameterSet2026COMPUTAEX()

    shots = 20
    initial_seed = 20260909

    counts = Counter()
    failures = []
    wall_times = []

    for shot in range(shots):
        seed = initial_seed + shot

        ns.sim_reset()
        ns.set_random_state(seed=seed)

        master = DEGGAMasterProtocol(
            targets_j,
            partition,
            processor_type,
            parameter_set,
            processor_kwargs={"noiseless": False},
            verbose=False
        )

        wall_start = time.perf_counter()

        master.start()
        ns.sim_run()

        wall_time = time.perf_counter() - wall_start

        counts[master.result] += 1
        wall_times.append(wall_time)

        if not master.success:
            failures.append({
                "shot": shot,
                "seed": seed,
                "result": master.result
            })

        print(
            f"[{shot + 1:02d}/{shots}] "
            f"seed={seed}, "
            f"result={master.result}, "
            f"success={master.success}, "
            f"wall={wall_time:.2f} s"
        )

    successes = shots - len(failures)

    print()
    print("=" * 60)
    print("IDEAL DEGGA VALIDATION")
    print("=" * 60)
    print("Processor:", processor_type.__name__)
    print("Targets:", targets_j)
    print("Partition:", partition)
    print("Shots:", shots)
    print("Counts:", dict(counts))
    print("Successes:", successes)
    print("Failures:", len(failures))
    print("Success probability:", successes / shots)
    print(f"Mean wall time: {np.mean(wall_times):.3f} s")
    print(f"Wall-time standard deviation: {np.std(wall_times):.3f} s")

    if failures:
        print("Failed executions:")
        for failure in failures:
            print(failure)

    
