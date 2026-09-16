# Diamond color centers emulation on NetSquid
This repository contains a NetSquid-based distributed color center emulation platform. The framework allow us to run algorithms using NV or SnV QPUs with accurate noise dynamics. 

The long-term goal of the project is to extend this framework scale. Increasing the number of qubits and QPUs is not manageable in practice with laptops, so more powerful systems are needed.

## Project Overview
The code models a distributed quantum-computing architecture where each colour-centre node contains:
- One electronic spin used as communication/control/processing qubit.
- Nuclear spins used as data/memory qubits.
- an emission position for photonic entanglement generation
We include examples of application: DEGA and DEGGA


## Repository Structure
- DEGA.py 
- DEGGA.py
- primitives.py 
- processors.py 
- nv_2026.py 
- snv_2026.py 
- entanglement.py 
- telegate.py 

`DEGA.py`
We implement Distributed Exact Grover's Algorithm as an example of application on this emulator.

`DEGGA.py`
We implement Distributed Exact Generalized Grover's Algorithm as an example of application on this emulator.

`primitives.py`
Contains a set of hardware-aware primitives used by the algorithms.
- Initialization of qubits
- Bit flip
- Hadamard gates
- Physical CNOT
- Multi-controlled X/Z and phase gates
- Measurement
All of these primitives are built knowing that electron spin is our intermediary and processing qubit

`processors.py`
Defines the color-center quantum processors. We can consider them as the digital twins for the centers. These processors inherit from NetSquid's `QuantumProcessor` and define the physical instructions available in the emulated hardware. These instructions are chosen according to experiments. The structure of this class is taken from NetSquid-NV.

`nv_2026.py` and `snv_2026.py`
Parameter sets for the NV and SnV processor model. It contains values for:
- Coherence times
- Gate duration and fidelities
- Initialization and measurement errors
- Photon emission 
- Detection
These values are taken from recent experiments and their corresponding papers.

`entanglement.py`
Contains protocol useful to generate entanglement between quantum nodes. 

`telegate.py`
This module contains protocols involving the application of gates between two distant qubits 

## Physical Model

The processor model follows a colour-centre architecture with:

position 0       -> electronic spin / communication qubit
positions 1..n   -> nuclear spin data qubits
last position    -> photon emission position

Only operations available as physical instructions in the processor model are used directly. Higher-level logical gates are decomposed into these physical instructions.

For example, logical operations between two nuclear-spin qubits are mediated through the electronic spin using SWAP operations and electron-carbon two-qubit gates

## Requirements

This project requires:

- Python 3
- NumPy
- NetSquid
- netsquid-nv
- Custom local modules included in this repository

NetSquid is not distributed through the standard PyPI index. It must be installed following the official NetSquid installation instructions.

## Development Status

This repository is part of an ongoing research and emulation project. The current task is the implementation in the Lusitania supercomputer, controlled by COMPUTAEX (Extremadura Supercomputing Center)
