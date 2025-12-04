# S4: Shallower Single Setting Shadow

**Efficient Quantum Shadow Tomography via Optimized SIC-POVM Circuits. The core algorithm is modular and applicable to any task requiring IC-POVMs.**

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org/)
[![Qiskit](https://img.shields.io/badge/Qiskit-1.0%2B-6929C4)](https://qiskit.org/)
[![License](https://img.shields.io/badge/License-MIT-green)](https://opensource.org/licenses/MIT)
[![DOI](https://img.shields.io/badge/DOI-10.1103%2FPhysRevApplied.23.014021-blue)](https://doi.org/10.1103/PhysRevApplied.23.014021)

## 📖 Overview

**S4 (Shallower Single Setting Shadow)** is a quantum software framework designed to implement **Symmetric Informationally Complete POVMs (SIC-POVMs)** on gate-based quantum computers with minimal circuit depth.

The toolkit implements algorithms based on `Phys. Rev. Applied 23, 014021 (2025)`. Standard implementations of general POVMs often require significant overhead in terms of CNOT gates, making them vulnerable to hardware noise. This project implements a compilation strategy that reduces the cost of single-qubit SIC-POVMs from the standard **3 CNOTs** down to **1 CNOT** (or **at most 2 for general minimal IC-POVMs**).

This optimization enables **noise-resilient Shadow Estimation**, allowing for high-fidelity prediction of quantum state properties (such as GHZ state fidelity) even under realistic thermal relaxation ($T_1/T_2$) noise.

## 🛠️ Installation
### 1. Clone the repository:
```bash
git clone https://github.com/Yomure/Shallow-Single-Setting-Shadow.git
cd Shallow-Single-Setting-Shadow
```
### 2. Create a virtual environment (Recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```
### 3. Install dependencies:
```bash
pip install -r requirements.txt
```

## 📂 Project Structure

```text
S4-Project/
├── data/
│   └── sic_phi_vectors/       # Pre-computed fiducial vectors for SIC-POVMs
├── packages/                  # Core library
│   ├── decomposition.py       # Numerical tools (Gram-Schmidt, KAK)
│   ├── linalg.py              # Linear algebra helpers (Fidelity, Trace distance)
│   ├── povms.py               # Naimark dilation & measurement logic
│   ├── states.py              # Quantum state preparation (GHZ, Random, etc.)
│   ├── unitary.py             # S4 Compilation algorithms & gate synthesis
│   └── utils.py               # General utilities & visualization
├── examples/
│   ├── S4_Exp_GHZ_Fidelity_T1T2.ipynb        # Main Experiment: Benchmarking under noise
│   ├── S4_Exp_GHZ_Fidelity_noiseless.ipynb   # Main Experiment: Benchmarking without noise
│   ├── S4_Demo_Arbitrary_SIC_to_Bell.ipynb   # Theory: Verifying the compilation logic
├── requirements.txt           # Python dependencies
└── README.md                  # Project documentation
```

## 🚀 Usage & Examples

### 1. Running Benchmarks
To reproduce the results from the paper or test the noise resilience:
- Open `examples/S4_Exp_GHZ_Fidelity_T1T2.ipynb` in Jupyter Notebook.
- This notebook sets up a noise model, compiles the optimized circuit, and runs Shadow Estimation.
### 2. Understanding the Compilation
To understand how an arbitrary (SIC-)POVM is transformed into the optimized circuit:
- Check `examples/S4_Demo_Arbitrary_SIC_to_Bell.ipynb`.


## 📚 Citation
If you use this code or algorithm in your research, please cite the following paper:
```
@article{PhysRevApplied.23.014021,
  title = {Circuit Optimization of Informationally Complete Positive Operator-Valued Qubit Measurements for Shadow Estimation},
  author = {You, Zhou and Liu, Qing and Zhou, You},
  journal = {Phys. Rev. Applied},
  volume = {23},
  issue = {1},
  pages = {014021},
  numpages = {21},
  year = {2025},
  month = {Jan},
  publisher = {American Physical Society},
  doi = {10.1103/PhysRevApplied.23.014021},
  url = {https://link.aps.org/doi/10.1103/PhysRevApplied.23.014021}
}
```# Shallow-Singgle-Setting-Shadow
