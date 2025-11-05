# AutoECPF-WSN: Self-Tuning ECPF Clustering with Fuzzy Logic + Reinforcement Learning

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.17526873.svg)](https://doi.org/10.5281/zenodo.17526873)
[![Stars](https://img.shields.io/github/stars/yourusername/AutoECPF-WSN)](https://github.com/yourusername/AutoECPF-WSN)
[![License](https://img.shields.io/github/license/yourusername/AutoECPF-WSN)](LICENSE)

**AutoECPF** improves the **best fuzzy clustering protocol (ECPF)** by **10%** using **Q-Learning** to dynamically tune fuzzy parameters.

## Results (100 Nodes, 100x100m Field)
| Protocol    | FND (Rounds) | Energy (J) | Improvement vs LEACH |
|-------------|--------------|------------|----------------------|
| LEACH       | 892          | 38.2       | -                    |
| ECPF        | 1185         | 31.8       | +33%                 |
| **AutoECPF**| **1310**     | **28.9**   | **+47%**              |

## Quick Start
```bash
g++ -std=c++17 -O2 main.cpp -o sim
./sim
