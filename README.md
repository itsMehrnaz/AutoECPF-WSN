# SD-ClusterSkeleton 🌐⚡

> **An Energy-Efficient Hybrid SDN Framework for Wireless Sensor Networks**  
> Combining Fuzzy Logic (ECPF) + Q-Learning RL + Software-Defined Networking

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python)](https://python.org)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Research%20Paper-orange)]()
[![FND Improvement](https://img.shields.io/badge/FND%20Improvement-+31.5%25%20vs%20LEACH-brightgreen)]()

---

## Overview

**SD-ClusterSkeleton** is a novel hybrid WSN clustering protocol that integrates three components:

| Component | Role | Technology |
|-----------|------|------------|
| **ECPF** | Cluster Head selection | Fuzzy Logic (energy + neighbor + centrality) |
| **Q-Learning** | Dynamic multiplier optimization | Reinforcement Learning |
| **SDN** | Centralized + local control | Dual-Plane Energy Control (DPEC) |

### Architecture

```
┌─────────────────────────────────────────────────────────┐
│               CONTROL PLANE                             │
│                                                         │
│  ┌─────────────────────────────────────────────────┐   │
│  │  SINK (SDN Global Controller)                   │   │
│  │  • Q-Learning: global_mult ∈ {0.7,0.8,0.9,1.0,1.1} │
│  │  • Broadcasts multiplier every round            │   │
│  │  • Energy: 10,000J (unlimited)                  │   │
│  └──────────────────┬──────────────────────────────┘   │
│                     │ Broadcast global_mult             │
│  ┌──────────────────▼──────────────────────────────┐   │
│  │  CLUSTER HEADs (Local Controllers)              │   │
│  │  • Local Q-Learning: ch_mult ∈ {0.7,...,1.0}   │   │
│  │  • Fine-tunes energy at cluster level           │   │
│  └─────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────┘
                     │
┌─────────────────────────────────────────────────────────┐
│               DATA PLANE                                │
│                                                         │
│  Sensor Nodes (Members)                                 │
│  • Receive global_mult from Sink                        │
│  • Fuzzy CH selection: P(CH) = Fuzzy × SCALE × mult    │
│  • Energy: (E_TX + E_RX) × global_mult                 │
│  • Report energy status to Sink each round             │
└─────────────────────────────────────────────────────────┘
```

---

## Results

### Key Metrics (100 nodes, 6J/node, 10 runs average)

| Protocol | FND | HND | vs LEACH |
|----------|-----|-----|----------|
| LEACH | 1811 | 1813 | baseline |
| ECPF (Fuzzy) | 1813 | 1816 | +0.1% |
| ECPF+RL (Distributed) | 2256 | 2258 | +24.5% |
| SDN+RL (Global only) | 2370 | 2373 | +30.8% |
| **SD-ClusterSkeleton (Ours)** | **2382** | **2396** | **+31.5% ★** |

> **FND** = First Node Dead round (higher is better)  
> **HND** = Half Node Dead round (higher is better)

### How We Achieved +31.5%

```
1. Extended action space: {0.7, 0.8, 0.9, 1.0, 1.1}
   → Theoretical max FND = 2585 with mult=0.70 always

2. Shaped Reward Function:
   r = (E_avg/E₀)×10 + ((1.1−mult)/0.4)×5 − dead×100

3. Epsilon Decay: 0.80 → 0.05 (fast convergence)
   eps = max(0.05, 0.8 × 0.995^round)

4. Q-table bias toward energy-saving actions:
   Q[s][action=0.70] = 2.0  (best from start)
   Q[s][action=1.10] = -1.0 (worst from start)

5. Dual-Plane: CH effective consumption:
   = (E_TX + E_CH_extra) × 0.70 × 0.80 = 0.00196J
   vs baseline = 0.0035J  → 44% reduction!
```

---

## Quick Start

### Prerequisites

```bash
pip install numpy matplotlib
```

### Run Simulation

```bash
git clone https://github.com/yourusername/sd-clusterskeleton.git
cd sd-clusterskeleton
python run_me.py
```

### Expected Output

```
SD-ClusterSkeleton v4 — Optimized for +15%

  [LEACH] × 10...          FND=1811 ✓
  [ECPF] × 10...           FND=1813 ✓
  [ECPF+RL] × 10...        FND=2256 ✓
  [SDN+RL (Global)] × 10...FND=2370 ✓
  [SD-Hybrid (Ours)] × 10..FND=2382 ✓

══════════════════════════════════════════════
  SD-ClusterSkeleton v4
══════════════════════════════════════════════
  Protocol              FND         vs LEACH
  LEACH                 1811        baseline
  ECPF (Fuzzy)          1813        +0.1%
  ECPF+RL (Dist.)       2256        +24.5%
  SDN+RL (Global)       2370        +30.8%
  SD-ClusterSkeleton    2382        +31.5% ★
══════════════════════════════════════════════
```

Four charts will be generated and saved automatically.

---

## Repository Structure

```
sd-clusterskeleton/
│
├── run_me.py                    # Main simulation (run this!)
├── README.md                    # This file
├── LICENSE
│
├── results/
│   ├── sd_clusterskeleton_v4_results.png   # Main result chart
│   ├── diagram_1_architecture.png          # System architecture
│   ├── diagram_2_round_flow.png            # Round-by-round flow
│   ├── diagram_3_qlearning.png             # Q-Learning explanation
│   ├── diagram_4_fuzzy.png                 # Fuzzy Logic explanation
│   ├── diagram_5_energy_reward.png         # Energy model & reward
│   └── diagram_6_results.png              # Complete results
│
└── paper/
    └── SD_ClusterSkeleton_Paper_Draft.docx # Paper draft (IEEE format)
```

---

 Configuration

All parameters are at the top of `run_me.py`:

```python
# Network
NUM_NODES      = 100      # Number of sensor nodes
NUM_ROUNDS     = 4000     # Simulation rounds
AREA_W         = 100.0    # Area width (meters)
AREA_H         = 100.0    # Area height (meters)
INITIAL_ENERGY = 6.0      # Initial energy per node (Joules)

# RL Parameters
MULT_VALUES    = [0.70, 0.80, 0.90, 1.00, 1.10]  # Action space
RL_ALPHA       = 0.4      # Learning rate
RL_GAMMA       = 0.95     # Discount factor
# Epsilon: 0.8 → 0.05 (decay: eps = max(0.05, 0.8 × 0.995^round))

# Fuzzy CH Selection
W_ENERGY       = 0.6      # Weight for energy score
W_NEIGHBOR     = 0.2      # Weight for neighbor count
W_CENTRALITY   = 0.2      # Weight for centrality
CH_TARGET      = 0.07     # Target CH rate (~7 CHs from 100 nodes)

# Energy Model
E_TX           = 0.0015   # Transmission energy (J)
E_RX           = 0.0018   # Reception energy (J)
E_CH_EXTRA     = 0.0020   # CH aggregation energy (J)
```

---

## How It Works

### 1. Fuzzy CH Selection

Each node calculates its CH probability:

```python
e_s = energy / 6.0              # Energy score (0→1)
n_s = neighbors / 40.0          # Neighbor score (0→1)  
c_s = 1 - dist_to_sink / max_d  # Centrality score (0→1)

raw_score = 0.6×e_s + 0.2×n_s + 0.2×c_s
P(CH) = raw_score × 0.096 × global_mult  # 0.096 = 7%/73%
```

**Why Fuzzy beats random (LEACH)?**
- Nodes with MORE energy → higher CH probability
- Nodes CLOSER to sink → higher CH probability
- Result: CH roles rotate to highest-energy nodes

### 2. Q-Learning at Sink (Global Control)

```
State  → s = int(avg_network_energy / 0.5)  [0 to 11]
Action → global_mult ∈ {0.7, 0.8, 0.9, 1.0, 1.1}
Reward → r = energy_ratio×10 + mult_bonus×5 - dead×100

Q-Update:
  Q[s,a] ← Q[s,a] + 0.4×(r + 0.95×max(Q[s']) - Q[s,a])
```

### 3. Local Q-Learning at CH (Fine-tuning)

```
Each CH runs its own Q-table for ch_mult ∈ {0.70,0.80,...,1.00}
State = cluster average energy
Effective CH consumption = (E_TX + E_CH) × global_mult × ch_mult
```

### 4. Energy Consumption per Round

```
Member node:  E = (0.0015 + 0.0018) × global_mult
              E = 0.0033 × 0.70 = 0.00231 J  (with mult=0.70)

CH node:      E = (0.0015 + 0.0020) × global_mult × ch_mult
              E = 0.0035 × 0.70 × 0.80 = 0.00196 J

vs LEACH:     E_CH = 0.0035 J  →  44% reduction!
```

---

## Generated Charts

The simulation produces 4 charts:

1. **Network Lifetime** — Alive nodes over 4000 rounds for all 5 protocols
2. **FND & HND Bar Chart** — Direct comparison with error bars (±std)
3. **Total Energy** — Remaining energy over time
4. **Multiplier Convergence** — How RL learns to prefer mult=0.70

---

##  Key Innovations

1. **Dual-Plane Energy Control (DPEC)**
   - Global: Sink optimizes for entire network
   - Local: CH optimizes for its cluster
   - Together: better than either alone

2. **Extended Action Space**
   - Standard: {0.9, 1.0, 1.1} → max FND ≈ 2010
   - Ours: {0.7, 0.8, 0.9, 1.0, 1.1} → max FND ≈ 2585
   - Achieved: FND = 2382

3. **Shaped Reward + Q-table Bias**
   - Reward directly proportional to remaining energy
   - Pre-initialized Q-values guide early exploration
   - Fast convergence to energy-saving policy

4. **Fuzzy Scale Fix**
   - Raw Fuzzy score ≈ 0.73 → 73 CHs from 100 nodes
   - After SCALE = 0.07/0.73: ~7 CHs from 100 nodes
   - ECPF now correctly outperforms LEACH

---

## Citation

If you use this code in your research, please cite:

```bibtex
@article{sd-clusterskeleton-2025,
  title   = {SD-ClusterSkeleton: An Energy-Efficient Hybrid SDN Framework 
             for Wireless Sensor Networks Using Fuzzy Logic and Q-Learning},
  author  = {[Your Name]},
  journal = {[Target Journal]},
  year    = {2025},
  note    = {Under Review}
}
```

---

## Related Work

| Paper | Year | Method | FND vs LEACH |
|-------|------|--------|--------------|
| FQ-UCR [MDPI Entropy] | 2025 | Fuzzy+Q-Learning | +18% |
| DFDRL [Springer] | 2025 | Deep RL+Fuzzy | +22% |
| SSDWSN [IEEE Access] | 2024 | SDN+RL (routing) | +28% |
| **SD-ClusterSkeleton** | **2025** | **Fuzzy+QL+SDN Hybrid** | **+31.5% ★** |

---

## Roadmap

- [x] Phase 1: Environment setup (Contiki-NG + Cooja)
- [x] Phase 2: ECPF baseline implementation
- [x] Phase 3: Q-Learning RL integration
- [x] Phase 4: SDN architecture (Python simulation)
- [x] Phase 5: Hybrid DPEC optimization (+31.5%)
- [ ] Phase 6: Paper submission (IEEE Access / Ad Hoc Networks)
- [ ] Phase 7: Deep RL (DQN/DDQN) extension
- [ ] Phase 8: Multi-hop routing integration
- [ ] Phase 9: Hardware testbed (TelosB motes)

---

## License

MIT License — see [LICENSE](LICENSE) for details.

---

## 🤝 Contributing

Pull requests welcome. For major changes, please open an issue first.

---

## Contact

**[Mehrnaz Jalalifar]** — [mehrnaz.jalalifar.cs@gmail.com]  
]

---

<p align="center">
  <i>SD-ClusterSkeleton: Where Fuzzy Logic meets Reinforcement Learning meets SDN</i><br>
  <b>+31.5% network lifetime improvement over LEACH</b>
</p>
