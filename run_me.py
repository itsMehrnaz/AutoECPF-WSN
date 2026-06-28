import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import random
from dataclasses import dataclass, field
from typing import List, Optional


NUM_NODES      = 100
NUM_ROUNDS     = 4000     
AREA_W         = 100.0
AREA_H         = 100.0
SINK_X         = 50.0
SINK_Y         = 50.0
INITIAL_ENERGY = 6.0
SINK_ENERGY    = 10000.0
E_TX           = 0.0015
E_RX           = 0.0018
E_CH_EXTRA     = 0.0020

# RL
NUM_STATES     = 12        
RL_ALPHA       = 0.4      
RL_GAMMA       = 0.95     


MULT_VALUES    = [0.70, 0.80, 0.90, 1.00, 1.10]
NUM_ACTIONS    = len(MULT_VALUES)


W_ENERGY       = 0.6       
W_NEIGHBOR     = 0.2
W_CENTRALITY   = 0.2
CH_TARGET      = 0.07   
FUZZY_SCALE    = CH_TARGET / 0.73


CH_MULT_VALUES = [0.70, 0.80, 0.85, 0.90, 0.95, 1.00]
NUM_CH_ACTIONS = len(CH_MULT_VALUES)



@dataclass
class SensorNode:
    node_id:   int
    x:         float
    y:         float
    energy:    float = INITIAL_ENERGY
    is_dead:   bool  = False
    is_ch:     bool  = False
    round_dead: int  = -1
    mult:      float = 0.9    

    # Global RL
    q_table:   np.ndarray = field(
        default_factory=lambda: np.zeros((NUM_STATES, NUM_ACTIONS)))
    rl_state:  int  = NUM_STATES - 1
    rl_action: int  = 2   

    # CH local RL
    ch_q:      np.ndarray = field(
        default_factory=lambda: np.zeros((NUM_STATES, NUM_CH_ACTIONS)))
    ch_action: int  = 2  
    ch_mult:   float = 0.85

    def get_state(self, e: Optional[float] = None) -> int:
        val = e if e is not None else self.energy

        s = int(val / 0.5)
        return max(0, min(NUM_STATES - 1, s))

    def distance_to_sink(self) -> float:
        return np.sqrt((self.x-SINK_X)**2 + (self.y-SINK_Y)**2)

    def fuzzy_ch_prob(self, nb: int) -> float:
        e_s = max(0.0, min(1.0, self.energy / INITIAL_ENERGY))
        n_s = max(0.0, min(1.0, nb / 40.0))
        max_d = np.sqrt(AREA_W**2 + AREA_H**2) / 2
        c_s = max(0.0, min(1.0, 1.0 - self.distance_to_sink() / max_d))
        raw  = W_ENERGY*e_s + W_NEIGHBOR*n_s + W_CENTRALITY*c_s
        p    = raw * FUZZY_SCALE * self.mult
        return max(0.0, min(1.0, p))

    def consume_energy(self, local_mult: float = 1.0):
        if self.is_ch:
            consumed = (E_TX + E_CH_EXTRA) * self.mult * local_mult
        else:
            consumed = (E_TX + E_RX) * self.mult
        self.energy -= consumed
        self.energy  = max(0.0, self.energy)

    def choose_action(self, q, n_act, eps) -> int:
        if random.random() < eps:
            return random.randint(0, n_act - 1)
        return int(np.argmax(q[self.get_state()]))

    def update_q(self, q, s, a, r, ns, n_act):
        old      = q[s][a]
        best     = np.max(q[ns])
        q[s][a]  = old + RL_ALPHA * (r + RL_GAMMA * best - old)


class SDNSink:
    def __init__(self):
        # FIX 5: Q-table با bias به سمت mult کم
        self.q_table = np.zeros((NUM_STATES, NUM_ACTIONS))

        for s in range(NUM_STATES):
            self.q_table[s][0] = 2.0   # mult=0.70
            self.q_table[s][1] = 1.5   # mult=0.80
            self.q_table[s][2] = 1.0   # mult=0.90
            self.q_table[s][3] = 0.0   # mult=1.00
            self.q_table[s][4] = -1.0  # mult=1.10

        self.rl_state    = NUM_STATES - 1
        self.rl_action   = 2    
        self.global_mult = 0.9
        self.prev_avg_e  = INITIAL_ENERGY
        self.mult_history  = []
        self.reward_history = []

    def get_state(self, avg_e: float) -> int:
        s = int(avg_e / 0.5)
        return max(0, min(NUM_STATES - 1, s))

    def compute_reward(self, cur_avg, prev_avg, dead_cnt,
                        alive_cnt, mult, rnd):
       
        r = 0.0


        energy_ratio = cur_avg / INITIAL_ENERGY
        r += energy_ratio * 10.0


        mult_bonus = (1.1 - mult) / 0.4   # 0 تا 1
        r += mult_bonus * 5.0


        if cur_avg < 3.0:
            r -= (3.0 - cur_avg) * 5.0
        if cur_avg < 1.5:
            r -= (1.5 - cur_avg) * 20.0


        if dead_cnt > 0:
            r -= dead_cnt * 100.0


        r += (alive_cnt / NUM_NODES) * 2.0

        return r

    def update(self, nodes: List[SensorNode], rnd: int):
        alive    = [n for n in nodes if not n.is_dead]
        dead_cnt = sum(1 for n in nodes if n.is_dead)
        if not alive:
            return

        avg_e  = np.mean([n.energy for n in alive])
        reward = self.compute_reward(avg_e, self.prev_avg_e,
                                      dead_cnt, len(alive),
                                      self.global_mult, rnd)
        ns     = self.get_state(avg_e)
        old_q  = self.q_table[self.rl_state][self.rl_action]
        best_n = np.max(self.q_table[ns])
        self.q_table[self.rl_state][self.rl_action] = (
            old_q + RL_ALPHA * (reward + RL_GAMMA * best_n - old_q))


        eps = max(0.05, 0.8 * (0.995 ** rnd))

        self.rl_state  = ns
        if random.random() < eps:
            self.rl_action = random.randint(0, NUM_ACTIONS - 1)
        else:
            self.rl_action = int(np.argmax(self.q_table[ns]))

        self.global_mult = MULT_VALUES[self.rl_action]
        self.prev_avg_e  = avg_e
        self.mult_history.append(self.global_mult)
        self.reward_history.append(reward)

    def broadcast(self, nodes):
        for n in nodes:
            if not n.is_dead:
                n.mult = self.global_mult


def ch_local_update(ch: SensorNode, members: List[SensorNode], rnd: int):
    if not members:
        return
    cluster_avg = np.mean([m.energy for m in members])
    state  = ch.get_state(cluster_avg)
    eps    = max(0.05, 0.6 * (0.997 ** rnd))


    e_ratio = cluster_avg / INITIAL_ENERGY
    reward  = e_ratio * 8.0
    reward += (1.0 - CH_MULT_VALUES[ch.ch_action]) * 4.0
    if cluster_avg < 2.0:
        reward -= (2.0 - cluster_avg) * 10.0

    ns = state
    ch.update_q(ch.ch_q, state, ch.ch_action, reward, ns, NUM_CH_ACTIONS)
    ch.ch_action = ch.choose_action(ch.ch_q, NUM_CH_ACTIONS, eps)
    ch.ch_mult   = CH_MULT_VALUES[ch.ch_action]


def run_simulation(protocol: str, seed: int = 42) -> dict:
    random.seed(seed)
    np.random.seed(seed)

    nodes = []
    for i in range(NUM_NODES):
        n = SensorNode(i+1, random.uniform(0,AREA_W),
                       random.uniform(0,AREA_H))
        if protocol in ('sdn_rl', 'hybrid', 'ecpf_rl'):

            for s in range(NUM_STATES):
                n.q_table[s][0] = 1.5   # 0.70
                n.q_table[s][1] = 1.0   # 0.80
                n.q_table[s][2] = 0.5   # 0.90
        nodes.append(n)

    sink = SDNSink()

    alive_h  = []
    energy_h = []
    mult_h   = []
    fnd = hnd = lnd = -1

    for rnd in range(1, NUM_ROUNDS + 1):
        alive = [n for n in nodes if not n.is_dead]
        if not alive:
            pad = NUM_ROUNDS - len(alive_h)
            alive_h  += [0]*pad
            energy_h += [0.0]*pad
            mult_h   += [mult_h[-1] if mult_h else 0.9]*pad
            break

        # Global SDN
        if protocol in ('sdn_rl', 'hybrid'):
            sink.update(alive, rnd)
            sink.broadcast(alive)


        eps = max(0.05, 0.8 * (0.995 ** rnd))


        for node in alive:
            nb = random.randint(5, 30)
            if protocol == 'leach':
                node.mult  = 1.0
                node.is_ch = random.random() < 0.05
            elif protocol == 'ecpf':
                node.mult  = 1.0
                node.is_ch = random.random() < node.fuzzy_ch_prob(nb)
            elif protocol == 'ecpf_rl':
                node.rl_action = node.choose_action(
                    node.q_table, NUM_ACTIONS, eps)
                node.mult  = MULT_VALUES[node.rl_action]
                node.is_ch = random.random() < node.fuzzy_ch_prob(nb)
            else:
                node.is_ch = random.random() < node.fuzzy_ch_prob(nb)

        # CH local RL (hybrid)
        if protocol == 'hybrid':
            chs = [n for n in alive if n.is_ch]
            mems = [n for n in alive if not n.is_ch]
            for ch in chs:
                near = sorted(mems,
                    key=lambda m: np.hypot(m.x-ch.x, m.y-ch.y))[:8]
                ch_local_update(ch, near, rnd)


        for node in alive:
            lm = (node.ch_mult
                  if (protocol == 'hybrid' and node.is_ch)
                  else 1.0)
            e_before = node.energy
            node.consume_energy(lm)

            if protocol == 'ecpf_rl':
                ns     = node.get_state()
                e_r    = node.energy / INITIAL_ENERGY
                reward = e_r * 10.0
                reward += (1.1 - node.mult) / 0.4 * 5.0
                if node.energy < 3.0:
                    reward -= (3.0 - node.energy) * 5.0
                node.update_q(node.q_table, node.rl_state,
                               node.rl_action, reward, ns, NUM_ACTIONS)
                node.rl_state  = ns
                node.rl_action = node.choose_action(
                    node.q_table, NUM_ACTIONS, eps)
                node.mult = MULT_VALUES[node.rl_action]

            if node.energy <= 0.001 and not node.is_dead:
                node.is_dead    = True
                node.round_dead = rnd

        alive_cnt = sum(1 for n in nodes if not n.is_dead)
        tot_e     = sum(n.energy for n in nodes)
        cm = (sink.global_mult
              if protocol in ('sdn_rl','hybrid')
              else np.mean([n.mult for n in alive]))

        alive_h.append(alive_cnt)
        energy_h.append(tot_e)
        mult_h.append(cm)

        if fnd < 0 and alive_cnt < NUM_NODES: fnd = rnd
        if hnd < 0 and alive_cnt <= NUM_NODES//2: hnd = rnd
        if lnd < 0 and alive_cnt == 0: lnd = rnd

    return dict(protocol=protocol, alive=alive_h,
                energy=energy_h, mult=mult_h,
                fnd=fnd, hnd=hnd, lnd=lnd,
                sink=sink if protocol in ('sdn_rl','hybrid') else None)


def run_multiple(protocol: str, runs: int = 10) -> dict:
    aa, ae, am, af, ah = [], [], [], [], []
    for seed in range(runs):
        r   = run_simulation(protocol, seed)
        pad = NUM_ROUNDS - len(r['alive'])
        aa.append(r['alive']  + [0]*pad)
        ae.append(r['energy'] + [0.0]*pad)
        lm = r['mult'][-1] if r['mult'] else 0.9
        am.append(r['mult']   + [lm]*pad)
        if r['fnd'] > 0: af.append(r['fnd'])
        if r['hnd'] > 0: ah.append(r['hnd'])
    return dict(
        protocol    = protocol,
        alive_mean  = np.mean(aa, axis=0),
        alive_std   = np.std(aa,  axis=0),
        energy_mean = np.mean(ae, axis=0),
        energy_std  = np.std(ae,  axis=0),
        mult_mean   = np.mean(am, axis=0),
        fnd_mean    = np.mean(af) if af else 0,
        fnd_std     = np.std(af)  if af else 0,
        hnd_mean    = np.mean(ah) if ah else 0,
        hnd_std     = np.std(ah)  if ah else 0,
    )


def plot_results(results: dict):
    rounds = np.arange(1, NUM_ROUNDS + 1)
    colors = dict(leach='#e74c3c', ecpf='#f39c12',
                  ecpf_rl='#3498db', sdn_rl='#9b59b6',
                  hybrid='#2ecc71')
    labels = dict(
        leach   = 'LEACH',
        ecpf    = 'ECPF (Fuzzy)',
        ecpf_rl = 'ECPF+RL (Distributed)',
        sdn_rl  = 'SDN+RL (Global)',
        hybrid  = 'SD-ClusterSkeleton Hybrid (Ours)')

    fig = plt.figure(figsize=(18, 12))
    fig.suptitle(
        'SD-ClusterSkeleton v4: ECPF + Q-Learning + SDN Hybrid\n'
        f'({NUM_NODES} Nodes | {NUM_ROUNDS} Rounds | {INITIAL_ENERGY}J | 10 runs)',
        fontsize=13, fontweight='bold')
    gs = gridspec.GridSpec(2, 3, hspace=0.42, wspace=0.35)


    ax1 = fig.add_subplot(gs[0, :2])
    for k, r in results.items():
        lw = 2.8 if k == 'hybrid' else 1.8
        ls = '-' if k in ('hybrid','sdn_rl') else '--'
        ax1.plot(rounds, r['alive_mean'],
                 color=colors[k], label=labels[k],
                 linewidth=lw, linestyle=ls)
        ax1.fill_between(rounds,
                         np.maximum(r['alive_mean']-r['alive_std'], 0),
                         np.minimum(r['alive_mean']+r['alive_std'], NUM_NODES),
                         color=colors[k], alpha=0.10)
    ax1.set_xlabel('Round', fontsize=11)
    ax1.set_ylabel('Alive Nodes', fontsize=11)
    ax1.set_title('Network Lifetime — Alive Nodes Over Time', fontsize=12)
    ax1.legend(fontsize=8, loc='upper left')
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, NUM_ROUNDS)
    ax1.set_ylim(0, NUM_NODES + 5)


    ax2 = fig.add_subplot(gs[0, 2])
    protos = list(results.keys())
    x  = np.arange(len(protos))
    w  = 0.35
    fv = [results[p]['fnd_mean'] for p in protos]
    hv = [results[p]['hnd_mean'] for p in protos]
    fs = [results[p]['fnd_std']  for p in protos]
    hs = [results[p]['hnd_std']  for p in protos]
    b1 = ax2.bar(x-w/2, fv, w, yerr=fs, label='FND',
                 color=[colors[p] for p in protos], capsize=3, alpha=0.9)
    b2 = ax2.bar(x+w/2, hv, w, yerr=hs, label='HND',
                 color=[colors[p] for p in protos], capsize=3, alpha=0.55)
    for bar in list(b1)+list(b2):
        h = bar.get_height()
        if h > 50:
            ax2.text(bar.get_x()+bar.get_width()/2, h+20,
                     f'{int(h)}', ha='center', va='bottom', fontsize=7)
    ax2.set_xticks(x)
    ax2.set_xticklabels([labels[p].split('(')[0][:10]
                         for p in protos], fontsize=7, rotation=20)
    ax2.set_ylabel('Round')
    ax2.set_title('FND & HND Comparison', fontsize=12)
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3, axis='y')


    ax3 = fig.add_subplot(gs[1, :2])
    for k, r in results.items():
        lw = 2.8 if k == 'hybrid' else 1.8
        ls = '-' if k in ('hybrid','sdn_rl') else '--'
        ax3.plot(rounds, r['energy_mean'],
                 color=colors[k], label=labels[k],
                 linewidth=lw, linestyle=ls)
        ax3.fill_between(rounds,
                         np.maximum(r['energy_mean']-r['energy_std'], 0),
                         r['energy_mean']+r['energy_std'],
                         color=colors[k], alpha=0.10)
    ax3.set_xlabel('Round', fontsize=11)
    ax3.set_ylabel('Total Remaining Energy (J)', fontsize=11)
    ax3.set_title('Total Network Energy Over Time', fontsize=12)
    ax3.legend(fontsize=8)
    ax3.grid(True, alpha=0.3)
    ax3.set_xlim(0, NUM_ROUNDS)


    ax4 = fig.add_subplot(gs[1, 2])
    for k in ('sdn_rl', 'hybrid'):
        if k not in results: continue
        m   = results[k]['mult_mean']
        ax4.plot(rounds, m, color=colors[k], alpha=0.3, linewidth=0.8)
        win = 150
        if len(m) >= win:
            roll = np.convolve(m, np.ones(win)/win, mode='valid')
            ax4.plot(rounds[win-1:], roll, color=colors[k],
                     linewidth=2.5, label=f'{labels[k][:18]} (avg)')
    for v, c, l in zip([0.7,0.8,0.9,1.0,1.1],
                        ['darkgreen','green','lime','orange','red'],
                        ['0.70','0.80','0.90','1.00','1.10']):
        ax4.axhline(y=v, color=c, linestyle=':', alpha=0.6, linewidth=1)
    ax4.set_xlabel('Round', fontsize=11)
    ax4.set_ylabel('Global Multiplier', fontsize=11)
    ax4.set_title('RL Multiplier Convergence (Epsilon Decay)', fontsize=12)
    ax4.set_ylim(0.60, 1.20)
    ax4.legend(fontsize=8)
    ax4.grid(True, alpha=0.3)

    plt.savefig('sd_clusterskeleton_v4_results.png',
                dpi=150, bbox_inches='tight')
    plt.show()
    print("نمودار: sd_clusterskeleton_v4_results.png")


def print_table(results: dict):
    leach_fnd = results['leach']['fnd_mean']
    print("\n" + "═"*75)
    print(f"  SD-ClusterSkeleton v4 — {NUM_NODES} nodes | "
          f"{NUM_ROUNDS} rounds | 10 runs")
    print("═"*75)
    print(f"  {'Protocol':<30} {'FND':>12} {'HND':>12} {'vs LEACH':>10}")
    print("─"*75)
    lbls = dict(leach='LEACH', ecpf='ECPF (Fuzzy)',
                ecpf_rl='ECPF+RL (Distributed)',
                sdn_rl='SDN+RL (Global only)',
                hybrid='SD-ClusterSkeleton Hybrid')
    for k, r in results.items():
        fnd  = r['fnd_mean']
        hnd  = r['hnd_mean']
        fs   = r['fnd_std']
        hs   = r['hnd_std']
        impr = ((fnd-leach_fnd)/leach_fnd*100) if leach_fnd else 0
        mark = " ★" if k == 'hybrid' else ""
        print(f"  {lbls[k]:<30} {fnd:>5.0f}±{fs:<4.0f}  "
              f"{hnd:>5.0f}±{hs:<4.0f}  "
              f"{'+' if impr>=0 else ''}{impr:.1f}%{mark}")
    print("═"*75)
    hybrid_impr = ((results['hybrid']['fnd_mean']-leach_fnd)/leach_fnd*100)
    print(f"\n  هدف: +15%   نتیجه: +{hybrid_impr:.1f}%")
    if hybrid_impr >= 15:
        print("  به هدف رسیدیم!")
    else:
        print(f"    {15-hybrid_impr:.1f}% بیشتر نیاز است")


if __name__ == "__main__":
    print("SD-ClusterSkeleton v4 — Optimized for +15%\n")
    print("تغییرات کلیدی:")
    print("  ✓ MULT_VALUES: [0.7, 0.8, 0.9, 1.0, 1.1] (5 action)")
    print("  ✓ Shaped Reward: energy_ratio × 10 + mult_bonus × 5")
    print("  ✓ Epsilon decay: 0.8 × 0.995^round → 0.05")
    print("  ✓ Q-table bias: action کم‌مصرف از ابتدا بالاتر")
    print("  ✓ NUM_ROUNDS: 4000 (واضح‌تر)")
    print()

    protocols = ['leach','ecpf','ecpf_rl','sdn_rl','hybrid']
    names     = ['LEACH','ECPF','ECPF+RL','SDN+RL','SD-Hybrid']
    results   = {}

    for proto, name in zip(protocols, names):
        print(f"  [{name}] × 10...", end=" ", flush=True)
        results[proto] = run_multiple(proto, runs=10)
        r = results[proto]
        print(f"FND={r['fnd_mean']:.0f} ✓")

    print_table(results)
    plot_results(results)
