// main.cpp - AutoECPF: Energy-efficient Clustering with Fuzzy + RL
// GitHub: https://github.com/yourusername/AutoECPF-WSN
#include <bits/stdc++.h>
using namespace std;

const int N = 100;
struct Node { double x, y, energy; string type; };
vector<Node> nodes(N);
mt19937 rng(chrono::steady_clock::now().time_since_epoch().count());

// --- Fuzzy Logic ---
float trimf(float x, float a, float b, float c) {
    if (x <= a || x >= c) return 0;
    return x <= b ? (x - a)/(b - a) : (c - x)/(c - b);
}

float fuzzy_inference(float e, float deg, float cent) {
    float low_e = trimf(e, 0, 0, 0.3f), med_e = trimf(e, 0.2, 0.5, 0.8), high_e = trimf(e, 0.7, 1, 1);
    float low_d = trimf(deg, 0, 2, 5), med_d = trimf(deg, 3, 5, 8), high_d = trimf(deg, 6, 10, 10);
    float low_c = trimf(cent, 0, 30, 60), med_c = trimf(cent, 40, 70, 100), high_c = trimf(cent, 80, 100, 100);

    float r1 = min({high_e, high_d, low_c}); 
    float r2 = min({high_e, med_d, low_c});  
    float r3 = min({med_e, high_d, low_c});  

    float num = r1*0.9 + r2*0.7 + r3*0.7;
    float den = r1 + r2 + r3;
    return den > 0 ? num/den : 0;
}

// --- RL for AutoECPF ---
class AutoECPF {
    vector<vector<float>> Q = vector(100, vector<float>(3, 0));
    float alpha = 0.1, gamma = 0.9, eps = 0.3;
public:
    float multiplier = 1.0;
    int get_state() {
        float avg_e = 0; int alive = 0;
        for (auto& n : nodes) if (n.energy > 0) { avg_e += n.energy; alive++; }
        avg_e /= alive; alive /= 10;
        return min((int)(avg_e*10), 9)*10 + min(alive, 9);
    }
    void select_CH() {
        int state = get_state();
        int action = (rand()%100 < eps*100) ? rand()%3 : 
                     max_element(Q[state].begin(), Q[state].end()) - Q[state].begin();
        multiplier = 0.9f + action * 0.1f;

        for (int i = 0; i < N; i++) {
            if (nodes[i].energy <= 0) continue;
            float e = nodes[i].energy / 0.5;
            int deg = 0; 
            float cent = hypot(nodes[i].x-50, nodes[i].y-50);
            float prob = fuzzy_inference(e, deg, cent) * multiplier;
            if (uniform_real_distribution<>(0,1)(rng) < prob) {
                nodes[i].type = "CH";
            }
        }

        float reward = alive_count() / 100.0;
        int next = get_state();
        Q[state][action] += alpha * (reward + gamma * *max_element(Q[next].begin(), Q[next].end()) - Q[state][action]);
    }
    int alive_count() { int c=0; for (auto& n:nodes) if (n.energy>0) c++; return c; }
};

int main() {

    AutoECPF sim;
    int round = 0;
    while (sim.alive_count() > 0) {
        sim.select_CH();
        round++;
        if (sim.alive_count() < 100) { cout << "FND: " << round << endl; break; }
    }
    return 0;
}