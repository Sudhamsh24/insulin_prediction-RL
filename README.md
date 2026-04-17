Hybrid SAC-TD3 Artificial Pancreas System (APS)
This repository contains a safety-critical Reinforcement Learning (RL) framework for autonomous insulin dosing in Type 1 Diabetes management. By merging Soft Actor-Critic (SAC) and Twin Delayed Deep Deterministic Policy Gradient (TD3), this project achieves superior glycemic control in the biorealistic simglucose simulator.

📌 Project Overview
Diabetes management requires precise insulin dosing where incorrect decisions lead to severe hypoglycemia. This project develops a closed-loop controller that balances aggressive glucose correction with a multi-layered safety framework.

Key Results
TD3 Baseline: 72.27% Time in Range (TIR)

Hybrid SAC-TD3: 78.78% Time in Range (TIR)

Improvement: +6.51% absolute gain in TIR using the hybrid architecture.

🏗️ System Architecture
The agent is trained on a cohort of 10 virtual adult patients using a Realistic Meal Scenario (stochastic meal times and carb counts).

1. Hybrid SAC-TD3 Agent
The model leverages:

TD3’s Clipped Double-Q Learning: To prevent the overestimation of Q-values.

SAC’s Entropy Regularization: To encourage robust exploration and prevent the policy from collapsing into sub-optimal local minima.

2. Multi-Layered Safety Framework
Safety is enforced at three distinct levels:

State Level: Includes a PK/PD-modeled Insulin On Board (IOB) feature to prevent insulin stacking.

Reward Level: An exponentially punitive reward function for any blood glucose (BG) readings below 70 mg/dL.

Action Level: A Rule-Based Safety Layer that acts as a hard fail-safe, blocking or reducing insulin doses based on real-time BG velocity and IOB limits.

📂 Repository Structure
Bash
├── agents/
│   ├── sac_agent.py          # SAC Implementation
│   ├── td3_agent.py          # TD3 Implementation
│   └── hybrid_agent.py       # Integrated SAC-TD3 Hybrid
├── utils/
│   ├── safety_layer.py       # Rule-based clinical constraints
│   ├── state_management.py   # IOB calculation and Reward shaping
│   └── replay_buffer.py      # Experience replay
├── td3_complete_training.py  # Main parallel training script
└── results/                  # Evaluation plots and summary CSVs
🚀 Getting Started
Prerequisites
Python 3.9+

simglucose (T1D Simulator)

gymnasium

pytorch

Installation
Bash
git clone https://github.com/your-username/hybrid-aps-rl.git
cd hybrid-aps-rl
pip install -r requirements.txt
Running Training
The project uses multiprocessing to train across all 10 adult patients simultaneously across available GPUs:

Bash
python td3_complete_training.py
📊 Evaluation
The evaluation scenario tests the agent against a standardized day (3 meals: 45g, 70g, 80g). The hybrid model demonstrates a smoother glucose trajectory and faster recovery from postprandial spikes without triggering hypoglycemic events.

📜 Abstract
Diabetes management requires precise insulin dosing, where incorrect predictions can lead to severe health consequences. This project proposes a hybrid Reinforcement Learning approach using SAC and TD3 architectures to capture complex insulin response dynamics. By integrating a layered safety framework—including IOB-aware states and hard clinical constraints—the hybrid model achieves an average TIR of 78.78%, outperforming the standalone TD3 baseline (72.27%). This system provides a reliable, scalable, and safety-first solution for autonomous glucose regulation.
