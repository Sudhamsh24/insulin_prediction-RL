# 🩺 Hybrid SAC-TD3 Agent

A safety-critical Reinforcement Learning (RL) framework for autonomous insulin dosing in Type 1 Diabetes management.
This project combines Soft Actor-Critic (SAC) and Twin Delayed Deep Deterministic Policy Gradient (TD3) to achieve robust and safe glycemic control in the simglucose simulator.

## 📌 Project Overview

Diabetes management requires precise insulin dosing, where incorrect decisions can lead to severe hypoglycemia or hyperglycemia.

This project implements a closed-loop Artificial Pancreas System (APS) that:

Maintains glucose within a safe range
Balances aggressive correction with safety constraints
Uses a hybrid RL approach for improved performance and stability
##📊 Key Results
Model	Time in Range (TIR)
SAC Baseline  70.85
TD3 Baseline	72.27%
Hybrid SAC-TD3	78.78%

✅ +6.51% absolute improvement in glycemic control using the hybrid model

## 🏗️ System Architecture

The agent is trained on a cohort of 10 virtual adult patients using a realistic stochastic meal scenario.

🤖 Hybrid SAC-TD3 Agent

This model combines the strengths of both algorithms:

TD3 (Twin Delayed DDPG)
Clipped Double-Q Learning
Reduces Q-value overestimation
SAC (Soft Actor-Critic)
Entropy Regularization
Encourages stable and robust exploration
🛡️ Multi-Layered Safety Framework

Safety is enforced at three levels:

1. State-Level Safety
Incorporates Insulin On Board (IOB)
Prevents insulin stacking using PK/PD modeling
2. Reward-Level Safety
Strong penalty for:
Blood Glucose < 70 mg/dL
Uses exponential punishment for hypoglycemia
3. Action-Level Safety
Rule-based safety layer:
Blocks unsafe insulin doses
Uses BG trends + IOB constraints
## 📂 Repository Structure
├── agents/
│   ├── sac_agent.py        # SAC implementation
│   ├── td3_agent.py        # TD3 implementation
│   └── hybrid_agent.py     # Hybrid SAC-TD3 agent
│
├── utils/
│   ├── safety_layer.py     # Rule-based safety constraints
│   ├── state_management.py # IOB + reward shaping
│   └── replay_buffer.py    # Experience replay
│
├── td3_complete_training.py  # Main training script
├── results/                  # Evaluation plots & metrics
└── README.md
## 🚀 Getting Started
📦 Prerequisites
Python 3.9+
simglucose (T1D Simulator)
gymnasium
pytorch
🔧 Installation
git clone https://github.com/your-username/hybrid-aps-rl.git
cd hybrid-aps-rl
pip install -r requirements.txt
▶️ Run Training

Train across 10 virtual patients in parallel:

python td3_complete_training.py
## 📊 Evaluation

The model is tested on a standardized meal scenario:

3 meals: 45g, 70g, 80g carbs
📈 Observations:
Smoother glucose trajectories
Faster recovery from meal spikes
No hypoglycemic events
## 📜 Abstract

Diabetes management requires precise insulin dosing, where incorrect predictions can lead to severe health consequences. This project proposes a hybrid Reinforcement Learning framework combining SAC and TD3 to model complex insulin-glucose dynamics.

By integrating a multi-layered safety framework—including IOB-aware states, reward penalties, and rule-based action constraints—the system achieves 78.78% Time in Range (TIR), outperforming the TD3 baseline (72.27%).

This work demonstrates a reliable, scalable, and safety-first approach for autonomous glucose regulation.

## ⭐ Future Work
Real patient data validation
Deployment on embedded medical devices
Integration with continuous glucose monitoring (CGM) systems
