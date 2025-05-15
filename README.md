
# 🧠 RAG + PPO: Reinforcement Learning Meets Retrieval in Social Dilemmas

## 📌 Overview
This project explores a hybrid RAG (Retrieval-Augmented Generation) and PPO (Proximal Policy Optimization) approach to teach LLM-based agents socially adaptive behavior in the **Iterated Prisoner's Dilemma** — a classic setting for studying trust, cooperation, and betrayal.

## 🧠 Motivation
Traditional reinforcement learning agents struggle with **human-like reasoning** in social games. By introducing **retrieval mechanisms**, we allow agents to reference **past behaviors** and adapt more intelligently — similar to how humans learn from memory.

---

## ⚙️ Architecture

### ✅ Components:
- **Environment**: Iterated Prisoner's Dilemma (IPD)
- **Agent**: Language model (LLM-based policy)
- **Retriever**: Fetches similar behavioral episodes
- **History Buffer**: Stores `(opponent policy, moves, rewards)`
- **PPO Trainer**: Optimizes the agent’s strategy over time

### 🔄 Workflow:
1. Agent plays repeated rounds of IPD.
2. Every *N* steps:
   - Retrieve top-*k* similar past episodes.
   - Combine with current observation as input.
3. Agent generates next action (Cooperate/Defect).
4. Environment returns reward.
5. PPO updates the policy based on total episodic return.

---

## 🧪 Evaluation: Fixed Opponent Benchmarks

| Opponent Policy        | Description                              | PPO Behavior Summary             | Success |
|------------------------|------------------------------------------|----------------------------------|---------|
| Tit for Tat            | Cooperates, mimics last move             | Learned to cooperate             | ✅      |
| Always Defect          | Always defects                           | Learned to defect in return      | ✅      |
| Always Cooperate       | Always cooperates                        | Exploited with defect            | ✅      |
| Friedman               | Punishes once defected                   | Early defection triggered loss   | ❌      |
| Joss                   | 90% cooperation, 10% random defect       | Mostly cooperative               | ✅      |
| Tester                 | Cooperates, then defects permanently     | Detected pattern, adapted        | ✅      |
| Backstabber            | Cooperates 5 rounds, then defects always | Delayed detection, adapted late  | ✅      |

---

## 💡 Why It Matters

- 🧠 **Memory-Augmented Decision-Making**: Simulates how humans use memory + feedback.
- 📚 **Education Modeling**: Extendable to student-agent modeling based on thinking patterns.
- 🤖 **Social Agent Foundations**: Step toward adaptive, context-aware, LLM-powered agents in dynamic interactions.

---

## 🛠️ Future Directions
- Expand to 1v1 adaptive LLM vs LLM agents
- Fine-tune models on specific interaction traits (e.g., trust, retaliation)
- Visualize evolving policy strategies over time

---

## 📁 Folder Structure
rag-ppo-ipd/
├── agents/ # LLM policy + retriever wrapper
├── env/ # Prisoner's Dilemma environment
├── trainer/ # PPO implementation
├── data/ # Episode history buffer
├── notebooks/ # Experiments and visualizations
└── README.md # You are here 🚀

yaml
Copy
Edit

---

## 📜 Citation / Inspiration
If you're inspired by this work, feel free to fork, cite, or reach out!


