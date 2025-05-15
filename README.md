🧠 Retrieval-Augmented Reinforcement Learning (RAG + PPO) in Social Dilemmas
🧩 Problem Overview
In multi-agent environments like the Iterated Prisoner's Dilemma, teaching LLM-based agents to behave in socially intelligent ways (cooperation, retaliation, forgiveness) remains a challenge. This project explores a hybrid architecture combining:

Retrieval-Augmented Generation (RAG): Supplies behavioral context/examples from prior interactions.

Proximal Policy Optimization (PPO): Reinforces actions that lead to higher long-term rewards (e.g., cooperation payoffs).

🧭 Architecture
🔄 Training Workflow
Environment: Iterated Prisoner's Dilemma against fixed policies (e.g., Tit for Tat, Always Defect).

Agent: A language model that learns to play optimally via PPO, conditioned on past rounds and retrieved samples.

History Buffer: Stores tuples of (opponent policy, recent moves, outcomes) every few episodes.

Retriever: Every N steps, retrieves similar past behaviors from buffer to guide generation (using cosine or embedding similarity).

Generator: Generates next move (Cooperate or Defect) based on:

Current observation (history, opponent behavior)

Retrieved memory

Policy Update: PPO optimizes the generator by comparing rewards over episodes and updating the behavior policy.

🔧 Key Mechanism: RAG-PPO Fusion
Every 10 steps:

Retrieve top-k past behaviors using similarity.

Fuse retrieved examples into model input.

Generate action → get reward → update PPO.

This blends memory and learning, enabling better adaptation across strategies.

🧪 Evaluation: Fixed Policy Opponents
Opponent Policy	Description	PPO Behavior Summary	Success ✅/❌
Tit for Tat	Cooperates, mimics last move	Learned to cooperate	✅
Always Defect	Always defects	Learned to defect in return	✅
Always Cooperate	Always cooperates	Exploited with defect	✅
Friedman	Punishes once defected	Early defection triggered loss	❌
Joss	90% cooperation, 10% random defect	Mostly cooperative	✅
Tester	Cooperates, then defects permanently	Detected pattern, adapted	✅
Backstabber	Cooperates 5 rounds, then defects always	Delayed detection, adapted late	✅

🧠 Why This Matters
Human-like Generalization: Mimics how people use memory + real-time feedback to behave.

Educational Modeling: Could be extended to simulate student learning responses in dynamic settings.

Multi-Agent Foundations: A step toward building socially intelligent LLM agents capable of adapting to evolving behaviors, not just static prompts.


