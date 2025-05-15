import random
import numpy as np
import gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from langchain_chroma import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain_ollama import OllamaLLM
from get_embedding_function import get_embedding_function

# === Constants ===
CHROMA_PATH = "chroma"

PROMPT_TEMPLATE = """
Context:
{context}
You are an expert strategy selector whose aim is to maximise rewards. Based on the following context, choose exactly ONE bootstrapping policy from the list below that is most suitable.
cooperation is 0 and defection is 1 
Available Policies:
- alternating_policy
- random_policy
- cooperate_for_five_defect_for_five
- tit_for_tat_bootstrap
- tit_for_two_tat_bootstrap
Question: Which one bootstrapping policy should the agent use.

Respond with the name of the selected policy from the list above and also why with respect to some past behaviour that is opponent action if available mention the iterations (mandatory).
give in below format
policy name:
"""

# === Preload Ollama Model ===
ollama_model = OllamaLLM(model="mistral")

# === Bootstrapping Policies ===
def alternating_policy(step): return 0 if step % 2 == 0 else 1
def random_policy(step): return random.choice([0, 1])
def cooperate_for_five_defect_for_five(step): return 0 if (step // 5) % 2 == 0 else 1
def tit_for_tat_bootstrap(agent_history, opponent_history): return 0 if not opponent_history else opponent_history[-1]
def tit_for_two_tat_bootstrap(agent_history, opponent_history):
    return 0 if len(opponent_history) < 2 else (1 if opponent_history[-1] == 1 and opponent_history[-2] == 1 else 0)

bootstrap_policies = {
    "alternating_policy": lambda h, o: alternating_policy(len(h)),
    "random_policy": lambda h, o: random_policy(len(h)),
    "cooperate_for_five_defect_for_five": lambda h, o: cooperate_for_five_defect_for_five(len(h)),
    "tit_for_tat_bootstrap": tit_for_tat_bootstrap,
    "tit_for_two_tat_bootstrap": tit_for_two_tat_bootstrap,
}

from langchain_core.documents import Document  # Add this import

def query_rag_bootstrap_policy(agent_history, opponent_history, reward_history, iteration_no):
    import re

    # Initialize embedding function and Chroma database
    embedding_function = get_embedding_function()
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

    # === STEP 1: Store trimmed history data with iteration number ===
    trimmed_agent = agent_history[-30:]
    trimmed_opponent = opponent_history[-30:]
    trimmed_rewards = reward_history[-30:]

    history_data = {
        "agent_actions": trimmed_agent,
        "opponent_actions": trimmed_opponent,
        "reward_history": trimmed_rewards,
        "iteration": iteration_no,
        "context_type": "history_before_policy_selection"
    }
    print(history_data)
    # ✅ Wrap as LangChain Document
    doc = Document(
        page_content=str(history_data),
        metadata={"type": "interaction_history", "iteration": iteration_no}
    )
    db.add_documents([doc])
    
    # === STEP 2: Retrieve similar contexts and generate policy recommendation ===
    question = "Choose one bootstrapping policy from: " + ', '.join(bootstrap_policies.keys())
    search_results = db.similarity_search_with_score(question, k=10)
    context_text = "\n".join([result[0].page_content for result in search_results])

    prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE).format(context=context_text, question=question)
    response = ollama_model.invoke(prompt).strip()

    # Extract policy name using regex
    valid_policies = list(bootstrap_policies.keys())
    pattern = r'\b(' + '|'.join(re.escape(policy) for policy in valid_policies) + r')\b'
    match = re.search(pattern, response)
    selected_policy = match.group(0) if match else None

    print(response)
    print(f"[RAG SELECTOR] Selected bootstrap policy: {selected_policy}")

    return selected_policy


# === Opponent Strategies ===
def tit_for_tat(agent_history, opponent_history): return 'C' if not agent_history else ('C' if agent_history[-1] == 0 else 'D')
def tit_for_two_tat(agent_history, opponent_history): return 'C' if len(agent_history) < 2 else ('D' if agent_history[-1] == 1 and agent_history[-2] == 1 else 'C')
def always_defect(agent_history, opponent_history): return 'D'

# === Environment Classes ===
class PrisonersDilemmaEnv(gym.Env):
    def __init__(self, strategy, history_len=20):
        super().__init__()
        self.strategy = strategy
        self.history_len = history_len
        self.action_space = gym.spaces.Discrete(2)  # C=0, D=1
        self.observation_space = gym.spaces.Box(low=0, high=5, shape=(3 * history_len,), dtype=np.float32)
        self.reset()

    def reset(self):
        self.agent_history = []
        self.opponent_history = []
        self.reward_history = []
        self.step_count = 0
        return self._get_obs()

    def step(self, action):
        action = int(action)
        opponent_move = self.strategy(self.agent_history, self.opponent_history)
        opponent_action = 0 if opponent_move == 'C' else 1
        reward = self.get_reward(action, opponent_action)

        self.agent_history.append(action)
        self.opponent_history.append(opponent_action)
        self.reward_history.append(reward)

        self.agent_history = self.agent_history[-self.history_len:]
        self.opponent_history = self.opponent_history[-self.history_len:]
        self.reward_history = self.reward_history[-self.history_len:]

        obs = self._get_obs()
        self.step_count += 1

        # === PRINT FOR TRAINING ===
        print(f"[TRAIN] Step {self.step_count:03}: Agent={'C' if action==0 else 'D'}, "
              f"Opponent={'C' if opponent_action==0 else 'D'}, Reward={reward}")

        return obs, reward, False, {}

    def _get_obs(self):
        pad = lambda x: x + [0] * (self.history_len - len(x))
        return np.array(pad(self.agent_history) + pad(self.opponent_history) + pad(self.reward_history), dtype=np.float32)

    def get_reward(self, a, b):
        return { (0, 0): 3, (1, 1): 1, (0, 1): 0, (1, 0): 5 }[(a, b)]

class ScheduledBootstrapEnv(PrisonersDilemmaEnv):
    def __init__(self, strategy, switch_every=10000, bootstrap_steps=10, history_len=10):
        super().__init__(strategy, history_len)
        self.switch_every = switch_every
        self.bootstrap_steps = bootstrap_steps
        self.current_bootstrap_policy = None
        self.last_bootstrap_phase = -1
    check=0
    def step(self, action):
        current_phase = self.step_count // self.switch_every
        phase_offset = self.step_count % self.switch_every
        print(current_phase)
        print(phase_offset)
       
        if (phase_offset < self.bootstrap_steps) & (phase_offset==0):
            bootstrap_policy = query_rag_bootstrap_policy(self.agent_history,self.opponent_history,self.reward_history,self.step_count).strip().lower()
            check=1
            if bootstrap_policy in bootstrap_policies:
                self.current_bootstrap_policy = bootstrap_policy
                self.last_bootstrap_phase = current_phase
            if self.current_bootstrap_policy in bootstrap_policies:
                action = bootstrap_policies[self.current_bootstrap_policy](self.agent_history, self.opponent_history)

        return super().step(action)

# === Training / Testing ===
def make_vec_env():
    strategies = [tit_for_two_tat, always_defect]
    envs = [lambda s=s: ScheduledBootstrapEnv(s) for s in strategies]
    return DummyVecEnv(envs)
log_path = "./ppo_logs/"
from stable_baselines3.common.logger import configure

def train_ppo():
    print("🚀 Starting PPO Training...")

    # Set up environment and log path
    env = make_vec_env()

    # Create PPO model
    model = PPO("MlpPolicy", env, verbose=1)

    # Set up custom logging
    log_path = "./ppo_logs/"
    model.set_logger(configure(log_path, ["stdout", "csv"]))

    # Train the model and track statistics
    model.learn(total_timesteps=130000)

    # Save the trained model
    model.save("ppo_vs_all_strategies_scheduled")
    print("✅ Training Complete and Model Saved.")

    print(f"\n📊 Training logs can be seen in terminal.")


def test_ppo():
    print("🔍 Loading model...")
    model = PPO.load("ppo_vs_all_strategies_scheduled")
    strategies = [("Tit for Tat", tit_for_tat), ("Tit for Two Tat", tit_for_two_tat), ("Always Defect", always_defect)]

    for name, strategy in strategies:
        print(f"\n🧪 Testing against strategy: {name}")
        env = ScheduledBootstrapEnv(strategy)
        obs = env.reset()

        agent_seq, opp_seq, rewards = [], [], []

        for i in range(200):
            action, _ = model.predict(obs)
            obs, reward, _, _ = env.step(action)

            agent_seq.append(env.agent_history[-1])
            opp_seq.append(env.opponent_history[-1])
            rewards.append(reward)

            print(f"Step {i+1:03}: Agent={'C' if agent_seq[-1]==0 else 'D'}, "
                  f"Opponent={'C' if opp_seq[-1]==0 else 'D'}, Reward={reward}")

        print("\n📈 Summary:")
        print("Agent   :", ''.join(['C' if a==0 else 'D' for a in agent_seq]))
        print("Opponent:", ''.join(['C' if a==0 else 'D' for a in opp_seq]))
        print("Rewards :", rewards)
        print("Total Reward:", sum(rewards))

# === Entry Point ===
if __name__ == "__main__":
    train_ppo()
    test_ppo()