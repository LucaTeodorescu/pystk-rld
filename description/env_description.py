import gymnasium as gym
import pystk2_gymnasium
from pystk2_gymnasium import AgentSpec

if __name__ == '__main__':
        
    ListOfEnvs = [
        "supertuxkart/full-v0",
        "supertuxkart/simple-v0",
        "supertuxkart/flattened-v0",
        "supertuxkart/flattened_continuous_actions-v0",
        "supertuxkart/flattened_multidiscrete-v0",
        "supertuxkart/flattened_discrete-v0",
    ]

    with open("output_description.txt", "a") as f:
        for envname in ListOfEnvs:
            env = gym.make(envname, agent=AgentSpec(use_ai=False))
            actionspace = env.action_space
            observationspace = env.observation_space
            print("===============================", file=f)
            print(f"ENV: {envname}", file=f)
            print(f"The action space is {actionspace}", file=f)
            print(f"The observation space is {observationspace}", file=f)
        