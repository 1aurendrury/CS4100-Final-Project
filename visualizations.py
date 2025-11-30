from cc_optimizer_gym import *
from Q_learning import *

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd


cards_df = pd.read_csv("credit_cards/cards.csv")
transactions_df = pd.read_csv("test_data/fake_transactions.csv")

# create environment (reward_type = points, cashback, both)
env = CreditCardEnv(cards_df, transactions_df, reward_type="both")

# train Q-learning
episode = 40000
Q_table, all_rewards = Q_learning(env, episodes=episode)

recommended_card = recommend_cards_by_category(env, Q_table)
total_reward, per_card_rewards = evaluate_policy(env, Q_table)
category_rewards = compute_category_rewards(env, recommended_card)


def plot_training_rewards(all_rewards, episode):
    """ create plot to show training rewards over time """

    plt.plot(all_rewards, color="blue", label="Rewards per Episode")
    plt.plot(np.cumsum(all_rewards) / np.arange(1, len(all_rewards) + 1), color="red", label="Running Average")
    plt.title(f"Training Rewards Over {episode} Episodes")
    plt.xlabel("Number of Episode")
    plt.ylabel("Total Reward")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"running_average_{episode}.png", dpi=300, bbox_inches="tight")
    plt.show()


def plot_per_card_rewards(per_card_rewards, env, episode):
    """ create a bar plot showing total rewards earned per card """

    # get card names and rewards
    card_names = [env.cards.iloc[i]["card_name"] for i in per_card_rewards.keys()]
    rewards = list(per_card_rewards.values())

    sns.barplot(x = card_names, y = rewards)
    plt.xticks(rotation=80)
    plt.title(f"Total Rewards Earned Per Card for {episode} Episodes")
    plt.xlabel("Card")
    plt.ylabel("Total Rewards")
    plt.tight_layout()
    plt.savefig(f"per_card_rewards_{episode}.png", dpi=300, bbox_inches="tight")
    plt.show()
    
def plot_policy_heatmap(Q_table, env, episode):
    """ plot a heatmap of the learned policy: for each state (category, bucket), highlight which card has the highest Q-value. """

    # convert Q-table to a df
    df = pd.DataFrame.from_dict(Q_table, orient="index")
    df.columns = env.cards["card_name"].tolist()

    # compute the best action (argmax) for each state
    best_actions = df.idxmax(axis=1)

    # create a matrix df 1 where card == best card for that state, else 0
    policy_matrix = pd.DataFrame(0, index=df.index, columns=df.columns)
    
    # iterate over the df index directly to ensure index format matches
    for state in df.index:
        best_card = best_actions.at[state]
        policy_matrix.at[state, best_card] = 1

    # sort states by category first, then by bucket number
    sorted_states = sorted(policy_matrix.index, key=lambda x: (x[0], x[1]))
    policy_matrix_sorted = policy_matrix.loc[sorted_states]

    # plot with viridis colormap
    plt.figure(figsize=(16, max(8, len(policy_matrix_sorted) * 0.3)))
    sns.heatmap(policy_matrix_sorted,
                cmap="viridis",
                xticklabels=env.cards["card_name"].tolist(),
                yticklabels=[str(s) for s in policy_matrix_sorted.index],
                cbar_kws={"label": "Policy (1 = Best Card)"},
                annot=False)

    plt.title(f"Policy Heatmap (Best Card per State) after {episode} Episodes")
    plt.xlabel("Cards")
    plt.ylabel("States (Category, Amount Bucket)")
    plt.tight_layout()
    plt.savefig(f"policy_heatmap_{episode}.png", dpi=300, bbox_inches="tight")
    plt.show()



# call the functions to create the plots
plot_training_rewards(all_rewards, episode)
plot_per_card_rewards(per_card_rewards, env, episode)
plot_policy_heatmap(Q_table, env, episode)
