from cc_optimizer_gym import *
from Q_learning import *

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd


cards_df = pd.read_csv("creditcards/cards.csv")
transactions_df = pd.read_csv("testtransactiondata/fake_transactions.csv")

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


def plot_q_table_heatmap(Q_table, env, episode):
    """ create Q-table heatmap to show what the model learned per category/card
    used AI to help fix the layout of the plot so that everything could be displayed evenly """

    # convert Q_table dictionary to a df
    df = pd.DataFrame.from_dict(Q_table, orient="index")

    card_names = env.cards["card_name"].tolist()
    df.columns = card_names

    # normalize each row for comparison to see which card is best
    df_norm = df.div(df.max(axis=1), axis=0).fillna(0)

    plt.figure(figsize=(16, max(8, len(df_norm) * 0.3)))

    sns.heatmap(df_norm,
                cmap="viridis",
                xticklabels=card_names,
                yticklabels=[str(s) for s in df_norm.index],
                cbar_kws={"label": "Normalized Q-value"},
                annot=False)

    plt.title(f"Q-Table Heatmap for {episode} Episodes")
    plt.xlabel("Cards")
    plt.ylabel("States (Category, Amount Bucket)")
    plt.tight_layout()
    plt.savefig(f"heatmap_{episode}.png", dpi=300, bbox_inches="tight")
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


# call the functions to create the plots
plot_training_rewards(all_rewards, episode)
plot_q_table_heatmap(Q_table, env, episode)
plot_per_card_rewards(per_card_rewards, env, episode)
