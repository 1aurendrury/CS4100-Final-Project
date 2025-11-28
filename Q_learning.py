import argparse
import pickle
import time
from collections import defaultdict
import numpy as np
import pandas as pd
from tqdm import tqdm
from cc_optimizer_gym import CreditCardEnv

BOLD = "\033[1m"
RESET = "\033[0m"
# 15% boost for reusing same card
REUSE_BONUS_FACTOR = 0.15
# points conversion rate: 100 points = $1, so 1 point = $0.01
POINTS_PER_DOLLAR = 100.0


def load_cards(path):
    df = pd.read_csv(path)
    required = {"card_name", "reward_type", "annual_fee_usd", "reward_category_map"}
    if not required.issubset(df.columns):
        raise ValueError(f"cards csv is missing the following columns: {required - set(df.columns)}")
    return df

def load_transactions(path):
    df = pd.read_csv(path)
    required = {"amount", "category"}
    if not required.issubset(df.columns):
        raise ValueError(f"transactions csv is missing the following columns: {required - set(df.columns)}")
    return df


def hash(obs):
    if obs is None:
        return None
    category = obs["category"]
    bucket = min(int(obs["amount"] // 20), 10)
    return (category, bucket)


def Q_learning(
    env,
    episodes=2000,
    gamma=0.95,
    epsilon=1.0,
    decay=0.995,
):
    # initialize Q-table, counts, action space, and update interval
    Q_table = {}
    counts = {}
    actions = env.action_space.n

    # loop through episodes with progress bar
    for ep in tqdm(range(episodes), desc="Training Q-learning"):

        # reset the environment
        obs, _, done, _ = env.reset()
        if done:
            break
        
        # hash the observation to get the state
        state = hash(obs)
        if state not in Q_table:
            Q_table[state] = np.zeros(actions)
            counts[state] = np.zeros(actions)
            
        # loop through steps until done
        while not done:
            # choose action based on epsilon-greedy policy
            if np.random.rand() < epsilon:
                action = env.action_space.sample()
            else:
                qvals = Q_table[state]
                action = np.random.choice(np.flatnonzero(qvals == qvals.max()))
            # take the action and get the next observation, reward, done, and info
            next_obs, reward, done, info = env.step(action)
            
            # if there is no next observation, set the next state to None
            if next_obs is None:
                next_state = None
            else:
                # hash the next observation to get the next state
                next_state = hash(next_obs)
                # if the next state is not in the Q-table, initialize it to 0
                if next_state not in Q_table:
                    Q_table[next_state] = np.zeros(actions)
                    counts[next_state] = np.zeros(actions)
            
            # increment the count for the action in the current state
            counts[state][action] += 1
            # calculate the learning rate
            lr = 1.0 / (1.0 + counts[state][action])
        
            # if there is no next state, set the target to the reward
            if next_state is None:
                target = reward
            else:
                # calculate the target
                target = reward + gamma * Q_table[next_state].max()
    
            # update the Q-value for the action in the current state
            Q_table[state][action] += lr * (target - Q_table[state][action])
            
            # update the state
            state = next_state
            
        # decay the epsilon value
        epsilon *= decay
        
    return Q_table



def evaluate_policy(env, Q_table):
    # initialize total reward and per card reward
    total = 0.0
    per_card = defaultdict(float)
    
    # reset the environment
    obs, _, done, _ = env.reset()
    # if done, return 0 and empty dictionary
    if done:
        return 0, {}

    # loop through steps until done
    while not done:
        # hash the observation to get the state
        state = hash(obs)
        # if state is not in the Q-table, choose a random action
        if state not in Q_table:
            action = np.random.randint(env.action_space.n)
        else:
            # if state is in the Q-table, choose the action with the highest Q-value
            qvals = Q_table[state]
            action = np.random.choice(np.flatnonzero(qvals == qvals.max()))

        # take the action and get the next observation, reward, done, and info
        _, r, done, info = env.step(action)
        # increment the total reward and per card reward
        total += r
        per_card[action] += r
        obs = info.get("raw_txn", None)
        
    # return the total reward and per card reward
    return total, per_card


def recommend_cards_by_category(env, Q_table):
    # initialize categories and chosen cards
    categories = sorted(env.transactions["category"].unique())
    chosen_cards = set()
    recommendations = {}
    
    # loop through categories
    for cat in categories:
        # collect Q-vectors for this category
        states = [s for s in Q_table if s is not None and s[0] == cat]
        if not states:
            continue
        
        # stack the Q-values for the category
        qvals = np.stack([Q_table[s] for s in states], axis=0)
        mean_q = qvals.mean(axis=0)
        
        # add the reuse bonus factor if there are chosen cards
        if chosen_cards:
            for card in chosen_cards:
                mean_q[card] += REUSE_BONUS_FACTOR * mean_q.max()
                
        # choose the best card for the category
        best_card = int(mean_q.argmax())
        best_q = float(mean_q[best_card])
        
        # add the best card to the recommendations
        recommendations[cat] = {
            "card_idx": best_card,
            "q_value": best_q,
        }
        
        # add the best card to the chosen cards
        chosen_cards.add(best_card)

    return recommendations


def compute_category_rewards(env, recommendations):
    # initialize results
    results = {}
    # loop through categories
    for cat, info in recommendations.items():
        # get the card index and card
        card_idx = info["card_idx"]
        card = env.cards.iloc[card_idx]
        # get the category map and reward type for the card
        mapping = env.card_category_maps[card_idx]
        reward_type = str(card["reward_type"]).lower()
        
        # get the category multiplier for the category
        if cat in mapping:
            mult = mapping[cat]
        elif "other" in mapping:
            mult = mapping["other"]
        else:
            mult = 1.0
            
        # get the category transactions
        tx_cat = env.transactions[env.transactions["category"] == cat]
        raw_points = 0.0
        raw_cash = 0.0
        
        # loop through the category transactions and calculate the raw rewards
        for _, txn in tx_cat.iterrows():
            amount = float(txn["amount"])
            if reward_type == "points":
                raw_points += amount * mult
            elif reward_type == "cashback":
                raw_cash += amount * mult / 100.0
                
        # add the results to the dictionary
        results[cat] = {
            "card_idx": card_idx,
            "reward_type": reward_type,
            "mult": mult,
            "raw_points": raw_points,
            "raw_cash": raw_cash,
        }

    return results


def print_category_recommendations(env, recs, cat_rewards):
    print(f"\n{BOLD}Category Recommendations:{RESET}")
    
    for cat, info in recs.items():
        card_idx = info["card_idx"]
        card = env.cards.iloc[card_idx]
        qval = info["q_value"]
        
        reward_data = cat_rewards[cat]
        reward_type = reward_data["reward_type"]
        multiplier = reward_data["mult"]

        print(f"\n{BOLD}{cat}{RESET} -> {card['card_name']} ({card['issuer']})")
        print(f"   - Q-learning value for this category: {qval:.2f}")
        print(f"   - Multiplier for this category: {multiplier:.1f}x" 
              if reward_type=="points"
              else f"   - Multiplier for this category: {multiplier:.1f}% cashback")

        if reward_type == "points":
            print(f"   - Estimated raw reward (this category only): {reward_data['raw_points']:.0f} points")
        else:
            print(f"   - Estimated raw reward (this category only): ${reward_data['raw_cash']:.2f} cashback")

        print(f"   - Annual fee: ${float(card['annual_fee_usd']):.2f}")



def print_global_summary(env, recs, cat_rewards):
    print(f"\n{BOLD}Global Summary:{RESET}")
    
    # unique cards selected
    selected = {info["card_idx"] for info in recs.values()}
    total_fees = sum(float(env.cards.iloc[c]["annual_fee_usd"]) for c in selected)
    
    # total raw rewards
    total_points = sum(v["raw_points"] for v in cat_rewards.values())
    total_cash = sum(v["raw_cash"] for v in cat_rewards.values())

    print(f"\nTotal annual fees (unique cards only): ${total_fees:.2f}")
    
    if total_points > 0:
        points_dollar_value = total_points / POINTS_PER_DOLLAR
        print(f"Total raw points earned: {total_points:.0f} points (${points_dollar_value:.2f})")
        
        # for points mode, show net points after converting fees to points (100 points = $1)
        if env.reward_type == "points":
            fees_in_points = total_fees * POINTS_PER_DOLLAR
            net_points = total_points - fees_in_points
            net_points_dollar_value = net_points / POINTS_PER_DOLLAR
            print(f"Annual fees in points: {fees_in_points:.0f} points (${total_fees:.2f})")
            print(f"Net points after fees: {net_points:.0f} points (${net_points_dollar_value:.2f})")
    
    if total_cash > 0:
        print(f"Total raw cashback earned: ${total_cash:.2f}")
        
    # net reward (raw - fees)
    # convert points to dollars for calculation purposes (100 points = $1 again here)
    net_value = total_cash + (total_points / POINTS_PER_DOLLAR) - total_fees
    print(f"\n{BOLD}Net estimated yearly value: ${net_value:.2f}{RESET}")


def spending_summary(df):
    print(f"\n{BOLD}Spending Summary:{RESET}")
    
    print("\nTop categories:")
    cat_totals = df.groupby("category")["amount"].sum().sort_values(ascending=False)
    for cat, amount in cat_totals.items():
        print(f"  - {cat}: ${amount:.2f}")
    
    if "brand" in df.columns:
        print("\nTop brands:")
        brand_totals = df.groupby("brand")["amount"].sum().sort_values(ascending=False)
        for b, amount in brand_totals.items():
            print(f"  - {b}: ${amount:.2f}")



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cards", required=False) # we can use default cards.csv if not provided
    parser.add_argument("--transactions", required=True)
    parser.add_argument("--mode", choices=["points", "cashback", "both"], default="points")
    parser.add_argument("--episodes", type=int, default=5000)
    # added flag to save Q-table as a pickle file
    parser.add_argument("--save", type=str, default=None)
    args = parser.parse_args()
    
    # load cards and transactions, set up environment
    cards_df = load_cards(args.cards)
    tx_df = load_transactions(args.transactions)
    env = CreditCardEnv(cards_df, tx_df, reward_type=args.mode)
    
    # train Q-learning model
    print(f"\nTraining Q-learning model for {args.episodes} episodes...")
    qstart = time.time()
    Q = Q_learning(env, episodes=args.episodes)
    print(f"Training complete in {time.time()-qstart:.2f}s")
    
    # get recommendations for best card per category
    recs = recommend_cards_by_category(env, Q)

    # get category raw rewards
    cat_rewards = compute_category_rewards(env, recs)

    # print summaries and results
    print_category_recommendations(env, recs, cat_rewards)
    print_global_summary(env, recs, cat_rewards)
    spending_summary(tx_df)

    # save Q-table if desired as a pickle file
    if args.save:
        with open(args.save, "wb") as f:
            pickle.dump(Q, f)
        print(f"\nSaved Q-table to {args.save}")


if __name__ == "__main__":
    main()

