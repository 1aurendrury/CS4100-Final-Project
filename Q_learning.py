import argparse
from cc_optimizer_gym import CreditCardEnv
import os
import sys
import time
import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm
from collections import defaultdict # Import defaultdict for the update counter.
import matplotlib.pyplot as plt # Import matplotlib for plotting rewards.

# IMPORTANT FRIENDLY REMINDER :D
# below are some functions that I had in mind implementing when creating cc_optimizer_gym.py
# note that we might not need all of them, but they are just some things that I thought of when creating the gym/RL env
# note also that cc_optimizer_gym.py was developed with the following args in mind (can use argparse for these):
# python Q_learning.py --cards xxx.csv --transactions xxx.csv --mode points/cashback/both --episodes 2000 (xxxx)

# TODO: might need functions for loading csv cards and transactions maybe? - can do this in another file also
# TODO: will probably need a hashing function, similar to PA2

def hash(obs):
    """
    Convert an observation dictionary to a unique hashable state identifier.
    Parameters:
    - obs (dict): Observation from the environment containing 'category' and 'amount'.
    Returns:
    - tuple: A hashable tuple representing the state (category, amount_bin).
    """
    # If the observation is None (end of episode), then return the terminal tuple to signal the end state to the agent.
    if obs is None:
        return ("terminal",)
    
    # Extract the category from the observation.
    category = obs["category"]
    
    # Extract the amount from the observation and bin it into ranges.
    # We bin the amount to reduce the state space size (continuous -> discrete).
    amount = obs["amount"]
    
    # Create bins for the amount: 0-50, 50-100, 100-200, 200+.
    # This groups similar purchase amounts together to help the agent generalize.
    if amount < 50:
        amount_bin = "0-50"
    elif amount < 100:
        amount_bin = "50-100"
    elif amount < 200:
        amount_bin = "100-200"
    else:
        amount_bin = "200+"
    
    # Return a tuple of (category, amount_bin) as the state identifier.
    return (category, amount_bin)

env = None  # Global environment variable, will be set in main().

# TODO: implement Q-learning algorithm here
# one thing I thought of is that perhaps we have a bonus for using the same card multiple times in a row (if applicable) to reduce overall annual fees?
# we can make this a small bonus, maybe 0.15 or something

def Q_learning(
    env,
    episodes=5000,
    gamma=0.95,
    epsilon=1.0,
    decay=0.997,
):
    """
    Run Q-learning algorithm for a specified number of episodes.
    Parameters:
    - env (CreditCardEnv): The credit card environment to train on.
    - episodes (int): Number of episodes to run.
    - gamma (float): Discount factor.
    - epsilon (float): Exploration rate.
    - decay (float): Rate at which epsilon decays. Epsilon should be decayed as epsilon = epsilon * decay after each episode.
    Returns:
    - Q_table (dict): Dictionary containing the Q-values for each state-action pair.
    """
    Q_table = {}
    
    # ==== Complete the Q-learning function ====

    # num_updates is a dictionary mapping (state_id, action_index) to the count of how many times it has been updated.
    # Use defaultdict so it automatically initializes new keys with a value of 0.
    num_updates = defaultdict(int)
    
    # Store the total reward for each episode; used for plotting.
    rewards_per_episode = []
    
    # Iterate over each episode.
    for episode in tqdm(range(episodes)):

        # "call obs, reward, done, info = env.reset() at the start of each episode."
        # This resets the game environment and returns the tuple (obs, reward, done, info).
        obs, _, done, _ = env.reset()  # Get the starting observation and done flag.
        
        # Initialize variables to track the state of the current episode.
        total_episode_reward = 0  # Sum of rewards collected in this episode.
        
        # Iterate if the current episode is not "done" (episode has not ended yet).
        while not done:
            
            # Get the unique hashable state for the current observation.
            s = hash(obs)
            
            # ==== Epsilon-Greedy Action Selection ====
            
            # First, check if this state "s" has been observed before.
            if s not in Q_table:
                # If this state is a new state, then add it to the Q-table.
                # Initialize Q-values for all actions from this state to 0.
                # env.action_space.n gives the total number of actions (credit cards).
                Q_table[s] = np.zeros(env.action_space.n)
            
            # Then, decide whether to explore (random action) or exploit (best action).
            # "Use epsilon-greedy action selection when choosing actions."
            if np.random.rand() < epsilon: # Generate a random number between 0.0 and 1.0 and compare to epsilon.
                # Explore: choose a random action from the available action space.
                a = env.action_space.sample()
            else:
                # Exploit: choose the action "a" that has the highest Q-value in Q_table[s].
                a = np.argmax(Q_table[s])
            
            # ==== Take Action and Observe Outcome ====
            
            # Take the chosen action "a" in the environment.
            # env.step() returns the next observation, reward, if the episode is done, and extra info.
            next_obs, reward, done, info = env.step(a)
            
            # Add the received reward to the total for this episode.
            total_episode_reward += reward
            
            # Get the unique hashable state for the next state.
            s_prime = hash(next_obs)
            
            # ==== Q-Value Update ====
            
            # Check if the next state "s_prime" has been observed before.
            if s_prime not in Q_table:
                # If this state is a new state, then add it to the Q-table with all-zero Q-values.
                Q_table[s_prime] = np.zeros(env.action_space.n)
            
            # Increment the update count for this specific (state, action) pair.
            num_updates[(s, a)] += 1
            
            # Calculate the learning rate "eta" for this (state, action) pair.
            # "Use a learning-rate schedule per (s,a) pair, i.e. eta = 1/(1 + N(s,a)) where N(s,a) is the number of updates applied to that pair so far."
            eta = 1.0 / (1.0 + num_updates[(s, a)])
            
            # Get the largest Q-value for the next state (max Q(s', a')).
            # This is the estimate of the optimal future value V*(s').
            max_q_prime = np.max(Q_table[s_prime])
            
            # Calculate the temporal difference target value for the update.
            if done:
                # If the episode is done, then the episode is over, and there is no future value.
                # The target is the current reward.
                target = reward
            else:
                # If the episode is not done, then there is a future value.
                # The target is the current reward + discounted optimal future value (gamma * V*(s')).
                target = reward + gamma * max_q_prime
            
            # Apply the Q-learning update rule.
            # Q(s,a) = (1-eta) * Q(s,a) + eta * (target)
            Q_table[s][a] = (1.0 - eta) * Q_table[s][a] + eta * target
            
            # The next observation becomes the current observation for the next loop iteration.
            obs = next_obs
            
        # ==== End of Episode ====
        # Add the total reward from this episode to the rewards list for plotting.
        rewards_per_episode.append(total_episode_reward)
        
        # Multiply epsilon by the decay rate to gradually reduce exploration.
        epsilon = epsilon * decay
    
    # ==== End of Training: All Episodes Done ====

    # ==== Plotting Rewards ====

    # Convert rewards_per_episode to a numpy array to avoid matplotlib errors.
    rewards_per_episode = np.array(rewards_per_episode)

    # Calculate the running average of the rewards.
    # np.cumsum() creates a cumulative sum (e.g., [1, 2, 3] -> [1, 3, 6]).
    # Divide by the episode number (1, 2, 3, ...) to get the average.
    # Add 1 to np.arange(), because episode indices are 0-based (0, 1, 2...).
    running_avg = np.cumsum(rewards_per_episode) / (np.arange(episodes) + 1)
    
    # Create the reward plot.
    plt.figure(figsize=(12, 6))
    plt.plot(rewards_per_episode, label='Episode Reward', alpha=0.3)
    plt.plot(running_avg, label='Running Average Reward', color='red')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title(f'Training Rewards ({episodes} Episodes, Decay={decay})')
    plt.legend()
    
    # Save the plot to a PNG file.
    plot_filename = f'reward_plot_{episodes}_{decay}.png'
    plt.savefig(plot_filename)
    
    # Print a message to confirm the plot was saved.
    print(f"Reward plot saved to {plot_filename}")
    
    # Return the filled Q-table.
    return Q_table

# TODO: implement evaluation function here where we evaluate the current policy and return the total reward and per card reward
# here we could get the Q-values for each action and choose the action with the highest Q-value

def evaluate_policy(env, Q_table):
    """
    Evaluate the learned policy by running through all transactions once.
    Parameters:
    - env (CreditCardEnv): The credit card environment to evaluate on.
    - Q_table (dict): The learned Q-table containing state-action values.
    Returns:
    - total_reward (float): The total reward earned during evaluation.
    - card_rewards (dict): Dictionary mapping card index to total reward earned by that card.
    """
    # Reset the environment to start evaluation.
    obs, _, done, _ = env.reset()
    
    # Initialize total reward counter.
    total_reward = 0
    
    # Initialize dictionary to track rewards per card.
    card_rewards = defaultdict(float)
    
    # Loop through all transactions until done.
    while not done:
        # Get the hashed state for the current observation.
        s = hash(obs)
        
        # Check if this state exists in the Q-table.
        if s in Q_table:
            # Choose the action with the highest Q-value (exploit only, no exploration).
            a = np.argmax(Q_table[s])
        else:
            # If state not in Q-table, then choose a random action as fallback.
            a = env.action_space.sample()
        
        # Take the chosen action in the environment.
        next_obs, reward, done, info = env.step(a)
        
        # Add the reward to the total reward.
        total_reward += reward
        
        # Add the reward to the total of the specific card.
        card_rewards[a] += reward
        
        # Move to the next observation.
        obs = next_obs
    
    # Return the total reward and per-card rewards.
    return total_reward, dict(card_rewards)

# TODO: implement category recommendations function here based on current/given Q-table
# here we could return a dictionary of categories and the best card for that category

def recommend_cards_by_category(env, Q_table):
    """
    Recommend the best card for each spending category based on the Q-table.
    Parameters:
    - env (CreditCardEnv): The credit card environment with card information.
    - Q_table (dict): The learned Q-table containing state-action values.
    Returns:
    - recommendations (dict): Dictionary mapping category to best card index.
    """
    # Initialize dictionary to store recommendations.
    recommendations = {}
    
    # Get all unique categories from the transactions.
    categories = env.transactions["category"].unique()
    
    # For each category, find the best card based on Q-values across all amount bins.
    for category in categories:
        # Track the best card and its average Q-value across all amount bins.
        best_card = None
        best_avg_q = float('-inf')
        
        # Define the amount bins we used in the hash function.
        amount_bins = ["0-50", "50-100", "100-200", "200+"]
        
        # For each card, calculate its average Q-value across all amount bins for this category.
        for card_idx in range(env.num_cards):
            # Only consider cards that match the reward type we're optimizing for.
            card = env.cards.iloc[card_idx]
            card_reward_type = str(card["reward_type"]).lower()
            
            # Skip this card if its reward type doesn't match what we're optimizing for.
            if env.reward_type == "points" and card_reward_type != "points":
                continue
            elif env.reward_type == "cashback" and card_reward_type != "cashback":
                continue
            # If mode is "both", then include all cards.
            
            # Calculate the average Q-value for this card across all amount bins.
            q_values = []
            for amount_bin in amount_bins:
                # Create the state tuple for this category and amount bin.
                state = (category, amount_bin)
                
                # Check if this state exists in the Q-table.
                if state in Q_table:
                    # Get the Q-value for this card in this state.
                    q_values.append(Q_table[state][card_idx])
            
            # If we found Q-values for this card, then calculate the average.
            if q_values:
                avg_q = np.mean(q_values)
                
                # Update the best card if this card has a higher average Q-value.
                if avg_q > best_avg_q:
                    best_avg_q = avg_q
                    best_card = card_idx
        
        # If no card was found (should not happen), then default to card 0.
        if best_card is None:
            # Find the first card that matches the reward type.
            for card_idx in range(env.num_cards):
                card = env.cards.iloc[card_idx]
                card_reward_type = str(card["reward_type"]).lower()
                if env.reward_type == "points" and card_reward_type == "points":
                    best_card = card_idx
                    break
                elif env.reward_type == "cashback" and card_reward_type == "cashback":
                    best_card = card_idx
                    break
            # If still None, then default to 0.
            if best_card is None:
                best_card = 0
        
        # Store the recommendation for this category.
        recommendations[category] = best_card
    
    # Return the recommendations dictionary.
    return recommendations

# TODO: implement reward calculation here based on category recommendations
# aka take their current transaction data and multiply by the multiplier for the category for the respective card

def compute_category_rewards(env, recommendations):
    """
    Compute the total rewards that would be earned using the recommended cards.
    Parameters:
    - env (CreditCardEnv): The credit card environment with card and transaction information.
    - recommendations (dict): Dictionary mapping category to recommended card index.
    Returns:
    - category_rewards (dict): Dictionary mapping category to total reward earned in that category.
    """
    # Initialize dictionary to store rewards per category.
    category_rewards = defaultdict(float)
    
    # Track which cards have been used to account for annual fees only once.
    used_cards = set()
    
    # Iterate through each transaction in the environment.
    for _, txn in env.transactions.iterrows():
        # Get the category and amount for this transaction.
        category = txn["category"]
        amount = float(txn["amount"])
        
        # Get the recommended card for this category.
        card_idx = recommendations.get(category, 0)
        
        # Get the card information.
        card = env.cards.iloc[card_idx]
        category_map = env.card_category_maps[card_idx]
        
        # Find the multiplier for this category.
        if category in category_map:
            multiplier = category_map[category]
        else:
            # Use default multiplier of 1.0 if category not found.
            multiplier = 1.0
        
        # Get the reward type for the card.
        reward_type = str(card["reward_type"]).lower()
        
        # Calculate the reward based on the reward type of the environment.
        if env.reward_type == "points":
            # Points are worth face value (1x = 1 point per dollar).
            reward = amount * multiplier if reward_type == "points" else 0.0
        elif env.reward_type == "cashback":
            # Cashback is a percentage, so multiply by 0.01 to convert. (3x = 3% = 0.03)
            reward = amount * multiplier * 0.01 if reward_type == "cashback" else 0.0
        else:
            reward = 0.0
        
        # Subtract annual fee only the first time we use a card.
        if card_idx not in used_cards:
            fee = float(card["annual_fee_usd"])
            reward -= fee
            used_cards.add(card_idx)
        
        # Add the reward to the total of this category.
        category_rewards[category] += reward
    
    # Return the category rewards dictionary.
    return dict(category_rewards)

def print_category_recommendations(env, recs, cat_rewards):
    """
    Print a summary of card recommendations by category.
    Parameters:
    - env (CreditCardEnv): The credit card environment with card information.
    - recs (dict): Dictionary mapping category to recommended card index.
    - cat_rewards (dict): Dictionary mapping category to total reward in that category.
    Returns:
    - None
    """
    # Print header for the recommendations section.
    print("\n" + "="*80)
    print("CARD RECOMMENDATIONS BY CATEGORY")
    print("(These show potential rewards if you consistently use the recommended card)")
    print("="*80)
    
    # Calculate the total potential reward across all categories.
    total_potential_reward = sum(cat_rewards.values())
    
    # Track which cards are recommended and their total usage.
    card_usage = defaultdict(lambda: {"categories": [], "raw_rewards": 0.0, "spending": 0.0})
    
    # Sort categories alphabetically for consistent output.
    sorted_categories = sorted(recs.keys())
    
    # Iterate through each category, print details, and collect card usage information.
    for category in sorted_categories:
        # Get the recommended card index.
        card_idx = recs[category]
        
        # Get the card information from the environment.
        card = env.cards.iloc[card_idx]
        reward_type = str(card["reward_type"]).lower()
        
        # Calculate the total spending in this category.
        category_spending = env.transactions[env.transactions["category"] == category]["amount"].sum()
        
        # Get the multiplier of the card for this category if available.
        category_map = env.card_category_maps[card_idx]
        multiplier = category_map.get(category, 1.0)
        
        # Calculate raw reward for this category (before fees).
        if env.reward_type == "points":
            raw_reward = category_spending * multiplier if reward_type == "points" else 0.0
        elif env.reward_type == "cashback":
            raw_reward = category_spending * multiplier * 0.01 if reward_type == "cashback" else 0.0
        else:
            raw_reward = 0.0
        
        # Store card usage information for the card-level summary later.
        card_usage[card_idx]["categories"].append(category)
        card_usage[card_idx]["raw_rewards"] += raw_reward
        card_usage[card_idx]["spending"] += category_spending
        
        # Print the category and recommended card details.
        print(f"\nCategory: {category}")
        print(f"  Recommended Card: {card['card_name']} (Card Index: {card_idx})")
        print(f"  Issuer: {card['issuer']}")
        print(f"  Reward Type: {card['reward_type']}")
        print(f"  Reward Multiplier: {multiplier}x")
        print(f"  Total Spending in Category: ${category_spending:.2f}")
        print(f"  Raw Rewards Earned in Category: {raw_reward:.2f}")
    
    # Print card-level summary showing annual fees.
    print(f"\n{'='*80}")
    print("CARD-LEVEL SUMMARY (Annual Fees Applied Here)")
    print(f"{'='*80}")
    
    # Iterate through each card that was recommended for at least one category.
    for card_idx in sorted(card_usage.keys()):
        # Get the card information.
        card = env.cards.iloc[card_idx]
        annual_fee = float(card["annual_fee_usd"])
        
        # Get usage information for this card.
        usage = card_usage[card_idx]
        categories_used = ", ".join(usage["categories"])
        raw_rewards = usage["raw_rewards"]
        total_spending = usage["spending"]
        net_rewards = raw_rewards - annual_fee
        
        # Print card summary.
        print(f"\nCard {card_idx}: {card['card_name']}")
        print(f"  Used for categories: {categories_used}")
        print(f"  Total spending on this card: ${total_spending:.2f}")
        print(f"  Raw rewards earned: {raw_rewards:.2f}")
        print(f"  Annual fee: ${annual_fee:.2f}")
        print(f"  Net rewards (after fee): {net_rewards:.2f}")
    
    # Print the total potential reward.
    print(f"\n{'='*80}")
    print(f"TOTAL POTENTIAL REWARD (after all annual fees): {total_potential_reward:.2f}")
    print(f"{'='*80}")

# TODO: implement spending summary here (print out spending by category and brand)
# i think since we also have the data we should print a summary of spending by category and brand too bc why not

def spending_summary(df):
    """
    Print a summary of spending by category and brand.
    Parameters:
    - df (DataFrame): The transactions dataframe.
    Returns:
    - None
    """
    # Print header for the spending summary section.
    print("\n" + "="*80)
    print("SPENDING SUMMARY")
    print("="*80)
    
    # Calculate total spending.
    total_spending = df["amount"].sum()
    print(f"\nTotal Spending: ${total_spending:.2f}")
    
    # Calculate spending by category.
    print("\n--- Spending by Category ---")
    category_spending = df.groupby("category")["amount"].sum().sort_values(ascending=False)
    for category, amount in category_spending.items():
        # Calculate percentage of total spending.
        percentage = (amount / total_spending) * 100
        print(f"  {category}: ${amount:.2f} ({percentage:.1f}%)")
    
    # Calculate spending by brand.
    print("\n--- Top 10 Spending by Brand ---")
    brand_spending = df.groupby("brand")["amount"].sum().sort_values(ascending=False).head(10)
    for brand, amount in brand_spending.items():
        # Calculate percentage of total spending.
        percentage = (amount / total_spending) * 100
        print(f"  {brand}: ${amount:.2f} ({percentage:.1f}%)")

# TODO: implement main function here where we call everything else and print summaries when done w/ training

def main():
    """
    Main function to run the credit card optimizer.
    Parses command-line arguments, loads data, trains the Q-learning agent,
    and prints evaluation results and recommendations.
    """
    # Set up argument parser for command-line arguments.
    parser = argparse.ArgumentParser(description="Credit Card Rewards Optimizer using Q-Learning")
    parser.add_argument("--cards", required=False, default="cards.csv", help="Path to cards CSV file")
    parser.add_argument("--transactions", required=True, help="Path to transactions CSV file")
    parser.add_argument("--mode", choices=["points", "cashback", "both"], default="points", help="Reward optimization mode")
    parser.add_argument("--episodes", type=int, default=5000, help="Number of training episodes")
    args = parser.parse_args()
    
    # Print starting message.
    print("="*80)
    print("CREDIT CARD REWARDS OPTIMIZER")
    print("="*80)
    print(f"Loading data...")
    
    # Load the cards CSV file into a pandas DataFrame.
    try:
        cards_df = pd.read_csv(args.cards)
        print(f"Loaded {len(cards_df)} cards from {args.cards}")
    except FileNotFoundError:
        print(f"Error: Cards file '{args.cards}' not found.")
        sys.exit(1)
    
    # Load the transactions CSV file into a pandas DataFrame.
    try:
        transactions_df = pd.read_csv(args.transactions)
        print(f"Loaded {len(transactions_df)} transactions from {args.transactions}")
    except FileNotFoundError:
        print(f"Error: Transactions file '{args.transactions}' not found.")
        sys.exit(1)
    
    # Create the credit card environment with the loaded data.
    print(f"Creating environment with reward mode: {args.mode}")
    env = CreditCardEnv(cards_df, transactions_df, reward_type=args.mode)
    
    # Print environment information.
    print(f"Environment created with {env.num_cards} cards and {len(env.transactions)} transactions")
    
    # Print spending summary before training.
    spending_summary(transactions_df)
    
    # Train the Q-learning agent.
    print(f"\nTraining Q-learning agent for {args.episodes} episodes...")
    Q_table = Q_learning(env, episodes=args.episodes)
    
    # Print training completion message.
    print(f"\nTraining complete! Q-table has {len(Q_table)} states.")
    
    # Evaluate the learned policy.
    print("\nEvaluating learned policy...")
    total_reward, card_rewards = evaluate_policy(env, Q_table)

    # Print evaluation results.
    print("\n" + "="*80)
    print("EVALUATION RESULTS")
    print("(These show actual rewards earned during the Q-learning agent's evaluation run)")
    print("="*80)
    print(f"\nTotal Reward Earned: {total_reward:.2f}")
    print("\n--- Rewards by Card ---")
    for card_idx in sorted(card_rewards.keys()):
        # Get the card name for this index.
        card_name = env.cards.iloc[card_idx]["card_name"]
        reward = card_rewards[card_idx]
        print(f"  Card {card_idx} ({card_name}): {reward:.2f}")

    # Get category recommendations based on the Q-table.
    print("\nGenerating category recommendations...")
    recommendations = recommend_cards_by_category(env, Q_table)

    # Compute rewards for each category using recommendations.
    category_rewards = compute_category_rewards(env, recommendations)

    # Print the category recommendations.
    print_category_recommendations(env, recommendations, category_rewards)

    # Calculate and print the difference between potential and actual rewards.
    total_potential = sum(category_rewards.values())
    print(f"\n{'='*80}")
    print("COMPARISON")
    print(f"{'='*80}")
    print(f"Actual Reward (Q-learning agent's choices): {total_reward:.2f}")
    print(f"Potential Reward (following recommendations): {total_potential:.2f}")
    print(f"Difference: {total_potential - total_reward:.2f}")
    if total_potential > total_reward:
        improvement_pct = ((total_potential - total_reward) / abs(total_reward)) * 100 if total_reward != 0 else 0
        print(f"Potential improvement: {improvement_pct:.1f}%")
    print(f"{'='*80}")

if __name__ == "__main__":
    main()
