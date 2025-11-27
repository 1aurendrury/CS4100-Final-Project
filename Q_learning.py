import argparse
from cc_optimizer_gym import CreditCardEnv

# IMPORTANT FRIENDLY REMINDER :D
# below are some functions that I had in mind implementing when creating cc_optimizer_gym.py
# note that we might not need all of them, but they are just some things that I thought of when creating the gym/RL env

# note also that cc_optimizer_gym.py was developed with the following args in mind (can use argparse for these):
# python Q_learning.py --cards xxx.csv --transactions xxx.csv --mode points/cashback/both --episodes 2000 (xxxx)


# TODO: might need functions for loading csv cards and transactions maybe? - can do this in another file also


# TODO: will probably need a hashing function, similar to PA2
def hash(obs):
    return None

# TODO: implement Q-learning algorithm here
# one thing I thought of is that perhaps we have a bonus for using the same card multiple times in a row (if applicable) to reduce overall annual fees?
# we can make this a small bonus, maybe 0.15 or something
def Q_learning(
    episodes=2000,
    gamma=0.95,
    epsilon=1.0,
    decay=0.995,
):
    return None



# TODO: implement evaluation function here where we evaluate the current policy and return the total reward and per card reward
# here we could get the Q-values for each action and choose the action with the highest Q-value
def evaluate_policy(env, Q_table):
    return None


# TODO: implement category recommendations function here based on current/given Q-table
# here we could return a dictionary of categories and the best card for that category
def recommend_cards_by_category(env, Q_table):
    return None


# TODO: implement reward calculation here based on category recommendations
# aka take their current transaction data and multiply by the multiplier for the category for the respective card
def compute_category_rewards(env, recommendations):
    return None


# TODO: implement recommendations summary here (print out best card per category and the card's q-value, annual fee, estimated raw reward, etc.)
# this will essentially be breakdown by card (ex. use this card for dining, groceries, etc.)
def print_category_recommendations(env, recs, cat_rewards):
    return None


# TODO: implement spending summary here (print out spending by category and brand)
# i think since we also have the data we should print a summary of spending by category and brand too bc why not
def spending_summary(df):
    return None


# TODO: implement main function here where we call everything else and print summaries when done w/ training

def main():
    # feel free to change these as needed!
    parser = argparse.ArgumentParser()
    parser.add_argument("--cards", required=False) # we can use default cards.csv if not provided
    parser.add_argument("--transactions", required=True)
    parser.add_argument("--mode", choices=["points", "cashback", "both"], default="points")
    parser.add_argument("--episodes", type=int, default=5000)
    args = parser.parse_args()
    
    return None

if __name__ == "__main__":
    main()

