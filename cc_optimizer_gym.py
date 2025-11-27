import numpy as np
import pandas as pd
from gym.spaces import Discrete


def parse_reward_category_map(raw_rewards_map: str):
    """helper function to turn cards.csv reward strings like '4x:dining;3x:gas' into a dict mapping the card's categories to multipliers"""
    rewards_map = {}
    # if the reward string is empty, return an empty map
    if pd.isna(raw_rewards_map):
        return rewards_map

    # split the reward string into tokens
    for token in str(raw_rewards_map).split(";"):
        token = token.strip()
        # if the token is empty or does not contain a "x:" multiplier, skip it
        if not token or "x:" not in token:
            continue
        # split the token into the multiplier and the category
        token_segments = token.split("x:", 1)
        multiplier = token_segments[0]
        category = token_segments[1]
        # convert the multiplier to a float and add the category to the map
        multiplier = float(multiplier)
        rewards_map[category.strip()] = multiplier
    return rewards_map


class CreditCardEnv:
    """environment for learning which credit card to use for each transaction"""
    # initialize the environment with the cards and transactions dataframes, and the reward type
    # ideally this will take in cards.csv, transactions.csv, and the reward type as arguments
    # these will probably be arg parsed in the Q_learning.py file
    # note that everything below is subject to change, but this is the general idea atm since Q_learning.py isn't implemented yet!
    
    def __init__(self, cards_df, transactions_df, reward_type="points"):
        # make copies of cards and transactions dfs so we don't edit the originals
        self.cards = cards_df.reset_index(drop=True).copy()
        self.transactions = transactions_df.reset_index(drop=True).copy()

        # optimize for points, cashback, or both, raise error if not one of the three (points is the default)
        reward_type = reward_type.lower()
        if reward_type not in {"points", "cashback", "both"}:
            raise ValueError("reward_type must be points, cashback, or both")
        self.reward_type = reward_type

        # parse reward multipliers for each card into a map of categories to multipliers
        # this ends up looking like: [{'dining': 4.0, 'gas': 3.0}, {'dining': 4.0, 'gas': 3.0}, ...] per card
        self.card_category_maps = []
        for _, card_row in self.cards.iterrows():
            reward_map_str = card_row.get("reward_category_map", "")
            category_map = parse_reward_category_map(reward_map_str)
            self.card_category_maps.append(category_map)

        # set the number of cards and the action space (using gym Discrete space)
        self.num_cards = len(self.cards)
        self.action_space = Discrete(self.num_cards)

        # initialize the transaction index and the cards used this episode to track rewards and fees
        self._transaction_index = 0
        self._used_cards_this_episode = set()
        

    def _get_current_transaction(self):
        """helper function to get/break down the current transaction we're processing for the current step in the episode"""
        # return none if we've reached the end of the transactions
        if self._transaction_index >= len(self.transactions):
            return None
        # otherwise get the current transaction row and return a dict with the relevant information
        row = self.transactions.iloc[self._transaction_index]
        return {
            "index": self._transaction_index,
            "amount": float(row["amount"]),
            "category": row["category"],
            "raw_txn": row,
        }

    def _reward_for_card(self, card_idx, txn):
        """helper function to calculate the reward for using a given card on a given transaction"""
        # get the amount, category, and card from the transaction and card
        amount = txn["amount"]
        category = txn["category"]
        card = self.cards.iloc[card_idx]
        category_map = self.card_category_maps[card_idx]

        # find the multiplier for this category using the category map
        if category in category_map:
            multiplier = category_map[category]
        else:
            # if the category is not found in the category map, use a multiplier of 1.0 as a default (typical for most cards)
            # note that this is different from the default multiplier of 1.0 for cards with no category map!
            # also note that this won't ever be reached with our fake data, but it's good to have for when we use real data (tbd in a later PR)
            multiplier = 1.0

        # get the reward type for the card (points or cashback)
        reward_type = str(card["reward_type"]).lower()

        # only count rewards if they match what we're optimizing for, otherwise set reward to 0.0
        if self.reward_type == "points":
            reward = amount * multiplier if reward_type == "points" else 0.0
        elif self.reward_type == "cashback":
            reward = amount * multiplier if reward_type == "cashback" else 0.0
        else:
            # otherwise, set reward to 0.0
            reward = 0.0


        # subtract annual fee only the first time we use a card in an episode to avoid double counting annual fees at the end
        if card_idx not in self._used_cards_this_episode:
            fee = float(card["annual_fee_usd"])
            reward -= fee
            self._used_cards_this_episode.add(card_idx)

        return reward

    def reset(self):
        """reset the environment for a new episode"""
        # reset the transaction index and used cards set
        self._transaction_index = 0
        self._used_cards_this_episode = set()

        # get the first transaction and return the observation, reward, done, and info
        txn = self._get_current_transaction()
        # if there are no transactions, return None, 0.0, True, {}
        if txn is None:
            return None, 0.0, True, {}

        # otherwise return the observation, reward, done, and info
        obs = {
            "txn_index": txn["index"],
            "category": txn["category"],
            "amount": txn["amount"],
        }
        return obs, 0.0, False, {"raw_txn": txn["raw_txn"]}

    def step(self, action):
        """perform one step: use the chosen card and move to next transaction"""
        # get the current transaction and return None, 0.0, True, {} if there is no current transaction to process
        txn = self._get_current_transaction()
        if txn is None:
            return None, 0.0, True, {}

        # otherwise calculate the reward for using the chosen card on the current transaction (not Q-value, just the points/cashback value)
        reward = self._reward_for_card(action, txn)
        # increment the transaction index and get the next transaction
        self._transaction_index += 1
        next_txn = self._get_current_transaction()

        # if there are no more transactions, return None, reward, True, {}
        if next_txn is None:
            return None, reward, True, {
                "card_index": action,
                "raw_txn": txn["raw_txn"],
            }

        # otherwise return the observation, reward, done, and info for the next transaction
        next_obs = {
            "txn_index": next_txn["index"],
            "category": next_txn["category"],
            "amount": next_txn["amount"],
        }
        return next_obs, reward, False, {"card_index": action, "raw_txn": txn["raw_txn"]}
