import numpy as np
import pandas as pd
from gym.spaces import Discrete

# points conversion rate: 100 points = $1, so 1 point = $0.01
POINTS_PER_DOLLAR = 100.0


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
        try:
            multiplier = float(multiplier)
            rewards_map[category.strip()] = multiplier
        except ValueError:
            # skip invalid multiplier values (shouldn't ever happen with cards.csv but good practice)
            continue
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
            
        # collect all known categories for future matching
        # this is needed to normalize the transaction categories to standard names found in the cards.csv file for matching (when we use real data)
        known_categories = set()
        for m in self.card_category_maps:
            known_categories.update(m.keys())
        known_categories.add("other")
        self.known_categories = sorted(known_categories)
        
        # initialize a set to track unknown categories that don't match up with any of the categories in cards.csv
        self._unknown_category_seen = set()
        
        # raise error if the transactions df does not contain a 'category' column, which is required for processing transactions
        if "category" not in self.transactions.columns:
            raise ValueError("Given transactions csv must contain a 'category' column")
        
        # normalize the transaction categories to standard names found in the cards.csv file for matching
        self.transactions["category"] = self.transactions["category"].apply(self._normalize_category_name)

        # set the number of cards and the action space (using gym Discrete space)
        self.num_cards = len(self.cards)
        self.action_space = Discrete(self.num_cards)

        # initialize the transaction index and the cards used this episode to track rewards and fees
        self._transaction_index = 0
        self._used_cards_this_episode = set()
        
    
    def _normalize_category_name(self, raw_cat):
        """normalize transaction categories to the known categories found in the cards.csv file"""
        # if the category is missing, return 'other'
        if pd.isna(raw_cat):
            return "other"

        # convert the category to a string and strip whitespace
        cat = str(raw_cat).strip().lower()
    
        # if the category is already in the known categories, return it without modification
        if cat in self.known_categories:
            return cat
        
        # try to match common transaction patterns to known categories found in cards.csv
        if any(p in cat for p in ["restaurant", "dining", "cafe", "food", "bar"]):
            return "dining"
        if any(p in cat for p in ["grocery", "market", "supermarket", "grocery store", "grocery shop"]):
            return "groceries"
        if any(p in cat for p in ["gas", "fuel", "petrol"]):
            return "gas"
        if any(p in cat for p in ["airfare", "air travel", "airline", "flight"]):
            return "air_travel"
        if any(p in cat for p in ["hotel", "inn", "resort", "hotel booking", "hotel reservation"]):
            return "hotels"
        if any(p in cat for p in ["uber", "lyft", "taxi", "cab", "ride"]):
            return "ride_share"
        if any(p in cat for p in ["stream", "netflix", "spotify", "hulu", "hbo max", "disney+", "paramount+"]):
            return "streaming"
        if any(p in cat for p in ["online", "amazon", "ecommerce", "shop", "shop online"]):
            return "online_shopping"
        if "misc" in cat or "other" in cat:
            return "other"

        # give up and use 'other' if the category is not found in the cards.csv file
        if cat not in self._unknown_category_seen:
            # print warning for unknown categories, use this for debugging now, might remove later
            print(f"[WARNING] Unknown category '{cat}' was mapped to 'other'")
            self._unknown_category_seen.add(cat)

        return "other"


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
        elif "other" in category_map:
            multiplier = category_map["other"]
        else:
            # if the category is not found in the category map, use a multiplier of 1.0 as a default (typical for most cards)
            multiplier = 1.0

        # get the reward type for the card (points or cashback)
        reward_type = str(card["reward_type"]).lower()

        # only count rewards if they match what we're optimizing for, otherwise set reward to 0.0
        # note that this checks the reward type arg and the reward type for the card
        if self.reward_type == "points" and reward_type == "points":
            # for points mode: amount * multiplier gives points (ex. 4x = 4 points per dollar)
            reward = amount * multiplier
        elif self.reward_type == "cashback" and reward_type == "cashback":
            # for cashback: amount * multiplier * 0.01 converts percentage to decimal (ex. 6x = 6% = 0.06)
            reward = amount * multiplier * 0.01
        elif self.reward_type == "both":
            # "both" mode: calculate both points and cashback, but convert to USD for consistent units (100 points = $1 again here))
            if reward_type == "points":
                # convert points to USD (100 points = $1, so divide by POINTS_PER_DOLLAR)
                reward = amount * multiplier / POINTS_PER_DOLLAR
            elif reward_type == "cashback":
                reward = amount * multiplier * 0.01
        else:
            # optimization mode doesn't match card reward type, set reward to 0.0
            reward = 0.0

        # subtract annual fee only the first time we use a card in an episode to avoid double counting annual fees at the end
        if card_idx not in self._used_cards_this_episode:
            fee = float(card["annual_fee_usd"])
            # fees are in USD, so we need to ensure reward is also in USD before subtracting
            # for points rewards, convert fee to points equivalent (100 points = $1)
            if self.reward_type == "points" and reward_type == "points":
                # reward is in points, convert fee to points equivalent
                reward -= fee * POINTS_PER_DOLLAR
            else:
                # for cashback or "both" mode, rewards are already in USD, so subtract fee directly
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
