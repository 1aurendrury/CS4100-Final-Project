import pandas as pd
import numpy as np
import random
from faker import Faker

fake = Faker()

# categories pulled from the cards.csv file
categories = [
    "dining",
    "groceries",
    "gas",
    "air_travel",
    "hotels",
    "ride_share",
    "streaming",
    "online_shopping",
    "other"
]

# some random brands that correspond to the categories above
brands_by_category = {
    "dining": ["Starbucks", "Chipotle", "McDonald's", "Dunkin'"],
    "groceries": ["Whole Foods", "Trader Joe's", "Walmart", "Stop and Shop"],
    "gas": ["Shell", "Mobil", "BP"],
    "air_travel": ["Delta", "United", "American Airlines"],
    "hotels": ["Marriott", "Hilton", "Hyatt"],
    "ride_share": ["Uber", "Lyft"],
    "streaming": ["Netflix", "Spotify", "Hulu"],
    "online_shopping": ["Amazon", "eBay", "Etsy"],
    "other": ["CVS", "Walgreens", "Target"]
}

# generate 100 fake credit card transactions using the data above
rows = []
for x in range(100):
    category = random.choice(categories)
    merchant = random.choice(brands_by_category[category])
    amount = round(np.random.uniform(3, 300), 2)
    date = fake.date_between(start_date="-3y", end_date="today")
    brand = merchant
    rows.append([date, merchant, brand, amount, category])

df = pd.DataFrame(rows, columns=["date", "merchant", "brand", "amount", "category"])
df.to_csv("fake_transactions.csv", index=False)
