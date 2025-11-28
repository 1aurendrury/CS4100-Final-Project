import streamlit as st
import pandas as pd
from cc_optimizer_gym import CreditCardEnv
from Q_learning import Q_learning, recommend_cards_by_category, compute_category_rewards

# quick wip of a streamlit app for the credit card optimizer, runs on localhost:8501 when you run 'streamlit run gui.py' in terminal
# todos for the AM: fix the graphs to be descending by value, fix points calculation, seems slightly off atm (might be for Q-learning)
st.set_page_config(page_title="Credit Card Optimizer", layout="wide")
st.title("Credit Card Q-Learning Optimizer")
st.markdown("""Upload your credit card dataset and transaction history, 
            and let the reinforcement learning model recommend the **best credit card for each of your spending categories**. 
            If you do not upload a credit cards file, we will use the system default one (30+ of the most popular credit cards in the US).""")

# upload cards and transactions files
cards_file = st.file_uploader("Upload credit cards (optional)", type=["csv"])
tx_file = st.file_uploader("Upload transactions (required)", type=["csv"])

# select reward type
mode = st.selectbox("Optimize for:", ["points", "cashback", "both"])

# select number of training episodes
episodes = st.slider("Number of Training Episodes", 500, 20000, 5000, step=500)

# run button to start the optimization
run_button = st.button("Run Optimization")

# actually run Q-learning now using our code :D
if run_button:
   # load transactions
   if tx_file is None:
       st.error("Please upload a transactions csvfile.")
       st.stop()

   # load transactions dataframe
   tx_df = pd.read_csv(tx_file)

   # load cards dataframe
   if cards_file is None:
       st.warning("No credit cards file uploaded - using default dataset (30+ of the most popular credit cards in the US).")
       cards_path = "creditcards/cards.csv"

       # try to load default cards file
       try:
           cards_df = pd.read_csv(cards_path)
       except Exception as e:
           st.error(f"Default credit cards file not found at {cards_path}")
           st.stop()
   else:
       # load uploaded cards file
       cards_df = pd.read_csv(cards_file)

   # build environment
   st.success("Files loaded successfully!")

   # build environment with uploaded cards and transactions - similar to Q_learning.py main
   env = CreditCardEnv(cards_df, tx_df, reward_type=mode)

   # train Q-learning model
   st.write("Training Q-learning model...")

   # Create a progress bar widget
   progress_bar = st.progress(0)
   # Create a text widget to show episode number
   progress_text = st.empty()

   # Define a callback function that updates the Streamlit progress bar
   def update_progress(current_episode, total_episodes):
    # Calculate progress as a percentage (0.0 to 1.0)
    progress = current_episode / total_episodes
    # Update the progress bar
    progress_bar.progress(progress)
    # Update the text to show current episode
    progress_text.text(f"Episode {current_episode} / {total_episodes}")

   # Train the Q-learning model with the progress callback
   Q = Q_learning(
       env,
       episodes=episodes,
       progress_callback=update_progress,  # Pass the callback function
       use_tqdm=False,  # Disable tqdm in Streamlit
    )

   # Clear the progress widgets after training is complete
   progress_bar.empty()
   progress_text.empty()

   st.success("Training complete! Running recommendations...")

   # get recommendations for best card per category - similar to Q_learning.py main again
   recs = recommend_cards_by_category(env, Q)
   cat_rewards = compute_category_rewards(env, recs)

   # print results table
   st.header("Best Card Per Spending Category")

   # loop through categories and add to rows
   rows = []
   for cat, info in recs.items():
       card_idx = info["card_idx"]
       card = env.cards.iloc[card_idx]
       reward_data = cat_rewards[cat]

       reward_type = reward_data["reward_type"]
       multiplier = reward_data["mult"]
       fee = float(card["annual_fee_usd"])

       raw_reward = reward_data["raw_points"] if reward_type == "points" else reward_data["raw_cash"]

       rows.append([
           cat,
           f"{card['card_name']} ({card['issuer']})",
           f"{multiplier:.1f}x" if reward_type == "points" else f"{multiplier:.1f}%",
           f"${fee:.2f}",
           f"{raw_reward:.0f} points" if reward_type == "points" else f"${raw_reward:.2f}"
       ])

   df_out = pd.DataFrame(rows, columns=["Category", "Recommended Card", "Multiplier", "Annual Fee", "Estimated Raw Reward"])
   # print results table to the app
   st.table(df_out)

   # print global summary
   st.header("Global Summary")
   
   # get unique cards selected
   selected_cards = {info["card_idx"] for info in recs.values()}
   total_fees = sum(float(env.cards.iloc[c]["annual_fee_usd"]) for c in selected_cards)

   # get total raw rewards
   total_points = sum(v["raw_points"] for v in cat_rewards.values())
   total_cash = sum(v["raw_cash"] for v in cat_rewards.values())

   # calculate net value
   net_value = total_cash + (total_points * 0.01) - total_fees

   # display all three sections side by side in columns
   col1, col2, col3 = st.columns(3)
   
   with col1:
       st.subheader("Annual Fees")
       st.write(f"Total annual fees (unique cards only): **${total_fees:.2f}**")

   with col2:
       st.subheader("Raw Rewards")
       if total_points > 0:
           st.write(f"Total points earned: **{total_points:.0f} points**")
       if total_cash > 0:
           st.write(f"Total cashback earned: **${total_cash:.2f}**")

   with col3:
       st.subheader("Net Estimated Yearly Value")
       st.write(f"**${net_value:.2f}** after annual fees")


   # print spending summary
   st.header("Spending Summary")

   cat_totals = tx_df.groupby("category")["amount"].sum().sort_values(ascending=False)
   st.write("**Spending by Category:**")
   st.bar_chart(cat_totals)


   if "brand" in tx_df.columns:
       brand_totals = tx_df.groupby("brand")["amount"].sum().sort_values(ascending=False)
       st.write("**Top Merchants/Brands:**")
       st.bar_chart(brand_totals)

   st.success("Credit card optimizer complete!")
