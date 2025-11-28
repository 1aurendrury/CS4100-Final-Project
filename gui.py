import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from cc_optimizer_gym import CreditCardEnv
from Q_learning import Q_learning, recommend_cards_by_category, compute_category_rewards, evaluate_policy

# ==== Helper Functions for Plotting, Based on Functions from visualizations.py ====

def plot_training_rewards_streamlit(all_rewards):
    """ create plot to show training rewards over time """
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(all_rewards, color="blue", alpha=0.3, label="Rewards per Episode")
    
    # Plot running average
    running_avg = np.cumsum(all_rewards) / np.arange(1, len(all_rewards) + 1)
    ax.plot(running_avg, color="red", linewidth=2, label="Running Average")
    ax.set_title(f"Training Rewards Over {len(all_rewards)} Episodes", fontsize=14, fontweight='bold')
    ax.set_xlabel("Episode Number", fontsize=12)
    ax.set_ylabel("Total Reward", fontsize=12)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    return fig

def plot_q_table_heatmap_streamlit(Q_table, env):
    """ create Q-table heatmap to show what the model learned per category/card """

    # convert Q_table dictionary to a df
    df = pd.DataFrame.from_dict(Q_table, orient="index")

    card_names = env.cards["card_name"].tolist()
    df.columns = card_names
    
    # normalize each row for comparison
    df_norm = df.div(df.max(axis=1), axis=0).fillna(0)
    
    # Plot the heatmap
    fig, ax = plt.subplots(figsize=(16, max(8, len(df_norm) * 0.3)))
    sns.heatmap(df_norm,
                cmap="viridis",
                xticklabels=card_names,
                yticklabels=[str(s) for s in df_norm.index],
                cbar_kws={"label": "Normalized Q-value"},
                annot=False,
                ax=ax)
    ax.set_title("Q-Table Heatmap: Learned Card Values by State", fontsize=14, fontweight='bold')
    ax.set_xlabel("Credit Cards", fontsize=12)
    ax.set_ylabel("States (Category, Amount Bucket)", fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()

    return fig

def plot_per_card_rewards_streamlit(per_card_rewards, env):
    """ create a bar plot showing total rewards earned per card """
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # get card names and rewards
    card_indices = list(per_card_rewards.keys())
    card_names = [env.cards.iloc[i]["card_name"] for i in card_indices]
    rewards = list(per_card_rewards.values())
    
    # Create a dataframe for easier plotting
    df_rewards = pd.DataFrame({
        'Card': card_names,
        'Reward': rewards
    })
    
    # Sort by reward value (descending)
    df_rewards = df_rewards.sort_values('Reward', ascending=False)
    
    # Create bar plot
    colors = ['green' if r > 0 else 'red' for r in df_rewards['Reward']]
    ax.bar(range(len(df_rewards)), df_rewards['Reward'], color=colors, alpha=0.7)
    ax.set_xticks(range(len(df_rewards)))
    ax.set_xticklabels(df_rewards['Card'], rotation=45, ha='right')
    ax.set_title("Total Rewards Earned Per Card (During Evaluation)", fontsize=14, fontweight='bold')
    ax.set_xlabel("Credit Card", fontsize=12)
    ax.set_ylabel("Total Reward", fontsize=12)
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()

    return fig

# ==== Streamlit Webpage ====

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
   Q, all_rewards_lst = Q_learning(
       env,
       episodes=episodes,
       progress_callback=update_progress,  # Pass the callback function
       use_tqdm=False,  # Disable tqdm in Streamlit
    )

   # Clear the progress widgets after training is complete
   progress_bar.empty()
   progress_text.empty()

   st.success("Training complete! Running recommendations...")
   
   # Evaluate the policy to get per-card rewards
   total_reward, per_card_rewards = evaluate_policy(env, Q)
   
   # get recommendations for best card per category
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
   
   # ===== Plot Visualizations =====

   st.header("Training & Performance Visualizations")
   
   # Plot 1: Training Rewards
   st.subheader("1. Training Progress")
   st.write("This chart shows how the agent's performance improved over training episodes.")
   fig1 = plot_training_rewards_streamlit(all_rewards)
   st.pyplot(fig1)
   plt.close(fig1)  # Close to free memory
   
   # Plot 2: Q-Table Heatmap
   st.subheader("2. Q-Table Heatmap")
   st.write("This heatmap shows which cards the agent learned are best for different spending categories and amounts.")
   fig2 = plot_q_table_heatmap_streamlit(Q, env)
   st.pyplot(fig2)
   plt.close(fig2)  # Close to free memory
   
   # Plot 3: Per-Card Rewards
   st.subheader("3. Rewards Earned Per Card")
   st.write("This chart shows the total rewards (or losses from fees) for each card used during evaluation.")
   fig3 = plot_per_card_rewards_streamlit(per_card_rewards, env)
   st.pyplot(fig3)
   plt.close(fig3)  # Close to free memory
   
   st.success("Credit card optimizer complete!")
