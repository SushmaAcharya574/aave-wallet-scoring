# DeFi Wallet Credit Scoring using Aave V2 Transactions

This project uses historical DeFi transactions from the Aave V2 protocol to assign a **credit score (0–1000)** to each wallet. The score reflects how trustworthy or risky a wallet's financial behavior is — much like a credit rating.

# Project Structure

```
project-root/
├── data/                         # Raw input file
├── output/                       # Generated results & plots
├── score_wallets.py              # Feature extraction + scoring
├── train_model.py                # ML model training and evaluation
```

# Method Overview

### 1. Feature Engineering (`score_wallets.py`)

- Parses raw transactions (`deposit`, `borrow`, `repay`, etc.)
- Computes behavior-based features:
  - Total amounts in USD
  - Count of actions
  - Repay-to-borrow ratio
  - Liquidations
- Assigns a **score out of 1000** using simple rule logic.

### 2. Model Training (`train_model.py`)

- Trains a **Random Forest Regressor** to predict credit scores from features.
- Splits into training/test sets (80/20).
- Evaluates using:
  - Mean Absolute Error (MAE)
  - R² Score

---

## How to Run

> Requires Python 3.8+  
> Install dependencies:

```bash
pip install pandas matplotlib scikit-learn joblib tqdm
```

### 1. Score the Wallets

```bash
python score_wallets.py
```

- Generates: `wallet_scores.json`, `features.csv`, `score_distribution.png`

### 2. Train the ML Model

```bash
python train_model.py
```

- Generates: `model.pkl`, `prediction_vs_actual.png`

---

## Outputs

- `output/wallet_scores.json` → Credit scores by wallet
- `output/score_distribution.png` → Histogram of wallet scores
- `output/prediction_vs_actual.png` → Model prediction vs actual scores
- `output/features.csv` → Wallet features + scores
- `output/model.pkl` → Trained model for deployment

---

## References

- [Aave V2 Protocol](https://docs.aave.com/)
- [Polygon Network](https://polygon.technology/)
- Machine Learning: `RandomForestRegressor` (from `sklearn.ensemble`)

For insights, see [`analysis.md`](analysis.md)
