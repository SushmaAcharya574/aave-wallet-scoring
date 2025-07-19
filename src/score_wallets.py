import json
from collections import defaultdict
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
# 1. Load data
with open('./data/user-wallet-transactions.json') as f:
    data = json.load(f)

# 2. Feature extraction
wallets = defaultdict(lambda :{
    'deposit':0, 'borrow':0, 'repay':0, 'redeem':0, 'deposit_count':0, 'borrow_count':0, 'repay_count':0, 'liquidations':0
})

for tx in tqdm(data):
    wallet = tx['userWallet']
    action = tx['action'].lower()
    
    action_data = tx.get('actionData', {})
    raw_amount = float(action_data.get('amount', 0))
    price_usd = float(action_data.get('assetPriceUSD', 1))
    
    #Convert tp real USD value 
    amount = (raw_amount * price_usd ) /1e6
    
    if action == 'deposit':
        wallets[wallet]['deposit'] += amount
        wallets[wallet]['deposit_count'] += 1
    elif action == 'borrow':
        wallets[wallet]['borrow'] += amount
        wallets[wallet]['borrow_count'] += 1
    elif action == 'repay':
        wallets[wallet]['repay'] += amount
        wallets[wallet]['repay_count'] += 1 
    elif action == 'redeem':
        wallets[wallet]['redeem'] += amount
    elif action == 'liquidations':
        wallets[wallet]['liquidation'] += 1
        
# 3. Scoring logic
wallet_scores = {}
for wallet, info in wallets.items():
    score = 600
    
    #review for repaying fully
    if info['borrow'] > 0:
        repay_ratio = info['repay']/info['borrow']
        if repay_ratio >= 1:
            score +=200;
        elif repay_ratio >= 0.5:
            score += 50
        else :
            score -= 100
    else :
        repay_ratio = None
        
    #Large deposits = positive signal
    if info['deposit'] >= 1000:
        score += 100
    
    #Liquidation = negative
    score -= info['liquidations']*200
    
    #Never repaid after borrowing
    if info['borrow'] > 0 and info['repay'] == 0:
        score -= 100
    
    #Clip score between 0 and 1000
    score = max(0,min(1000, score))
    wallet_scores[wallet] = round(score)
    
# 4. Save scores to file
os.makedirs('./output', exist_ok =True)
with open('./output/wallet_scores.json', 'w') as file:
    json.dump(wallet_scores, file, indent = 2)

# 5. Plot histogram
scores = list(wallet_scores.values())
bins = list(range(0,1100,100))
plt.hist(scores, bins = bins, edgecolor = 'black')
plt.title('Wallet credit score Distribrution')
plt.xlabel('Score Range')
plt.ylabel('Number of wallets')
plt.grid(True)          
plt.savefig('./output/score_distribution.png')

# 6. Save features to CSV for ML training
rows = []
for wallet, info in wallets.items():
    rows.append({
        'wallet': wallet,
        'deposit': info['deposit'],
        'borrow': info['borrow'],
        'repay': info['repay'],
        'redeem': info['redeem'],
        'deposit_count': info['deposit_count'],
        'borrow_count': info['borrow_count'],
        'repay_count': info['repay_count'],
        'liquidations': info['liquidations'],
        'repay_ratio': (info['repay'] / info['borrow']) if info['borrow'] else 0,
        'score': wallet_scores[wallet]
    })

df = pd.DataFrame(rows)
df.to_csv('./output/features.csv', index=False)
print("Saved wallet features to features.csv")
