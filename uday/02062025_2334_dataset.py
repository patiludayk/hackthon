# Sample synthetic dataset for Power BI FpML trade visualization
import pandas as pd
import random
from datetime import datetime, timedelta

# Define synthetic parties and regions
parties = ['Party A', 'Party B', 'Party C', 'Party D', 'Party E']
regions = {
    'Party A': 'California',
    'Party B': 'New York',
    'Party C': 'Illinois',
    'Party D': 'Texas',
    'Party E': 'Florida'
}

# Generate synthetic trade data
n_trades = 1000
start_date = datetime(2021, 1, 1)

data = []
for i in range(n_trades):
    buyer = random.choice(parties)
    seller = random.choice([p for p in parties if p != buyer])
    trade_date = start_date + timedelta(days=random.randint(0, 1300))
    fixed_rate = round(random.uniform(1.5, 4.5), 2)
    period = random.choice(['M', 'Q', 'Y'])
    multiplier = random.choice([1, 3, 6, 12])

    data.append({
        'TradeID': f'TR{i:04d}',
        'TradeDate': trade_date.date(),
        'BuyerParty': buyer,
        'SellerParty': seller,
        'FixedRate': fixed_rate,
        'Period': period,
        'Multiplier': multiplier,
        'Region': regions[seller]  # Region by seller party
    })

# Save to CSV or Excel for Power BI import
df = pd.DataFrame(data)
df.to_csv("fpml_trade_sample.csv", index=False)
print("Generated sample FpML trade dataset: fpml_trade_sample.csv")

# Preview first few rows
print(df.head())
