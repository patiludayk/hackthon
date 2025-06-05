import os
import xml.etree.ElementTree as ET
import pandas as pd

# Path to folder containing FpML files
fpml_folder = "./output"

# Namespace for FpML 5.13
ns = {'fpml': 'http://www.fpml.org/FpML-5/confirmation'}

# Parse FpML XML files
def parse_fpml(filepath):
    try:
        tree = ET.parse(filepath)
        root = tree.getroot()

        party1 = root.find('.//fpml:party[@id="party1"]/fpml:partyId', ns).text
        party2 = root.find('.//fpml:party[@id="party2"]/fpml:partyId', ns).text

        currency1 = root.find('.//fpml:exchangedCurrency1/fpml:paymentAmount/fpml:currency', ns).text
        amount1 = float(root.find('.//fpml:exchangedCurrency1/fpml:paymentAmount/fpml:amount', ns).text)

        currency2 = root.find('.//fpml:exchangedCurrency2/fpml:paymentAmount/fpml:currency', ns).text
        rate = float(root.find('.//fpml:exchangeRate/fpml:rate', ns).text)

        scenario = "before_tariff" if "before" in filepath else "after_tariff"

        return {
            "party1": party1,
            "party2": party2,
            "currency_pair": f"{currency1}/{currency2}",
            "notional": amount1,
            "rate": rate,
            "scenario": scenario
        }
    except Exception as e:
        print(f"Error in {filepath}: {e}")
        return None

# Extract trade list
trade_data = []
for file in os.listdir(fpml_folder):
    if file.endswith(".xml"):
        record = parse_fpml(os.path.join(fpml_folder, file))
        if record:
            trade_data.append(record)

df = pd.DataFrame(trade_data)

# Aggregate: per party1, currency pair, and scenario
agg = df.groupby(['party1', 'scenario', 'currency_pair']).agg(
    trade_count=('notional', 'count'),
    total_notional=('notional', 'sum'),
    avg_notional=('notional', 'mean'),
    avg_rate=('rate', 'mean')
).reset_index()

# Pivot to create before/after columns
before = agg[agg['scenario'] == 'before_tariff'].rename(columns={
    'trade_count': 'before_trade_count',
    'total_notional': 'before_total_notional',
    'avg_notional': 'before_avg_notional',
    'avg_rate': 'before_avg_rate'
}).drop(columns='scenario')

after = agg[agg['scenario'] == 'after_tariff'].rename(columns={
    'trade_count': 'after_trade_count',
    'total_notional': 'after_total_notional',
    'avg_notional': 'after_avg_notional',
    'avg_rate': 'after_avg_rate'
}).drop(columns='scenario')

# Merge and calculate behavior shifts
merged = pd.merge(before, after, on=['party1', 'currency_pair'], how='outer').fillna(0)
merged['notional_change_pct'] = ((merged['after_total_notional'] - merged['before_total_notional']) /
                                 merged['before_total_notional'].replace(0, 1)) * 100
merged['rate_sensitivity'] = merged['after_avg_rate'] - merged['before_avg_rate']
merged['trade_count_change'] = merged['after_trade_count'] - merged['before_trade_count']

# Save to CSV
merged.to_csv("enhanced_party_behavior_features.csv", index=False)
print("✅ Generated: enhanced_party_behavior_features.csv")
