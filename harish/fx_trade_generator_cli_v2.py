import os
import sys
import yaml
import uuid
import random
from datetime import datetime, timedelta
from jinja2 import Environment, FileSystemLoader


def load_client_profiles(yaml_file):
    with open(yaml_file, "r") as f:
        return yaml.safe_load(f)["clients"]


def generate_trade_data(client, trade_index):
#     today = datetime.today()
    today = datetime.strptime(client["trade_date"],"%Y-%m-%d")
    trade_date = today - timedelta(days=random.randint(2, 10))
    value_date = trade_date + timedelta(days=random.choice([2, 5, 7, 10, 30]))

    amount1 = random.randint(*client["amount_range"])
    fx_rate = round(random.uniform(*client["rate_range"]), 4)
    amount2 = round(amount1 * fx_rate, 2)

    return {
        "uuid": str(uuid.uuid4()),
        "trade_id": f"{client['client_id']}_trade_{trade_index}",
        "creation_timestamp": datetime.now().isoformat(),
        "correlation_id": str(uuid.uuid4()),
        "sequence_number": trade_index,

        "trade_date": trade_date.strftime("%Y-%m-%d"),
        "value_date": value_date.strftime("%Y-%m-%d"),

        "currency1": client["base_currency"],
        "currency2": client["preferred_currency"],
        "amount1": amount1,
        "amount2": amount2,
        "rate": fx_rate,
        "spot_rate": fx_rate,
        "forward_points": 0.0,

        "party1": {
            "party_id": client["client_id"],
            "party_name": client["party_name"]
        },
        "party2": {
            "party_id": client["counterparty_id"],
            "party_name": client["counterparty_name"]
        }
    }


def render_trade_template(env, template_file, trade_data):
    template = env.get_template(template_file)
    return template.render(trade=trade_data)


def write_trade_file(output_dir, trade_xml, client_id, index):
    os.makedirs(output_dir, exist_ok=True)
    file_path = os.path.join(output_dir, f"{client_id}_trade_{index}" + "_before_extra.xml")
    with open(file_path, "w") as f:
        f.write(trade_xml)


def main(template_file, yaml_file, output_dir, num_trades):
    env = Environment(loader=FileSystemLoader(os.path.dirname(template_file)))
    template_name = os.path.basename(template_file)
    clients = load_client_profiles(yaml_file)

    trades_per_client = int(num_trades) // len(clients)
    for client in clients:
        for i in range(trades_per_client):
            trade_data = generate_trade_data(client, i + 1)
            xml_output = render_trade_template(env, template_name, trade_data)
            write_trade_file(output_dir, xml_output, client["client_id"], i + 1)


if __name__ == "__main__":
    if len(sys.argv) != 5:
        print("Usage: python fx_trade_generator_cli.py <template_file> <yaml_file> <output_dir> <num_trades>")
        sys.exit(1)

    main(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])
