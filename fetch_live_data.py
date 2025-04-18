from websocket import WebSocketApp
import json
import csv
from datetime import datetime

symbol = "btcusdt"


def on_message(ws, message):
    data = json.loads(message)
    price = data['p']
    timestamp = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] Price: {price}")

    with open("price_data.csv", "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([timestamp, price])


def on_error(ws, error):
    print("Error:", error)


def on_close(ws, close_status_code, close_msg):
    print("WebSocket closed")


def on_open(ws):
    payload = {
        "method": "SUBSCRIBE",
        "params": [f"{symbol}@trade"],
        "id": 1
    }
    ws.send(json.dumps(payload))


if __name__ == "__main__":
    ws = WebSocketApp("wss://stream.binance.com:9443/ws",
                      on_open=on_open,
                      on_message=on_message,
                      on_error=on_error,
                      on_close=on_close)
    ws.run_forever()
