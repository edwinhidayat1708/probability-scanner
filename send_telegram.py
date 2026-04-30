# send_telegram.py
import requests

TOKEN = "8671405778:AAET2ilEnkz8XMYedOR5k0d1UG1gRGpLe9w"
CHAT_ID = "5900350160"

def send_message(text):
    url = f"https://api.telegram.org/bot{TOKEN}/sendMessage"
    requests.post(url, data={
        "chat_id": CHAT_ID,
        "text": text
    })

def send_file(file_path):
    url = f"https://api.telegram.org/bot{TOKEN}/sendDocument"
    with open(file_path, 'rb') as f:
        requests.post(url, data={"chat_id": CHAT_ID}, files={"document": f})

if __name__ == "__main__":
    send_message("✅ Screener selesai dijalankan")
    send_file("output/live_signal.csv")