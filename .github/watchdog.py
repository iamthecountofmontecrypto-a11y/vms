"""
Monitor watchdog -- runs on GitHub's servers (see workflows/watchdog.yml),
completely independent of the PC running vms_fault_monitor.py.

The monitor only alerts while it's running. If the PC sleeps, reboots or
loses internet, or the monitor/sync stops, alerts silently stop. This
checks the published heartbeat.txt and sends ONE Telegram message when it
goes stale, and one when it recovers. It never contacts combinedcontrol.

A stop done through the GUI (graceful) writes monitor_stopped.txt, which
is treated as deliberate: no alert.
"""
import json
import os
import sys
import urllib.parse
import urllib.request
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

STALE_MINUTES = 30
DASHBOARD_URL = "https://iamthecountofmontecrypto-a11y.github.io/vms/"
STATE_PATH = ".watchdog/state.json"
LONDON = ZoneInfo("Europe/London")  # the monitor writes naive UK local times


def read_time(path):
    try:
        with open(path, encoding="utf-8") as f:
            return datetime.fromisoformat(f.read().strip()).replace(tzinfo=LONDON)
    except (OSError, ValueError):
        return None


def send(text):
    token, chat = os.environ.get("TELEGRAM_BOT_TOKEN"), os.environ.get("TELEGRAM_CHAT_ID")
    if not token or not chat:
        sys.exit("TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID repo secrets are not set")
    data = urllib.parse.urlencode({"chat_id": chat, "text": text}).encode()
    urllib.request.urlopen(f"https://api.telegram.org/bot{token}/sendMessage", data=data, timeout=15).read()


def main():
    os.makedirs(os.path.dirname(STATE_PATH), exist_ok=True)
    try:
        with open(STATE_PATH, encoding="utf-8") as f:
            state = json.load(f)
    except (OSError, ValueError):
        state = {}

    now = datetime.now(LONDON)
    heartbeat = read_time("heartbeat.txt")
    stopped = read_time("monitor_stopped.txt")
    if os.environ.get("SEND_TEST") == "true":
        send(f"VMS watchdog test: Telegram works. Last monitor check "
             f"{heartbeat:%H:%M} ({int((now - heartbeat).total_seconds() // 60)} min ago)." if heartbeat
             else "VMS watchdog test: Telegram works, but heartbeat.txt is missing.")
    if heartbeat is None:
        sys.exit("heartbeat.txt missing or unreadable")

    age = now - heartbeat
    deliberate = stopped is not None and stopped >= heartbeat
    print(f"heartbeat {heartbeat:%Y-%m-%d %H:%M} ({int(age.total_seconds() // 60)} min old), "
          f"deliberate stop: {deliberate}, alerted_for: {state.get('alerted_for')}")

    if age > timedelta(minutes=STALE_MINUTES) and not deliberate:
        if state.get("alerted_for") != heartbeat.isoformat():
            send(f"VMS MONITOR SILENT: no check since {heartbeat:%H:%M} ({int(age.total_seconds() // 60)} min ago).\n"
                 f"Sign and fault alerts are NOT being sent. The PC may be off/asleep or offline, "
                 f"or the monitor or sync has stopped.\nCheck: {DASHBOARD_URL}")
            state["alerted_for"] = heartbeat.isoformat()
    elif state.get("alerted_for") and age <= timedelta(minutes=STALE_MINUTES):
        send(f"VMS MONITOR BACK: checks resumed (last check {heartbeat:%H:%M}).\nCheck: {DASHBOARD_URL}")
        state["alerted_for"] = None

    with open(STATE_PATH, "w", encoding="utf-8") as f:
        json.dump(state, f)


if __name__ == "__main__":
    main()
