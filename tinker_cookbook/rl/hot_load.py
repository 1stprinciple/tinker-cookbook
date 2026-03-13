import time
from typing import Any, Dict

import requests


def get_current_hot_load_snapshot(base_url: str) -> str:
    """
    Fetches the currently active hot load snapshot identity.
    """
    # .rstrip('/') prevents double slashes if the user passes 'http://localhost:8902/'
    url = f"{base_url.rstrip('/')}/v1/models/hot_load"

    try:
        response = requests.get(url, headers={"Content-Type": "application/json"})
        response.raise_for_status()

        data = response.json()
        replicas = data.get("replicas", [])

        if replicas:
            return replicas[0].get("current_snapshot_identity")

        return "No replicas found"

    except requests.exceptions.RequestException as e:
        return f"Request failed: {e}"


def hot_load_snapshot(base_url: str, identity: str) -> Dict[str, Any]:
    """
    Triggers a hot load for a specific snapshot identity.
    """
    url = f"{base_url.rstrip('/')}/v1/models/hot_load"
    headers = {"Content-Type": "application/json"}
    payload = {"identity": identity}

    try:
        response = requests.post(url, json=payload, headers=headers)
        response.raise_for_status()
        return response.json()

    except requests.exceptions.RequestException as e:
        return {"error": f"Failed to reach server: {e}"}


def trigger_and_wait_for_hot_load(base_url: str, target_identity: str, interval: int = 10, timeout_limit: int = 300) -> bool:
    """
    Triggers the load and polls every 'interval' seconds until successful
    or until 'timeout_limit' (300 seconds default) is reached.
    """
    print(f"🚀 Initiating hot load for: '{target_identity}' at {base_url}")

    current_snapshot = get_current_hot_load_snapshot(base_url)
    if target_identity == current_snapshot:
        print(f"✅ Target snapshot '{target_identity}' is already active.")
        return True
    load_result = hot_load_snapshot(base_url, target_identity)

    # Check if the returned dictionary contains an error
    if "error" in load_result:
        print(f"❌ Failed to trigger hot load: {load_result['error']}")
        return False

    start_time = time.time()

    while True:
        current_snapshot = get_current_hot_load_snapshot(base_url)
        print(f"Checking status... Current: '{current_snapshot}' | Target: '{target_identity}'")

        if current_snapshot == target_identity:
            print("✅ Success! Snapshot is now active.")
            return True

        # Safety break if it takes too long
        if (time.time() - start_time) > timeout_limit:
            print("❌ Timeout reached. Hot load failed or is taking too long.")
            return False

        time.sleep(interval)
        time.sleep(interval)