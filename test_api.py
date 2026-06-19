import requests
import json
import time

BASE_URL = "http://127.0.0.1:8000"

# Sample inputs mimicking different clinical states and dashboard profiles
TEST_CASES = [
    {
        "name": "Standard Case (High Sleep, Good Mood)",
        "payload": {
            "user_id": "user_happy_001",
            "message": "I had a great day today and slept really well last night!",
            "logs": {
                "sleep_hours": 8.5,
                "mood_score": 8.0,
                "productivity_score": 7.0,
                "social_score": 7.0,
                "session_frequency": 4.0
            }
        }
    },
    {
        "name": "High Risk Case (Depressed Message, Low Sleep & Social)",
        "payload": {
            "user_id": "user_at_risk_002",
            "message": "I feel so lonely and hopeless. I can't keep going like this, everything feels like too much effort.",
            "logs": {
                "sleep_hours": 3.0,
                "mood_score": 2.0,
                "productivity_score": 1.0,
                "social_score": 1.0,
                "session_frequency": 1.0
            }
        }
    },
    {
        "name": "Missing Features Case (Some logs omitted - Tests FT-Transformer learned masking)",
        "payload": {
            "user_id": "user_partial_003",
            "message": "Just checking in, doing fine.",
            "logs": {
                "sleep_hours": 6.5,
                "mood_score": 5.0
                # productivity, social, and session_frequency are omitted
            }
        }
    },
    {
        "name": "New User Case (No logs at all - Tests fallback to global means)",
        "payload": {
            "user_id": "user_new_004",
            "message": "Hello, I just downloaded the app.",
            "logs": {}
        }
    }
]

def run_test_pipeline():
    print("=" * 70)
    print(" Kenko Layer 1: Integration & End-to-End Test Pipeline ".center(70, "="))
    print("=" * 70)

    # 1. Health check
    try:
        health_resp = requests.get(f"{BASE_URL}/health")
        if health_resp.status_code == 200:
            status = health_resp.json()
            print(f"[SUCCESS] Health check passed. Status: {status}")
        else:
            print(f"[FAIL] Health check failed with status: {health_resp.status_code}")
            return
    except requests.exceptions.ConnectionError:
        print(f"[ERROR] Could not connect to API server at {BASE_URL}.")
        print("Please ensure the FastAPI server is running: .\\augur\\Scripts\\python.exe -m uvicorn layer1.api:app --reload --port 8000")
        return

    # 2. Run Test Cases
    headers = {"Content-Type": "application/json"}
    
    for case in TEST_CASES:
        name = case["name"]
        payload = case["payload"]
        
        print("\n" + "-" * 70)
        print(f"Running Test Case: {name}")
        print(f"Input Message: '{payload['message']}'")
        print(f"Input Logs: {payload['logs']}")
        print("-" * 70)
        
        t0 = time.perf_counter()
        try:
            response = requests.post(f"{BASE_URL}/layer1/process", headers=headers, json=payload)
            latency_ms = (time.perf_counter() - t0) * 1000
            
            if response.status_code == 200:
                result = response.json()
                print(f"[SUCCESS] Response received in {latency_ms:.1f}ms")
                print(f"  - Fused Vector Dim: {result['dim']} (Text: {result['text_embedding']['dim']} + Tabular: {result['tabular_embedding']['dim']})")
                print(f"  - Preliminary Risk Score: {result['preliminary_risk_score']:.4f}")
                print(f"  - Imputed/Missing Features: {result['tabular_embedding']['missing_features']}")
                
                # Check risk alerting
                risk = result['preliminary_risk_score']
                if risk > 0.7:
                    print("  [ALERT] High risk level detected! (Forwarding to Layer 3 Safety Guardrails)")
                elif risk > 0.4:
                    print("  [WARN] Moderate risk level detected.")
                else:
                    print("  [OK] Low risk level detected.")
            else:
                print(f"[FAIL] API returned status {response.status_code}")
                print(f"Details: {response.text}")
        except Exception as e:
            print(f"[ERROR] Exception occurred: {e}")

    print("\n" + "=" * 70)
    print(" Test Pipeline Run Complete ".center(70, "="))
    print("=" * 70)

if __name__ == "__main__":
    run_test_pipeline()
