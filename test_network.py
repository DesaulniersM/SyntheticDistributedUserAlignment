import requests
import time
import subprocess

def test_communication():
    print("--- Starting Network Test ---")
    
    # 1. Check if containers are up (Assuming they were started via docker-compose)
    try:
        r1 = requests.get("http://localhost:8001/")
        r2 = requests.get("http://localhost:8002/")
        print(f"Node 1 status: {r1.json()}")
        print(f"Node 2 status: {r2.json()}")
    except Exception as e:
        print(f"Error connecting to nodes: {e}")
        print("Make sure to run 'docker-compose up -d' first.")
        return

    # 2. Test HTTP Communication (Node 1 -> Node 2)
    print("Testing HTTP Message...")
    msg_payload = {
        "sender": "node1",
        "content": "Hello via HTTP!",
        "data": {"test": 123}
    }
    try:
        resp = requests.post("http://localhost:8002/receive_message", json=msg_payload)
        print(f"HTTP Response from Node 2: {resp.json()}")
    except Exception as e:
        print(f"HTTP Test failed: {e}")

    # 3. Test UDP Communication (Trigger Node 1 to send UDP to Node 2)
    print("Testing UDP Message (Node 1 -> Node 2)...")
    udp_payload = {
        "target_host": "node2",
        "target_port": 9000,
        "content": "Hello via UDP!",
        "data": {"value": 42}
    }
    try:
        resp = requests.post("http://localhost:8001/send_udp", json=udp_payload)
        print(f"Trigger UDP Response: {resp.json()}")
        
        print("Checking Node 2 logs for received UDP message...")
        time.sleep(1) # Give it a second to process
        logs = subprocess.check_output(["docker-compose", "logs", "node2"]).decode()
        if "UDP received" in logs and "Hello via UDP!" in logs:
            print("SUCCESS: UDP message received by Node 2!")
        else:
            print("FAILURE: UDP message not found in Node 2 logs.")
            print("Last 10 lines of logs:")
            print("".join(logs.splitlines()[-10:]))
            
    except Exception as e:
        print(f"UDP Test failed: {e}")

if __name__ == "__main__":
    test_communication()