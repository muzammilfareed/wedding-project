#!/usr/bin/env python3
import os
import subprocess

def kill_port(port):
    try:
        # Find process using the port
        result = subprocess.run(
            ["lsof", "-t", f"-i:{port}"],
            capture_output=True,
            text=True
        )
        pids = result.stdout.strip().split("\n")
        for pid in pids:
            if pid:
                print(f"Killing process on port {port} with PID: {pid}")
                os.system(f"kill -9 {pid}")
        if not pids or not pids[0]:
            print(f"No process found on port {port}.")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    kill_port(8888)
