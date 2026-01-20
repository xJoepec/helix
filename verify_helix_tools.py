import subprocess
import sys
import os
import time

def run_command(command, description):
    print(f"\n{'='*80}")
    print(f"TESTING: {description}")
    print(f"COMMAND: {command}")
    print(f"{'='*80}\n")
    
    try:
        # Run command and capture output
        result = subprocess.run(
            command,
            shell=True,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        print("✅ SUCCESS")
        print("Output head (first 10 lines):")
        print('\n'.join(result.stdout.splitlines()[:10]))
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ FAILED with exit code {e.returncode}")
        print("STDOUT:")
        print(e.stdout)
        print("STDERR:")
        print(e.stderr)
        return False

def main():
    results = []
    
    # Ensure we are in the right directory
    cwd = os.getcwd()
    print(f"Working directory: {cwd}")

    # 1. Test Helix Demo
    cmd1 = "./helix demo --non-interactive --no-show --epochs 1 --samples 100"
    results.append(run_command(cmd1, "Helix Demo (Standard Experiment)"))

    # 2. Test Helix Reveals
    cmd2 = "./helix reveals"
    results.append(run_command(cmd2, "Helix Reveals (Invariants Showcase)"))

    # 3. Test Helix HelixEnv (OA Environment)
    cmd3 = "./helix helixenv --samples 100 --epochs 1 --no-ulam"
    results.append(run_command(cmd3, "Helix HelixEnv (OA Environment)"))

    # 4. Test Helix K-Theory (requires data)
    # Ensure data exists
    if not os.path.exists("test_ktheory_data.npy"):
        try:
            import numpy as np
            X = np.random.randn(100, 10).astype(np.float32)
            np.save("test_ktheory_data.npy", X)
            print("Created temporary test_ktheory_data.npy")
        except ImportError:
            print("Skipping K-Theory test: numpy not found to generate data")
            results.append(False)
    
    if os.path.exists("test_ktheory_data.npy"):
        cmd4 = "./helix ktheory --data-x test_ktheory_data.npy --width 8 --max-depth 2"
        results.append(run_command(cmd4, "Helix K-Theory Analysis"))
    
        # 5. Test Helix Analyze (Custom Analysis)
        cmd5 = "./helix analyze --data-x test_ktheory_data.npy --epochs 1 --no-ulam"
        results.append(run_command(cmd5, "Helix Analyze (Custom Experiment)"))
    else:
        print("Skipping K-Theory/Analyze tests due to missing data")
        results.append(False) # K-theory
        results.append(False) # Analyze

    # Summary
    print(f"\n{'='*80}")
    print("TEST SUMMARY")
    print(f"{'='*80}")
    tests = [
        "Helix Demo",
        "Helix Reveals",
        "Helix HelixEnv",
        "Helix K-Theory",
        "Helix Analyze"
    ]
    
    clean_pass = True
    for i, res in enumerate(results):
        status = "PASS" if res else "FAIL"
        print(f"{tests[i]}: {status}")
        if not res:
            clean_pass = False
            
    if clean_pass:
        print("\nAll tests passed successfully! 🚀")
        sys.exit(0)
    else:
        print("\nSome tests failed. Please review logs.")
        sys.exit(1)

if __name__ == "__main__":
    main()
