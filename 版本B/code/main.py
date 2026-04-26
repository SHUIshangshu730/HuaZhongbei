"""
主程序：串联所有子问题，汇总结果
"""
import json, os, sys, subprocess
sys.path.insert(0, '.')

PYTHON = "/d/modex/Modex-MH-Agent/runtime/python/python.exe"

def run_module(module):
    print(f"\n{'='*50}")
    print(f"运行: {module}")
    print('='*50)
    result = subprocess.run(
        [PYTHON, "-m", module],
        capture_output=False, text=True
    )
    return result.returncode == 0

if __name__ == "__main__":
    os.makedirs("figures", exist_ok=True)
    success = {}

    success["problem1"] = run_module("code.problem1")
    success["problem2"] = run_module("code.problem2")
    success["problem3"] = run_module("code.problem3")
    success["sensitivity"] = run_module("code.sensitivity_analysis")

    # 汇总所有结果
    all_results = {"run_status": success}
    for fname in ["problem1_results", "problem2_results", "problem3_results", "sensitivity_results"]:
        path = f"figures/{fname}.json"
        if os.path.exists(path):
            with open(path, encoding="utf-8") as f:
                all_results[fname] = json.load(f)

    with open("figures/all_results.json", "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    print("\n所有结果已汇总到 figures/all_results.json")
    print(f"运行状态: {success}")
