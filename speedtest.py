import os
import subprocess

# 定义需要测试的核心数
# core_counts = [2, 4, 6, 8, 12, 16, 20, 24, 28, 32]
core_counts = [4]

# 创建日志目录
log_dir = "log"
os.makedirs(log_dir, exist_ok=True)

# 循环遍历每个核心数，运行测试命令并保存输出
for num in core_counts:
    print(f"Running with {num} cores...")
    log_file = os.path.join(log_dir, f"{num}C")

    # 构造 mpirun 命令
    command = ["mpirun", "-np", str(num), "python", "DynamicPF.py"]

    # 打开日志文件并执行命令
    with open(log_file, "w") as f:
        subprocess.run(command, stdout=f, stderr=subprocess.STDOUT)

    print(f"Finished running with {num} cores. Output saved to {log_file}.")

print("All tests completed.")
