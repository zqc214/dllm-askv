import re
from collections import defaultdict

def analyze_log(filename="log.txt"):
    # 存储统计数据
    stats = {
        'Dual Cache': {'nfe': [], 'time': [], 'acc_rate': [], 'count': 0},
        'Prefix Cache': {'nfe': [], 'time': [], 'acc_rate': [], 'count': 0}
    }
    
    current_mode = None
    current_problem_time = 0
    current_problem_nfe = 0
    current_problem_tokens = 0
    
    with open(filename, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            line = line.strip()
            
            # 1. 检测模式切换
            if "[Dual Cache Mode]" in line:
                current_mode = 'Dual Cache'
                current_problem_time = 0
            elif "[Prefix Cache Mode]" in line:
                current_mode = 'Prefix Cache'
                current_problem_time = 0
            
            # 2. 解析 Block 信息
            # 格式: Block 0: NFE=45, AccRate=2.10 toks/step, Time=1.200s
            if line.startswith("Block") and "Time=" in line and current_mode:
                try:
                    # 提取 Time
                    time_part = line.split("Time=")[1].replace("s", "")
                    block_time = float(time_part)
                    current_problem_time += block_time
                    
                    # 提取 AccRate
                    acc_part = line.split("AccRate=")[1].split(" ")[0]
                    stats[current_mode]['acc_rate'].append(float(acc_part))
                except:
                    pass

            # 3. 解析 Total NFE (通常标志着一个问题结束)
            if "Total NFE:" in line and current_mode:
                try:
                    nfe = int(line.split(":")[1])
                    stats[current_mode]['nfe'].append(nfe)
                    stats[current_mode]['time'].append(current_problem_time)
                    stats[current_mode]['count'] += 1
                except:
                    pass

    # --- 输出报告 ---
    print(f"{'Metric':<20} | {'Dual Cache (Optimized)':<25} | {'Prefix Cache (Baseline)':<25} | {'Diff'}")
    print("-" * 85)
    
    metrics = ['Avg NFE', 'Avg Time (s)', 'Avg AccRate']
    keys = ['nfe', 'time', 'acc_rate']
    
    for metric, key in zip(metrics, keys):
        val_dual = sum(stats['Dual Cache'][key]) / len(stats['Dual Cache'][key]) if stats['Dual Cache'][key] else 0
        val_prefix = sum(stats['Prefix Cache'][key]) / len(stats['Prefix Cache'][key]) if stats['Prefix Cache'][key] else 0
        
        diff = val_dual - val_prefix
        diff_pct = (diff / val_prefix * 100) if val_prefix > 0 else 0
        
        print(f"{metric:<20} | {val_dual:<25.4f} | {val_prefix:<25.4f} | {diff_pct:+.1f}%")

    print("-" * 85)
    print(f"Problems Analyzed    | {stats['Dual Cache']['count']:<25} | {stats['Prefix Cache']['count']:<25} |")

if __name__ == "__main__":
    # 如果你的日志文件不叫 log.txt，请修改这里
    analyze_log("/home/qchzhao/Fast-dLLM-sparsekv/llada/humaneval_askv_eval_20260107_165726.log")