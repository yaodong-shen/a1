import os
import sys
import time
from collections import Counter
import itertools


def gpu_info():
    task_status = os.popen('nvidia-smi | grep python').read().split('|')
    task_status.pop()
    gpu_status = os.popen('nvidia-smi | grep %').read().split('|')

    gpu_num = (len(gpu_status) - 1) // 4
    gpu_memory_availables = []
    gpu_powers = []
    tasks = [0] * gpu_num

    if len(task_status) > 1:
        task_counter = Counter([int(v.split()[0]) for v in task_status[1::2]])
        for k, v in task_counter.items():
            tasks[k] += v

    for i in range(gpu_num):
        left_memory = int(gpu_status[2 + 4 * i].split('/')[1].split('M')[0].strip()) - int(
            gpu_status[2 + 4 * i].split('/')[0].split('M')[0].strip())
        gpu_memory_availables.append(left_memory)
        gpu_power = int(gpu_status[1 + 4 * i].split('   ')[-1].split('/')[0].split('W')[0].strip())
        gpu_powers.append(gpu_power)

    return gpu_powers, gpu_memory_availables, tasks


def narrow_setup(interval=600, m_need=10000, times=0, gpus=[0], max_tasks=3):
    cnt = 0
    while True:  # set waiting condition
        gpu_powers, gpu_memory_availables, tasks = gpu_info()
        max_available_memory_gpu_id = -1
        max_available_memory = -1
        for idx, (m, t) in enumerate(zip(gpu_memory_availables, tasks)):
            if idx in gpus and m > max_available_memory and m > m_need and t < max_tasks and gpu_powers[idx] > 0:
                max_available_memory = m
                max_available_memory_gpu_id = idx

        cnt += 1
        symbol = f'monitoring times: {cnt}' + '|'
        if max_available_memory_gpu_id == -1:
            gpu_memory_str = 'max gpu available memory:%d MiB |' % max(gpu_memory_availables)
            task_str = 'all tasks:%d |' % sum(tasks)
        else:
            gpu_memory_str = 'max gpu available memory:%d MiB |' % max_available_memory
            task_str = 'task number: %d' % tasks[max_available_memory_gpu_id]
            cmd = run(times, max_available_memory_gpu_id)
            times += 1
            print(f'No.{times} experiment is running')

        sys.stdout.write(symbol + '\n' + gpu_memory_str + '\n' + task_str + '\n\n')
        sys.stdout.flush()

        time.sleep(interval)


def run(times, gpu):
    # 定义参数列表
    seed_list = [1]  # 原始代码中的 seed 计算方式
    pr_list = [0.05, 0.1, 0.15, 0.2]
    imb_ratio_list = [5, 10]  # 添加imb_ratio可选值

    # 生成所有参数组合
    combos = list(itertools.product(seed_list, pr_list, imb_ratio_list))
    total = len(combos)

    # 如果超出组合总数则退出
    if times >= total:
        print("All experiments finished.")
        exit(0)

    # 选择当前参数组合
    seed, pr, imb_ratio = combos[times]

    # 构造命令
    cmd = (
        f"CUDA_VISIBLE_DEVICES={gpu} nohup python train_dualmatch.py "
        f"--dataset cifar100 --epochs 800 --batch-size 256 --lr 0.01 --wd 1e-3 --t 2 --save_ckpt "
        f"--imb_ratio {imb_ratio} --rho 0.2 --partial_rate {pr} --seed {seed} >> output.log &"
    )

    print(cmd)
    # os.system(cmd)
    return cmd


if __name__ == '__main__':
    narrow_setup(interval=70, m_need=6000, times=0, gpus=list(range(8)), max_tasks=2)
