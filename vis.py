import matplotlib.pyplot as plt
import numpy as np

# 数据
threads = [1, 2, 4, 8, 16, 32, 64, 128]
times = [0.0215886, 0.0627011, 0.00786238, 0.00663097, 0.00717736, 0.0488605, 0.0100836, 0.0681726]

# 对数尺度
log_threads = np.log2(threads)
log_times = np.log10(times)

# 理想缩放
ideal_times = times[0] / np.array(threads)

# 绘图
plt.figure(figsize=(8, 6))
plt.plot(threads, times, 'o-', label='Actual Timing', color='gold')
plt.plot(threads, ideal_times, '--', label='Ideal Scaling', color='skyblue')

# 标记百分比
percentages = [100 * t / times[0] for t in times]
for i, (x, y, p) in enumerate(zip(threads, times, percentages)):
    plt.text(x, y, f'{int(p)}%', fontsize=10, ha='right' if i == 0 else 'left')

# 设置对数刻度
plt.xscale('log', base=2)
plt.yscale('log', base=10)

# 图表装饰
plt.xlabel('Number of Threads')
plt.ylabel('Time')
plt.title('(a)')
plt.legend()
plt.grid(True, which="both", ls="--")

# 显示图表
plt.show()
