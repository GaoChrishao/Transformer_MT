import math


def adjust_learning_rate(d_model ,step, optimizer, warmup_steps=1000 * 8):
    lr = 1 / math.sqrt(d_model) * min(math.pow(step, -0.5), step * math.pow(warmup_steps, -1.5))
    for param_group in optimizer.param_groups:
        # 在每次更新参数前迭代更改学习率
        param_group["lr"] = lr



