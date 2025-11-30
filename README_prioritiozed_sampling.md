# 优先采样（Prioritized Sampling）功能说明

## 概述

本实现为 PPO 算法添加了优先采样（Prioritized Sampling）功能，允许在训练过程中根据不同的策略有选择地抽取 transition 样本，而不是完全随机采样。这可以帮助模型更关注重要的或困难的样本，从而提升训练效率。

## 功能特性

### 1. 多种采样策略

实现了三种基于不同指标的优先采样策略：

- **`'random'`** (默认): 完全随机采样，保持原有行为
- **`'advantage'`**: 基于 advantages 的加权采样
- **`'reward'`**: 基于 rewards 的加权采样  
- **`'delta'`**: 基于 TD 误差（deltas）的加权采样

### 2. 逆权重采样机制

当前实现使用**逆权重采样**（Inverse Weight Sampling），这意味着：
- 对于高权重的样本（如高 advantage、高 reward、高 delta），其**逆权重较低**
- 采样时**优先选择逆权重高的样本**，即优先选择权重低的样本
- 这种设计有助于关注"困难"样本或"价值估计误差大"的样本，可能有助于提升模型的鲁棒性

### 3. 灵活的配置选项

- **`reshuffle_per_epoch`**: 控制是否在每个 epoch 开始时重新打乱索引
- **向后兼容**: 默认行为与原始实现完全一致

## 实现细节

### 数据结构

在 `RolloutStorage` 中添加了 `deltas` 字段来存储 TD 误差：

```python
self.deltas = torch.zeros(
    num_transitions_per_env, num_envs, 1, device=self.device
)
```

### TD 误差计算

在 `compute_returns` 方法中计算并存储 TD 误差：

```python
delta = (
    self.rewards[step]
    + next_is_not_terminal * gamma * next_values
    - self.values[step]
)
self.deltas[step] = delta
```

### 权重计算

不同策略的权重计算方式：

1. **Advantage 策略**:
   ```python
   advantages_flat = advantages.squeeze(-1)
   weights = torch.softmax(advantages_flat / (advantages_flat.std() + 1e-8), dim=0)
   ```

2. **Reward 策略**:
   ```python
   rewards_flat = self.rewards.flatten(0, 1).squeeze(-1)
   rewards_min = rewards_flat.min()
   rewards_shifted = rewards_flat - rewards_min + 1e-8
   weights = rewards_shifted / rewards_shifted.sum()
   ```

3. **Delta 策略**:
   ```python
   deltas_flat = self.deltas.flatten(0, 1).squeeze(-1)
   deltas_min = deltas_flat.min()
   deltas_shifted = deltas_flat - deltas_min + 1e-8
   weights = deltas_shifted / deltas_shifted.sum()
   ```

### 逆权重采样

使用逆权重进行采样：

```python
inv_weights = (weights.max() - weights).clamp_min(1e-8)
inv_weights = inv_weights / inv_weights.sum()
indices = torch.multinomial(inv_weights, num_samples=num_samples, replacement=True)
```

## 使用方法

### 在 PPO 中启用优先采样

在 `ppo.py` 的 `update` 方法中，可以指定采样策略：

```python
generator = self.storage.mini_batch_generator(
    self.num_mini_batches, 
    self.num_learning_epochs, 
    sampling_strategy='advantage'  # 或 'reward', 'delta'
)
```

### 配置参数

```python
def mini_batch_generator(
    self, 
    num_mini_batches: int, 
    num_epochs: int,
    sampling_strategy: Optional[str] = None,  # 'random', 'advantage', 'reward', 'delta'
    reshuffle_per_epoch: bool = False,  # 是否每个 epoch 重新打乱
):
```

## 实验建议

### 1. 策略选择

- **`'advantage'`**: 适合关注高优势样本，可能有助于快速学习好的策略
- **`'reward'`**: 适合关注高奖励样本，有助于学习成功经验
- **`'delta'`**: 适合关注价值估计误差大的样本，有助于提升价值函数估计的准确性

### 2. 逆权重 vs 正权重

当前实现使用逆权重采样，优先选择权重低的样本。如果需要优先选择权重高的样本，可以修改代码：

```python
# 改为正权重采样（优先选择高权重样本）
indices = torch.multinomial(
    weights, 
    num_samples=num_samples, 
    replacement=True
)
```

### 3. 混合策略

可以考虑实现混合策略，例如：
- 一定比例使用优先采样，一定比例使用随机采样
- 根据训练阶段动态调整采样策略

### 4. 性能监控

建议监控以下指标来评估优先采样的效果：
- 训练收敛速度
- 最终性能
- 样本利用率
- 价值函数估计误差

## 注意事项

1. **内存开销**: 添加了 `deltas` 字段，略微增加内存使用
2. **计算开销**: 加权采样需要额外的权重计算和采样操作，但开销较小
3. **采样偏差**: 优先采样会引入偏差，需要根据具体任务评估影响
4. **超参数敏感性**: 不同采样策略可能对超参数（如 learning rate）的敏感性不同

## 代码位置

- 主要实现: `mani-centric-wbc/legged_gym/rsl_rl/storage/rollout_storage.py`
- 使用位置: `mani-centric-wbc/legged_gym/rsl_rl/algorithms/ppo.py`

## 使用训练好的模型进行测试

### 实验文件夹命名规则

在 `mani-centric-wbc/wandb/` 目录下，有多个实验文件夹，命名规则如下：

- **`a-p+_r=f`**: advantage 策略 + 逆权重（inverse） + reshuffle=False
- **`a+p+_r=t`**: advantage 策略 + 正权重（positive） + reshuffle=True
- **`a+p+_r=f`**: advantage 策略 + 正权重 + reshuffle=False
- **`a-p+_r=t`**: advantage 策略 + 逆权重 + reshuffle=True
- **`d-p+_r=f`**: delta 策略 + 逆权重 + reshuffle=False
- **`d+p+_r=t`**: delta 策略 + 正权重 + reshuffle=True
- **`d+p+_r=f`**: delta 策略 + 正权重 + reshuffle=False
- **`d-p+_r=t`**: delta 策略 + 逆权重 + reshuffle=True
- **`random`**: 随机采样（baseline）

每个实验文件夹的 `files/` 目录中包含：
- `model_2000.pt`: 训练 2000 次迭代后的模型权重
- `config.pkl`: 训练时使用的配置文件（用于恢复环境设置）

### 使用 play.py 进行可视化测试

`play.py` 脚本可以加载训练好的模型并进行可视化测试或录制视频。

#### 基本用法

```bash
cd mani-centric-wbc
python scripts/play.py \
    --ckpt_path wandb/a+p+_r=t/files/model_2000.pt \
    --trajectory_file_path data/tossing.pkl \
    --visualize \
    --num_steps 1000
```

#### 参数说明

- `--ckpt_path`: 模型文件路径（必需）
- `--trajectory_file_path`: 轨迹文件路径（必需），用于定义任务轨迹
- `--visualize`: 启用可视化（可选）
- `--record_video`: 录制视频（可选）
- `--device`: 设备，默认为 `cuda:0`
- `--num_envs`: 环境数量，默认为 1（可视化时强制为 1）
- `--num_steps`: 运行步数，默认为 1000，使用 `-1` 表示无限运行

#### 示例：测试不同采样策略的模型

```bash
# 测试 advantage + 正权重 + reshuffle=True 的模型
python scripts/play.py \
    --ckpt_path wandb/a+p+_r=t/files/model_2000.pt \
    --trajectory_file_path data/tossing.pkl \
    --visualize \
    --record_video

# 测试 delta + 逆权重 + reshuffle=False 的模型
python scripts/play.py \
    --ckpt_path wandb/d-p+_r=f/files/model_2000.pt \
    --trajectory_file_path data/tossing.pkl \
    --visualize

# 测试随机采样 baseline
python scripts/play.py \
    --ckpt_path wandb/random/files/model_2000.pt \
    --trajectory_file_path data/tossing.pkl \
    --visualize
```

### 使用 evaluate.py 进行批量评估

`evaluate.py` 脚本可以进行批量评估并生成统计信息。

#### 基本用法

1. 修改配置文件 `config/eval.yaml`，设置模型路径：

```yaml
ckpt_path: wandb/a+p+_r=t/files/model_2000.pt
```

2. 运行评估：

```bash
cd mani-centric-wbc
python scripts/evaluate.py
```

#### 在配置文件中指定模型路径

也可以通过命令行参数覆盖配置：

```bash
python scripts/evaluate.py ckpt_path=wandb/a+p+_r=t/files/model_2000.pt
```

### 模型文件位置

所有训练好的模型文件位于 `mani-centric-wbc/wandb/` 目录下：

```
wandb/
├── a-p+_r=f/files/model_2000.pt      # advantage + 逆权重 + reshuffle=False
├── a+p+_r=t/files/model_2000.pt      # advantage + 正权重 + reshuffle=True
├── a+p+_r=f/files/model_2000.pt      # advantage + 正权重 + reshuffle=False
├── a-p+_r=t/files/model_2000.pt      # advantage + 逆权重 + reshuffle=True
├── d-p+_r=f/files/model_2000.pt      # delta + 逆权重 + reshuffle=False
├── d+p+_r=t/files/model_2000.pt      # delta + 正权重 + reshuffle=True
├── d+p+_r=f/files/model_2000.pt      # delta + 正权重 + reshuffle=False
├── d-p+_r=t/files/model_2000.pt      # delta + 逆权重 + reshuffle=True
└── random/files/model_2000.pt        # 随机采样 baseline
```

### 注意事项

1. **配置文件**: `play.py` 会自动从模型文件同目录下的 `config.pkl` 加载训练配置，确保环境设置一致。

2. **轨迹文件**: 需要提供相应的轨迹文件（如 `data/tossing.pkl`、`data/pushing.pkl` 等）来定义任务。

3. **设备兼容性**: 如果模型在不同设备上训练，使用 `--device` 参数指定正确的设备。

4. **模型迭代次数**: 除了 `model_2000.pt`，某些实验文件夹可能还包含其他迭代次数的模型（如 `model_500.pt`、`model_1000.pt`、`model_1500.pt`），可以根据需要选择。

5. **输出文件**: 
   - `play.py` 会在 wandb 运行目录下生成 `logs.zarr` 和 `logs.pkl` 文件，包含状态、动作和任务日志
   - 如果启用 `--record_video`，会生成 `video.mp4` 文件



