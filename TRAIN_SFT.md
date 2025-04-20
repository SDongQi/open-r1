# SFT（监督微调）训练流程详解

本文档详细解释了以下SFT（Supervised Fine-Tuning，监督微调）命令的执行流程：

```bash
ACCELERATE_LOG_LEVEL=info accelerate launch --config_file recipes/accelerate_configs/zero3.yaml \
    src/open_r1/sft.py \
    --config recipes/Qwen2.5-1.5B-Instruct/sft/config_demo.yaml
```

## 1. 命令解析

### 环境变量设置
- `ACCELERATE_LOG_LEVEL=info`：设置Accelerate库的日志级别为info，便于查看训练过程中的详细信息。

### 分布式训练配置
- `accelerate launch`：使用Hugging Face的Accelerate库启动分布式训练。
- `--config_file recipes/accelerate_configs/zero3.yaml`：指定分布式训练配置文件，采用DeepSpeed ZeRO-3优化策略。

### 训练脚本与配置
- `src/open_r1/sft.py`：指定SFT训练脚本路径。
- `--config recipes/Qwen2.5-1.5B-Instruct/sft/config_demo.yaml`：指定训练配置文件，包含模型、数据集和训练参数信息。

## 2. DeepSpeed ZeRO-3配置解析

DeepSpeed ZeRO-3 是微软开发的深度学习优化库 DeepSpeed 中的核心技术，属于其​​零冗余优化器（ZeRO）​​的第三阶段优化，旨在通过显存分片和通信优化实现超大规模模型的高效训练。
1、​​全分片存储机制​​
    ZeRO-3 在 ZeRO-1（优化器状态分片）和 ZeRO-2（梯度分片）的基础上，进一步​​将模型参数分片存储到不同 GPU 上。
2、动态通信调度
3、​​显存效率提升

`zero3.yaml`文件定义了以下配置：

```yaml
compute_environment: LOCAL_MACHINE    # 计算环境设置为本地机器
debug: false                          # 是否启用调试模式，设置为否
deepspeed_config:                     # DeepSpeed配置部分
  deepspeed_multinode_launcher: standard  # 多节点启动器类型，使用标准模式
  offload_optimizer_device: none      # 优化器状态不卸载到CPU或NVMe
  offload_param_device: none          # 模型参数不卸载到CPU或NVMe
  zero3_init_flag: true               # 启用ZeRO-3初始化标志
  zero3_save_16bit_model: true        # 保存模型时使用16位精度
  zero_stage: 3                       # 使用ZeRO优化的第3阶段（完全分片）
distributed_type: DEEPSPEED           # 分布式类型设置为DeepSpeed
downcast_bf16: 'no'                   # 不将FP32降级为BF16
machine_rank: 0                       # 当前机器在多机集群中的排名
main_training_function: main          # 主训练函数名称
mixed_precision: bf16                 # 使用bfloat16混合精度训练
num_machines: 1                       # 使用的机器数量，这里为单机
num_processes: 8                      # 总进程数，通常对应GPU数量
rdzv_backend: static                  # 集合点后端设置为静态
same_network: true                    # 所有节点在同一网络
tpu_env: []                           # TPU环境设置（未使用）
tpu_use_cluster: false                # 不使用TPU集群
tpu_use_sudo: false                   # 不使用sudo运行TPU命令
use_cpu: false                        # 不使用CPU进行训练
```

主要特点：
- 使用DeepSpeed分布式训练框架
- 采用ZeRO Stage-3优化，可以显著减少GPU内存使用
- 使用bf16混合精度训练
- 配置为8个GPU进程（通常为单节点8卡）
- 启用模型参数分片存储

## 3. 模型与数据集配置

`config_demo.yaml`文件定义：

```yaml
# 模型参数
model_name_or_path: Qwen/Qwen2.5-1.5B-Instruct   # 预训练模型的名称或路径，使用千问2.5系列1.5B指令模型
model_revision: main                             # 模型版本，使用主分支
torch_dtype: bfloat16                            # PyTorch数据类型，使用bfloat16
attn_implementation: flash_attention_2           # 注意力机制实现，使用FlashAttention 2优化性能

# 数据训练参数
dataset_name: open-r1/OpenR1-Math-220k           # 训练数据集名称，来自HuggingFace Hub
dataset_num_proc: 48                             # 数据处理并行数，使用48个进程
```

- 使用的基础模型：Qwen2.5-1.5B-Instruct（千问2.5系列1.5B参数量指令模型）
- 注意力实现：使用FlashAttention 2优化性能
- 训练数据集：open-r1/OpenR1-Math-220k（数学推理数据集）
- 数据处理并行数：48线程

## 4. 训练配置详解

```yaml
# SFT训练器配置
bf16: true                                # 使用bfloat16混合精度训练
do_eval: false                            # 是否进行评估，这里不进行
eval_strategy: 'no'                       # 评估策略设置为不评估（若评估可以实现早停机制，但进行评估会消耗更多资源）
gradient_accumulation_steps: 1            # 梯度累积步数，每1步更新一次参数
gradient_checkpointing: true              # 启用梯度检查点以节省显存
gradient_checkpointing_kwargs:            # 梯度检查点的额外参数
  use_reentrant: false                    # 不使用可重入方式，提高稳定性
hub_model_id: Qwen2.5-1.5B-Open-R1-Distill  # HuggingFace Hub上的模型ID
hub_strategy: every_save                  # Hub上传策略，每次保存时上传
learning_rate: 5.0e-05                    # 学习率设置为5e-5
log_level: info                           # 日志级别设置为info
logging_steps: 5                          # 每5步记录一次日志
logging_strategy: steps                   # 按步数记录日志
lr_scheduler_type: cosine_with_min_lr     # 学习率调度器类型，使用带最小学习率的余弦退火
lr_scheduler_kwargs:                      # 学习率调度器额外参数
  min_lr_rate: 0.1                        # 最小学习率为初始学习率的10%
packing: false                            # 是否启用序列打包，这里不启用
max_length: 16384                         # 最大序列长度为16384个token
max_steps: -1                             # 最大训练步数，-1表示由epoch决定（完成所有epoch才停止）
num_train_epochs: 1                       # 训练轮数为1个epoch
output_dir: data/Qwen2.5-1.5B-Open-R1-Distill  # 输出目录
overwrite_output_dir: true                # 如果输出目录已存在则覆盖
per_device_eval_batch_size: 16            # 每个设备的评估批量大小
per_device_train_batch_size: 16           # 每个设备的训练批量大小
push_to_hub: true                         # 是否将模型推送到HuggingFace Hub
report_to:                                # 向哪些平台报告训练过程
- wandb                                   # 使用Weights & Biases跟踪训练
save_strategy: "steps"                    # 按步数保存模型
save_steps: 100                           # 每100步保存一次
save_total_limit: 1                       # 最多保存1个检查点（节省磁盘空间）
seed: 42                                  # 随机种子设置为42
use_liger_kernel: true                    # 使用Liger优化CUDA内核
warmup_ratio: 0.05                        # 学习率预热比例为5%
```

训练超参数解析：
- 使用bf16混合精度训练
- 使用梯度检查点（gradient checkpointing）节省显存
- 学习率：5.0e-05，使用余弦退火调度器
- 最大序列长度：16384 tokens
- 训练1个epoch
- 每个设备的批量大小：16
- 开启liger kernel优化内核
- 预热比例：5%
- 训练结果将推送到Hugging Face Hub
- 使用Weights & Biases（wandb）跟踪训练过程

## 5. SFT训练流程解析

基于`src/open_r1/sft.py`脚本，训练流程如下：

### 5.1 初始化阶段
1. 解析命令行参数和配置文件
2. 设置随机种子确保可复现性
3. 初始化日志系统
4. 检查是否存在检查点以便从断点恢复训练

### 5.2 数据准备阶段
1. 使用Hugging Face Datasets库加载指定数据集
2. 加载并配置分词器（tokenizer）
3. 处理训练和评估数据集

### 5.3 模型准备阶段
1. 加载预训练模型
2. 配置模型参数，如注意力机制、数据类型等
3. 应用PEFT（Parameter-Efficient Fine-Tuning）配置（如果有）

### 5.4 训练器初始化
```python
trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset[script_args.dataset_train_split],
    eval_dataset=dataset[script_args.dataset_test_split] if training_args.eval_strategy != "no" else None,
    processing_class=tokenizer,
    peft_config=get_peft_config(model_args),
    callbacks=get_callbacks(training_args, model_args),
)
```

使用TRL库的SFTTrainer进行训练，它是Transformer Reinforcement Learning的简写，专门为LLM训练设计。

### 5.5 训练循环
1. 检查并加载检查点（如果存在）
2. 执行训练循环
   ```python
   train_result = trainer.train(resume_from_checkpoint=checkpoint)
   ```
3. 记录训练指标
4. 保存训练状态

### 5.6 模型保存和评估
1. 保存最终模型和训练状态
   ```python
   trainer.save_model(training_args.output_dir)
   ```
2. 创建模型卡片（只通过主进程保存，避免冲突）
3. 恢复KV缓存以加速推理（训练时不使用kv缓存，因为所有时间步的QKV同时计算，无需逐token生成。而现在要推理了，开始使用kv缓存）
4. 可选：执行最终评估
5. 如果配置了push_to_hub，将模型推送到Hugging Face Hub

## 6. 模型优化技术

该训练流程中使用了多种优化技术：

1. **DeepSpeed ZeRO-3**：优化GPU内存使用，实现更大模型的训练
2. **混合精度训练**：使用bf16减少内存使用并加速训练
3. **梯度检查点**：通过重计算中间激活值减少内存使用
4. **FlashAttention 2**：优化注意力计算性能
5. **梯度累积**：可以通过增加gradient_accumulation_steps来模拟更大的批处理大小
6. **Liger内核**：针对Transformer模型的优化CUDA内核实现

## 7. 适用场景与注意事项

该SFT训练流程特别适合：
- 基于高质量数据集对预训练模型进行指令微调
- 在数学推理等特定领域增强模型能力
- 利用分布式训练提高大型语言模型训练效率

注意事项：
- 对于不同的基础模型，可能需要调整chat template和EOS token
- 需要根据实际GPU数量和大小调整batch size和梯度累积步数
- 训练过程中应密切监控损失值和学习率变化

## 8. 总结

此SFT训练命令执行了一个完整的监督微调流程，使用DeepSpeed ZeRO-3优化内存使用，在open-r1/OpenR1-Math-220k数据集上微调Qwen2.5-1.5B-Instruct模型。整个流程结合了多种优化技术，可以高效地完成大型语言模型的特定领域微调。 