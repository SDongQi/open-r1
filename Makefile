# 定义伪目标，这些目标不会创建实际文件
.PHONY: style quality

# 确保在脚本中测试本地检出的代码，而不是预先安装的代码（不要使用引号！）
export PYTHONPATH = src

# 定义需要检查的目录
check_dirs := src tests


# 开发依赖安装
install:
	# 创建Python 3.11虚拟环境并激活
	uv venv openr1 --python 3.11 && . openr1/bin/activate && uv pip install --upgrade pip
	# 安装vllm库
	uv pip install vllm==0.8.3
	# 安装setuptools
	uv pip install setuptools
	# 安装flash-attn（无构建隔离）
	uv pip install flash-attn --no-build-isolation
	# 安装当前项目（开发模式）
	GIT_LFS_SKIP_SMUDGE=1 uv pip install -e ".[dev]"

# 代码格式化
style:
	# 使用ruff格式化代码
	ruff format --line-length 119 --target-version py310 $(check_dirs) setup.py
	# 使用isort排序导入
	isort $(check_dirs) setup.py

# 代码质量检查
quality:
	# 使用ruff进行代码检查
	ruff check --line-length 119 --target-version py310 $(check_dirs) setup.py
	# 使用isort检查导入排序
	isort --check-only $(check_dirs) setup.py
	# 使用flake8进行代码检查
	flake8 --max-line-length 119 $(check_dirs) setup.py

# 运行常规测试
test:
	pytest -sv --ignore=tests/slow/ tests/

# 运行耗时测试
slow_test:
	pytest -sv -vv tests/slow/

# 评估部分

# 运行模型评估
evaluate:
	# 根据PARALLEL参数设置并行参数
	$(eval PARALLEL_ARGS := $(if $(PARALLEL),$(shell \
		if [ "$(PARALLEL)" = "data" ]; then \
			echo "data_parallel_size=$(NUM_GPUS)"; \
		elif [ "$(PARALLEL)" = "tensor" ]; then \
			echo "tensor_parallel_size=$(NUM_GPUS)"; \
		fi \
	),))
	# 如果使用tensor并行，设置VLLM工作进程多进程方法为spawn
	$(if $(filter tensor,$(PARALLEL)),export VLLM_WORKER_MULTIPROC_METHOD=spawn &&,) \
	# 设置模型参数
	MODEL_ARGS="pretrained=$(MODEL),dtype=bfloat16,$(PARALLEL_ARGS),max_model_length=32768,max_num_batched_tokens=32768,gpu_memory_utilization=0.8,generation_parameters={max_new_tokens:32768,temperature:0.6,top_p:0.95}" && \
	# 根据任务类型运行不同的评估
	if [ "$(TASK)" = "lcb" ]; then \
		lighteval vllm $$MODEL_ARGS "extended|lcb:codegeneration|0|0" \
			--use-chat-template \
			--output-dir data/evals/$(MODEL); \
	else \
		lighteval vllm $$MODEL_ARGS "lighteval|$(TASK)|0|0" \
			--use-chat-template \
			--output-dir data/evals/$(MODEL); \
	fi
