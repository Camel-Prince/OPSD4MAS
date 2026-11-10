# OPSD4MAS FW4 部署与运行指南

这是独立发布仓库，默认运行论文的 FW4 方法。外部模型负责提取事实和标注关系；特权教师仍是当前角色策略。正式推理无需外部 API 或未来事实。完整主结果见[英文首页](../README.md)。

## 1. 安装环境

在 Linux x86_64、四张 80GB NVIDIA GPU 的机器上，从本仓库根目录逐块执行。不要套用其他 CoMAS/vLLM 环境的版本。

```bash
conda create -n opsd4mas python=3.10 -y
conda activate opsd4mas
python -m pip install --upgrade pip
python -m pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124
python -m pip install -r requirements.txt
python scripts/install_flash_attention.py
python -m pip install -e . --no-deps
python -m pip check
python scripts/check_environment.py
```

FlashAttention 安装器只寻找官方发布中与当前 Python、PyTorch、C++ ABI 匹配的 wheel。找不到就停止，不会悄悄开始长时间源码编译。这个环境组合依据保留的 SGLang 版本及其依赖约束整理，仍需在实际 GPU 主机验收；它不是完整恢复的历史环境快照。[环境说明](INSTALLATION.md)。

## 2. 下载基础模型

```bash
python -c "from huggingface_hub import snapshot_download; snapshot_download('Qwen/Qwen3-4B', local_dir='data/models/Qwen3-4B')"
```

默认模型目录在 `configs/fw4.env` 中设置。已有模型可以通过 `export MODEL_PATH=/你的模型目录` 指定。

## 3. 检查并准备数据

需要 DAPO 训练 Parquet，以及论文对应的六数据集原始 Parquet。六集目录布局为 `AMC23/test.parquet` 等。当前仓库不打包数据内容；原始数据的公共下载地址与再分发条件尚待补齐，不能把任意同名版本当作等价输入。

```bash
export TRAIN_SOURCE=/你的训练数据/dapo-math-17k.parquet
export BENCHMARK_SOURCE=/你的六集数据目录
python scripts/prepare_data.py --train-source "$TRAIN_SOURCE" --test-source "$BENCHMARK_SOURCE" --output data/prepared
```

脚本按源 ID 去重，核验 17,917 个训练问题。六个测试集题数分别为 AMC23 83、AIME24 30、AIME25 30、MATH-500 500、Olympiad-Bench 675、Minerva 272，并核对题目与答案的内容指纹。当前公开 `math-ai/amc23` 的 40 题版本不能替代论文的 83 题输入。

不额外指定验证集时，保留历史上 AMC23+AIME24+AIME25 的 reference 组合。**它与三个测试集重叠，不能称为独立验证集。** 新实验如需独立选择 checkpoint，加上 `--validation-source /你的独立验证集.parquet`，并记录与历史协议的差异。[数据说明](DATA.md)。

## 4. 配置自己的 API

```bash
cp .env.example .env.local
```

用编辑器填写 `.env.local` 中的 endpoint、模型名、密钥、TPM、RPM 和并发额度。它已被 Git 忽略。填好后执行：

```bash
set -a
source .env.local
set +a
```

这些额度是本任务的份额，不是整个账户额度。其他服务器或任务共用账户时必须预留份额。训练会使用付费 API，dry-run 不发请求。当前默认路径对 extraction 和 relation 的每次调用及重试共用进程内限流；不能保证其他进程也遵守同一份额度。[API 说明](API.md)。

## 5. 检查并正式训练

```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3
python scripts/check_environment.py --inputs --gpu
```

环境检查不加载完整模型。`--gpu` 会在四张空闲卡上做小型 CUDA 和 FlashAttention 算子测试。

```bash
env -u BASH_ENV bash scripts/train_fw4.sh configs/fw4.env --dry-run
```

确认打印出的参数正确后，正式前台启动：

```bash
env -u BASH_ENV bash scripts/train_fw4.sh configs/fw4.env
```

终端持续显示进度，日志同时保存在打印出的新结果目录。不会后台启动队列或 GPU filler。默认每 10 步保存 checkpoint 和事实产物，不自动轮换删除旧 checkpoint；需自行规划磁盘空间。若结果目录已存在会停止，不覆盖。

核心配置是每批 32 题、每题 8 条 MAS 轨迹、最多三轮 Solver–Verifier 交互、future window 4、每角色每 GPU actor micro 2、评分 micro 8；默认关闭 PowerOPD。两套策略共置在同一组四张卡上，不是每角色分两张卡。其他完整设置见 `configs/fw4.env` 与英文首页。

公开入口默认从头训练到 550 步上限、两个数据集 epoch。论文对应历史 run 从 step20 续训；公开 fresh 配置不承诺逐位复现当时训练轨迹或主表分数。完整环境、早期配置和主表 checkpoint 对应关系尚未全部恢复。

## 6. 从 checkpoint 继续训练

```bash
env -u BASH_ENV bash scripts/train_fw4.sh configs/fw4.env --resume-from /你的checkpoint/global_step_100
```

需要完整的两个角色、四 rank model/optim/extra_state 分片及 `data.pt`。不能把训练中尚未写完的目录当作完整 checkpoint。恢复结果写入新目录，源 checkpoint 保留。

## 7. 跑六个测试集

不需要 API 密钥。保留基础模型 tokenizer，指定完整四卡 FSDP checkpoint：

```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3
env -u BASH_ENV bash scripts/evaluate.sh --checkpoint /你的checkpoint/global_step_100 --datasets all --dry-run
```

正式运行：

```bash
env -u BASH_ENV bash scripts/evaluate.sh --checkpoint /你的checkpoint/global_step_100 --datasets all
```

只跑前三集：

```bash
env -u BASH_ENV bash scripts/evaluate.sh --checkpoint /你的checkpoint/global_step_100 --datasets AMC23 AIME24 AIME25
```

终端显示各集结果及最终汇总，每集保存原始 token、回答、轨迹判分和 `result.json`，根目录保存 `results.json` 与 `results.md`。p1 是每题四条完整轨迹的平均正确率，p4 是其中至少一条成功；同一轨迹中不同 agent 的回答不算四次采样。

仅重新统计已有判分：

```bash
python scripts/evaluation_records.py /你的结果/predictions/predictions.jsonl --questions 83
```

这条命令重算指标，不更换评分器。Minerva 继续保留原评分器的适用边界，统一离线判分需要另行处理。[评测说明](EVALUATION.md)。
