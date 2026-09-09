# AGENTS.md

## 项目背景
这是一个 **6 个月 GPU 学习计划** 的代码仓库。用中文向用户解释所有内容。

## 环境 / 工具链
- **不能直接在宿主机跑 nvcc**：宿主机没有 CUDA。所有 CUDA 编译和运行都必须通过 apptainer 容器执行。
- 容器镜像通过环境变量 `cuSif` 指定（在自己的 shell 里 `export cuSif=/path/to/cuda.sif`，不入库）。
- 容器内 nvcc 路径: `/usr/local/cuda/bin/nvcc` (CUDA 13.2)，**不在默认 PATH，必须用全路径**。
- GPU 架构: `sm_89`。
- 运行 GPU 程序需要 `apptainer exec --nv`。

## 构建 / 运行命令
- 编译: `make` (产物输出到 `build/`)，清理: `make clean`
- 编译并运行: `./run.sh [目标名] [程序参数...]`
  - `./run.sh` 无参数 = 编译全部并依次运行
  - `./run.sh vecadd_2_3` = 只编译并运行该目标

## 约定
- 每个 `.cu` 源文件 = 一个可执行目标，产物统一放 `build/`。
- 修改 GPU 架构改 `Makefile` 的 `ARCH`；容器路径通过环境变量 `cuSif` 或 `make CUSIF=...` 指定。
