# GPU-learning

6 个月 GPU 学习计划的代码仓库。所有 CUDA 代码都通过 apptainer 容器 (`cuSif`) 编译和运行。

## 环境

- 编译器: 容器内 `/usr/local/cuda/bin/nvcc` (CUDA 13.2)
- 容器镜像: `$cuSif` (定义在 `~/.bashrc`)
- GPU 架构: `sm_89` (L40), 如需修改改 `Makefile` 里的 `ARCH`
- `/export`、`/rapid` 已由站点默认挂载

## 用法

编译 (在容器内跑 nvcc, 产物输出到 `build/`):

```bash
make            # 编译所有 .cu
make clean      # 清理 build/
```

编译 + 运行 (推荐用 run.sh, 自动带 --nv GPU):

```bash
./run.sh                  # 编译全部并依次运行
./run.sh vecadd_2_3       # 只编译并运行 vecadd_2_3
./run.sh vecadd_2_3 arg1  # 运行时把 arg1 传给程序
```

可用环境变量 `cuSif` 覆盖容器镜像路径; 需额外绑定目录时用 `make BIND="--bind /some/path"`。

## Week 1

### 9.7
Finish the chapter 2m add kernel, refresh the memory ,running on one L40 gpu card
