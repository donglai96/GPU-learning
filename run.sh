#!/usr/bin/env bash
# 通过 apptainer 容器 (cuSif) 编译并运行 CUDA 程序
#
# 用法:
#   ./run.sh                 # 编译所有 .cu 并运行全部生成的程序
#   ./run.sh vecadd_2_3      # 只编译并运行 vecadd_2_3
#   ./run.sh vecadd_2_3 arg1 # 编译并运行, 把 arg1 传给程序
#
set -euo pipefail

# 容器镜像 (与 ~/.bashrc 中的 cuSif 保持一致), 可用环境变量覆盖
CUSIF="${cuSif:-/export/Public/donglma/container/Alma10-CUDA-13.2-Algo-Dev-1.4.0.sif}"

RUN_IN=(apptainer exec --nv "$CUSIF")

cd "$(dirname "$0")"

if [[ $# -eq 0 ]]; then
    # 无参数: 编译全部, 依次运行
    make CUSIF="$CUSIF"
    for bin in build/*; do
        [[ -x "$bin" ]] || continue
        echo "==================== run $bin ===================="
        "${RUN_IN[@]}" "./$bin"
    done
else
    # 第一个参数是目标名, 其余参数传给程序
    target="$1"; shift
    make "build/$target" CUSIF="$CUSIF"
    echo "==================== run build/$target ===================="
    "${RUN_IN[@]}" "./build/$target" "$@"
fi
