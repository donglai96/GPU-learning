# GPU 学习项目 Makefile
# 所有 nvcc 编译都通过 apptainer 容器 (cuSif) 执行

# 容器镜像路径, 从环境变量 cuSif 读取 (在自己的 shell 里设置, 不入库)
# 例如: export cuSif=/path/to/cuda.sif
CUSIF ?= $(cuSif)
# 如需额外绑定目录, 设置: make BIND="--bind /foo"
BIND ?=

# 在容器内执行命令的封装 (--nv 开启 GPU)
RUN_IN = apptainer exec --nv $(BIND) "$(CUSIF)"

# 容器内 nvcc 不在默认 PATH, 使用全路径
NVCC  = $(RUN_IN) /usr/local/cuda/bin/nvcc
ARCH  = sm_89
FLAGS = -O3 -std=c++17 -arch=$(ARCH) -lineinfo

SRCS = $(wildcard *.cu)
BINS = $(patsubst %.cu,build/%,$(SRCS))

all: $(BINS)

build/%: %.cu | build
	$(NVCC) $(FLAGS) -o $@ $<

build:
	mkdir -p build

clean:
	rm -rf build

.PHONY: all clean
