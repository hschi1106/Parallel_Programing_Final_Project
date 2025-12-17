# Makefile
NVCC      := nvcc
CXX       := g++

# 加入 -ccbin $(CXX) 明確指定 host compiler
# 根據你的錯誤訊息，你可能需要移除 -std=c++17 改用 -std=c++14，
# 因為舊版 CUDA (如 10.x 或 11.0) 有時不完全支援 c++17 作為 host compiler flag
# 但我們先試試看保留 c++17 並加上 -ccbin
NVCCFLAGS := -ccbin $(CXX) -O3 -std=c++17 -arch=sm_60 --use_fast_math -Xcompiler -Wall
CXXFLAGS  := -O3 -std=c++17 -Wall
TARGET    := gpg_parallel

SRCS_CPP  := main.cpp gpg_cpu.cpp
SRCS_CU   := gpg_cuda.cu

OBJS_CPP  := $(SRCS_CPP:.cpp=.o)
OBJS_CU   := $(SRCS_CU:.cu=.o)
OBJS      := $(OBJS_CPP) $(OBJS_CU)

INCLUDES  := -I.

all: $(TARGET)

$(TARGET): $(OBJS)
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) $^ -o $@

%.o: %.cpp
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

%.o: %.cu
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) -c $< -o $@

clean:
	rm -f $(OBJS) $(TARGET)

.PHONY: all clean