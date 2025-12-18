# Makefile
NVCC      := nvcc
CXX       := g++

# 針對平行歸約和較大的暫存器使用量，建議至少 sm_60 (Pascal) 以上
# -maxrregcount 可以限制暫存器用量避免 occupancy 驟降，但這邊先讓編譯器決定
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