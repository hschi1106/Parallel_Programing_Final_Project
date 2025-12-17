NVCC       := nvcc
CXX        := g++

TARGET     := gpg_parallel

INCLUDES   := -I.

CXXFLAGS   := -O3 -std=c++17 $(INCLUDES)
NVCCFLAGS  := -O3 -std=c++17 $(INCLUDES) \
              -gencode arch=compute_89,code=sm_89 \
              -gencode arch=compute_89,code=compute_89

SRCS_CPP   := main.cpp gpg_cpu.cpp
SRCS_CU    := gpg_cuda.cu

OBJS_CPP   := $(SRCS_CPP:.cpp=.o)
OBJS_CU    := $(SRCS_CU:.cu=.o)
OBJS       := $(OBJS_CPP) $(OBJS_CU)

all: $(TARGET)

$(TARGET): $(OBJS)
	$(NVCC) $(NVCCFLAGS) $^ -o $@

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

%.o: %.cu
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

clean:
	rm -f $(OBJS) $(TARGET)

.PHONY: all clean
