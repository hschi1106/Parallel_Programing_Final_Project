NVCC      := nvcc
CXXFLAGS  := -O3 -std=c++17
TARGET    := gpg_parallel

SRCS_CPP  := main.cpp gpg_cpu.cpp
SRCS_CU   := gpg_cuda.cu

OBJS_CPP  := $(SRCS_CPP:.cpp=.o)
OBJS_CU   := $(SRCS_CU:.cu=.o)
OBJS      := $(OBJS_CPP) $(OBJS_CU)

INCLUDES  := -I.

all: $(TARGET)

$(TARGET): $(OBJS)
	$(NVCC) $(CXXFLAGS) $(INCLUDES) $^ -o $@

%.o: %.cpp
	$(NVCC) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

%.o: %.cu
	$(NVCC) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

clean:
	rm -f $(OBJS) $(TARGET)

.PHONY: all clean
