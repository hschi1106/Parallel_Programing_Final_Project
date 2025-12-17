# Build: make
# Override GPU arch: make ARCH=86

TARGET    := gpg_parallel
BUILD_DIR := build

CXX  := g++
NVCC := nvcc

ARCH ?= 89

INCLUDES  := -Iinclude -Isrc

CXXFLAGS  := -O3 -std=c++17 $(INCLUDES) -MMD -MP
NVCCFLAGS := -O3 -std=c++17 $(INCLUDES) -MMD -MP \
            -gencode arch=compute_$(ARCH),code=sm_$(ARCH) \
            -gencode arch=compute_$(ARCH),code=compute_$(ARCH)

SRCS_CPP := src/main.cpp src/gomea.cpp
SRCS_CU  := src/fitness_cuda.cu

OBJS_CPP := $(patsubst src/%.cpp,$(BUILD_DIR)/%.o,$(SRCS_CPP))
OBJS_CU  := $(patsubst src/%.cu,$(BUILD_DIR)/%.o,$(SRCS_CU))
OBJS     := $(OBJS_CPP) $(OBJS_CU)
DEPS     := $(OBJS:.o=.d)

all: $(TARGET)

$(TARGET): $(OBJS)
	$(NVCC) $(NVCCFLAGS) $^ -o $@

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

$(BUILD_DIR)/%.o: src/%.cpp | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(BUILD_DIR)/%.o: src/%.cu | $(BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

-include $(DEPS)

clean:
	rm -rf $(BUILD_DIR) $(TARGET)

.PHONY: all clean
