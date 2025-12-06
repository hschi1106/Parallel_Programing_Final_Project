CXX      = g++
CXXFLAGS = -O3 -std=c++17 -Wall -Wextra -march=native
TARGET   = gpg_serial
SRC      = main.cpp

all: $(TARGET)

$(TARGET): $(SRC)
	$(CXX) $(CXXFLAGS) $< -o $@

run: $(TARGET)
	./$(TARGET)

clean:
	rm -f $(TARGET)
