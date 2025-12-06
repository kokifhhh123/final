CXX = nvcc
CXXFLAGS = -std=c++17 -O3
LDFLAGS = -lpng -lz

SRC_DIR  = .
OBJ_DIR  = ../object
BIN_DIR  = ../bin

TARGET  = main
BIN     = $(BIN_DIR)/$(TARGET)

SRCS = $(wildcard $(SRC_DIR)/*.cu)
OBJS = $(patsubst $(SRC_DIR)/%.cu, $(OBJ_DIR)/%.o, $(SRCS))

all: $(BIN)

$(BIN): $(OBJS)
	@mkdir -p $(BIN_DIR)
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDFLAGS)

# Compile .cu -> .o
$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cu
	@mkdir -p $(OBJ_DIR)
	$(CXX) $(CXXFLAGS) -c $< -o $@

debug: CXXFLAGS += -G -DDEBUG
debug: clean all

clean:
	rm -rf $(OBJ_DIR) $(BIN)

.PHONY: all clean debug
