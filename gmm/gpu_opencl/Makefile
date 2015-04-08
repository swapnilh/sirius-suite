CXX = g++
CXX_FLAGS = -Wno-deprecated-declarations -Wall -g
LINK_FLAGS += -I/opt/AMDAPP/include/ -l OpenCL

INPUT = ../input/gmm_data.txt

# File names
EXEC = gmm_scoring
SOURCES = gmm_scoring.cpp
OBJECTS = $(SOURCES:.cpp=.o)

.PHONY: all test

all : $(EXEC)
$(EXEC): $(OBJECTS) Makefile
	$(CXX) $(CXX_FLAGS) $(OBJECTS) -o $(EXEC) $(LINK_FLAGS)

# To obtain object files
%.o: %.cpp Makefile
	$(CXX) -c $(CXX_FLAGS) $< -o $@ $(LINK_FLAGS)

# test
test: CXX_FLAGS += -DTESTING
test: all
	@./$(EXEC) $(INPUT)
	-@diff -q ../input/$(EXEC).baseline ../input/$(EXEC).gpu

# To remove generated files
clean:
	$(RM) -f $(EXEC) *.o ../input/$(EXEC).gpu
