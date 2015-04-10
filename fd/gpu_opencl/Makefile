CXX = g++
CXX_FLAGS = -O3 #-I//opt/opencv-2.4.11/modules/core/include/ -I//opt/opencv-2.4.11/include/ -I/opt/opencv-2.4.11/modules/features2d/include/ -I/opt/opencv-2.4.11/modules/flann/include/ -I/opt/opencv-2.4.11/ -I/opt/opencv-2.4.11/modules/nonfree/include/ -I/opt/opencv-2.4.11/modules/highgui/include/
LINK_FLAGS = -lopencv_core \
			 -lopencv_highgui \
			 -lopencv_imgproc \
			 -lopencv_nonfree \
			 -lopencv_ocl \
			 -lopencv_features2d

INPUT = ../input/2048x2048.jpg

# File names
EXEC = surf-fd
SOURCES = surf-fd.cpp ../../utils/timer.cpp
OBJECTS = $(SOURCES:.cpp=.o)

.PHONY: all test

all : $(EXEC)
$(EXEC): $(OBJECTS) Makefile
	$(CXX) $(OBJECTS) -o $(EXEC) $(LINK_FLAGS)

# To obtain object files
%.o: %.cpp Makefile
	$(CXX) -c $(CXX_FLAGS) $(EXTRA_FLAGS) $< -o $@

# test
test: CXX_FLAGS += -DTESTING
test: all
	@./$(EXEC) $(INPUT)

# To remove generated files
clean:
	$(RM) -f $(EXEC) *.o ../input/$(EXEC).gpu
