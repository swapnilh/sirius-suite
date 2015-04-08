OS := $(shell uname)
OPTIONS:= 

ifeq ($(OS),Darwin)
	OPTIONS += -framework OpenCL
else
	OPTIONS += -I/opt/AMDAPP/include/ -l OpenCL
endif

main: gmm_scoring.cpp
	g++ -Wall -g gmm_scoring.cpp -o gmm_scoring $(OPTIONS)

clean:
	rm -rf gmm_scoring
