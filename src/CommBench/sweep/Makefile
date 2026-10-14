export OMPI_CXX := /home/intel/dpcpp_nightly/bin/dpclang++

CXX = mpicxx
CXXFLAGS = -fsycl -fsycl-targets=spir64 -O2 -Ishim -I/home/intel/dpcpp_nightly/include/sycl -I/usr/include/level_zero
LDFLAGS = -fsycl -lze_loader

TARGET = sweep
OBJECTS = main.o

all: $(TARGET)

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(TARGET): $(OBJECTS)
	$(CXX) -o $@ $(OBJECTS) $(LDFLAGS)

clean:
	rm -f $(TARGET) *.o
