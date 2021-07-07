INC=-I ../Tools/
CXXFLAGS=-std=c++11 -O3 -march=native
LIBFLAGS=-pthread -fopenmp

all: clean xreg_train xreg_predict xreg_metric

xreg_train:
	$(CXX) -o xreg_train $(CXXFLAGS) $(INC) xreg_train.cpp xreg.cpp ../Tools/*.cpp $(LIBFLAGS)

xreg_predict:
	$(CXX) -o xreg_predict $(CXXFLAGS) $(INC) xreg_predict.cpp xreg.cpp ../Tools/*.cpp $(LIBFLAGS)

clean:
	rm -f xreg_train xreg_predict xreg_metric

xreg_metric:
	$(CXX) -o xreg_metric $(CXXFLAGS) $(INC) xreg_metric.cpp $(LIBFLAGS)