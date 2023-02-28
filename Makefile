# Makefile, Justin Thoreson

CPPFLAGS = -std=c++20 -Wall -Werror -pedantic -ggdb
VGFLAGS = --suppressions=valgrind.supp --leak-check=full --show-leak-kinds=all
PROGRAMS = MNISTKMeansParallel MNISTKMeansSequential

all : $(PROGRAMS)

MNISTImage.o : MNISTImage.cpp MNISTImage.h
	mpic++ $(CPPFLAGS) $< -c -o $@

MNISTKMeansSequential.o : MNISTKMeansSequential.cpp MNISTKMeans.h MNISTImage.h
	mpic++ $(CPPFLAGS) $< -c -o $@

MNISTKMeansSequential : MNISTKMeansSequential.o MNISTImage.o 
	mpic++ $(CPPFLAGS) $^ -o $@

mnistseq : MNISTKMeansSequential
	./$<

MNISTKMeansParallel.o : MNISTKMeansParallel.cpp MNISTKMeansMPI.h MNISTImage.h
	mpic++ $(CPPFLAGS) $< -c -o $@

MNISTKMeansParallel : MNISTKMeansParallel.o MNISTImage.o
	mpic++ $(CPPFLAGS) $^ -o $@

mnist% : MNISTKMeansParallel
	mpirun -n $* ./$<

valgrind : MNISTKMeansParallel
	mpirun -n 2 valgrind $(VGFLAGS) ./$<

clean :
	rm -f $(PROGRAMS) *.o *.html
