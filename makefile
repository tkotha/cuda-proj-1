all:
	nvcc SDH.cu -o SDH
run:
	./SDH 10000 500.0
runtofile:
	./SDH 10000 500.0 > output.txt
clean:
	rm -rf SDH
