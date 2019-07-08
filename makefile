all:
	nvcc proj3/proj3.cu -o proj3/proj3
proj1:
	nvcc SDH.cu -o SDH
commit:
	git add -A .
	git commit -m "project3 1st attempt"
	git push origin
pull:
	git pull origin
run:
	./proj3/proj3 10 4
run-proj1:
	./SDH 1000 500.0
runtofile-proj1:
	./SDH 1000 500.0 > output.txt
clean:
	rm -rf SDH
