all:
	nvcc SDH.cu -o SDH
commit:
	git add -A .
	git commit -m "project 2 2nd attempt.. step 1: get the tiling algorithm to work"
	git push origin
pull:
	git pull origin
run:
	./SDH 1000 500.0
runtofile:
	./SDH 1000 500.0 > output.txt
clean:
	rm -rf SDH
