all:
	nvcc SDH.cu -o SDH
commit:
	git add -A .
	git commit -m "project 2 3rd attempt.. get the histogram priv to be accurate"
	git push origin
pull:
	git pull origin
run:
	./SDH 1000 500.0
runtofile:
	./SDH 1000 500.0 > output.txt
clean:
	rm -rf SDH
