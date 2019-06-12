all:
	nvcc SDH.cu -o SDH
commit:
	git add -A .
	git commit -m "working on project 2 code now.. step 0: make sure we have data working in SOA format"
	git push origin
pull:
	git pull origin
run:
	./SDH 10000 500.0
runtofile:
	./SDH 10000 500.0 > output.txt
clean:
	rm -rf SDH
