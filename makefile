all:
	nvcc SDH.cu -o SDH
commit:
	git add -A .
	git commit -m "okay... we are restarting now... I have no idea what happened there"
	git push origin
pull:
	git pull origin
run:
	./SDH 10000 500.0
runtofile:
	./SDH 10000 500.0 > output.txt
clean:
	rm -rf SDH
