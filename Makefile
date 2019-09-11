
docs:
	sphinx-apidoc -a -f -F -o docs . *.py tests
	$(MAKE) -C docs html
