.PHONY: inline ui clean update

OS := $(shell uname)

ifneq (,$(findstring NT-,$(OS)))
DFLAGS=-DMS_WIN64
PYTHON=python
else
PYTHON=python3
endif

inline:
	$(PYTHON) setup.py build_ext $(DFLAGS) --inplace

ui:
	pyuic4 pymea/ui/PyMEAMainWindow.ui -o pymea/ui/main_window.py

clean:
	-@rm -rf pymea/mea_cython.c pymea/mea_cython.so pymea/mea_cython.pyd 2>/dev/null 
	-@rm -rf dist
	-@rm -rf build

update:
	@echo "updating..."
	git pull
	$(PYTHON) setup.py build_ext $(DFLAGS) --inplace
	@echo "done."
