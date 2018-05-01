.PHONY: inline ui clean update resources test

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
	pyuic5 pymea/ui/PyMEAMainWindow.ui -o pymea/ui/main_window.py
	pyuic5 pymea/ui/MEAToolsMainWindow.ui -o pymea/ui/mea_tools_window.py

clean:
	-@rm -rf pymea/mea_cython.c pymea/mea_cython.so pymea/mea_cython.pyd 2>/dev/null 
	-@rm -rf dist
	-@rm -rf build

update:
	@echo "updating..."
	git pull
	$(PYTHON) setup.py build_ext $(DFLAGS) --inplace
	@echo "done."

resources:
	pyrcc5 -py3 pymea/rsc/resources.qrc -o pymea/rsc.py

test:
	python3 ~/mea-tools/mea-runner.py view ~/Dropbox/neuron/mea/pymea/2014-10-30_I9119_Stimulate_D3.h5

test-ui:
	python3 ~/mea-tools/mea_tools_runner.py
