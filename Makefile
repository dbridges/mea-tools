inline:
	python3 setup.py build_ext --inplace

ui:
	pyside-uic pymea/ui/PyMEAMainWindow.ui -o pymea/ui/main_window.py
