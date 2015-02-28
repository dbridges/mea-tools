inline:
	python3 setup.py build_ext --inplace

inlinew:
	python setup.py build_ext -DMS_WIN64 --inplace

ui:
	pyside-uic pymea/ui/PyMEAMainWindow.ui -o pymea/ui/main_window.py
