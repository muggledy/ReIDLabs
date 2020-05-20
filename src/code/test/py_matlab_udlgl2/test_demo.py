import matlab.engine
eng = matlab.engine.start_matlab() #https://stackoverflow.com/questions/46141631/running-matlab-using-python-gives-no-module-named-matlab-engine-error
eng.main_script(nargout=0)