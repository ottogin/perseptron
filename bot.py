import pyximport, numpy
pyximport.install(setup_args={"include_dirs":numpy.get_include()}, reload_support=True)
import cbot
cbot.run()
