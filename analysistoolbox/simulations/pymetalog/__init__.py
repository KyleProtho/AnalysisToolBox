from .metalog import metalog
from .class_method import rmetalog, plot, qmetalog, pmetalog, dmetalog, summary, update

# Create a pymetalog module that can be imported
class pymetalog:
    def __init__(self):
        self.metalog = metalog
        self.rmetalog = rmetalog
        self.plot = plot
        self.qmetalog = qmetalog
        self.pmetalog = pmetalog
        self.dmetalog = dmetalog
        self.summary = summary
        self.update = update

