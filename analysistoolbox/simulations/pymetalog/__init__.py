from .metalog import metalog
from .class_method import rmetalog, plot, qmetalog, pmetalog, dmetalog, summary, update

# Create a pymetalog module that can be imported
class pymetalog:
    def __init__(self, x, bounds=[0,1], boundedness='u', term_limit=13, term_lower_bound=2, step_len=.01, probs=None, fit_method='any', penalty=None, alpha=0.):
        self.metalog = metalog(x, bounds, boundedness, term_limit, term_lower_bound, step_len, probs, fit_method, penalty, alpha)
        self.rmetalog = rmetalog(self.metalog)
        self.plot = plot(self.metalog)
        self.qmetalog = qmetalog(self.metalog)
        self.pmetalog = pmetalog(self.metalog)
        self.dmetalog = dmetalog(self.metalog)
        self.summary = summary(self.metalog)
        self.update = update(self.metalog)

