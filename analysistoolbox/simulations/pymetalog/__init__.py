from .metalog import metalog
from .class_method import rmetalog, plot, qmetalog, pmetalog, dmetalog, summary, update

# Create a pymetalog module that can be imported
class pymetalog:
    def __init__(self, x=None, bounds=[0,1], boundedness='u', term_limit=13, term_lower_bound=2, step_len=0.01, probs=None, fit_method='any', penalty=None, alpha=0.0, new_data=None, n=1, generator='rand', term=2):
        self.x = x
        self.bounds = bounds
        self.boundedness = boundedness
        self.term_limit = term_limit
        self.term_lower_bound = term_lower_bound
        self.step_len = step_len
        self.probs = probs
        self.fit_method = fit_method
        self.penalty = penalty
        self.alpha = alpha
        self.new_data = new_data
        self.n = n
        self.generator = generator
        self.term = term
        self.my_metalog = None
    
    def metalog(self, x=None, bounds=None, boundedness=None, term_limit=None, term_lower_bound=None, step_len=None, probs=None, fit_method=None, penalty=None, alpha=None):
        """
        Fit a metalog distribution to the data
        """
        if self.my_metalog is None:
            # Use provided parameters or fall back to instance variables
            x_val = x if x is not None else self.x
            bounds_val = bounds if bounds is not None else self.bounds
            boundedness_val = boundedness if boundedness is not None else self.boundedness
            term_limit_val = term_limit if term_limit is not None else self.term_limit
            term_lower_bound_val = term_lower_bound if term_lower_bound is not None else self.term_lower_bound
            step_len_val = step_len if step_len is not None else self.step_len
            probs_val = probs if probs is not None else self.probs
            fit_method_val = fit_method if fit_method is not None else self.fit_method
            penalty_val = penalty if penalty is not None else self.penalty
            alpha_val = alpha if alpha is not None else self.alpha
            
            self.my_metalog = metalog(x_val, bounds_val, boundedness_val, term_limit_val, term_lower_bound_val, step_len_val, probs_val, fit_method_val, penalty_val, alpha_val)
        return self.my_metalog

    def rmetalog(self, m=None, n=1, term=2, generator='rand'):
        """
        Take random draws from the metalog distribution
        
        Parameters:
        -----------
        m : object, optional
            A metalog object. If None, uses the class's metalog object
        n : int, default=1
            Number of random draws to take
        term : int, default=2
            Term to use for the metalog distribution
        generator : str, default='rand'
            Random number generator to use
        """
        # Use the provided metalog object or the class's metalog object
        m = m if m is not None else self.metalog()
        return rmetalog(m, n=n, term=term, generator=generator)

    # def plot(self, m=None):
    #     """
    #     Plot the metalog distribution
        
    #     Parameters:
    #     -----------
    #     m : object, optional
    #         A metalog object. If None, uses the class's metalog object
    #     """
    #     m = m if m is not None else self.metalog()
    #     return plot(m)

    def qmetalog(self, m=None, y=None, term=2):
        """
        Calculate the quantile of the metalog distribution
        
        Parameters:
        -----------
        m : object, optional
            A metalog object. If None, uses the class's metalog object
        y : list or float
            Probability values to calculate quantiles for
        term : int, default=2
            Term to use for the metalog distribution
        """
        m = m if m is not None else self.metalog()
        return qmetalog(m, y=y, term=term)

    def pmetalog(self, m=None, q=None, term=2):
        """
        Calculate the probability of the metalog distribution
        
        Parameters:
        -----------
        m : object, optional
            A metalog object. If None, uses the class's metalog object
        q : list or float
            Quantile values to calculate probabilities for
        term : int, default=2
            Term to use for the metalog distribution
        """
        m = m if m is not None else self.metalog()
        return pmetalog(m, q=q, term=term)

    def dmetalog(self, m=None, q=None, term=2):
        """
        Calculate the density of the metalog distribution
        
        Parameters:
        -----------
        m : object, optional
            A metalog object. If None, uses the class's metalog object
        q : list or float
            Quantile values to calculate densities for
        term : int, default=2
            Term to use for the metalog distribution
        """
        m = m if m is not None else self.metalog()
        return dmetalog(m, q=q, term=term)

    def summary(self, m=None):
        """
        Return a summary of the metalog distribution
        
        Parameters:
        -----------
        m : object, optional
            A metalog object. If None, uses the class's metalog object
        """
        m = m if m is not None else self.metalog()
        return summary(m)

    # def update(self, m=None, new_data=None, penalty=None, alpha=0.0):
    #     """
    #     Update the metalog distribution with new data
        
    #     Parameters:
    #     -----------
    #     m : object, optional
    #         A metalog object. If None, uses the class's metalog object
    #     new_data : list or array
    #         New data to update the distribution with
    #     penalty : float or None, default=None
    #         Penalty parameter for the update
    #     alpha : float, default=0.0
    #         Alpha parameter for the update
    #     """
    #     m = m if m is not None else self.metalog()
    #     return update(m, new_data=new_data, penalty=penalty, alpha=alpha)
