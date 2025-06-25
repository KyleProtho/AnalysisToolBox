from .metalog import metalog
from .class_method import rmetalog, plot, qmetalog, pmetalog, dmetalog, summary, update

# Create a pymetalog module that can be imported
class pymetalog:
    def __init__(self, x=None, bounds=[0,1], boundedness='u', term_limit=13, term_lower_bound=2, step_len=0.01, probs=None, fit_method='any', penalty=None, alpha=0.0, new_data=None, random_draws=1, random_generator='rand', specified_term=2):
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
        self.random_draws = random_draws
        self.random_generator = random_generator
        self.specified_term = specified_term
        self.my_metalog = None
    
    def metalog(self, x=self.x, bounds=self.bounds, boundedness=self.boundedness, term_limit=self.term_limit, term_lower_bound=self.term_lower_bound, step_len=self.step_len, probs=self.probs, fit_method=self.fit_method, penalty=self.penalty, alpha=self.alpha):
        """
        Fit a metalog distribution to the data
        """
        if self.my_metalog is None:
            self.my_metalog = metalog(x, bounds, boundedness, term_limit, term_lower_bound, step_len, probs, fit_method, penalty, alpha)
        return self.my_metalog

    def rmetalog(self, metalog_obj=None, random_draws=1, specified_term=2, random_generator='rand'):
        """
        Take random draws from the metalog distribution
        
        Parameters:
        -----------
        metalog_obj : object, optional
            A metalog object. If None, uses the class's metalog object
        random_draws : int, default=1
            Number of random draws to take
        specified_term : int, default=2
            Term to use for the metalog distribution
        random_generator : str, default='rand'
            Random number generator to use
        """
        # Use the provided metalog object or the class's metalog object
        m_obj = metalog_obj if metalog_obj is not None else self.metalog()
        return rmetalog(m_obj, n=random_draws, term=specified_term, generator=random_generator)

    # def plot(self, metalog_obj=None):
    #     """
    #     Plot the metalog distribution
        
    #     Parameters:
    #     -----------
    #     metalog_obj : object, optional
    #         A metalog object. If None, uses the class's metalog object
    #     """
    #     m_obj = metalog_obj if metalog_obj is not None else self.metalog()
    #     return plot(m_obj)

    def qmetalog(self, metalog_obj=None, y=None, specified_term=2):
        """
        Calculate the quantile of the metalog distribution
        
        Parameters:
        -----------
        metalog_obj : object, optional
            A metalog object. If None, uses the class's metalog object
        y : list or float
            Probability values to calculate quantiles for
        specified_term : int, default=2
            Term to use for the metalog distribution
        """
        m_obj = metalog_obj if metalog_obj is not None else self.metalog()
        return qmetalog(m_obj, y=y, term=specified_term)

    def pmetalog(self, metalog_obj=None, q=None, specified_term=2):
        """
        Calculate the probability of the metalog distribution
        
        Parameters:
        -----------
        metalog_obj : object, optional
            A metalog object. If None, uses the class's metalog object
        q : list or float
            Quantile values to calculate probabilities for
        specified_term : int, default=2
            Term to use for the metalog distribution
        """
        m_obj = metalog_obj if metalog_obj is not None else self.metalog()
        return pmetalog(m_obj, q=q, term=specified_term)

    def dmetalog(self, metalog_obj=None, q=None, specified_term=2):
        """
        Calculate the density of the metalog distribution
        
        Parameters:
        -----------
        metalog_obj : object, optional
            A metalog object. If None, uses the class's metalog object
        q : list or float
            Quantile values to calculate densities for
        specified_term : int, default=2
            Term to use for the metalog distribution
        """
        m_obj = metalog_obj if metalog_obj is not None else self.metalog()
        return dmetalog(m_obj, q=q, term=specified_term)

    def summary(self, metalog_obj=None):
        """
        Return a summary of the metalog distribution
        
        Parameters:
        -----------
        metalog_obj : object, optional
            A metalog object. If None, uses the class's metalog object
        """
        m_obj = metalog_obj if metalog_obj is not None else self.metalog()
        return summary(m_obj)

    # def update(self, metalog_obj=None, new_data=None, penalty=None, alpha=0.0):
    #     """
    #     Update the metalog distribution with new data
        
    #     Parameters:
    #     -----------
    #     metalog_obj : object, optional
    #         A metalog object. If None, uses the class's metalog object
    #     new_data : list or array
    #         New data to update the distribution with
    #     penalty : float or None, default=None
    #         Penalty parameter for the update
    #     alpha : float, default=0.0
    #         Alpha parameter for the update
    #     """
    #     m_obj = metalog_obj if metalog_obj is not None else self.metalog()
    #     return update(m_obj, new_data=new_data, penalty=penalty, alpha=alpha)
