import sgld

class Variances:
    """
    this class will calculate standard deviations and batch standard deviations online
    we need to maintain state so we can support thinning and burn-in without storing
    all the parameters (memory constraints)
    """
    def __init__(burn_in=200, thinning=10):
        self.burn_in = burn_in
        self.thinning=thinning
        
        self.step = 0
        
        self.std = 0
        
        self.batch_std = 0
        
