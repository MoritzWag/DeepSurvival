

class Evaluator(nn.Module):
    """
    """
    def __init__(self, **kwargs):
        super(Evaluator, self).__init__(**kwargs)
    

    def coxphloss(self, y_true, y_pred):
        """
        """
        event, riskset = y_true
        predictions = y_pred 

    


    def cindex_metric(self):
        pass
