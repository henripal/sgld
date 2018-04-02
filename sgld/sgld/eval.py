import sgld
import numpy as np
from torch.autograd import Variable

def evaluate(model, test_loader):
    model.eval()
    outputs = []
    accuracies = []
    
    for data, target in test_loader:
        data, target = Variable(data, volatile=True), Variable(target, volatile=True)
        data = data.cuda()
        target = target.cuda()
        output = model(data)
        prediction = output.data.max(1)[1]   # first column has actual prob.
        val_accuracy = np.mean(prediction.eq(target.data))*100
        outputs.append(output)
        accuracies.append(val_accuracy)
        
    return np.mean(accuracies), output
    
    
    

class BatchEvaluator:
    def __init__(self, test_loader, burn_in=150, thinning=1):
        self.burn_in = burn_in
        self.thinning=thinning
        self.test_loader = test_loader
        
        self.cum_output = []
        self.accuracy = 0
        
        
        
    def iterate(self, step, model):
        if step > self.burn_in and step%self.thinning == 0:
            
            accuracies = []
            
            if self.cum_output:
                for i, (data, target) in enumerate(self.test_loader):
                    data, target = Variable(data, volatile=True), Variable(target, volatile=True)
                    data, target = data.cuda(), target.cuda()
                    output = model(data)
                    self.cum_output[i] = self.cum_output[i] + output
                    prediction = self.cum_output[i].data.max(1)[1]   # first column has actual prob.
                    val_accuracy = np.mean(prediction.eq(target.data))*100
                    accuracies.append(val_accuracy)
            else:
                for i, (data, target) in enumerate(self.test_loader):
                    data, target = Variable(data, volatile=True), Variable(target, volatile=True)
                    data, target = data.cuda(), target.cuda()
                    output = model(data)
                    self.cum_output.append(output)
                    prediction = self.cum_output[i].data.max(1)[1]   # first column has actual prob.
                    val_accuracy = np.mean(prediction.eq(target.data))*100
                    accuracies.append(val_accuracy)
                

            self.accuracy = np.mean(accuracies)
            
        return self.accuracy
        
        