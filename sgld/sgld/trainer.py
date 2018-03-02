from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
from numpy.random import RandomState
from .model import *
from .sgld_optimizer import *
from .preproc import *
import torch
import sgld
import xpbase

def runall(cuda_device,
    model_desc):
    
    
    
    
    db_string = "postgres://postgres:1418@localhost/experiments"
    xpbase.initialize(db_string)
    
    torch.cuda.set_device(cuda_device)
    model = MnistModel()
    train_loader, test_loader = sgld.make_datasets()
    model = model.cuda()
    optimizer = SGLD(model.parameters(), lr=model_desc['lr'])
    xp = train(
        model,
        train_loader,
        test_loader,
        optimizer,
        model_desc,
        cuda_device
    )

    return xp 

def sample(model_desc, model_params, percentage_tosample):
    """
    returns a stratified sample of the model params
    """
    result = []
    rng = RandomState(model_desc['seed'])
    for k, v in model_params.items():
        result.append(rng.choice(np.ravel(v),
                                 size=int(len(np.ravel(v))*percentage_tosample),
                                replace=False))
        
    result = np.hstack(result)
    
    return result
        
    

def train(model, train_loader, test_loader, optimizer, model_desc, cuda_device):
    i = 0 
    lossfunc = lambda x: sgld.lossrate(x,
                                       model_desc['a'],
                                       model_desc['b'],
                                       model_desc['gamma'])
#    lloss = []
#    acc = []
#    val = []
#    parameter_history = []
    xp = xpbase.start_experiment(gpu_id=cuda_device, model_desc=model_desc)
    val_accuracy=0
    stdev_acc = []
    std_median = 0
    std_max = 0

    for epoch in range(model_desc['epochs']):
        
        model.train()
        for data, target in train_loader:
            i += 1
            data, target = Variable(data), Variable(target)
            data = data.cuda()
            target = target.cuda()
            optimizer.zero_grad()
            output = model(data)
            loss = F.nll_loss(output, target)
            loss.backward()    # calc gradients
            optimizer.step(lossfunc(i), add_noise=model_desc['addnoise'])   # update gradients
            prediction = output.data.max(1)[1]   # first column has actual prob.
            accuracy = np.mean(prediction.eq(target.data))*100
            
            statedict = model.state_dict().copy()
            for k, v in statedict.items():
                statedict.update({k: v.cpu().numpy().tolist()})
            #parameter_history.append(statedict)
            
            
            stdev_acc.append(sample(model_desc, 
                                    statedict,
                                    model_desc['percentage_tosample']))
            
            if i%model_desc['step_samples'] == 0:
                sample_mat = np.vstack(stdev_acc)
                stdev_acc = []
                stdevs = np.std(sample_mat, axis = 0)
                std_median = np.median(stdevs)
                std_max = np.max(stdevs)
                
                
            if i%100 == 0:
                model.eval()
                for data, target in test_loader:
                    data, target = Variable(data, volatile=True), Variable(target, volatile=True)
                    data = data.cuda()
                    target = target.cuda()
                    output = model(data)
                    prediction = output.data.max(1)[1]   # first column has actual prob.
                    val_accuracy = np.mean(prediction.eq(target.data))*100
                    
                statedict = model.state_dict().copy()
                for k, v in statedict.items():
                    statedict.update({k: v.cpu().numpy().tolist()})
            else:
                state_dict = {}
                    
            xpbase.step_experiment(xp, i,
                                   trainloss=loss.data[0],
                                   trainacc=accuracy,
                                   valacc=val_accuracy,
                                   epoch=epoch,
                                   model_params={'statedict': statedict, 
                                                 'std_median': std_median,
                                                 'std_max': std_max}
                                  )
            
        print('Epoch: {}\tLoss: {:.3f}\tAcc: {:.3f}\tVal Acc: {:.3f}'.format(epoch,
                                                                np.mean(loss.data[0]),
                                                                np.mean(accuracy),
                                                                np.mean(val_accuracy)))
        
    print('Total number of steps: {}'.format(i))
    
    xpbase.end_experiment(xp)

    return xp