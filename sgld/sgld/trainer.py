from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np

def train(model, epochs, train_loader, test_loader, lossfunc, addnoise, optimizer):
    i = 0 
    lloss = []
    acc = []
    val = []
    parameter_history = []

    for epoch in range(epochs):
        epoch_loss = []
        epoch_acc = []
        epoch_val = []
        
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
            optimizer.step(lossfunc(i), add_noise=addnoise)   # update gradients
            prediction = output.data.max(1)[1]   # first column has actual prob.
            accuracy = np.mean(prediction.eq(target.data))*100
            epoch_acc.append(accuracy)
            epoch_loss.append(loss.data[0])
            
            bigparams = []
            for p in model.parameters():
                bigparams.append(p.data.cpu().numpy().ravel())
            parameter_history.append(np.hstack(bigparams))
                
        model.eval()
        for data, target in test_loader:
            data, target = Variable(data, volatile=True), Variable(target, volatile=True)
            data = data.cuda()
            target = target.cuda()
            output = model(data)
            prediction = output.data.max(1)[1]   # first column has actual prob.
            val_accuracy = np.mean(prediction.eq(target.data))*100
            epoch_val.append(val_accuracy)
            
        lloss.extend(epoch_loss)
        acc.extend(epoch_acc)
        val.extend(epoch_val)
        print('Epoch: {}\tLoss: {:.3f}\tAcc: {:.3f}\tVal Acc: {:.3f}'.format(epoch,
                                                                np.mean(epoch_loss),
                                                                np.mean(epoch_acc),
                                                                np.mean(epoch_val)))
        
    print('Total number of steps: {}'.format(i))
    parameter_history = np.vstack(parameter_history)

    return lloss, acc, val, parameter_history