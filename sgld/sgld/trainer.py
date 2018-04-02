from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
from numpy.random import RandomState
from .model import *
from .sgld_optimizer import *
from .preproc import *
import torch
import sgld
import labnotebook

def runall(cuda_device,
    model_desc):




    db_string = "postgres://postgres:1418@localhost/experiments"
    labnotebook.initialize(db_string)

    torch.cuda.set_device(cuda_device)
    torch.manual_seed(model_desc['seed'])
    model = MnistModel()
    train_loader, test_loader = sgld.make_datasets()
    model = model.cuda()
    if model_desc['optimizer'][:4] == 'sgld':
        optimizer = eval(model_desc['optimizer'])(model.parameters(),
                         lr=model_desc['lr'],
                         addnoise=model_desc['addnoise'])
    else:
        optimizer = eval(model_desc['optimizer'])(model.parameters(),
                         lr=model_desc['lr'])
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
    xp = labnotebook.start_experiment(gpu_id=cuda_device, model_desc=model_desc)
    val_accuracy=0
    stdev_acc = []
    std_median = 0
    std_max = 0
    batch_evaluator = sgld.BatchEvaluator(test_loader)

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
            if model_desc['parametric_step']: # custom lr?
                if 'a' in model_desc.keys():
                    optimizer.step(lossfunc(i))   # update gradients
                elif 'lr_epoch' in model_desc.keys():
                    optimizer.step(model_desc['lr']/(epoch//model_desc['lr_epoch'] + 1))
            else:
                optimizer.step()
            prediction = output.data.max(1)[1]   # first column has actual prob.
            accuracy = np.mean(prediction.eq(target.data))*100

            statedict = model.state_dict().copy()
            for k, v in statedict.items():
                statedict.update({k: v.cpu().numpy().tolist()})
            #parameter_history.append(statedict)


            stdev_acc.append(sample(model_desc,
                                    statedict,
                                    model_desc['percentage_tosample']))
            batch_accuracy = batch_evaluator.iterate(i, model)

            if i%model_desc['step_samples'] == 0:
                sample_mat = np.vstack(stdev_acc)
                stdev_acc = []
                stdevs = np.std(sample_mat, axis = 0)
                std_median = np.median(stdevs)
                std_max = np.max(stdevs)


            if i%1000 == 0:
                #save statedict
                statedict = model.state_dict().copy()
                for k, v in statedict.items():
                    statedict.update({k: v.cpu().numpy().tolist()})
            else:
                state_dict = {}

            labnotebook.step_experiment(xp, i,
                                   trainloss=loss.data[0],
                                   trainacc=accuracy,
                                   valacc=val_accuracy,
                                   epoch=epoch,
                                   custom_fields={'std_median': std_median,
                                                 'std_max': std_max,
                                                 'batch_accuracy': batch_accuracy},
                                   model_params=state_dict
                                  )

        # validate
        val_accuracy, _ = sgld.evaluate(model, test_loader)

        print('Epoch: {}\tLoss: {:.3f}\tAcc: {:.3f}\tVal Acc: {:.3f}'.format(epoch,
                                                                np.mean(loss.data[0]),
                                                                np.mean(accuracy),
                                                                val_accuracy))

    print('Total number of steps: {}'.format(i))

    labnotebook.end_experiment(xp, final_trainloss=np.mean(loss.data[0]),
                          final_trainacc=np.mean(accuracy),
                          final_valacc=val_accuracy)

    return xp
