import time
import copy
import torch
from torchnet import meter
from torch.autograd import Variable

from configs.mura_config import mura_config
from utils import plot_training

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

data_cat = ['train', 'valid'] # data categories

def train_model(model, criterion, optimizer, dataloaders, scheduler, 
                dataset_sizes, num_epochs, tensorboard_writer=None):
    tensorboard_writer.add_text("config", mura_config.__str__())
    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    costs = {x:[] for x in data_cat} # for storing costs per epoch
    accs = {x:[] for x in data_cat} # for storing accuracies per epoch
    print('Train batches:', len(dataloaders['train']))
    print('Valid batches:', len(dataloaders['valid']), '\n')
    for epoch in range(num_epochs):
        confusion_matrix = {x: meter.ConfusionMeter(2, normalized=True) 
                            for x in data_cat}
        print('Epoch {}/{}'.format(epoch+1, num_epochs))
        print('-' * 10)
        # Each epoch has a training and validation phase
        for phase in data_cat:
            model.train(phase == 'train')
            running_loss = 0.0
            running_corrects = 0
            # Iterate over data.
            for i, data in enumerate(dataloaders[phase]):
                if i > 100:
                    break
                # get the inputs
                print(i, end='\r')
                inputs = data['images'][0]
                labels = data['label'].type(torch.FloatTensor)
                # wrap them in Variable
                inputs = Variable(inputs.to(device))
                labels = Variable(labels.to(device))
                # zero the parameter gradients
                optimizer.zero_grad()
                # forward
                outputs = model(inputs)
                outputs = torch.mean(outputs)
                loss = criterion(outputs, labels, phase)
                running_loss += loss.data[0]
                # backward + optimize only if in training phase
                if phase == 'train':
                    #loss.backward()
                    loss.sum().backward()
                    optimizer.step()
                # statistics
                if device.type =="cpu":
                    preds = torch.FloatTensor([(outputs.data > 0.5).item()])
                else:
                    preds = (outputs.data > 0.5).type(torch.cuda.FloatTensor)
                running_corrects += torch.sum(preds == labels.data)
                #confusion_matrix[phase].add(preds, labels.data)
                confusion_matrix[phase].add(preds.data, labels.data)
            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects / dataset_sizes[phase]

            costs[phase].append(epoch_loss)
            accs[phase].append(epoch_acc)
            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))
            print('Confusion Meter:\n', confusion_matrix[phase].value())
            # deep copy the model
            if phase == 'valid':
                scheduler.step(epoch_loss)
                if epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(model.state_dict())
        time_elapsed = time.time() - since
        print('Time elapsed: {:.0f}m {:.0f}s'.format(
                time_elapsed // 60, time_elapsed % 60))
        print()
        # Tensorboard
        tensorboard_writer.add_scalar(f'{phase} loss', epoch_loss, epoch)
        tensorboard_writer.add_scalar(f'{phase} accuracy', epoch_acc, epoch)
    # Training done
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best valid Acc: {:4f}'.format(best_acc))
    plot_training(costs, accs)
    # load best model weights
    model.load_state_dict(best_model_wts)
    return model


def get_metrics(model, criterion, dataloaders, dataset_sizes, phase='valid'):
    '''
    Loops over phase (train or valid) set to determine acc, loss and 
    confusion meter of the model.
    '''
    confusion_matrix = meter.ConfusionMeter(2, normalized=True)
    running_loss = 0.0
    running_corrects = 0
    for i, data in enumerate(dataloaders[phase]):
        print(i, end='\r')
        if device.type == "cpu":
            labels = data['label'].type(torch.FloatTensor)
        else:
            labels = data['label'].type(torch.cuda.FloatTensor)
        inputs = data['images'][0]
        # wrap them in Variable
        inputs = Variable(inputs.to(device))
        labels = Variable(labels.to(device))
        # forward
        outputs = model(inputs)
        outputs = torch.mean(outputs)
        loss = criterion(outputs, labels, phase)
        # statistics
        running_loss += loss.data[0] * inputs.size(0)
        if device.type == "cpu":
            #preds = (outputs.data > 0.5).type(torch.FloatTensor)
            preds = torch.FloatTensor([(outputs.data > 0.5).item()])
        else:
            preds = (outputs.data > 0.5).type(torch.cuda.FloatTensor)
        running_corrects += torch.sum(preds.data == labels.data)
        confusion_matrix.add(preds, labels.data)

    loss = running_loss / dataset_sizes[phase]
    acc = running_corrects / dataset_sizes[phase]
    print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, loss, acc))
    print('Confusion Meter:\n', confusion_matrix.value())
