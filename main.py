import torch

from configs.mura_config import mura_config
from densenet import densenet169
from utils import n_p, get_count
from train import train_model, get_metrics
from pipeline import get_study_level_data, get_dataloaders
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime

# #### load study level dict data
def run(device, writer):
    torch.multiprocessing.freeze_support()
    study_data = get_study_level_data(mura_config["data"]["study_types"])
    # test
    # #### Create dataloaders pipeline
    data_cat = ['train', 'valid']  # data categories
    dataloaders = get_dataloaders(study_data, batch_size=mura_config["train"]["batch_size"])
    dataset_sizes = {x: len(study_data[x]) for x in data_cat}

    # #### Build model
    # tai = total abnormal images, tni = total normal images
    tai = {x: get_count(study_data[x], mura_config['data']['class_names'][1]) for x in data_cat}
    tni = {x: get_count(study_data[x], mura_config['data']['class_names'][0]) for x in data_cat}
    Wt1 = {x: n_p(tni[x] / (tni[x] + tai[x])) for x in data_cat}
    Wt0 = {x: n_p(tai[x] / (tni[x] + tai[x])) for x in data_cat}

    print('tai:', tai)
    print('tni:', tni, '\n')
    print('Wt0 train:', Wt0['train'])
    print('Wt0 valid:', Wt0['valid'])
    print('Wt1 train:', Wt1['train'])
    print('Wt1 valid:', Wt1['valid'])

    class Loss(torch.nn.modules.Module):
        def __init__(self, Wt1, Wt0):
            super(Loss, self).__init__()
            self.Wt1 = Wt1
            self.Wt0 = Wt0

        def forward(self, inputs, targets, phase):
            loss = - (self.Wt1[phase] * targets * inputs.log() + self.Wt0[phase] * (1 - targets) * (1 - inputs).log())
            return loss

    #model = densenet169(pretrained=True)
    model = torch.hub.load('pytorch/vision:v0.10.0', 'densenet121', pretrained=True)
    model = model.to(device)

    criterion = Loss(Wt1, Wt0)
    optimizer = torch.optim.Adam(model.parameters(), lr=mura_config['train']['learn_rate'])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                                                           patience=mura_config['train']['early_stopping_patience'],
                                                           verbose=True)
    # #### Train model
    model = train_model(model, criterion, optimizer, dataloaders, scheduler, dataset_sizes,
                        num_epochs=mura_config['train']['epochs'],
                        tensorboard_writer=writer)

    torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            }, 'saved_models/model.pth')

    get_metrics(model, criterion, dataloaders, dataset_sizes)


if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    log_dir = 'runs/mura/' + datetime.now().strftime("%Y-%m-%d--%H.%M_") + device.type
    writer = SummaryWriter(log_dir)
    print(f"Writing tensorboard to {log_dir}")
    run(device, writer)
