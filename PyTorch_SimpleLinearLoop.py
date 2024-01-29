import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics
import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
from torch.utils.data import Dataset, DataLoader

import numpy as np
from scipy.io import loadmat

class MyDataset(Dataset):
    def __init__(self, dipoles_list, leadfield, dipole_locations, dipole_orientations):
        super(MyDataset, self).__init__()

        self.dipoles_list = dipoles_list

        self.leadfield = leadfield
        self.dipole_locations = dipole_locations
        self.dipole_orientations = dipole_orientations

        self.range = torch.tensor(np.amax(leadfield,axis=1) - np.amin(leadfield,axis=1), dtype=torch.float32)

        self.noise_level = 100.0 # Approximately 100.0 for 10% noise, 10.0 for 1% noise, etc.
    
    def __getitem__(self, idx):
        dipole_idx = self.dipoles_list[idx]
        dipole_location = self.dipole_locations[dipole_idx]
        dipole_orientation = self.dipole_orientations[dipole_idx]
        leadfield_dipole = self.leadfield[:,dipole_idx]
        
        # Convert data types to single precision floats and longs
        dipole_idx = torch.tensor(dipole_idx, dtype=torch.long)
        dipole_location = torch.tensor(dipole_location, dtype=torch.float32)
        dipole_orientation = torch.tensor(dipole_orientation, dtype=torch.float32)
        leadfield_dipole = torch.tensor(leadfield_dipole, dtype=torch.float32)

        # Add noise to the leadfield
        noise = torch.randn_like(leadfield_dipole)*self.noise_level
        leadfield_dipole = leadfield_dipole + noise

        #print(torch.mean(torch.abs(noise)/self.range))

        return leadfield_dipole, dipole_idx, dipole_location, dipole_orientation, 

    def __len__(self):
        return len(self.dipoles_list)


class LitNetwork(pl.LightningModule):
    def __init__(self,in_channels,batch_size=1):
        super(LitNetwork, self).__init__()

        self.linear1 = nn.Linear(in_channels, 100)
        self.linear2 = nn.Linear(100, 1000)
        self.linear3 = nn.Linear(1000, 1000)
        self.linear4 = nn.Linear(1000, 100)
        self.linear5a = nn.Linear(100, 3)
        self.linear5b = nn.Linear(100, 3)
        
        self.loss_func1 = torch.nn.MSELoss()
        self.loss_func2 = torch.nn.CosineSimilarity()

        self.b = batch_size

    def forward(self, x):
        x = self.linear1(x)
        x = F.relu(x)
        x = self.linear2(x)
        x = F.relu(x)
        x = self.linear3(x)
        x = F.relu(x)
        x = self.linear4(x)
        x = F.relu(x)
        loc = self.linear5a(x)
        ang = self.linear5b(x)

        loc = F.tanh(loc)*150 # Approximate skull boundaries
        ang = F.normalize(ang, p=2, dim=1)

        return loc,ang

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)
        return optimizer

    def training_step(self, data, batch_idx):
        leadfield, dipole_idx, location, orientation = data[0], data[1], data[2], data[3]

        out_loc,out_ang = self.forward(leadfield)
        #dist_loss = self.loss_func1(outs[:,:3], location)
        #angle_loss = self.loss_func2(outs[:,3:], orientation).mean()
        dist_loss = self.loss_func1(out_loc, location)
        angle_loss = 1 - self.loss_func2(out_ang, orientation).mean()
        loss = dist_loss * angle_loss
        self.log("train_dist_loss",dist_loss,batch_size=self.b,sync_dist=True)
        self.log("train_angle_loss",angle_loss,batch_size=self.b,sync_dist=True)
        self.log("train_loss",loss,prog_bar=True,on_step=False,on_epoch=True,batch_size=self.b,sync_dist=True)
        return loss
    
    def validation_step(self, val_data, batch_idx):
        leadfield, dipole_idx, location, orientation = val_data[0], val_data[1], val_data[2], val_data[3]

        out_loc,out_ang = self.forward(leadfield)
        dist_loss = self.loss_func1(out_loc, location)
        angle_loss = 1 - self.loss_func2(out_ang, orientation).mean()
        loss = dist_loss * angle_loss
        self.log("val_dist_loss",dist_loss,batch_size=self.b,prog_bar=True,on_step=False,on_epoch=True,sync_dist=True)
        self.log("val_angle_loss",angle_loss,batch_size=self.b,prog_bar=True,on_step=False,on_epoch=True,sync_dist=True)
        self.log("val_loss",loss,batch_size=self.b,prog_bar=True,on_step=False,on_epoch=True,sync_dist=True)
        return None

    def test_step(self, test_data, batch_idx):
        leadfield, dipole_idx, location, orientation = test_data[0], test_data[1], test_data[2], test_data[3]

        out_loc,out_ang = self.forward(leadfield)
        dist_loss = self.loss_func1(out_loc, location)
        angle_loss = 1 - self.loss_func2(out_ang, orientation).mean()
        loss = dist_loss*angle_loss
        self.log("test_dist_loss",dist_loss,batch_size=self.b,prog_bar=True,on_step=False,on_epoch=True,sync_dist=True)
        self.log("test_angle_loss",angle_loss,batch_size=self.b,prog_bar=True,on_step=False,on_epoch=True,sync_dist=True)
        self.log("test_loss",loss,batch_size=self.b,prog_bar=True,on_step=False,on_epoch=True,sync_dist=True)
        return None


def train_network(workers=8):

    eeg_data = loadmat("ernie_eeg_simulations.mat")
    leadfield = eeg_data["leadfield"]
    #leadfield_laplacian = eeg_data["surface_laplacian_for_EEG"]
    dipole_locations = eeg_data["dipole_locations"]
    dipole_orientations = eeg_data["dipole_orientations"]
    #electrode_locations = eeg_data["electrodes"]["node"]

    num_dipoles = dipole_locations.shape[0]

    # Generate a train, validation, and test split over the num_dipoles
    split = torch.randperm(num_dipoles, generator=torch.Generator().manual_seed(42))
    train = split[:int(num_dipoles*0.72)]
    val = split[int(num_dipoles*0.72):int(num_dipoles*0.8)]
    test = split[int(num_dipoles*0.8):]

    train_dataset = MyDataset(train,leadfield,dipole_locations,dipole_orientations)
    validation_dataset = MyDataset(val,leadfield,dipole_locations,dipole_orientations)
    test_dataset = MyDataset(test,leadfield,dipole_locations,dipole_orientations)

    b = 512
    train_loader = DataLoader(train_dataset,batch_size=b,num_workers=workers,persistent_workers=True)
    val_loader = DataLoader(validation_dataset,batch_size=b,num_workers=workers,persistent_workers=True)
    test_loader = DataLoader(test_dataset,batch_size=b,num_workers=workers,persistent_workers=True)

    model = LitNetwork(leadfield.shape[0],b)
    checkpoint = pl.callbacks.ModelCheckpoint(monitor='val_loss', save_top_k=1, mode='min')
    logger = pl_loggers.TensorBoardLogger(save_dir="my_logs")
    #logger = pl_loggers.CSVLogger(save_dir="my_logs",name="my_csv_logs")

    device = "gpu" # Use 'mps' for Mac M1 or M2 Core, 'gpu' for Windows with Nvidia GPU, or 'cpu' for Windows without Nvidia GPU

    trainer = pl.Trainer(max_epochs=75, accelerator=device, callbacks=[checkpoint], logger=logger)
    trainer.fit(model,train_loader,val_loader)
        
    trainer.test(ckpt_path="best", dataloaders=test_loader)



if __name__ == "__main__":

    train_network(workers=8)

