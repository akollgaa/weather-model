import torch
from torch.utils.data import Dataset, DataLoader
from torch import nn
import glob
import os
import xarray as xr
import numpy as np
from math import prod

device = ("cuda" if torch.cuda.is_available() else "cpu")
data_path = "D:/Documents/Code/research/wrfout"
all_files = [os.path.basename(f) for f in glob.glob(data_path + "/wrfout*")][:35]
data_vars = [
    'T',
    'P',
    'PB',
    'QVAPOR',
    'QRAIN',
    'QSNOW',
]
epochs = 10

DATA_VARS_MIN = {'T': -8.417083740234375,
                 'P': -50.28515625,
                 'PB': 5420.28076171875,
                 'QVAPOR': 0.0,
                 'QRAIN': -1.3538986555136634e-12,
                 'QSNOW': -3.2677986886402296e-15,
                 'QGRAUP': 0.0,
                 'U': -14.949191093444824,
                 'V': -28.35787010192871,
                 'W': -5.611952781677246,
                 'QCLOUD': -7.121892919154105e-12}

DATA_VARS_MAX = {'T': 192.04437255859375,
                 'P': 1313.625,
                 'PB': 97532.5546875,
                 'QVAPOR': 0.02198491431772709,
                 'QRAIN': 0.0021312725730240345,
                 'QSNOW': 0.0001476061879657209,
                 'QGRAUP': 0.0,
                 'U': 26.568021774291992,
                 'V': 17.940385818481445,
                 'W': 8.062047958374023,
                 'QCLOUD': 0.002458590315654874}

def normalize(data, var) -> np.array:
    data = data.to_numpy()
    return (data - DATA_VARS_MIN[var]) / (DATA_VARS_MAX[var] - DATA_VARS_MIN[var])

# Combines array2 with array1
# There has to be a better way of doing this :(
def append_to_array(array1, array2):
    if array1 is None:
        return array2
    elif array1.shape == array2.shape:
        return torch.cat([torch.unsqueeze(array1, 0), torch.unsqueeze(array2, 0)], dim=0)
    else:
        return torch.cat([array1, torch.unsqueeze(array2, 0)], dim=0)


def load_data():
    data = {var: None for var in data_vars}
    print("\rLoading data", end='')
    for i in range(len(all_files)):
        ds = xr.open_dataset(f"{data_path}/{all_files[i]}")
        print(f"\rLoading data {i}/{len(all_files)}", end='')
        for var in data_vars:
            preprocessed_data = torch.from_numpy(normalize(ds[var], var))
            data[var] = append_to_array(data[var], preprocessed_data)
        ds.close()
    print("\rLoading data complete")
    return data

class CustomDataset(Dataset):
    def __init__(self, data, length, shape, time_offset):
        super().__init__()
        self.data = data
        self.length = length
        self.shape = shape  # time, height, row, column
        self.time_offset = time_offset

    def __len__(self):
        return self.length
    
    def gather_data(self, time, height, row, column):
        data = None
        for var in data_vars:
            # Must index at zero because wrf data has an extra outer dim
            data_point = self.data[var][time][0][height][row][column].clone().detach()
            data = append_to_array(data, data_point)
        return data

    def __getitem__(self, idx):
        column = idx % self.shape[3]
        row = (idx // self.shape[3]) % self.shape[2]
        height = (idx // (self.shape[3] * self.shape[2])) % self.shape[1]
        time = ((idx // (self.shape[3] * self.shape[2] * self.shape[1])) % self.shape[0]) + self.time_offset
        
        start_x = self.gather_data(time, height, row, column)
        end_x = self.gather_data(time+6, height, row, column)
        y = self.gather_data(time+3, height, row, column)

        x = torch.stack((start_x, end_x), dim=0)

        return x, y


class CustomModel(nn.Module):
    def __init__(self, input_shape):
        super().__init__()

        self.seq_module = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=(2, 1), stride=1),
            nn.ReLU(),
            nn.Conv2d(8, 16, kernel_size=(1, 2), stride=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=(1, 2), stride=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=(1, 2), stride=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=(1, 2), stride=1),
            nn.ReLU(),
            nn.Flatten(0, -1),
            nn.Linear(32 * prod((1, 2)), 64),
            nn.Linear(64, len(data_vars))
        )

        self.conv1 = nn.Conv2d(1, 16, kernel_size=(2, 1), stride=1)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(16, 32, kernel_size=(3, 1), stride=1)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=(3, 1), stride=1)
        self.conv4 = nn.Conv2d(32, 32, kernel_size=(3, 1), stride=1)
        self.flatten = nn.Flatten(0, -1)
        self.linear1 = nn.Linear(32 * prod((6, 1)), 64)
        self.linear2 = nn.Linear(64, len(data_vars))
        self.tanh = nn.Tanh()

    def concatenate_bounds(self, start, x, end, channels):
        new_x = torch.tensor([])
        for i in range(channels):
            new_x = torch.cat((new_x,
                               torch.unsqueeze(torch.cat((start, x[i], end), dim=0), dim=0)), dim=0)
        return new_x

    def forward(self, x):
        start = torch.unsqueeze(torch.squeeze(x).clone().detach()[0], dim=0)
        end = torch.unsqueeze(torch.squeeze(x).clone().detach()[1], dim=0)
        x = self.conv1(x)
        x = self.relu(x)
        x = self.concatenate_bounds(start, x, end, 16)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.concatenate_bounds(start, x, end, 32)
        x = self.conv3(x)
        x = self.relu(x)
        x = self.concatenate_bounds(start, x, end, 32)
        x = self.conv4(x)
        x = self.relu(x)
        x = self.flatten(x)
        x = self.linear1(x)
        x = self.tanh(x)
        return torch.unsqueeze(self.linear2(x), dim=0)


def calculate_average(x):
    output = []
    x = torch.squeeze(x)
    for i in range(len(data_vars)):
        avg = (x[0][i] + x[1][i]) / 2.0
        output.append(avg.item())
    return torch.unsqueeze(torch.tensor(output), dim=0)

def training_loop(data, model, loss_fn, optimizer, length, epoch):
    model.train()
    counter = 0
    total_loss = 0
    for batch, (x, y) in enumerate(data):
        counter += 1
        prediction = model(x)
        loss = loss_fn(prediction, y)
        total_loss += loss.item()

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 10 == 0:
            percent = int((counter / length) * 20)
            print(f"\rEpoch: {epoch} - {counter}/{length} - "
                  f"|{'#' * percent}{'-' * (20 - percent)}| - Loss: {loss.item():0.4e}"
                  f" - Avg Loss: {(total_loss/counter):0.4e}", end='')

    print(f"\rEpoch: {epoch} - Complete - Avg Loss: {(total_loss/counter):0.4e}")


def testing_loop(data, model, loss_fn):
    model.eval()
    counter = 0
    correct = 0
    total = 0
    total_loss = 0
    for x, y in data:
        counter += 1
        with torch.no_grad():
            prediction = model(x)
            average = calculate_average(x)
            prediction_loss = loss_fn(prediction, y).item()
            total_loss += prediction_loss
            average_loss = loss_fn(average, y).item()
            if prediction_loss < average_loss:
                correct += 1
            total += 1
        print(f"\rAccuracy: {(correct/total):0.2f}% - {correct}/{total} | "
              f"Pred_loss: {prediction_loss} Avg_loss: {average_loss}", end="")
    print(f"\rAccuracy: {(correct/total):0.2f}% - Validation Loss: {(total_loss/counter):0.4e}")


def main():
    validation_split = 0.8
    input_shape = (2, len(data_vars))
    model = CustomModel(input_shape)

    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters())

    data = load_data()
    for var in data:
        if data[var] is None:
            return
    data_shape = data[data_vars[0]].shape
    # The 30, 180, 180 are to adjust the data input so it is smaller so the model doesn't take forever to run.
    shape = (int((data_shape[0]*validation_split) - 6),  # Adjust for temporal changes
             data_shape[2] - 40,
             data_shape[3] - 200,
             data_shape[4] - 200)

    length = int(prod(shape))

    train_dataset_loader = DataLoader(CustomDataset(data, length, shape, 0),
                                      batch_size=1, shuffle=True)
    train_shape = (round((data_shape[0]*(1 - validation_split)) - 6),  # Adjust for temporal changes
                   data_shape[2] - 40,
                   data_shape[3] - 200,
                   data_shape[4] - 200)

    train_length = int(prod(shape))

    test_dataset_loader = DataLoader(CustomDataset(data, train_length, train_shape, int(data_shape[0]*validation_split)),
                                     batch_size=1, shuffle=False)

    for epoch in range(epochs):
        training_loop(train_dataset_loader, model, loss_fn, optimizer, length, epoch)
        testing_loop(test_dataset_loader, model, loss_fn)
    print("Done, hooray!")



if __name__ == "__main__":
    main()