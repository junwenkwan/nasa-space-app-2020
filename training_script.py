import numpy as np
import os
import argparse
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
from torch.utils.data import Dataset, DataLoader, random_split, ConcatDataset
import torch.optim as optim
from torch.autograd import Variable
import csv
import sys
sys.path.append(sys.path[0] + "/..")
from model.mlp import MLP
os.environ['CUDA_LAUNCH_BLOCKING']='1'

class FireDataset(Dataset):

    def __init__(self, csv_path):
        self.csv_path = csv_path
        self.data = []

        with open(csv_path) as csv_file:
            csv_reader = csv.DictReader(csv_file)
            line_count = 0
            for row in csv_reader:
                if line_count == 0:
                    line_count += 1
                self.data.append({'fire':row['fire'],'solar_insolation':row['solar_insolation'],'land_temperature':row['land_temperature'],
                                    'rainfall':row['rainfall']})
                line_count += 1

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        label = self.data[idx]['fire']
        land_temperature = self.data[idx]['land_temperature']
        solar_insolation = self.data[idx]['solar_insolation']
        rainfall = self.data[idx]['rainfall']
        parameters = np.array((float(land_temperature), float(solar_insolation), float(rainfall)))
        label = int(label)
        data = {"parameters": parameters, "class": label}

        return data

"""
TRAINING LOOP
"""
def do_train(model, device, trainloader, criterion, optimizer, epochs, weights_pth):
    model.train()

    for epoch in range(epochs):  # loop over the dataset multiple times

        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a dict of [inputs, class]
            inputs, cls = data['parameters'], data['class']
            inputs = Variable(inputs).float().cuda()
            cls = Variable(cls).float().cuda()
            # print(inputs)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)
            # print(outputs)
            loss = criterion(outputs, cls.view(-1,1))
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 10 == 9:    # print every 10 mini-batches
                print('[%d, %3d] loss: %.6f' %
                      (epoch + 1, i + 1, running_loss / 10))
                running_loss = 0.0

    torch.save(model.state_dict(), weights_pth)

    print('Finished Training')

"""
TESTING LOOP
"""
def do_test(model, device, testloader, weights_pth):
    model.load_state_dict(torch.load(weights_pth))
    model.eval()

    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            inputs, cls = data['parameters'].to(device), data['class'].to(device)
            inputs = Variable(inputs).float().cuda()
            cls = Variable(cls).float().cuda()
            outputs = model(inputs)
            predicted = (outputs >= 0.755).float()
            print('----------')
            print(predicted.flatten())
            print(cls)
            total += cls.size(0)
            correct += (predicted.flatten() == cls).sum().item()

    print('Accuracy of the network on the %d datapoints: %5f %%' % (len(testloader)*1024,
        100 * correct / total))

def main(args):
    input_size = 3
    output_size = 1

    # Make the training and testing set to be less bias
    fire_dataset = FireDataset(args.csv_path[0])

    fire_train, fire_test = random_split(fire_dataset, (round(0.7*len(fire_dataset)), round(0.3*len(fire_dataset))))

    trainloader = DataLoader(fire_train, batch_size=4096, shuffle=True, num_workers=2)
    testloader = DataLoader(fire_test, batch_size=512, shuffle=False, num_workers=2)

    save_weights_pth = args.weights_path[0]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MLP(input_size=input_size, output_size=output_size)
    model.to(device)

    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=5e-7)
    epochs = 30

    if args.eval_only:
        do_test(model, device, testloader, save_weights_pth)
    else:
        do_train(model, device, trainloader, criterion, optimizer, epochs, save_weights_pth)
        do_test(model, device, testloader, save_weights_pth)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Code to train feature vector network"
    )
    parser.add_argument(
        "--csv-path",
        default="./",
        nargs="+",
        metavar="csv",
        help="Path to the csv file",
        type=str
    )
    parser.add_argument(
        "--weights-path",
        default="./my_weights.pth",
        nargs="+",
        metavar="WEIGHTS_PATH",
        help="Path to save the weights file",
        type=str
    )
    parser.add_argument(
        "--eval-only",
        help="set model to evaluate only",
        action='store_true'
    )

    args = parser.parse_args()

    main(args)
