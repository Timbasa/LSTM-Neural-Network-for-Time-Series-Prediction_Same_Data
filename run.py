__author__ = "Jakob Aungiers"
__copyright__ = "Jakob Aungiers 2018"
__version__ = "2.0.0"
__license__ = "MIT"

import os
import json
import time
import math
import matplotlib.pyplot as plt
from core.data_processor import DataLoader
from core.model import Model
from core.LSTM import LSTM
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import math

# quantiles = [0.5, 0.9]
input_size = 2
hidden_size = 100
number_layer = 1
# output_layer = len(quantiles)
batch_size = 32
epoch = 50
output_layer = 10
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def plot_results(predicted_data, true_data):
    fig = plt.figure(facecolor='white')
    ax = fig.add_subplot(111)
    ax.plot(true_data, label='True Data')
    plt.plot(predicted_data, label='Prediction')
    plt.legend()
    plt.show()


def plot_results_multiple(predicted_data, true_data, prediction_len):
    fig = plt.figure(facecolor='white')
    ax = fig.add_subplot(111)
    ax.plot(true_data, label='True Data')
    # Pad the list of predictions to shift it in the graph to it's correct start
    for i, data in enumerate(predicted_data):
        padding = [None for p in range(i * prediction_len)]
        plt.plot(padding + data, label='Prediction')
        plt.legend()
    plt.show()


def plot_origin_picture(data):
    fig = plt.figure(facecolor='white')
    ax = fig.add_subplot(111)
    ax.plot(data, label='data')
    plt.legend()
    plt.show()


# pytorch lstm train the model
def train(model, x, y, optimizer, batch_size, epoch):
    loss_function = torch.nn.MSELoss()
    for e in range(1, epoch + 1):
        len_batch = math.ceil(x.size(0) / batch_size)
        for batch_idx in range(len_batch):
            # print(x[batch_idx])
            # output = model(x[batch_idx])
            if batch_size * (batch_idx + 1) > x.size(0):
                output = model(x[batch_idx * batch_size:])
                target = y[batch_idx * batch_size:]
            else:
                output = model(x[batch_idx * batch_size:(batch_idx + 1) * batch_size])
                target = y[batch_idx * batch_size:(batch_idx + 1) * batch_size]
            # output = output.view()
            # output = output.view(x.size()).to(device)
            loss = loss_function(output, target)
            # loss = loss_function(output, y[batch_idx])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if batch_idx % 10 == 0:
                print('{} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(e,
                                                                  batch_idx * batch_size, x.size(0),
                                                                  100. * batch_idx / len_batch, loss.item()))
            # if batch_idx % 10 == 0:
            #     print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
            #         e, batch_idx * len(x), len(x), 100. * batch_idx / len(x), loss.item()))


# pytorch lstm validate the result
def validation(model, x):
    # loss_function = torch.nn.MSELoss()
    # sum_loss = []
    # for batch_idx in range(len(x)):
    #     prediction = model(x)
    #     sum_loss.append(loss_function(y-prediction))
    # print('the test Loss is {:.6f}'.format(np.mean(sum_loss)))
    predicted = []
    # for batch_idx in range(len(x)):
    # x_ind = x[batch_idx]
    # p = model(x_ind).detach().numpy()
    p = model(x).detach().numpy()
    # predicted.append(p)

    # return np.reshape(np.asarray(predicted), (655,))
    return p


def main():
    configs = json.load(open('config.json', 'r'))
    if not os.path.exists(configs['model']['save_dir']): os.makedirs(configs['model']['save_dir'])

    data = DataLoader(
        os.path.join('data', configs['data']['filename']),
        configs['data']['train_test_split'],
        configs['data']['columns']
    )

    # train_data =[]
    # for i in range(len(data.data_train)):
    #     train_data.append(data.data_train[i][0])
    # plot_origin_picture(train_data)
    #
    # test_data = []
    # for i in range(len(data.data_test)):
    #     test_data.append(data.data_test[i][0])
    # plot_origin_picture(test_data)

    # build the model
    # model = Model()
    # model.build_model(configs)
    model = LSTM(input_size, hidden_size, number_layer, output_layer).to(device)
    optimizor = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)

    # plot_origin_picture(data.data_test)

    # get the data
    scaler = MinMaxScaler()
    scaler.fit(data.data_train)

    x, y = data.get_train_data(
        seq_len=configs['data']['sequence_length'],
        # normalise=configs['data']['normalise']
        normalise=scaler
    )

    # get the validation data
    x_test, y_test = data.get_test_data(
        seq_len=configs['data']['sequence_length'],
        normalise=scaler
    )

    # plot_origin_picture(y)
    # plot_origin_picture(y_test)

    # transfer the data to tensor
    x = torch.tensor(x, dtype=torch.float32).to(device)
    y = torch.tensor(y, dtype=torch.float32).to(device)
    # print(x)

    # in-memory training
    # model.train(
    #     x,
    #     y,
    #     epochs=configs['training']['epochs'],
    #     batch_size=configs['training']['batch_size'],
    #     save_dir=configs['model']['save_dir']
    # )
    # out-of memory generative training
    # steps_per_epoch = math.ceil((data.len_train - configs['data']['sequence_length']) / configs['training']['batch_size'])
    # model.train_generator(
    #     data_gen=data.generate_train_batch(
    #         seq_len=configs['data']['sequence_length'],
    #         batch_size=configs['training']['batch_size'],
    #         # normalise=configs['data']['normalise']
    #         normalise=scaler
    #     ),
    #     epochs=configs['training']['epochs'],
    #     batch_size=configs['training']['batch_size'],
    #     steps_per_epoch=steps_per_epoch,
    #     save_dir=configs['model']['save_dir']
    # )
    # train the model
    train(model, x, y, optimizor, batch_size=batch_size, epoch=epoch)

    # transfer them to tensor
    x_test = torch.tensor(x_test, dtype=torch.float32).to(device)
    # y_test = torch.tensor(y_test, dtype=torch.float32).to(device)

    # predict the result
    # predictions = model.predict_sequences_multiple(x_test, configs['data']['sequence_length'], configs['data']['sequence_length'])
    # predictions = model.predict_sequence_full(x_test, configs['data']['sequence_length'])
    # predictions = model.predict_point_by_point(x_test)
    predictions = validation(model, x_test)

    # plot the result
    # plot_results_multiple(predictions, y_test, configs['data']['sequence_length'])
    plot_results(predictions, y_test)


if __name__ == '__main__':
    main()
