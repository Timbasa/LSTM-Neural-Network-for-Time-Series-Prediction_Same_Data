import torch
import torch.nn as nn

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# RNN, many to many, lstm
class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, number_layer, output_layer, output_size= 1):
        super(LSTM, self).__init__()
        # self.hidden_size = hidden_size
        # self.num_layer = number_layer
        self.lstm = nn.LSTM(input_size=input_size,
                            hidden_size=hidden_size,
                            num_layers=number_layer,
                            batch_first=True,
                            dropout=0.2)
        # self.fc = nn.ModuleList([nn.Linear(self.hidden_size, output_layer) for _ in range(output_size)])
        # self.lstm = nn.LSTMCell(input_layer, hidden_layer)
        # self.lstm2 = nn.LSTMCell(hidden_layer, hidden_layer)
        self.out = nn.Linear(hidden_size, output_layer)


    def forward(self, x):
        # h0 = torch.zeros(self.num_layer, x.size(0), self.hidden_size).to(device)
        # c0 = torch.zeros(self.num_layer, x.size(0), self.hidden_size).to(device)
        # x = x.view(x.shape[0], -1, x.shape[1])
        # out, _ = self.lstm(x, (h0, c0))
        out, _ = self.lstm(x, None)
        # out = torch.cat([layer(out[-1, :, :]) for layer in self.fc], dim=1)
        # h0 = torch.zeros(x.size(0), self.hidden_layer).to(device)
        # c0 = torch.zeros(x.size(0), self.hidden_layer).to(device)
        # h1 = torch.zeros(x.size(0), self.hidden_layer).to(device)
        # c1 = torch.zeros(x.size(0), self.hidden_layer).to(device)
        # x = x.view(x.shape[0], x.shape[1])
        # out,_ = self.lstm(x, (h0, c0))
        # out,_ = self.lstm2(out, (h1, c1))
        out = self.out(out[:, -1, :])
        return out