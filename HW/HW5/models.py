
# you can import pretrained models for experimentation & add your own created models
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class myRNN(nn.Module):
    

    def __init__(self, args, input_size = 3, direction = 1):
        """
            A linear model for image classification.
        """

        super(myRNN, self).__init__()
        
        self.num_layers = args.lstm_layers
        self.input_size = input_size
        self.hidden_size = args.hidden_size
        self.batch_size = args.batch_size
        self.dropout = args.dropout
        self.direction = direction
        self.sequence = 1
        self.init_dim = self.num_layers * self.direction * self.hidden_size

        self.tanh = nn.Tanh()

        self.fc_init = nn.Linear(in_features = self.input_size, out_features = self.init_dim)
        
        self.lstm = nn.LSTM(num_layers = self.num_layers, input_size = self.input_size,
            hidden_size = self.hidden_size, batch_first = True, dropout = self.dropout,
            bidirectional = False if direction == 1 else True)
        
        self.sigmoid = nn.Sigmoid()

        self.fc = nn.Linear( in_features = self.hidden_size * self.direction * self.sequence, 
            out_features = self.sequence)


    def forward(self, x, mode = None):
        """
            feed-forward (vectorized) image into a linear model for classification.   
        """

        x = x.reshape(x.shape[0],1,self.input_size)
        x = x.to(torch.float)

        c_0 = None
        h_0 = None

        
        # *** Different initialization for h_0 and c_0
        
        '''
        temp = x[:,0,:]
        c0 = self.fc_init(temp)
        c0 = c0.reshape(self.num_layers * self.direction, temp.shape[0],  self.hidden_size)
        h0 = self.tanh(c0)   
        '''

        if c_0 == None:
            out , hidden = self.lstm(x)

        else:
            out , hidden = self.lstm(x , (h0.detach(), c0.detach()))
        

        out = self.sigmoid(out)

        out = out.view(out.shape[0], self.sequence * self.direction * self.hidden_size)

        '''
        out is the hidden unit for each time step of the last LSTM layer
        hn is the hidden unit for last time step of all the layers
        # shape of out is:
        # batch_size, sequence, hidden_size as batch_first = True
        # otherwise sequence, batch_size, hidden_size
        # so out.shape: torch.Size([1, 1, 4])
        # hn.shape-->  batch_size, num_layers, hidden_size
        '''

        out = self.fc(out)

        return out
