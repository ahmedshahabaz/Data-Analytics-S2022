# Deep learning course

import os, argparse

def get_parser(parser):

    #parser = argparse.ArgumentParser()

    '''
    Model Parameters
    '''
    parser.add_argument('--rnn_type', type = str, default = 'LSTM')

    parser.add_argument('--hidden_size', type=int, default=4)

    parser.add_argument('--lstm_layers', type=int, default=1)

    '''
    Training Parameters
    '''
    parser.add_argument('--num_epochs', type=int, default=100, help='maximum number of epochs')

    parser.add_argument('--batch_size', type=int, default=1)

    parser.add_argument('--learning_rate', type=float, default=.1, help='base learning rate')

    parser.add_argument('--clip', type=bool, default=False)

    parser.add_argument('--dropout', type=float, default=0, help='dropout ratio')
                                                                                                                        
    parser.add_argument('--device', type=int, default=0)

    '''
    Data Loader stuffs
    '''
    parser.add_argument('--data_split', type=float, default=0.33, help = "percentage of testing data")

    parser.add_argument('--data_dir', type=str, default='./')

    parser.add_argument('--data_file', type=str, default='airline-passengers.csv')

    parser.add_argument('--mode', type=str, default='train')

    parser.add_argument('--num_workers', type=int, default=8)

    parser.add_argument('--shuffle', type=bool, default=True)

    parser.add_argument('--comment', type=str, default = 'test', help='name for tensorboardX')

    args = parser.parse_args()

    return args
