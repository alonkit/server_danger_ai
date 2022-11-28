"""
***********************************************************************************************************************
    imports
***********************************************************************************************************************
"""

import our_src.training_utils as training_utils
import torch.nn as nn
import torch.optim as optim
import torch
import numpy as np

"""
***********************************************************************************************************************
    LSTMPredictor class
***********************************************************************************************************************
"""

class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size_for_lstm):
        super(Encoder, self).__init__()
        internal_hidden_dimension = 32
        num_layers = 2
        dropout = 0.03
        self.lstm = nn.LSTM(
                input_size=input_size,
                hidden_size=hidden_size_for_lstm,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout,
            )
        # self.linear = nn.Linear(
        #         in_features=hidden_size_for_lstm,
        #         out_features=hidden
        #     )
        
    def forward(self, x):
        output, (h_n, c_n) = self.lstm(x)
        # h_n = self.linear(h_n) #N*hidden
        return output, (h_n, c_n)
    
    def flatten_parameters(self):
        self.__seq_model[0].flatten_parameters()

class Decoder(nn.Module):
    def __init__(self, input_size,hidden_size_for_lstm , out_size):
        super(Decoder, self).__init__()
        num_layers = 2
        dropout = 0.03
        self.lstm = nn.LSTM(
                input_size=input_size,
                hidden_size=hidden_size_for_lstm,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout,
            )
        self.linear = nn.Linear(
                in_features=hidden_size_for_lstm,
                out_features=out_size
            )

            
    def forward(self, x, h_n, c_n):
        out = self.lstm(x,(h_n,c_n))[0]
        out= self.linear(out)
        return out
    
    def predict(self, x_0, h_0, c_0, prediction_len):
        # print('decoder', x_0.shape)
        x_i=x_0
        h_i, c_i = h_0, c_0 
        output=[]
        for i in range(prediction_len):
            x_i, (h_i, c_i) = self.lstm(x_i, (h_i, c_i))
            x_i = self.linear(x_i)
            # print(x_i.shape, h_i.shape, c_i.shape)
            output.append(x_i.squeeze(1).squeeze(1))
        return torch.cat(output,dim=0)
            
    def flatten_parameters(self):
        self.__seq_model[0].flatten_parameters()
        
class LSTMPredictor(nn.Module):
    def __init__(self, input_size, output_size):
        super(LSTMPredictor, self).__init__()
        hidden_size_for_lstm = 256
        self.encoder = Encoder(input_size, hidden_size_for_lstm)
        self.decoder=Decoder(input_size,hidden_size_for_lstm, output_size)
        print(self.encoder.parameters())
        print(self.decoder.parameters())

    def forward(self, x,y):
        y = y.unsqueeze(-1)
        # print(f'x={x.shape} , y={y.shape}')
        output, (h_n, c_n) = self.encoder(x)
        # print(f'h={h_n.shape} , c={c_n.shape}')
        out = self.decoder(torch.cat([x[:,-1:],y[:,:-1]],dim=1), h_n, c_n).squeeze(-1)
        # print(f'out={out.shape}')
        
        return out

    def predict(self,x, prediction_len):
        with torch.no_grad():
            x = torch.from_numpy(x).to(torch.float32).to(training_utils.device).unsqueeze(0).unsqueeze(-1)
            _, (h_n, c_n) = self.encoder(x)
            x_0=x[:,-1:]
            y = self.decoder.predict(x_0, h_n, c_n,prediction_len).numpy()
            y.astype(np.float64)
        return y
        
    
    def flatten_parameters(self):
        self.__seq_model[0].flatten_parameters()


"""
***********************************************************************************************************************
    Testable class
***********************************************************************************************************************
"""


class PytorchLSTMTester:
    def __init__(self, length_of_shortest_time_series, metric, app):
        # prepare parameters
        self.__msg = "[PytorchLSTMTester]"
        
        self.__model_input_length = length_of_shortest_time_series // 2
        self.__model_output_length = length_of_shortest_time_series // 2
        self.__model = LSTMPredictor(
            input_size=1,
            output_size=1,
        ).to(training_utils.device)
        self.__optimizer = optim.Adam(self.__model.parameters(), lr=0.01)
        self.__best_model = self.__model
        self.__criterion = nn.MSELoss()
        # print
        print(self.__msg, f"model = {self.__model}")
        print(self.__msg, f"optimizer = {self.__optimizer}")
        print(self.__msg, f"criterion = {self.__criterion}")

    """
    *******************************************************************************************************************
        API functions
    *******************************************************************************************************************
    """

    def learn_from_data_set(self, training_data_set):
        best_model =training_utils.train(
        training_data_set=training_data_set,
        model=self.__model,
        num_epochs=3,
        model_input_length=self.__model_input_length,
        model_output_length=self.__model_output_length,
        batch_size=64,
        criterion=self.__criterion,
        optimizer=self.__optimizer
        )
        return best_model


    def predict(self, ts_as_df_start, how_much_to_predict):
        x =ts_as_df_start['sample'].to_numpy()
        return self.__model.predict(x, how_much_to_predict)


"""
***********************************************************************************************************************
    main function
***********************************************************************************************************************
"""


def main(test_to_perform):
    import src.framework__test_bench as framework__test_bench
    tb = framework__test_bench.TestBench(
        class_to_test=PytorchLSTMTester,
        path_to_data="../data/",
        tests_to_perform=test_to_perform
    )
    tb.run_training_and_tests()


"""
***********************************************************************************************************************
    run main function
***********************************************************************************************************************
"""

if __name__ == "__main__":
    test_to_perform = (
        # Container CPU
        {"metric": "container_cpu", "app": "kube-rbac-proxy", "prediction length": 16, "sub sample rate": 30,
         "data length limit": 30},
        {"metric": "container_cpu", "app": "dns", "prediction length": 16, "sub sample rate": 30,
         "data length limit": 30}
        # {"metric": "container_cpu", "app": "collector", "prediction length": 16, "sub sample rate": 30,
        #  "data length limit": 30},
        # # Container Memory
        # {"metric": "container_mem", "app": "nmstate-handler", "prediction length": 16, "sub sample rate": 30,
        #  "data length limit": 30},
        # {"metric": "container_mem", "app": "coredns", "prediction length": 16, "sub sample rate": 30,
        #  "data length limit": 30},
        # {"metric": "container_mem", "app": "keepalived", "prediction length": 16, "sub sample rate": 30,
        #  "data length limit": 30},
        # # Node Memory
        # {"metric": "node_mem", "app": "moc/smaug", "prediction length": 16, "sub sample rate": 30,
        #  "data length limit": 30},
        # {"metric": "node_mem", "app": "emea/balrog", "prediction length": 16, "sub sample rate": 30,
        #  "data length limit": 30}
    )
    main(test_to_perform)
