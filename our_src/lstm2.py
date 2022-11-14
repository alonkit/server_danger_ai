"""
***********************************************************************************************************************
    imports
***********************************************************************************************************************
"""

import src.pytorch__driver_for_test_bench as pytorch__driver_for_test_bench
import torch.nn as nn
import torch.optim as optim

"""
***********************************************************************************************************************
    LSTMPredictor class
***********************************************************************************************************************
"""

class Encoder(nn.Module):
    def __init__(self, input_size, hidden):
        super(Encoder, self).__init__()
        hidden_size_for_lstm = 200
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
        self.linear = nn.Linear(
                in_features=hidden_size_for_lstm,
                out_features=hidden
            )
        
    def forward(self, x):
        output, (h_n, c_n) = self.lstm(x)
        output = self.linear(output[:,-1]) #N*hidden
        return output, (h_n, c_n)
    
    def flatten_parameters(self):
        self.__seq_model[0].flatten_parameters()

class Decoder(nn.Module):
    def __init__(self, input_size, out_size):
        super(Decoder, self).__init__()
        hidden_size_for_lstm = 200
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
        self.linear = nn.Linear(
                in_features=hidden_size_for_lstm,
                out_features=out_size
            )

            
    def forward(self, x, h_n, c_n):
        print(x.shape)
        exit()
        out = self.lstm(x,(h_n,c_n))[0]
        print(out.shape)
        out= self.linear(out)
        return out
    def flatten_parameters(self):
        self.__seq_model[0].flatten_parameters()
        
class LSTMPredictor(nn.Module):
    def __init__(self, input_size, output_size):
        super(LSTMPredictor, self).__init__()
        hidden_size_for_lstm = 200
        self.encoder = Encoder(input_size, hidden_size_for_lstm),
        self.decoder=Decoder(hidden_size_for_lstm, output_size),

    def forward(self, x):
        out = self.encoder
        return out

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
        self.__model = LSTMPredictor(
            input_size=1,
            output_size=1,
        ).to(pytorch__driver_for_test_bench.get_device())
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
        training_data_set=training_data_set,
        model=self.__model,
        num_epochs=30,
        model_input_length=self.__model_input_length,
        batch_size=64,
        criterion=self.__criterion,
        optimizer=self.__optimizer
        
        list_of_batch = __prepare_batches(
            training_data_set=training_data_set,
            model_input_length=model_input_length,
            batch_size=batch_size
            )
        epoch_time = 0
        training_start_time = time.time()
        min_sum_of_losses = float('inf')
        best_model = copy.deepcopy(model)
        for e in range(99999999):
            if (e >= num_epochs) and (time.time() - training_start_time > min_training_time_in_seconds):
                break
            epoch_start_time = time.time()
            sum_of_losses = __do_epoch(
                epoch_num=e, list_of_batch=list_of_batch, training_data_set=training_data_set, optimizer=optimizer,
                model=model, criterion=criterion
            )
            if sum_of_losses < min_sum_of_losses:
                min_sum_of_losses = sum_of_losses
                best_model = copy.deepcopy(model)
                assert not (best_model is model)  # assert different objects
            epoch_stop_time = time.time()
            epoch_time = epoch_stop_time - epoch_start_time
            avg_loss = sum_of_losses / len(list_of_batch)
            print(__msg, f"Epoch {e + 1} done. Epoch time was {epoch_time}. Average loss for the batches in epoch is {avg_loss}")
        return best_model


    def predict(self, ts_as_df_start, how_much_to_predict):
        self.__best_model.flatten_parameters()
        return pytorch__driver_for_test_bench.predict(
            ts_as_df_start=ts_as_df_start, how_much_to_predict=how_much_to_predict, best_model=self.__best_model
        )


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
