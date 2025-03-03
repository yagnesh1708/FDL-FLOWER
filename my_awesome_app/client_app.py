import torch
from flwr.client import ClientApp, NumPyClient
from flwr.common import Context
from my_awesome_app.task import Net, get_weights, set_weights, test, train_client


# Define Flower Client and client_fn
class FlowerClient(NumPyClient):
    def __init__(self, net, partition_id):
        self.net = net
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # Always using CUDA:0 if available
        print(f"Using device: {self.device}")
        self.net.to(self.device)  # Move model to the appropriate device
        self.partition_id = partition_id

    def fit(self, parameters, config):
        set_weights(self.net, parameters)
        
        train_loss = train_client(
            self.net,
            self.device,  # Pass device to training
            self.partition_id,
        )
        
        return (
            get_weights(self.net),
            200,
            {"train_loss": train_loss},
        )

    def evaluate(self, parameters, config):
        set_weights(self.net, parameters)
        loss, accuracy = test(self.net, self.device)
        return loss,100, {"Score": accuracy}


def client_fn(context: Context):
    # Load model and data
    net = Net(n_tags=87)
    partition_id = context.node_config["partition-id"]
    # num_partitions = context.node_config["num-partitions"]
    # trainloader, valloader = load_data(partition_id, num_partitions)
    # local_epochs = context.run_config["local-epochs"]
    
    # Return Client instance
    client = FlowerClient(net, partition_id)
    print(f"Client device: {client.device}")
    return client.to_client()


# Flower ClientApp
app = ClientApp(
    client_fn,
)
