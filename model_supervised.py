import torch
import math
import torch.nn as nn
import torch.nn.functional as F

class SupervisedNetwork(nn.Module):
    """
    Baseline supervised neural network for image classification
    """
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1, bias=True)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=True)
        self.bn2 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(2, 2)

        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=True)
        self.bn3 = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=True)
        self.bn4 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(2, 2)

        self.conv5 = nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=True)
        self.bn5 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(2, 2)

        self.fc1 = nn.Linear(128*4*4, 256, bias=True)
        self.dropout = nn.Dropout(0.5) # implement this as part of stage 3 properly!
        self.fc2 = nn.Linear(256, 10, bias=True)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool1(x)

        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = self.pool2(x)

        x = F.relu(self.bn5(self.conv5(x)))
        x = self.pool3(x)

        x = torch.flatten(x, 1)

        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)

        # x = F.softmax(x, dim=1)
        return x

def init_weights(m, type):
    """
    Initializes weights for the model based on the specified type of weight initialization
    
    :param m: the current layer of the model that this method is being applied to
    :param type: The initialization method, either random uniform, random normal, or He initialization
    """
    if not isinstance(m, (nn.Conv2d, nn.Linear)):
        return
    
    fanin, _ = nn.init._calculate_fan_in_and_fan_out(m.weight)

    if type == "random_uniform":
        """
        Initial weights are sampled from a uniform distribution with lower and upper bound -1/sqrt(fanin)
        and 1/sqrt(fanin) respectively, where fanin is the number of connections leading into a unit
        """
        bound = 1.0 / math.sqrt(fanin)
        nn.init.uniform_(m.weight, -bound, bound) # initializes weights using uniform distribution and modifies model weights directly

    elif type == "random_normal":
        """
        Initial weights are sampled from a normal distribution with mean 0 and a small standard deviation.
        """
        nn.init.normal_(m.weight, mean=0.0, std=0.01)

    elif type == "he":
        """
        Initial weights are sampled from a normal distritbution with mean 0 and a standard deviation equal to
        sqrt(2/fanin), where fanin is the number of connections leading into a unit
        """
        std = math.sqrt(2.0 / fanin)
        nn.init.normal_(m.weight, mean=0.0, std=std)

    else:
        raise ValueError(f"Invalid weight initialization type: {type}")

    if m.bias is not None:
        nn.init.zeros_(m.bias)