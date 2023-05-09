import torch as th
import torch.nn as nn
from gym import spaces

from torch.nn.utils import weight_norm

from stable_baselines3 import PPO
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.vec_env import VecFrameStack
from stable_baselines3.common.env_util import make_vec_env

class TCNExtractor(BaseFeaturesExtractor):
    """
    :param observation_space: (gym.Space)
    :param features_dim: (int) Number of features extracted.
        This corresponds to the number of unit for the last layer.
    """
    
    def __init__(self, observation_space: spaces.Box, features_dim: int = 64, input_size = 1, levels=2):
        super().__init__(observation_space, features_dim)
        self.input_size = input_size
        self.input_len = int(observation_space.shape[0]/input_size)
        
        num_channels = [self.input_len]*levels
        self.tcn = TemporalConvNet(self.input_size, num_channels)
        self.linear = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.input_len*self.input_len, features_dim),
            nn.ReLU(),
        )
        
    def forward(self, observations: th.Tensor) -> th.Tensor:
        observations = observations.reshape((-1, self.input_size, self.input_len))
        # y = self.tcn(observations)[:, :, -1]
        # # print(observations[1])
        # # print(self.tcn(observations)[1])
        # # print(y[1,:,-1])
        return self.linear(self.tcn(observations))
        # return self.linear(y)
    


class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2):
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=(kernel_size-1) * dilation_size, dropout=dropout)]

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


if __name__ == "__main__":
    from torchsummary import summary
    import numpy as np
    
    extractor = TCNExtractor(spaces.Box(low=-1, high=1, shape=(10,), dtype=np.float32), input_size = 2).cuda()
    obs = th.randn(32, 10, requires_grad=True)
    print(summary(extractor, (10,)))
    
    # extractor.cpu()
    # th.onnx.export(extractor, obs, 'tcn.onnx')
#     N_ENV = 32
#     env = make_vec_env("MountainCarContinuous-v0",
#                        wrapper_class = MountainCarContinuousNoVelObsWrapper,
#                        n_envs = N_ENV)
#     print(env.observation_space)
#     env = VecFrameStack(env, 4)
#     print(env.observation_space)
#     obs = env.reset()
#     # print(obs)
#     print(obs.shape)
#     print(env.action_space)

#     policy_kwargs = dict(
#         features_extractor_class=TCNExtractor,
#         # net_arch=dict(pi=[32, 32], vf=[32, 32])
#     )

#     model = PPO("MlpPolicy", 
#                 env,
#                 policy_kwargs=policy_kwargs,
#                 verbose=1)
    # print(model.policy)
    # obs = th.randn(4, 1, 4, requires_grad=True, device='cuda')
    # th.onnx.export(model.policy, obs,'tcb_ppo.onnx')
    # model.learn(1_000_000)