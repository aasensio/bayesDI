import torch
from torch import nn
from torch.nn import functional as F
from torch.nn import init

class ConvResidualBlock1d(nn.Module):
    def __init__(
        self,
        channels,
        context_channels=None,
        activation=F.relu,
        dropout_probability=0.0,
        use_batch_norm=False,
        zero_initialization=True,
    ):
        super().__init__()
        self.activation = activation

        if context_channels is not None:
            self.context_layer = nn.Conv1d(
                in_channels=context_channels,
                out_channels=channels,
                kernel_size=1,
                padding=0,
            )
        self.use_batch_norm = use_batch_norm
        if use_batch_norm:
            self.batch_norm_layers = nn.ModuleList(
                [nn.BatchNorm1d(channels, eps=1e-3) for _ in range(2)]
            )
        self.conv_layers = nn.ModuleList(
            [nn.Conv1d(channels, channels, kernel_size=3, padding=1) for _ in range(2)]
        )
        self.dropout = nn.Dropout(p=dropout_probability)
        if zero_initialization:
            init.uniform_(self.conv_layers[-1].weight, -1e-3, 1e-3)
            init.uniform_(self.conv_layers[-1].bias, -1e-3, 1e-3)

    def forward(self, inputs, context=None):
        temps = inputs
        if self.use_batch_norm:
            temps = self.batch_norm_layers[0](temps)
        temps = self.activation(temps)
        temps = self.conv_layers[0](temps)
        if self.use_batch_norm:
            temps = self.batch_norm_layers[1](temps)
        temps = self.activation(temps)
        temps = self.dropout(temps)
        temps = self.conv_layers[1](temps)
        if context is not None:
            temps = F.glu(torch.cat((temps, self.context_layer(context)), dim=1), dim=1)
        return inputs + temps


class ConvResidualNet1d(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        hidden_channels,
        context_channels=None,
        num_blocks=2,
        activation=F.relu,
        dropout_probability=0.0,
        use_batch_norm=False,
    ):
        super().__init__()
        
        self.context_channels = context_channels
        self.hidden_channels = hidden_channels
        if context_channels is not None:
            self.initial_layer = nn.Conv1d(
                in_channels=in_channels + context_channels,
                out_channels=hidden_channels,
                kernel_size=1,
                padding=0,
            )
        else:
            self.initial_layer = nn.Conv1d(
                in_channels=in_channels,
                out_channels=hidden_channels,
                kernel_size=1,
                padding=0,
            )
        self.blocks = nn.ModuleList(
            [
                ConvResidualBlock1d(
                    channels=hidden_channels,
                    context_channels=context_channels,
                    activation=activation,
                    dropout_probability=dropout_probability,
                    use_batch_norm=use_batch_norm,
                )
                for _ in range(num_blocks)
            ]
        )
        self.final_layer = nn.Conv1d(
            hidden_channels, out_channels, kernel_size=1, padding=0
        )

    def forward(self, inputs, context=None):        
        if context is None:
            temps = self.initial_layer(inputs.unsqueeze(-2))
        else:
            temps = self.initial_layer(torch.cat((inputs.unsqueeze(-2), context.unsqueeze(-2)), dim=1))
        for block in self.blocks:
            temps = block(temps, context.unsqueeze(-2))
        outputs = self.final_layer(temps)
        return outputs.view(outputs.size(0), -1)


def main():
    batch_size, channels, height = 100, 12, 64
    inputs = torch.rand(batch_size, channels, height)
    context = torch.rand(batch_size, channels // 2, height)
    net = ConvResidualNet1d(
        in_channels=channels,
        out_channels=2 * channels,
        hidden_channels=32,
        context_channels=channels // 2,
        num_blocks=2,
        dropout_probability=0.1,
        use_batch_norm=True,
    )
    # print(utils.get_num_parameters(net))
    outputs = net(inputs, context)
    print(outputs.shape)


if __name__ == "__main__":
    main()