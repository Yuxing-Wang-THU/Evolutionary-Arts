import torch.nn as nn

class CPPN_config:
    def __init__(self, dim_z, dim_x, dim_y, dim_c, hidden_size, scale) -> None:
        # dim_z: random vector size, for generating designs
        # dim_x: size of x
        # dim_y: size of y
        # dim_c: size of column, 3 is rgb, 1 is gray
        # hidden_size: hidden size of NN
        self.dim_z = dim_z
        self.dim_x = dim_x
        self.dim_y = dim_y
        self.dim_c = dim_c
        self.hidden_size = hidden_size
        self.scale = scale

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        m.weight.data.normal_(0.0, 1.0)
        if m.bias is not None:
            m.bias.data.fill_(0)


class CPPN(nn.Module):
    def __init__(self, config: CPPN_config):
        super(CPPN, self).__init__()
        dim_z = config.dim_z
        dim_c = config.dim_c
        ch = config.hidden_size

        self.l_z = nn.Linear(dim_z, ch)
        self.l_x = nn.Linear(1, ch, bias=False)
        self.l_y = nn.Linear(1, ch, bias=False)
        self.l_r = nn.Linear(1, ch, bias=False)

        self.nn_seq = nn.Sequential(
            nn.Tanh(),

            nn.Linear(ch, ch),
            nn.Tanh(),

            nn.Linear(ch, ch),
            nn.Tanh(),

            nn.Linear(ch, ch),
            nn.Tanh(),
            
            nn.Linear(ch, dim_c),
            nn.Sigmoid()
            )

        self._initialize()

    def _initialize(self):
        self.apply(weights_init)

    def forward(self, z, x, y, r):  
        output_z = self.l_z(z)
        output_x = self.l_x(x)
        output_y = self.l_y(y)
        output_r = self.l_r(r)
        
        u = output_z + output_x + output_y + output_r

        out = self.nn_seq(u)
        return out


