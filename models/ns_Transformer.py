import torch
import torch.nn as nn
from models.Transformer_EncDec import Decoder, DecoderLayer, Encoder, EncoderLayer
from models.SelfAttention_Family import DSAttention, AttentionLayer
from models.Embed import DataEmbedding


class Projector(nn.Module):
    '''
    MLP to learn the De-stationary factors
    '''
    def __init__(self, enc_in, seq_len, hidden_dims, hidden_layers, output_dim, kernel_size=3):
        super(Projector, self).__init__()

        padding = 1 if torch.__version__ >= '1.5.0' else 2
        self.series_conv = nn.Conv1d(in_channels=seq_len, out_channels=1, kernel_size=kernel_size, padding=padding, padding_mode='circular', bias=False)

        layers = [nn.Linear(2 * enc_in, hidden_dims[0]), nn.ReLU()]
        for i in range(hidden_layers-1):
            layers += [nn.Linear(hidden_dims[i], hidden_dims[i+1]), nn.ReLU()]
        
        layers += [nn.Linear(hidden_dims[-1], output_dim, bias=False)]
        self.backbone = nn.Sequential(*layers)

    def forward(self, x, stats):
        # x:     B x S x E
        # stats: B x 1 x E
        # y:     B x O
        batch_size = x.shape[0]
        x = self.series_conv(x)          # B x 1 x E
        x = torch.cat([x, stats], dim=1) # B x 2 x E
        x = x.view(batch_size, -1) # B x 2E
        y = self.backbone(x)       # B x O

        return y

class Model(nn.Module):
    """
    Non-stationary Transformer
    """
    def __init__(self):
        super(Model, self).__init__()
        # self.pred_len = 100
        # self.seq_len = 25
        # self.label_len = 25
        # self.output_attention = configs.output_attention
        self.d_model = 512
        self.dropout = 0.05
        self.n_heads = 8
        self.d_ff = 2048
        self.activation = 'gelu'
        self.e_layers = 2
        self.d_layers = 1
        self.c_out = 48
        self.enc_in = 48
        self.seq_len = 20
        self.p_hidden_dims = [64, 64]
        self.p_hidden_layers = 2
        # Embedding
        self.enc_embedding = DataEmbedding(self.enc_in, self.d_model,
                                           self.dropout)
        self.dec_embedding = DataEmbedding(self.c_out, self.d_model,
                                           self.dropout)
        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        DSAttention(attention_dropout=self.dropout,
                                      output_attention=False), self.d_model, self.n_heads),
                    self.d_model,
                    self.d_ff,
                    dropout=self.dropout,
                    activation=self.activation
                ) for l in range(self.e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(self.d_model)
        )
        # Decoder
        self.decoder = Decoder(
            [
                DecoderLayer(
                    AttentionLayer(
                        DSAttention(attention_dropout=self.dropout, output_attention=False),
                        self.d_model, self.n_heads),
                    AttentionLayer(
                        DSAttention(attention_dropout=self.dropout, output_attention=False),
                        self.d_model, self.n_heads),
                    self.d_model,
                    self.d_ff,
                    dropout=self.dropout,
                    activation=self.activation,
                )
                for l in range(self.d_layers)
            ],
            norm_layer=torch.nn.LayerNorm(self.d_model),
            projection=nn.Linear(self.d_model, self.c_out, bias=True)
        )

        # self.tau_learner   = Projector(enc_in=configs.enc_in, seq_len=configs.seq_len, hidden_dims=configs.p_hidden_dims, hidden_layers=configs.p_hidden_layers, output_dim=1)
        self.delta_learner = Projector(enc_in=self.enc_in, seq_len=self.seq_len, hidden_dims=self.p_hidden_dims, hidden_layers=self.p_hidden_layers, output_dim=self.seq_len)

    def forward(self, x_enc, x_dec):
        #x_enc: history B*25*48  x_dec: B*125*48

        enc_self_mask = None
        dec_self_mask = None
        dec_enc_mask = None
        x_raw = x_enc.clone().detach()#history B*25*48

        # Normalization
        mean_enc = x_enc.mean(1, keepdim=True).detach() # B x 1 x E
        x_enc = x_enc - mean_enc
        # std_enc = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5).detach() # B x 1 x E
        # x_enc = x_enc / std_enc
        #减均值，后面全补0
        x_dec_new = x_dec - mean_enc

        # tau = self.tau_learner(x_raw, std_enc).exp()     # B x S x E, B x 1 x E -> B x 1, positive scalar
        tau = None #标准差不需要
        # print('xxxxxxxxxxxxxxxxx', x_raw.size())
        # print('xxxxxxxxxxxxxxxxx', mean_enc.size())
        delta = self.delta_learner(x_raw, mean_enc)      # B x S x E, B x 1 x E -> B x S

        # Model Inference
        enc_out = self.enc_embedding(x_enc)
        enc_out, attns = self.encoder(enc_out, attn_mask=enc_self_mask, tau=tau, delta=delta)

        dec_out = self.dec_embedding(x_dec_new)
        dec_out = self.decoder(dec_out, enc_out, x_mask=dec_self_mask, cross_mask=dec_enc_mask, tau=tau, delta=delta)

        # De-normalization
        dec_out = dec_out + mean_enc

        # if self.output_attention:
        #     return dec_out[:, -self.pred_len:, :], attns
        # else:
        return dec_out  # [B, L, D]
