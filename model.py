import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
import math 
import numpy as np


from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree
from torch_geometric.nn import GCNConv, SAGEConv,TransformerConv,GATConv


def get_emb(sin_inp):
    """
    Gets a base embedding for one dimension with sin and cos intertwined
    """
    emb = torch.stack((sin_inp.sin(), sin_inp.cos()), dim=-1)
    return torch.flatten(emb, -2, -1)

class PositionalEncoding1D(nn.Module):
    def __init__(self, channels):
        """
        :param channels: The last dimension of the tensor you want to apply pos emb to.
        """
        super(PositionalEncoding1D, self).__init__()
        self.org_channels = channels
        channels = int(np.ceil(channels / 2) * 2)
        self.channels = channels
        inv_freq = 1.0 / (10000 ** (torch.arange(0, channels, 2).float() / channels))
        self.register_buffer("inv_freq", inv_freq)
        self.cached_penc = None

    def forward(self, tensor):
        """
        :param tensor: A 3d tensor of size (batch_size, x, ch)
        :return: Positional Encoding Matrix of size (batch_size, x, ch)
        """
        if len(tensor.shape) != 3:
            raise RuntimeError("The input tensor has to be 3d!")

        if self.cached_penc is not None and self.cached_penc.shape == tensor.shape:
            return self.cached_penc

        self.cached_penc = None
        batch_size, x, orig_ch = tensor.shape
        pos_x = torch.arange(x, device=tensor.device).type(self.inv_freq.type())
        sin_inp_x = torch.einsum("i,j->ij", pos_x, self.inv_freq)
        emb_x = get_emb(sin_inp_x)
        emb = torch.zeros((x, self.channels), device=tensor.device).type(tensor.type())
        emb[:, : self.channels] = emb_x

        self.cached_penc = emb[None, :, :orig_ch].repeat(batch_size, 1, 1)
        return self.cached_penc
        
class TransformerModel(nn.Module):
    def __init__(self, output_dim, d_input , d_model, nhead, num_layers, dropout=0.1):
        super(TransformerModel, self).__init__()
        self.d_model = d_model
        self.pose_embed = nn.Linear(d_input, d_model)
        self.tgt_query = nn.Embedding(output_dim+1, d_model)
        self.d_input = d_input

        self.class_token = torch.nn.Parameter(
            torch.randn(1, 1, self.d_input)
        )

        # self.pos_encoder = PositionalEncoding(d_model, dropout)

        # LAYERS
        self.positional_encoder = PositionalEncoding1D(d_model)

        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dropout=dropout,batch_first=True), 
            num_layers=num_layers)
        self.transformer_decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(d_model=d_model, nhead=nhead, dropout=dropout,batch_first=True), 
            num_layers=num_layers)

        self.fc = nn.Sequential(
            nn.Linear(d_model, 128),
            nn.Dropout(0.1),
            nn.Linear(128, output_dim)
        )
        self.fc_enc = nn.Sequential(
            nn.Linear(d_model, 128),
            nn.Dropout(0.1),
            nn.Linear(128, output_dim)
        )
        
        # self.fc_cls = nn.Sequential(
        #     nn.Linear(d_model, 128),
        #     nn.Dropout(0.1),
        #     nn.Linear(128, 2)
        # )
        self.fc_cls = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, 2))


        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_layers,
            num_decoder_layers=num_layers,
            dropout=dropout,
            batch_first=True,
        )



    def forward(self, poses,tgt=None):
        # Embedding layer for poses
        # poses = poses.view(1,-1,63)
        # pose_embedded = self.pose_embed(poses)
        # print(poses.shape)
        bs = poses.shape[0]

        poses = poses.view(bs,-1,self.d_input)

        poses = torch.cat([self.class_token, poses], dim=1)

        pose_embedded = self.pose_embed(poses)
        pos_embd = self.positional_encoder(pose_embedded)
        # print(self.tgt_query.shape)
        # Positional encoding
        batch_size , pose_len, pose_dim = pose_embedded.size()
        # pos = torch.arange(0, pose_len).unsqueeze(1).repeat(1, batch_size).to(pose_embedded.device)
        
        # # print(pos.shape)
        # pos_embedded = self.pos_embed(pos).permute(1, 0, 2) 
        src = pose_embedded + pos_embd.to(pose_embedded.device)
        
        # transformer_out = self.transformer(src, tgt)
        # Transformer encoder

        encoder = self.transformer_encoder(src)  # B x L x D
        encoder_out = self.fc_enc(encoder[:,1:,:])  # L x B x V
        cls_out = self.fc_cls(encoder[:,0,:])  # L x B x V


        if tgt!=None:
            tgt = tgt.reshape(bs,-1,1)
            pos_tgt = self.positional_encoder(tgt)
            tgt = self.tgt_query(tgt).reshape(bs,-1,self.d_model)
            tgt = tgt + pos_tgt.to(pose_embedded.device)

            tgt_mask = self.get_tgt_mask(tgt.size(1)).to(pose_embedded.device)
            decoder = self.transformer_decoder(tgt, encoder, tgt_mask=tgt_mask)
            logits = self.fc(decoder)  # L x B x V
            return  cls_out ,logits, encoder_out  # B x L x V

        return cls_out, encoder_out  # B x L x V
    
    @torch.no_grad()
    def return_scores(self, poses, strings, vocab_map):
        scores = []
        bs = poses.shape[0]
        poses = poses.view(bs,-1,self.d_input)

        poses = torch.cat([self.class_token, poses], dim=1)

        pose_embedded = self.pose_embed(poses)
        pos_embd = self.positional_encoder(pose_embedded)
        # print(self.tgt_query.shape)
        # Positional encoding
        batch_size , pose_len, pose_dim = pose_embedded.size()
        # pos = torch.arange(0, pose_len).unsqueeze(1).repeat(1, batch_size).to(pose_embedded.device)
        
        # # print(pos.shape)
        # pos_embedded = self.pos_embed(pos).permute(1, 0, 2) 
        src = pose_embedded + pos_embd.to(pose_embedded.device)
        
        # transformer_out = self.transformer(src, tgt)
        # Transformer encoder

        encoder = self.transformer_encoder(src)  # B x L x D
        encoder_out = self.fc_enc(encoder[:,1:,:])  # L x B x V
        cls_out = self.fc_cls(encoder[:,0,:])  # L x B x V

        for string in strings:
            if len(string)<1 :
                scores.append(0)
                continue
            criterion = nn.CrossEntropyLoss()
            tgt = torch.tensor([[32]], dtype=torch.long, device=poses.device)
            input = [32] + list(map(lambda x: vocab_map[x], string)) 
            grt =  torch.tensor(input[1:], dtype=torch.long, device=poses.device)
            tgt = torch.tensor(input[:-1], dtype=torch.long, device=poses.device)
            

            tgt = tgt.reshape(bs,-1,1)
            pos_tgt = self.positional_encoder(tgt)
            tgt = self.tgt_query(tgt).reshape(bs,-1,self.d_model)
            tgt = tgt + pos_tgt.to(pose_embedded.device)

            tgt_mask = self.get_tgt_mask(tgt.size(1)).to(pose_embedded.device)
            decoder = self.transformer_decoder(tgt, encoder, tgt_mask=tgt_mask)
            logits = self.fc(decoder)  # L x B x V
            loss = criterion(logits[0], grt)
            scores.append(-(len(input)-1)*loss.item())
        return scores


    def create_pad_mask(self, matrix: torch.tensor, pad_token: int):
        # If matrix = [1,2,3,0,0,0] where pad_token=0, the result mask is
        # [False, False, False, True, True, True]
        return (matrix == pad_token)

    def get_tgt_mask(self, size):
        # Generates a squeare matrix where the each row allows one word more to be seen
        mask = torch.tril(torch.ones(size, size) == 1) # Lower triangular matrix
        mask = mask.float()
        mask = mask.masked_fill(mask == 0, float('-inf')) # Convert zeros to -inf
        mask = mask.masked_fill(mask == 1, float(0.0)) # Convert ones to 0
        
        # EX for size=5:
        # [[0., -inf, -inf, -inf, -inf],
        #  [0.,   0., -inf, -inf, -inf],
        #  [0.,   0.,   0., -inf, -inf],
        #  [0.,   0.,   0.,   0., -inf],
        #  [0.,   0.,   0.,   0.,   0.]]
        return mask

def collate_fn(batch):
    # Pad sequences to the same length
    # Not used right now 

    poses, labels = zip(*batch)
    poses = pad_sequence(poses, batch_first=True)
    labels = pad_sequence(labels, batch_first=True)
    return poses, labels




class GCN(torch.nn.Module):
    def __init__(self, input_chanels, hidden_channels1, hidden_channels2, hidden_channels3, output_channels):
        super().__init__( )
        # torch.manual_seed(1234567)
        self.pose_embed = nn.Linear(input_chanels, hidden_channels1)
        self.positional_encoder = PositionalEncoding1D(hidden_channels1)

        self.transformer1 = TransformerConv(hidden_channels1, hidden_channels2, heads=8, dropout=0.1)
        self.head_transformer1 =nn.Linear(hidden_channels1*8, hidden_channels1)
        self.transformer2 = TransformerConv(hidden_channels1, hidden_channels2, heads=8, dropout=0.1)
        self.head_transformer2 =nn.Linear(hidden_channels1*8, hidden_channels1)

        self.conv3 = GCNConv(hidden_channels2, hidden_channels3)
        self.conv4 = GCNConv(hidden_channels3, output_channels)

        self.non_linearity = nn.ELU()
        self.activation = torch.nn.Sigmoid()

    def forward(self, data):
        # print(f'Inside model - num graphs: {data.num_graphs},', f'device: {data.batch.device}')
        x, edge_index = data.x, data.edge_index
        # import pdb; pdb.set_trace()

        x = x.view(-1,63)
        x = self.pose_embed(x)
        
        pos_embd = self.positional_encoder(x.unsqueeze(dim=0))
        x = x + pos_embd.to(x.device)[0]
        
        x = self.transformer1(x,edge_index)
        x = self.head_transformer1(x)
        x = self.transformer2(x,edge_index)
        x = self.head_transformer2(x)

        x = self.conv3(x,edge_index)
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv4(x,edge_index)
        # import pdb; pdb.set_trace()

        return x.unsqueeze(0)



    