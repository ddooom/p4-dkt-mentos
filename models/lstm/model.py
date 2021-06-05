import torch
import torch.nn as nn

import torch.nn as nn

class EmbeddingLayer(nn.Module):
    def __init__(self, args, hidden_dim):
        super(EmbeddingLayer, self).__init__()
        
        self.args = args
        self.device = args.device
        self.hidden_dim = hidden_dim
        
        labels_dim = self.hidden_dim // (len(args.n_embeddings) + 1)
        interaction_dim = self.hidden_dim - (labels_dim * len(args.n_embeddings)) 
        
        self.embedding_interaction = nn.Embedding(3, interaction_dim)
        self.embeddings = nn.ModuleDict({
            k: nn.Embedding(v + 1, labels_dim)  # plus 1 for padding
            for k, v in args.n_embeddings.items()
        })
        
    def forward(self, batch):
        embed_interaction = self.embedding_interaction(batch['interaction'])
        embed = torch.cat([embed_interaction] + [self.embeddings[k](batch[k]) for k in args.n_embeddings.keys()], 2)
        return embed
        

class LSTM(nn.Module):
    def __init__(self, args):
        super(LSTM, self).__init__()
        self.args = args
        self.device = args.device

        self.hidden_dim = self.args.hidden_dim
        self.n_layers = self.args.n_layers
        
        self.emb_layer = EmbeddingLayer(args, self.hidden_dim)
        self.comb_proj = nn.Linear(self.hidden_dim, self.hidden_dim)

        self.lstm = nn.LSTM(self.hidden_dim, self.hidden_dim, self.n_layers, batch_first=True)
        
        # Fully connected layer
        self.fc = nn.Linear(self.hidden_dim, 1)
        self.activation = nn.Sigmoid()

    def init_hidden(self, batch_size):
        h = torch.zeros(self.n_layers, batch_size, self.hidden_dim)
        h = h.to(self.device)

        c = torch.zeros(self.n_layers, batch_size, self.hidden_dim)
        c = c.to(self.device)

        return (h, c)

    def forward(self, batch):
        batch_size = batch['interaction'].size(0)                           
        embed = self.emb_layer(batch)
                                   
        X = self.comb_proj(embed)

        hidden = self.init_hidden(batch_size)
        out, hidden = self.lstm(X, hidden)
        out = out.contiguous().view(batch_size, -1, self.hidden_dim)

        out = self.fc(out)
        preds = self.activation(out).view(batch_size, -1)

        return preds
