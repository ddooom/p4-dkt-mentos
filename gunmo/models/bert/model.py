import torch
import torch.nn as nn
from transformers import BertConfig, BertModel


class EmbeddingLayer(nn.Module):
    def __init__(self, args, hidden_dim):
        super(EmbeddingLayer, self).__init__()

        self.args = args
        self.device = args.device
        self.hidden_dim = hidden_dim

        labels_dim = self.hidden_dim // (len(self.args.n_embeddings) + 1)
        interaction_dim = self.hidden_dim - (labels_dim * len(self.args.n_embeddings))

        self.embedding_interaction = nn.Embedding(3, interaction_dim)
        self.embeddings = nn.ModuleDict(
            {k: nn.Embedding(v + 1, labels_dim) for k, v in self.args.n_embeddings.items()}  # plus 1 for padding
        )

    def forward(self, batch):
        embed_interaction = self.embedding_interaction(batch["interaction"])
        embed = torch.cat(
            [embed_interaction] + [self.embeddings[k](batch[k]) for k in self.args.n_embeddings.keys()], 2
        )
        return embed


class LinearLayer(nn.Module):
    def __init__(self, args, hidden_dim):
        super(LinearLayer, self).__init__()

        self.args = args
        self.device = args.device

        self.hidden_dim = hidden_dim
        in_features = len(self.args.n_linears)
        self.fc_layer = nn.Linear(in_features, self.hidden_dim)

    def forward(self, batch):
        cont_v = torch.stack([batch[k] for k in self.args.n_linears]).permute(1, 2, 0)
        output = self.fc_layer(cont_v)
        return output


class Bert(nn.Module):
    def __init__(self, args):
        super(Bert, self).__init__()
        self.args = args
        self.device = args.device

        # Defining some parameters
        self.hidden_dim = self.args.hidden_dim
        self.n_layers = self.args.n_layers

        # Embedding
        # interaction은 현재 correct으로 구성되어있다. correct(1, 2) + padding(0)
        self.embedding_interaction = nn.Embedding(3, self.hidden_dim // 3)
        self.embedding_test = nn.Embedding(self.args.n_test + 1, self.hidden_dim // 3)
        self.embedding_question = nn.Embedding(self.args.n_questions + 1, self.hidden_dim // 3)
        self.embedding_tag = nn.Embedding(self.args.n_tag + 1, self.hidden_dim // 3)

        # embedding combination projection
        self.comb_proj = nn.Linear((self.hidden_dim // 3) * 4, self.hidden_dim)

        # Bert config
        self.config = BertConfig(
            3,  # not used
            hidden_size=self.hidden_dim,
            num_hidden_layers=self.args.n_layers,
            num_attention_heads=self.args.n_heads,
            max_position_embeddings=self.args.max_seq_len,
        )

        # Defining the layers
        # Bert Layer
        self.encoder = BertModel(self.config)

        # Fully connected layer
        self.fc = nn.Linear(self.args.hidden_dim, 1)

        self.activation = nn.Sigmoid()

    def forward(self, input):
        test, question, tag, _, mask, interaction, _ = input
        batch_size = interaction.size(0)

        # 신나는 embedding

        embed_interaction = self.embedding_interaction(interaction)
        embed_test = self.embedding_test(test)
        embed_question = self.embedding_question(question)
        embed_tag = self.embedding_tag(tag)

        embed = torch.cat([embed_interaction, embed_test, embed_question, embed_tag], 2)

        X = self.comb_proj(embed)

        # Bert
        encoded_layers = self.encoder(inputs_embeds=X, attention_mask=mask)
        out = encoded_layers[0]
        out = out.contiguous().view(batch_size, -1, self.hidden_dim)
        out = self.fc(out)
        preds = self.activation(out).view(batch_size, -1)

        return preds
