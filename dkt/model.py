import torch
import torch.nn as nn
import torch.nn.functional as F 
import numpy as np
import copy
import math

try:
    from transformers.modeling_bert import BertConfig, BertEncoder, BertModel    
except:
    from transformers.models.bert.modeling_bert import BertConfig, BertEncoder, BertModel    


class LSTM(nn.Module):

    def __init__(self, args):
        super(LSTM, self).__init__()
        self.args = args
        self.device = args.device

        self.args.hidden_dim = self.args.hidden_dim
        self.n_layers = self.args.n_layers

        # Embedding 
        # interaction은 현재 correct로 구성되어있다. correct(1, 2) + padding(0)
        self.embedding_interaction = nn.Embedding(3, self.args.hidden_dim)
        self.embedding_test = nn.Embedding(self.args.n_test + 1, self.args.hidden_dim)
        self.embedding_question = nn.Embedding(self.args.n_questions + 1, self.args.hidden_dim)
        self.embedding_tag = nn.Embedding(self.args.n_tag + 1, self.args.hidden_dim)

        # embedding combination projection
        self.comb_proj = nn.Linear((self.args.hidden_dim)*4, self.args.hidden_dim)

        self.lstm = nn.LSTM(self.args.hidden_dim,
                            self.args.hidden_dim,
                            self.n_layers,
                            batch_first=True)
        
        # Fully connected layer
        self.fc = nn.Linear(self.args.hidden_dim, 1)

        self.activation = nn.Sigmoid()

    def init_hidden(self, batch_size):
        h = torch.zeros(
            self.n_layers,
            batch_size,
            self.args.hidden_dim)
        h = h.to(self.device)

        c = torch.zeros(
            self.n_layers,
            batch_size,
            self.args.hidden_dim)
        c = c.to(self.device)

        return (h, c)

    def forward(self, input):

        test, question, tag, _, mask, interaction, _ = input

        batch_size = interaction.size(0)

        # Embedding

        embed_interaction = self.embedding_interaction(interaction)
        embed_test = self.embedding_test(test)
        embed_question = self.embedding_question(question)
        embed_tag = self.embedding_tag(tag)
        

        embed = torch.cat([embed_interaction,
                           embed_test,
                           embed_question,
                           embed_tag,], 2)

        X = self.comb_proj(embed)

        hidden = self.init_hidden(batch_size)
        out, hidden = self.lstm(X, hidden)
        out = out.contiguous().view(batch_size, -1, self.args.hidden_dim)

        out = self.fc(out)
        preds = self.activation(out).view(batch_size, -1)

        return preds


class LSTMATTN(nn.Module):
    
    def __init__(self, args):
        super(LSTMATTN, self).__init__()
        self.args = args
        self.device = args.device

        self.args.hidden_dim = self.args.hidden_dim
        self.n_layers = self.args.n_layers
        self.n_heads = self.args.n_heads
        self.drop_out = self.args.drop_out

        # Embedding 
        # interaction은 현재 correct로 구성되어있다. correct(1, 2) + padding(0)
        self.embedding_interaction = nn.Embedding(3, self.args.hidden_dim)
        self.embedding_test = nn.Embedding(self.args.n_test + 1, self.args.hidden_dim)
        self.embedding_question = nn.Embedding(self.args.n_questions + 1, self.args.hidden_dim)
        self.embedding_tag = nn.Embedding(self.args.n_tag + 1, self.args.hidden_dim)

        # embedding combination projection
        self.comb_proj = nn.Linear((self.args.hidden_dim)*4, self.args.hidden_dim)

        self.lstm = nn.LSTM(self.args.hidden_dim,
                            self.args.hidden_dim,
                            self.n_layers,
                            batch_first=True)
        
        self.config = BertConfig( 
            3, # not used
            hidden_size=self.args.hidden_dim,
            num_hidden_layers=1,
            num_attention_heads=self.n_heads,
            intermediate_size=self.args.hidden_dim,
            hidden_dropout_prob=self.drop_out,
            attention_probs_dropout_prob=self.drop_out,
        )
        self.attn = BertEncoder(self.config)            
    
        # Fully connected layer
        self.fc = nn.Linear(self.args.hidden_dim, 1)

        self.activation = nn.Sigmoid()

    def init_hidden(self, batch_size):
        h = torch.zeros(
            self.n_layers,
            batch_size,
            self.args.hidden_dim)
        h = h.to(self.device)

        c = torch.zeros(
            self.n_layers,
            batch_size,
            self.args.hidden_dim)
        c = c.to(self.device)

        return (h, c)

    def forward(self, input):

        test, question, tag, _, mask, interaction, _ = input

        batch_size = interaction.size(0)

        # Embedding

        embed_interaction = self.embedding_interaction(interaction)
        embed_test = self.embedding_test(test)
        embed_question = self.embedding_question(question)
        embed_tag = self.embedding_tag(tag)
        

        embed = torch.cat([embed_interaction,
                           embed_test,
                           embed_question,
                           embed_tag,], 2)

        X = self.comb_proj(embed)

        hidden = self.init_hidden(batch_size)
        out, hidden = self.lstm(X, hidden)
        out = out.contiguous().view(batch_size, -1, self.args.hidden_dim)
                
        extended_attention_mask = mask.unsqueeze(1).unsqueeze(2)
        extended_attention_mask = extended_attention_mask.to(dtype=torch.float32)
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        head_mask = [None] * self.n_layers
        
        encoded_layers = self.attn(out, extended_attention_mask, head_mask=head_mask)        
        sequence_output = encoded_layers[-1]
        
        out = self.fc(sequence_output)

        preds = self.activation(out).view(batch_size, -1)

        return preds

# feature 추가 #1~6 수정
class Bert(nn.Module):
    
    def __init__(self, args):
        super(Bert, self).__init__()
        self.args = args
        cate_size = len(self.args.cate_cols) + 1 # interaction
        cont_size = len(self.args.cont_cols)

        #1 Embedding 
        # interaction은 현재 correct으로 구성되어있다. correct(1, 2) + padding(0)
        self.embedding_interaction = nn.Embedding(3, self.args.hidden_dim)
        self.embedding_test = nn.Embedding(self.args.n_cols['testId'] + 1, self.args.hidden_dim)
        self.embedding_question = nn.Embedding(self.args.n_cols['assessmentItemID'] + 1, self.args.hidden_dim)
        self.embedding_tag = nn.Embedding(self.args.n_cols['KnowledgeTag'] + 1, self.args.hidden_dim)


        # embedding projection
        self.cate_proj = nn.Sequential(
            nn.Linear((self.args.hidden_dim) * cate_size, self.args.hidden_dim),
            nn.LayerNorm(self.args.hidden_dim),
        )

        self.cont_bn = nn.BatchNorm1d(cont_size)
        self.cont_proj = nn.Sequential(
            nn.Linear(cont_size, self.args.hidden_dim),
            nn.LayerNorm(self.args.hidden_dim),
        )

        self.comb_proj = nn.Sequential(
            nn.ReLU(),
            nn.Linear(self.args.hidden_dim * 2, self.args.hidden_dim),
            nn.LayerNorm(self.args.hidden_dim),
        )


        # Bert config
        self.config = BertConfig( 
            3, # not used
            hidden_size=self.args.hidden_dim,
            num_hidden_layers=self.args.n_layers,
            num_attention_heads=self.args.n_heads,
            intermediate_size=self.args.hidden_dim,
            hidden_dropout_prob=self.args.drop_out,
            attention_probs_dropout_prob=self.args.drop_out,
            max_position_embeddings=self.args.max_seq_len           
        )

        # Defining the layers
        # Bert Layer
        self.encoder = BertModel(self.config)  

        # Fully connected layer
        self.fc = nn.Linear(self.args.hidden_dim, 1)
       
        self.activation = nn.Sigmoid()

        def get_reg():
            return nn.Sequential(
            nn.Linear(self.args.hidden_dim, self.args.hidden_dim),
            nn.LayerNorm(self.args.hidden_dim),
            nn.Dropout(self.args.drop_out),
            nn.ReLU(),            
            nn.Linear(self.args.hidden_dim, self.args.hidden_dim),
        )     

        self.reg_layer = get_reg()


    def forward(self, inputs):
        #2 process_batch의 return
        test, question, tag, tail_prob, mask, interaction, gather_index, correct = inputs
        batch_size = interaction.size(0)

        #3 신나는 embedding
        embed_interaction = self.embedding_interaction(interaction)
        embed_test = self.embedding_test(test)
        embed_question = self.embedding_question(question)
        embed_tag = self.embedding_tag(tag)

        #4
        # (batch_size, max_seq_len, hidden_dim * n) : (64, 20, 64 * n) 
        cate_embed = torch.cat([
                    embed_interaction,
                    embed_tag,
                    embed_question,
                    
                    embed_test,
                ], 2)

        #5
        # (batch_size, max_seq_len * n) : (64, 20 * n) 
        cont_cat = torch.cat([
                tail_prob,            
        ], 1)
        
        # (batch_size, max_seq_len, n) : (64, 20, n) 
        cont_cat = cont_cat.view(batch_size, self.args.max_seq_len, -1)

        # (batch_size * max_seq_len, n) : (1280, 1)
        cont_bn_x = self.cont_bn(cont_cat.view(-1, cont_cat.size(-1)))

        # (batch_size, max_seq_len, n) : (64, 20, n) 
        cont_bn_x = cont_bn_x.view(batch_size, self.args.max_seq_len, -1) 

        # (batch_size, max_seq_len, hidden_dim) : (64, 20, 64)
        cate_X = self.cate_proj(cate_embed)
        cont_X = self.cont_proj(cont_cat)

        # (batch_size, max_seq_len, hidden_dim * 2) : [64, 20, 128]
        X = torch.cat([cate_X, cont_X], 2)

        # (batch_size, max_seq_len, hidden_dim) :[64, 20, 64]
        comb_X = self.comb_proj(X)

        # Bert
        encoded_layers = self.encoder(inputs_embeds=comb_X, attention_mask=mask)
        out = encoded_layers[0]
        
        
        # base
        # out = out.contiguous().view(batch_size, -1, self.args.hidden_dim)
        # out = self.fc(out)
        # preds = self.activation(out).view(batch_size, -1)

        # reg_layer
        out = self.reg_layer(out)
        out = self.fc(out)
        preds = self.activation(out).view(batch_size, -1)

        return preds