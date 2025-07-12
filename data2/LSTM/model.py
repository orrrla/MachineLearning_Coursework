import torch
from torch import nn
import math
import torch.nn.functional as F



class BiLSTM_Deep(nn.Module):
    def __init__(self, input_size=7, hidden_size=256, output_len=2160, num_layers=3, dropout=0.3):
        super().__init__()

        self.bilstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True,
            bidirectional=True
        )

        self.dropout = nn.Dropout(dropout)

        # 中间层提升非线性建模能力
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, output_len)
        )

    def forward(self, x):
        out, _ = self.bilstm(x)          # out: [B, T, hidden*2]
        out = self.dropout(out)
        out = out[:, -1, :]              # 只取最后一个时间步
        out = self.mlp(out)              # 输出: [B, output_len]
        return out


class BiLSTM_MultiOutput(nn.Module):
    def __init__(self, input_size=7, hidden_size=128, num_layers=2, dropout=0.3, output_dim=7):
        super().__init__()
        self.bilstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True,
            bidirectional=True
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size * 2, output_dim)  # 每个时间步输出 7 个特征

    def forward(self, x):
        out, _ = self.bilstm(x)         # [B, 2160, hidden*2]
        out = self.dropout(out)
        out = self.fc(out)              # [B, 2160, 7]
        return out

class BiLSTM_Forcast_MultiFeature(nn.Module):
    """
    这个模型接收一个历史序列，并预测紧接着的下一个时间步的所有特征。
    输入: [B, history_len, num_features]  (例如: [B, 2160, 7])
    输出: [B, num_features]             (例如: [B, 7]) -> 代表第2161个时间步的预测值
    """
    def __init__(self, input_size=7, hidden_size=128, num_layers=2, dropout=0.3):
        super().__init__()
        # input_size 就是你的特征数量
        self.feature_size = input_size 
        
        self.bilstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            batch_first=True,
            bidirectional=True
        )
        self.dropout = nn.Dropout(dropout)
        
        # 关键改动：FC层将总结向量映射到下一个时间步的7个特征
        self.fc = nn.Linear(hidden_size * 2, self.feature_size)

    def forward(self, x):
        # x shape: [B, history_len, 7]
        
        # BiLSTM编码整个历史序列
        out, _ = self.bilstm(x)         # out shape: [B, history_len, hidden_size * 2]
        
        # 我们只取最后一个时间步的输出作为对整个历史的总结
        summary = out[:, -1, :]         # summary shape: [B, hidden_size * 2]
        
        summary = self.dropout(summary)
        
        # 用这个总结来预测下一个时间步的7个特征
        prediction = self.fc(summary)   # prediction shape: [B, 7]
        
        return prediction


class BiLSTM_FullOutput(nn.Module):
    def __init__(self, input_size=7, hidden_size=128, output_len=2160, num_layers=2, dropout=0.3):
        super().__init__()
        self.bilstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            batch_first=True,
            bidirectional=True
        )
        self.fc = nn.Linear(hidden_size * 2, 1)  # 每个时间步输出一个值

    def forward(self, x):
        out, _ = self.bilstm(x)        # out: [B, 2160, hidden×2]
        out = self.fc(out).squeeze(-1) # shape: (B, 2160)
        return out




class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_seq_length=5000):
        super().__init__()
        position = torch.arange(max_seq_length).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_seq_length, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(0)]

class TransformerModel(nn.Module):
    def __init__(self, input_dim, d_model, nhead, num_layers, output_seq_len,dropout):
        super().__init__()
        
        self.input_embedding = nn.Linear(input_dim, d_model)
        self.positional_encoding = PositionalEncoding(d_model)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 2,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.output_layer = nn.Linear(96*d_model, output_seq_len)

    def forward(self, src, src_mask=None):
        # src shape: [batch_size, seq_len, features]
        
        # Embedding and positional encoding
        x = self.input_embedding(src)
        x = self.positional_encoding(x)
        
        # Transformer encoder
        # if src_mask is None:
        #     src_mask = self._generate_square_subsequent_mask(src.size(1))
        
        output = self.transformer_encoder(x, src_mask)
        
        # Predict next values
        predictions = self.output_layer(output.reshape(output.size()[0],-1))
        return predictions  # Return only the predicted sequence


class TransformerALL(nn.Module):
    def __init__(self, input_dim, d_model, nhead, num_layers, output_seq_len,dropout):
        super(TransformerALL, self).__init__()
        # 输入嵌入层
        self.embedding = nn.Linear(input_dim, d_model)
        # 位置编码
        self.positional_encoding = PositionalEncoding(d_model)
        # Transformer 编码器
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dropout=dropout)
        self.encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)
        # Transformer 解码器
        self.decoder_layer = nn.TransformerDecoderLayer(d_model=d_model, nhead=nhead, dropout=dropout)
        self.decoder = nn.TransformerDecoder(self.decoder_layer, num_layers=num_layers)
        # 输出层
        self.fc_out = nn.Linear(96*d_model, output_seq_len)

    def forward(self, src, tgt):
        # 输入嵌入和位置编码
        src = self.embedding(src)
        src=self.positional_encoding(src)
        tgt = self.embedding(tgt)
        tgt= self.positional_encoding(tgt)
        # 编码器提取特征
        memory = self.encoder(src)
        # 解码器生成目标序列
        output = self.decoder(tgt, memory)
        # 输出层映射到目标特征空间
        return self.fc_out(output.reshape(output.size(0),-1))


class Attention(nn.Module):
    def __init__(self, hidden_dim):
        super(Attention, self).__init__()
        self.hidden_dim = hidden_dim

        # 可学习的注意力参数
        self.attention_weights = nn.Linear(hidden_dim, hidden_dim)
        self.score = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, out, hidden):
        """
        Args:
            out: LSTM 的所有时间步输出 (batch_size, seq_len, hidden_dim)
            hidden: LSTM 的隐藏状态 (num_layers, batch_size, hidden_dim)
        Returns:
            context: 注意力加权后的上下文向量 (batch_size, hidden_dim)
            attention_weights: 注意力权重 (batch_size, seq_len)
        """
        # 取最后一层的隐藏状态
        hidden_last_layer = hidden[-1]  # (batch_size, hidden_dim)

        # 扩展 hidden 的维度以匹配 out 的时间步维度
        hidden_last_layer = hidden_last_layer.unsqueeze(1)  # (batch_size, 1, hidden_dim)

        # 计算注意力得分
        energy = torch.tanh(self.attention_weights(out) + hidden_last_layer)  # (batch_size, seq_len, hidden_dim)
        attention_scores = self.score(energy).squeeze(-1)  # (batch_size, seq_len)

        # 归一化得分
        attention_weights = torch.softmax(attention_scores, dim=-1)  # (batch_size, seq_len)

        # 计算上下文向量
        context = torch.bmm(attention_weights.unsqueeze(1), out).squeeze(1)  # (batch_size, hidden_dim)

        return context, attention_weights


class LSTM_with_Attention(nn.Module):
    def __init__(self,input_size, hidden_size, output_size, num_layers,dropout):
        super().__init__()
        # self.embedding = nn.Embedding(input_size, embedding_dim)
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True,dropout=dropout)
        self.fc = nn.Linear(hidden_size, output_size)
        self.attentions=Attention(hidden_size)    

    def forward(self, x):
        output, (hidden, cell) = self.lstm(x)

        # attn_output = self.attention_net(output, hidden)
        # attn_output = self.attention(output, hidden)
        attn_output,_ = self.attentions(output, hidden)

        return self.fc(attn_output)
    

class CNN_LSTM_Attention(nn.Module):
    def __init__(self,input_size, hidden_size, output_size, num_layers,dropout):
        super().__init__()
        # self.embedding = nn.Embedding(input_size, embedding_dim)
        self.cnn = nn.Conv1d(input_size, out_channels=32, kernel_size=3, padding=3 // 2)
        self.lstm = nn.LSTM(32, hidden_size, num_layers, batch_first=True,dropout=dropout)
        self.fc = nn.Linear(hidden_size, output_size)
        self.attentions=Attention(hidden_size)    

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = torch.relu(self.cnn(x))  # 经过卷积层
        x = x.permute(0, 2, 1)

        output, (hidden, cell) = self.lstm(x)

        # attn_output = self.attention_net(output, hidden)
        # attn_output = self.attention(output, hidden)
        attn_output,_ = self.attentions(output, hidden)

        return self.fc(attn_output)


class CNN_LSTM(nn.Module):
    def __init__(self, conv_input,input_size, hidden_size, num_layers, output_size):
        super(CNN_LSTM,self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.conv=nn.Conv1d(conv_input,conv_input,1)
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        x=self.conv(x)
        h0 = torch.zeros(self.num_layers,x.size(0), self.hidden_size) # 初始化隐藏状态h0
        c0 = torch.zeros(self.num_layers,x.size(0), self.hidden_size)  # 初始化记忆状态c0
        #print(f"x.shape:{x.shape},h0.shape:{h0.shape},c0.shape:{c0.shape}")
        out, _ = self.lstm(x, (h0, c0))  # LSTM前向传播
        out = self.fc(out[:, -1, :])  # 取最后一个时间步的输出作为预测结果
        return out