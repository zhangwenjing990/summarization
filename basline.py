
import torch
import torch.nn as nn
import os
import json
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
import glob
import tqdm
import math


MIN_FREQUENCY = 5
BATCH_SIZE = 32*8
EMBEDDING_DIM = 150
HIDDEN_SIZE = 128
NUM_HEADS = 8
HEAD_SIZE = 16
EPOCHS = 100
min_count = 32
max_sequence_len = 120

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



def load_vocab(vocab_path, min_count=32):
    if os.path.exists(vocab_path):
        chars, index2char, char2index = json.load(open(vocab_path))
    else:
        data = glob.glob(train_data) + glob.glob(valid_data)
        chars = {}
        for file in tqdm.tqdm(data):
            with open(file, 'r', encoding='utf-8') as f:
                new = f.read()
            for char in new:
                chars[char] = chars.get(char, 0) + 1

        chars = {i: j for i, j in chars.items() if j >= min_count}

        index2char = {'0': '<pad>', '1': '<unknown>', '2': '<start>', '3': '<end>'}
        char2index = {}
        for index, char in enumerate(chars):
            index2char[str(index + 4)] = char
            char2index[char] = index + 4
        json.dump([chars, index2char, char2index], open(vocab_path, 'w'))
        print('词典大小：{}'.format(len(chars)))

    return chars, index2char, char2index


def padding(batch):
    inputs, targets = zip(*batch)
    all_seq_len = [len(i) for i in inputs]
    sorted_seq = sorted(enumerate(all_seq_len), key=lambda x: x[1], reverse=True)
    sorted_inputs = tuple(inputs[i] for i, j in sorted_seq)
    sorted_targets = tuple(targets[i] for i, j in sorted_seq)
    padded_inputs = pad_sequence(sorted_inputs, batch_first=True, padding_value=0)
    padded_targets = pad_sequence(sorted_targets, batch_first=True, padding_value=0)

    return padded_inputs, padded_targets


class CustomDataset(Dataset):
    def __init__(self, data_path, char2index):
        super().__init__()
        self.files = glob.glob(data_path)
        self.char2index = char2index

    def __len__(self):
        return len(self.files)

    def __getitem__(self, item):
        file = self.files[item]
        with open(file, 'r', encoding='utf-8') as f:
            new = f.read()
        fs = new.split("[sep]")
        abstract, content = fs[0].strip(), fs[1].strip()
        inputs_token = [2] + [self.char2index.get(char, 1) for char in content][:max_sequence_len] + [3]
        targets_token = [2] + [self.char2index.get(char, 1) for char in abstract] + [3]

        return torch.tensor(inputs_token), torch.tensor(targets_token)

class MultiHeadAttention(nn.Module):
    def __init__(self, encode_hidden_size, decode_hidden_size, num_attention_heads, attention_head_size):
        super().__init__()
        self.num_attention_heads = num_attention_heads
        self.attention_head_size = attention_head_size
        self.W_Q = nn.Linear(decode_hidden_size, decode_hidden_size)
        self.W_k = nn.Linear(2 * encode_hidden_size, decode_hidden_size)
        self.W_V = nn.Linear(2*encode_hidden_size, decode_hidden_size)

    def transpose_for_scores(self,x):
        new_x_shape=x.size()[:-1]+(self.num_attention_heads,self.attention_head_size)
        x=x.view(*new_x_shape)
        return x.permute(0,2,1,3)

    def forward(self, decode_hidden_states, encode_outputs, attention_mask):
        # query:[batch, seq_dec, dec]
        # values:[batch, seq_enc, 2*enc]


        mask = (attention_mask > 0).unsqueeze(1)

        queries = self.W_Q(decode_hidden_states)
        keys = self.W_k(encode_outputs)
        values = self.W_V(encode_outputs)

        query_layer=self.transpose_for_scores(queries)
        key_layer=self.transpose_for_scores(keys)
        value_layer=self.transpose_for_scores(values)

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        attention_scores = attention_scores.masked_fill(~mask, -1e9)
        attention_probs = nn.Softmax(dim=-1)(attention_scores)
        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape=context_layer.size()[:-2]+(values.size()[-1], )
        context_layer=context_layer.view(*new_context_layer_shape)

        return context_layer

class CustomSeq2Seq(nn.Module):
    def __init__(self, vocab_size, embedding_dim, encode_hidden_size, decode_hidden_size):
        super().__init__()
        self.vocab_size = vocab_size
        self.encode_hidden_size = encode_hidden_size
        self.decode_hidden_size = decode_hidden_size

        self.embed = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.emb_norm = nn.LayerNorm(embedding_dim)
        # self.embed.weight.data[4:].copy_(torch.from_numpy(np.array(embeddings_weight, dtype='float32')))

        self.encoder = nn.GRU(embedding_dim, encode_hidden_size, bidirectional=True, batch_first=True)
        self.encoder_adapter = nn.Linear(2 * encode_hidden_size, decode_hidden_size)

        self.decoder = nn.GRU(embedding_dim, decode_hidden_size, bidirectional=False, batch_first=True)
        self.output = nn.Linear(decode_hidden_size, vocab_size)

        self.attention = MultiHeadAttention(encode_hidden_size, decode_hidden_size, NUM_HEADS, HEAD_SIZE)
        self.fc = nn.Linear(2*decode_hidden_size, decode_hidden_size)

        self.dropout = nn.Dropout(0.2)
        self.encode_layer_norm = nn.LayerNorm(self.encode_hidden_size)
        self.decode_layter_norm = nn.LayerNorm(self.decode_hidden_size)


    def encode(self, inputs):
        # inputs: [batch, seq]
        # encode_hidden_state: [2, batch, enc]
        inputs_length = (inputs>0).sum(-1)
        encode_embed = self.dropout(self.emb_norm(self.embed(inputs)))  # [batch, seq, emb]
        packed_encode_embed = pack_padded_sequence(encode_embed, inputs_length, batch_first=True)
        encode_output, encode_hidden_state = self.encoder(packed_encode_embed)  # [batch, seq, enc*2] | [2, batch, dec]
        encode_output = pad_packed_sequence(encode_output, batch_first=True)[0]
        output_size = encode_output.size()
        encode_output = encode_output.reshape(output_size[:2] + (2, self.encode_hidden_size))
        encode_output = self.encode_layer_norm(encode_output).reshape(output_size)
        encode_hidden_state = encode_hidden_state.permute(1, 0, 2).reshape(encode_hidden_state.size()[1], -1)  # [batch, 2*enc]
        adapted_hidden_state = torch.relu(self.encoder_adapter(encode_hidden_state)).unsqueeze(0)  # [1, batch, dec]

        return encode_output, adapted_hidden_state

    def decode(self, inputs, adapted_hidden_state, encode_output, attention_mask):
        # inputs:[batch, targets_seq]
        # encode_output:[batch, inputs_seq, enc]
        # decode_hidden_state:[1, batch, dec]
        # encoder_index:[batch, inputs_seq]
        decode_embed = self.dropout(self.emb_norm(self.embed(inputs)))
        decode_output, decode_hidden_state = self.decoder(decode_embed, adapted_hidden_state)
        decode_output = self.decode_layter_norm(decode_output)

        attention_vector = self.attention(decode_output, encode_output, attention_mask)
        output = torch.relu(self.fc(torch.cat((decode_output, attention_vector), dim=-1)))
        output = self.output(output)

        return output, decode_hidden_state

    def predicate(self, inputs, decode_hidden_state, encode_output, attention_mask):

        decode_embed = self.dropout(self.emb_norm(self.embed(inputs)))
        decode_output, decode_hidden_state = self.decoder(decode_embed, decode_hidden_state)
        decode_output = self.decode_layter_norm(decode_output)

        attention_vector = self.attention(decode_output, encode_output, attention_mask)
        output = torch.relu(self.fc(torch.cat((decode_output, attention_vector), dim=-1)))
        output = self.output(output)

        p_vocab = torch.softmax(output, dim=-1)

        return p_vocab.squeeze(1), decode_hidden_state

def mask(targets, inputs):
    targets = (targets > 0).unsqueeze(-1).to(dtype=torch.int64)
    inputs = (inputs > 0).unsqueeze(1).to(dtype=torch.int64)
    mask = torch.bmm(targets, inputs)
    return mask

def train(train_data, epochs):
    chars, index2char, char2index = load_vocab('seq2seq_config6.json')

    train_dataset = CustomDataset(train_data, char2index)
    train_data_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, collate_fn=padding, shuffle=True,
                                   num_workers=8, pin_memory=True)
    # train_data_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, collate_fn=padding, drop_last=True)
    print('训练集大小为:{}'.format(len(train_dataset)))

    model = CustomSeq2Seq(len(index2char), EMBEDDING_DIM, HIDDEN_SIZE//2, HIDDEN_SIZE).to(device)
    # model = torch.load('outputs/model1/all_epoch_25')
    loss_function = torch.nn.CrossEntropyLoss(ignore_index=0).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    total_num = sum(p.numel() for p in model.parameters())
    print('参数量:{}'.format(total_num))

    min_count, min_eval_loss = 0, float('inf')

    for epoch in range(0, epochs):

        model.train()
        iteration, epoch_train_loss, epoch_words = 0, 0, 0
        for inputs, targets in tqdm.tqdm(train_data_loader):
            model.train()
            attention_mask = mask(targets[:, 1:], inputs).to(device)
            inputs, targets = inputs.to(device), targets.to(device)
            encode_output, adapted_hidden_state = model.encode(inputs)
            decode_output, decode_hidden_state = model.decode(targets[:, :-1], adapted_hidden_state, encode_output, attention_mask)
            loss = loss_function(decode_output.reshape(-1, decode_output.shape[-1]), targets[:,1:].reshape(-1))

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            optimizer.step()

            iteration += 1
            iteration_words = torch.sum(targets[:, 1:] > 0).item()
            epoch_train_loss += loss.item() * iteration_words
            epoch_words += iteration_words

            if iteration % 2000 == 0:
                print('')
                print('train_loss:{}'.format(epoch_train_loss / epoch_words))
                eval_loss = evaluate(model, loss_function, valid_data, char2index)
                test(model, s1, char2index, index2char)
                test(model, s2, char2index, index2char)

        print('train_loss:{}'.format(epoch_train_loss / epoch_words))
        eval_loss = evaluate(model, loss_function, valid_data, char2index)

        if eval_loss < min_eval_loss:
            min_eval_loss = eval_loss
            best_model = model
            min_count = 0
        else:
            min_count += 1

        if min_count == 10:
            break

        if epoch % 5 == 0:
            model_name = 'outputs/model1/all_epoch_'+str(epoch)
            torch.save(model, model_name)


def evaluate(model, loss_function, valid_data, char2index):
    eval_dataset = CustomDataset(valid_data, char2index)

    eval_data_loader = DataLoader(eval_dataset, batch_size=BATCH_SIZE, collate_fn=padding, drop_last=True,
                                  num_workers=4, pin_memory=True)
    # eval_data_loader = DataLoader(eval_dataset, batch_size=BATCH_SIZE, collate_fn=padding, drop_last=True)

    model.eval()
    total_eval_loss, total_words, min_eval_loss = 0, 0, float('inf')
    with torch.no_grad():
        for inputs, targets in eval_data_loader:
            attention_mask = mask(targets[:, 1:], inputs).to(device)
            inputs, targets = inputs.to(device), targets.to(device)
            encode_output, adapted_hidden_state = model.encode(inputs)
            decode_output, decode_hidden_state = model.decode(targets[:, :-1], adapted_hidden_state, encode_output, attention_mask)
            loss = loss_function(decode_output.reshape(-1, decode_output.shape[-1]), targets[:,1:].reshape(-1))

            iteration_words = torch.sum(targets[:, 1:] > 0).item()
            total_eval_loss += loss.item() * iteration_words
            total_words += iteration_words

        average_loss = total_eval_loss / total_words
        print('evalu_loss:{}'.format(average_loss))
        return average_loss

def test(model, s, char2index, index2char, topk=3, maxlen=64):
    inputs = [2] + [char2index.get(char, 1) for char in s] + [3]
    encode_inputs = torch.tensor(inputs).expand((topk, len(inputs)))
    decode_inputs = torch.tensor([2]).expand((topk, 1))

    attention_mask = mask(decode_inputs, encode_inputs).to(device)
    encode_inputs, decode_inputs = encode_inputs.to(device), decode_inputs.to(device)
    state_indices = torch.tensor([[i] for i in range(topk)], device=device).expand(topk, topk).reshape(-1)
    output_tokens = torch.zeros([topk, maxlen], device=device)
    shift = torch.tensor([3], device=device).expand((topk, topk))


    model.eval()
    with torch.no_grad():
        encode_output, adapted_hidden_state = model.encode(encode_inputs)
        decode_hidden_state = adapted_hidden_state

        for i in range(maxlen):

            decode_output, decode_hidden_state = model.predicate(decode_inputs, decode_hidden_state, encode_output, attention_mask)
            topk_indices, topk_values = decode_output[:, 3:].topk(topk).indices, decode_output[:, 3:].topk(topk).values
            topk_indices = topk_indices+shift
            if i == 0:
                prob = topk_values[0, :].unsqueeze(1)
                decode_inputs = topk_indices[0, :].unsqueeze(1)
                output_tokens[:, i] = topk_indices[0, :]
            else:
                prob = (prob*topk_values).reshape(-1)
                indices = topk_indices.reshape(-1)
                prob_topk_indices, prob_topk_values = prob.topk(topk).indices, prob.topk(topk).values
                prob = prob_topk_values.unsqueeze(1)
                decode_inputs = indices[prob_topk_indices].unsqueeze(1)
                output_tokens = output_tokens[state_indices[prob_topk_indices], :]
                output_tokens[:, i] = indices[prob_topk_indices]
                decode_hidden_state = decode_hidden_state[:, state_indices[prob_topk_indices], :]
                best_t = indices[prob_topk_indices][0]

                if best_t.item() == 3:
                    print(''.join([index2char[str(int(token.item()))] for token in output_tokens[0, :i]]))
                    return ''.join([index2char[str(int(token.item()))] for token in output_tokens[0, :i]])



        print(''.join([index2char[str(int(token.item()))] for token in output_tokens[0, :]]))
        return ''.join([index2char[str(int(token.item()))] for token in output_tokens[0, :]])




if __name__ == '__main__':
    # 史上最贵微博：陈欧一条微博给聚美恢复10亿市值
    # 可穿戴技术十大设计原则
    s1 = '16日，聚美优品创始人兼CEO陈欧发出长微博《你永远不知道，陈欧这半年在做什么》，公开介绍聚美优品上市后所进行的重要业务转型。' \
         '当天聚美股价应声企稳，并已连续上涨两天，市值重上20亿美元，折算下来恢复了将近10亿人民币。'
    s2 = '本文总结了十个可穿戴产品的设计原则，而这些原则，同样也是笔者认为是这个行业最吸引人的地方：' \
         '1.为人们解决重复性问题；2.从人开始，而不是从机器开始；3.要引起注意，但不要刻意；4.提升用户能力，而不是取代人'

    train_data = './splited_data/train/*'
    valid_data = './splited_data/valid/*'

    train(train_data, EPOCHS)




