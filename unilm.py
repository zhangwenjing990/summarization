
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from config1 import Config

from glob import glob
from tqdm import tqdm

from utils import load_chinese_base_vocab, load_pretrained_bert, load_custom_model, Tokenizer
from bert_model import CustomUnilmModel


model_name = "bert"

cfg = Config()

s1 = '本文总结了十个可穿戴产品的设计原则，而这些原则，同样也是笔者认为是这个行业最吸引人的地方：' \
     '1.为人们解决重复性问题；2.从人开始，而不是从机器开始；3.要引起注意，但不要刻意；4.提升用户能力，而不是取代人'

s2 = '16日，聚美优品创始人兼CEO陈欧发出长微博《你永远不知道，陈欧这半年在做什么》，公开介绍聚美优品上市后所进行的重要业务转型。' \
     '当天聚美股价应声企稳，并已连续上涨两天，市值重上20亿美元，折算下来恢复了将近10亿人民币。'

class CustomDataset(Dataset):
    # 自定义数据集加载方式
    def __init__(self, data_path, tokenizer, max_seq_len):
        super().__init__()
        # self.files = glob(data_path + '*')[:16*1000*5]
        self.files = data_path
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        # self.unk = char2idx['[UNK]']
        # self.cls = char2idx['[CLS]']
        # self.sep = char2idx['[SEP]']

    def __len__(self):
        return len(self.files)

    def __getitem__(self, item):
        file = self.files[item]
        with open(file, "r", encoding="utf-8") as f:
            text = f.read()
        # 根据[sep]分割符号分割数据
        ### todo
        abstract, content = text.split('[sep]')
        abstract = abstract
        content = content

        # content_index = [self.char2idx.get(char, self.unk) for char in content]
        # title_index = [self.char2idx.get(char, self.unk) for char in abstract]
        #
        # # 如果序列长度超出指定范围，截掉content尾部的内容
        # input_index = [self.cls] + content_index[:self.max_seq_len - len(title_index) - 3] \
        #               + [self.sep] + title_index + [self.sep]
        # token_type_idx = [0] * (len(input_index) - len(title_index) - 1) + [1] * (len(title_index) + 1)

        input_index, token_type_idx = self.tokenizer.encode(content, abstract, self.max_seq_len)

        return torch.tensor(input_index), torch.tensor(token_type_idx)

def padding(batch):
    # 将数据padding到相同长度
    inputs, token_type_idx = zip(*batch)
    padded_inputs = pad_sequence(inputs, batch_first=True, padding_value=0)
    padded_token_type = pad_sequence(token_type_idx, batch_first=True, padding_value=0)
    padded_targets = (padded_inputs * padded_token_type)[:, 1:]
    return padded_inputs, padded_token_type, padded_targets




def unilm_mask(inputs, s):
    idxs = torch.cumsum(s, dim=1)
    mask = idxs[:, None, :] <= idxs[:, :, None]
    mask = mask[:, None].squeeze(1)
    return mask.to(dtype=torch.int64)


def train():
    # 加载数据
    char2idx, keep_tokens = load_chinese_base_vocab(cfg.vocab_path)
    tokenizer = Tokenizer(char2idx)
    # train_data = glob(cfg.train_data_path + '*')[16 * 1000 * 35:16 * 1000 * 40]
    train_data = glob(cfg.train_data_path + '*')[8 * 5000 * 5: 8 * 5000 * 10]
    train_dataset = CustomDataset(train_data, tokenizer, cfg.max_seq_len)
    train_dataloader = DataLoader(train_dataset, batch_size=cfg.batch_size, collate_fn=padding,
                                  shuffle=True, num_workers=4, pin_memory=True)

    # # debug
    # train_data = glob(cfg.test_data_path + '*')[:8 * 5000 * 5]
    # train_dataset = CustomDataset(train_data, tokenizer, cfg.max_seq_len)
    # train_dataloader = DataLoader(train_dataset, batch_size=cfg.batch_size, collate_fn=padding)
    # # debug
    # 加载模型
    model = CustomUnilmModel(len(char2idx))
    # model = load_pretrained_bert(model, cfg.pretrained_model_path, keep_tokens=keep_tokens).to(cfg.device)
    model = load_custom_model(model, cfg.save_model_path).to(cfg.device)

    loss_function = nn.CrossEntropyLoss(ignore_index=0).to(cfg.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learn_rate)
    # 迭代训练
    iteration, train_loss = 0, 0
    model.train()
    for inputs, token_type, targets in tqdm(train_dataloader, position=0, leave=True):
        attention_mask = unilm_mask(inputs, token_type).to(cfg.device)
        inputs, token_type, targets = inputs.to(cfg.device), token_type.to(cfg.device), targets.to(cfg.device)
        prediction = model(inputs, token_type, attention_mask)
        loss = loss_function(prediction[:,:-1,:].reshape(-1, prediction.shape[-1]), targets.reshape(-1))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        iteration += 1

        if iteration % cfg.print_loss_steps == 0:
            eval_loss = evaluate(model, tokenizer, loss_function)
            print('')
            print('train_loss:{}'.format(train_loss/cfg.print_loss_steps))
            print('evalu_loss:{}'.format(eval_loss))
            test_string(s1, tokenizer, model)
            test_string(s2, tokenizer, model)
            model.train()
            train_loss = 0

        if iteration % cfg.save_model_steps == 0:
            torch.save(model.state_dict(), cfg.save_model_path)





def evaluate(model, tokenizer, loss_function):
    # 加载验证集验证
    valid_data = glob(cfg.valid_data_path + '*')
    eval_dataset = CustomDataset(valid_data, tokenizer, cfg.max_seq_len)
    eval_dataloader = DataLoader(eval_dataset, batch_size=cfg.batch_size, collate_fn=padding,
                                  shuffle=True, num_workers=4, pin_memory=True)

    model.eval()
    iteration, eval_loss = 0, 0
    with torch.no_grad():
        for inputs, token_type, targets in eval_dataloader:
            attention_mask = unilm_mask(inputs, token_type).to(cfg.device)
            inputs, token_type, targets = inputs.to(cfg.device), token_type.to(cfg.device), targets.to(cfg.device)
            prediction = model(inputs, token_type, attention_mask)
            loss = loss_function(prediction[:, :-1, :].reshape(-1, prediction.shape[-1]), targets.reshape(-1))
            eval_loss += loss.item()
            iteration += 1
        return eval_loss / iteration



def test_string(content, tokenizer, model):
    topk = cfg.beam_size
    # 文本内容编码
    # content_idx = [char2idx.get(char, char2idx['[UNK]']) for char in content]
    # 允许输入的文本的最大长度
    max_content_len = cfg.max_seq_len - cfg.max_decode_len
    # 如果文本长度过长，进行截断处理
    # input_idx = [char2idx['[CLS]']] + content_idx[:max_content_len] + [char2idx['[SEP]']]
    input_idx, token_type_ids = tokenizer.encode(content, max_length=max_content_len)
    # 生成模型需要的输入数据
    inputs = torch.tensor(input_idx).expand(topk, len(input_idx))
    token_type = torch.tensor([0]).expand(inputs.size())
    attention_mask = unilm_mask(inputs, token_type)

    state_indices = torch.tensor([[i] for i in range(topk)]).expand(topk, topk).reshape(-1)
    output_tokens = torch.zeros([topk, cfg.max_decode_len])
    ones = torch.tensor([1]).expand(topk, 1)
    shift = torch.tensor([2]).expand((topk, topk))

    model.eval()
    with torch.no_grad():
        for t in range(cfg.max_decode_len):
            prediction = model(inputs.to(cfg.device), token_type.to(cfg.device), attention_mask.to(cfg.device))
            decode_t = prediction[:, -1, :].to('cpu')
            # 获取topk个输出（去除两个特殊的token：'[UNK]'和'[CLS]'）
            topk_indices, topk_values = decode_t[:, 2:].topk(topk).indices, decode_t[:, 2:].topk(topk).values
            # 由于去除了两个token，造成token的值减小了，所以这里要加上
            topk_indices = topk_indices + shift

            if t == 0:
                # topk输出概率
                prob = topk_values[0, :].unsqueeze(1)
                # 下一时间步的输入
                decode_inputs = topk_indices[0, :].unsqueeze(1)
                inputs = torch.cat((inputs, decode_inputs), dim=-1)
                token_type = torch.cat((token_type, ones), dim=-1)
                attention_mask = unilm_mask(inputs, token_type)
                # 记录当前时间步的topk输出
                output_tokens[:, t] = topk_indices[0, :]

            else:
                # 当前时间步的序列概率
                prob = (prob*topk_values).reshape(-1)
                # 获得topk个概率最大的输出
                indices = topk_indices.reshape(-1)
                prob_topk_indices, prob_topk_values = prob.topk(topk).indices, prob.topk(topk).values
                prob = prob_topk_values.unsqueeze(1)

                # 根据topk概率重置输入
                decode_inputs = indices[prob_topk_indices].unsqueeze(1)
                inputs = inputs[state_indices[prob_topk_indices], :]
                inputs = torch.cat((inputs, decode_inputs), dim=-1)
                token_type = torch.cat((token_type, ones), dim=-1)
                attention_mask = unilm_mask(inputs, token_type)

                # 重置输出
                output_tokens = output_tokens[state_indices[prob_topk_indices], :]
                output_tokens[:, t] = indices[prob_topk_indices]

                # 符合停止条件，则输出
                if sum(indices[prob_topk_indices] == 3) > 0:
                    stop_index = torch.nonzero(indices[prob_topk_indices] == 3)[0].item()
                    output = tokenizer.decode(output_tokens[stop_index, :t].numpy())
                    print(output)
                    return output

        output = tokenizer.decode(output_tokens[0, :].numpy())
        print(output)
        return output

    # 逐步解码



if __name__ == '__main__':
    train()