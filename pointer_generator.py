
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import glob
import tqdm

from model_v4 import *
from config_v2 import *


def padding(batch):
    x, x_ext, y, y_ext, len_xi_oovs = zip(*batch)
    max_ext_len = max(len_xi_oovs)
    len_x = [len(i) for i in x]
    sorted_x, sorted_x_ext, sorted_y, sorted_y_ext, sorted_len_x = [], [], [], [], []
    sorted_len_x_pair = sorted(enumerate(len_x), key=lambda x: x[1], reverse=True)
    for i, j in sorted_len_x_pair:
        sorted_x.append(x[i])
        sorted_x_ext.append(x_ext[i])
        sorted_y.append(y[i])
        sorted_y_ext.append(y_ext[i])
        sorted_len_x.append(len_x[i])

    padded_x = pad_sequence(sorted_x, padding_value=0)
    padded_x_ext = pad_sequence(sorted_x_ext, padding_value=0)
    padded_y = pad_sequence(sorted_y, padding_value=0)
    padded_y_ext = pad_sequence(sorted_y_ext, padding_value=0)
    x_mask = (padded_x > 0).to(dtype=torch.float).unsqueeze(-1)
    y_mask = (padded_y > 0).to(dtype=torch.float).unsqueeze(-1)
    return padded_x, padded_x_ext, padded_y, padded_y_ext, x_mask, y_mask, torch.tensor(sorted_len_x), max_ext_len


class CustomDataset(Dataset):
    def __init__(self, data_path, cfg):
        super().__init__()
        self.files = glob.glob(data_path)
        self.cfg = cfg

    def __getitem__(self, index):
        file = self.files[index]
        with open(file, 'r', encoding='utf-8') as f:
            new = f.read()
        fs = new.split("[sep]")
        abstract, contents = fs[0].strip(), fs[1].strip()

        w2i = self.cfg.w2i
        i2w = self.cfg.i2w
        dict_size = len(w2i)

        xi_oovs = []
        x_ext = []
        x = []
        contents_words = list(contents)
        for w in contents_words[:self.cfg.len_x] + ["<eos>"]:
            if w not in w2i:
                if w not in xi_oovs:
                    xi_oovs.append(w)
                x_ext.append(dict_size + xi_oovs.index(w))
                x.append(self.cfg.lfw_emb)
            else:
                x_ext.append(w2i[w])
                x.append(w2i[w])

        y_ext = []
        y = []
        abstract_words = list(abstract) + ["<eos>"]
        for w in abstract_words[:self.cfg.len_y]:
            if w not in w2i:
                if w in xi_oovs:
                    y_ext.append(dict_size + xi_oovs.index(w))
                else:
                    y_ext.append(self.cfg.lfw_emb)
                y.append(self.cfg.lfw_emb)
            else:
                y_ext.append(w2i[w])
                y.append(w2i[w])

        return torch.tensor(x), torch.tensor(x_ext), torch.tensor(y), torch.tensor(y_ext), len(xi_oovs)

    def __len__(self):
        return len(self.files)


def evaluate(model, cfg):
    valid_dataset = CustomDataset(cfg.valid_data_path, cfg)
    valid_data_loader = DataLoader(valid_dataset, batch_size=cfg.batch_size, collate_fn=padding, num_workers=4,
                                   pin_memory=True)

    model.eval()
    total_vocab_loss, total_coverage_loss, total_words = 0, 0, 0
    with torch.no_grad():
        for batch in valid_data_loader:
            x, x_ext, y, y_ext, x_mask, y_mask, len_x, max_ext_len = batch
            x, len_x, y, x_ext, y_ext = x.to(cfg.device), len_x.to(cfg.device), y.to(cfg.device), x_ext.to(
                cfg.device), y_ext.to(cfg.device)
            x_mask, y_mask = x_mask.to(dtype=torch.float32, device=cfg.device), y_mask.to(dtype=torch.float32,
                                                                                          device=cfg.device)
            y_prediction, vocab_loss, coverage_loss = model(x, len_x, y, x_mask, y_mask, x_ext, y_ext, max_ext_len)

            iteration_words = torch.sum(y_mask).item()
            total_vocab_loss += vocab_loss.item() * iteration_words
            total_coverage_loss += coverage_loss.item() * iteration_words if coverage_loss else 0
            total_words += iteration_words

        return total_vocab_loss / total_words, total_coverage_loss / total_words


def train():
    cfg = Configs()
    model = Model(cfg).to(cfg.device)

    total_num = sum(p.numel() for p in model.parameters())
    print('参数量:{}'.format(total_num))

    model = load_model('./outputs/models/' + 'epoch_v5_test1_1', model) #todo
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    train_dataset = CustomDataset(cfg.train_data_path, cfg)
    train_data_loader = DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True, collate_fn=padding, num_workers=4, pin_memory=True)

    # # debug
    # train_dataset = CustomDataset(cfg.valid_data_path, cfg)
    # train_data_loader = DataLoader(train_dataset, batch_size=cfg.batch_size, collate_fn=padding)

    for epoch in range(2, cfg.max_epoch):

        total_vocab_loss, total_coverage_loss, total_words = 0, 0, 0
        iteration = 0
        for batch in tqdm.tqdm(train_data_loader):
            model.train()
            model.zero_grad()

            x, x_ext, y, y_ext, x_mask, y_mask, len_x, max_ext_len = batch
            x, len_x, y, x_ext, y_ext = x.to(cfg.device), len_x.to(cfg.device), y.to(cfg.device), x_ext.to(
                cfg.device), y_ext.to(cfg.device)
            x_mask, y_mask = x_mask.to(dtype=torch.float32, device=cfg.device), y_mask.to(dtype=torch.float32,
                                                                                          device=cfg.device)
            y_prediction, vocab_loss, coverage_loss = model(x, len_x, y, x_mask, y_mask, x_ext, y_ext, max_ext_len)

            composite_loss = vocab_loss + coverage_loss  # todo

            if cfg.coverage:
                composite_loss.backward()
            else:
                vocab_loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.norm_clip)
            optimizer.step()

            iteration += 1
            iteration_words = torch.sum(y_mask).item()
            total_vocab_loss += vocab_loss.item() * iteration_words
            total_coverage_loss += coverage_loss.item() * iteration_words if coverage_loss else 0
            total_words += iteration_words

            if iteration % 1000 == 0:
                eval_loss, eval_loss_c = evaluate(model, cfg)
                print()
                print('epoch: {} | iterations: {}'.format(epoch, iteration))
                print('|-> total_train_loss: {} | total_train_loss_c: {}'.format(total_vocab_loss / total_words,
                                                                                 total_coverage_loss / total_words))
                print('|-> total_evalu_loss: {} | total_evalu_loss_c: {}'.format(eval_loss, eval_loss_c))
                f = './outputs/models/' + 'epoch_v5_test1_' + str(epoch)
                torch.save({"model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict()}, f)

        eval_loss, eval_loss_c = evaluate(model, cfg)
        print('epoch: {} | iterations: {}'.format(epoch, iteration))
        print(' total_train_loss: {} | total_train_loss_c: {}'.format(total_vocab_loss / total_words,
                                                                      total_coverage_loss / total_words))
        print(' total_evalu_loss: {} | total_evalu_loss_c: {}'.format(eval_loss, eval_loss_c))

        f = './outputs/models/' + 'epoch_v5_test1_' + str(epoch)
        torch.save({"model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict()}, f)

def load_model(f, model):
    checkpoint = torch.load(f)
    model.load_state_dict(checkpoint["model_state_dict"])
    return model

if __name__ == '__main__':
    train()
