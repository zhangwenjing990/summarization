# 尝试更改模型代码-精简代码
# 修改配置文件
# model_v3,gru_dec_v3


import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import glob
import tqdm

from model_v3 import *
from config import *

device = 'cuda' if torch.cuda.is_available() else 'cpu'

cudaid = 0
os.environ["CUDA_VISIBLE_DEVICES"] = str(cudaid)
cfg = Configs()
TRAINING_DATASET_CLS = Training
TESTING_DATASET_CLS = Testing

def init_modules():
    init_seeds()

    options = {}

    options["is_debugging"] = False
    options["is_predicting"] = False
    options[
        "model_selection"] = False  # When options["is_predicting"] = True, true means use validation set for tuning, false is real testing.

    options["cuda"] = cfg.CUDA and torch.cuda.is_available()
    options["device"] = torch.device("cuda" if options["cuda"] else "cpu")

    # in config.py
    options["cell"] = cfg.CELL
    options["copy"] = cfg.COPY
    options["coverage"] = cfg.COVERAGE
    options["is_bidirectional"] = cfg.BI_RNN
    options["avg_nll"] = cfg.AVG_NLL

    options["beam_decoding"] = cfg.BEAM_SEARCH  # False for greedy decoding

    assert TRAINING_DATASET_CLS.IS_UNICODE == TESTING_DATASET_CLS.IS_UNICODE
    options["is_unicode"] = TRAINING_DATASET_CLS.IS_UNICODE  # True Chinese dataet
    options["has_y"] = TRAINING_DATASET_CLS.HAS_Y

    options["has_learnable_w2v"] = True
    options[
        "omit_eos"] = False  # omit <eos> and continuously decode until length of sentence reaches MAX_LEN_PREDICT (for DUC testing data)
    options["prediction_bytes_limitation"] = False if TESTING_DATASET_CLS.MAX_BYTE_PREDICT == None else True

    assert options["is_unicode"] == True

    consts = {}

    consts["idx_gpu"] = cudaid

    consts["norm_clip"] = cfg.NORM_CLIP
    consts["dim_x"] = cfg.DIM_X
    consts["dim_y"] = cfg.DIM_Y
    consts["len_x"] = cfg.MAX_LEN_X + 1  # plus 1 for eos
    consts["len_y"] = cfg.MAX_LEN_Y + 1
    consts["num_x"] = cfg.MAX_NUM_X
    consts["num_y"] = cfg.NUM_Y
    consts["hidden_size"] = cfg.HIDDEN_SIZE

    consts["batch_size"] = 5 if options["is_debugging"] else TRAINING_DATASET_CLS.BATCH_SIZE
    if options["is_debugging"]:
        consts["testing_batch_size"] = 1 if options["beam_decoding"] else 2
    else:
        # consts["testing_batch_size"] = 1 if options["beam_decoding"] else TESTING_DATASET_CLS.BATCH_SIZE
        consts["testing_batch_size"] = TESTING_DATASET_CLS.BATCH_SIZE

    consts["min_len_predict"] = TESTING_DATASET_CLS.MIN_LEN_PREDICT
    consts["max_len_predict"] = TESTING_DATASET_CLS.MAX_LEN_PREDICT
    consts["max_byte_predict"] = TESTING_DATASET_CLS.MAX_BYTE_PREDICT
    consts["testing_print_size"] = TESTING_DATASET_CLS.PRINT_SIZE

    consts["lr"] = cfg.LR
    consts["beam_size"] = cfg.BEAM_SIZE

    consts["max_epoch"] = 150 if options["is_debugging"] else 9
    consts["print_time"] = 3
    consts["save_epoch"] = 1

    consts['train_data_path'] = './splited_data/train/*'
    consts['valid_data_path'] = './splited_data/valid/*'
    consts['test_data_path'] = './splited_data/test/*'

    assert consts["dim_x"] == consts["dim_y"]
    assert consts["beam_size"] >= 1

    modules = {}

    [_, dic, hfw, w2i, i2w, w2w] = pickle.load(open(cfg.cc.TRAINING_DATA_PATH + "dic.pkl", "rb"))

    consts["dict_size"] = len(dic)
    modules["dic"] = dic
    modules["w2i"] = w2i
    modules["i2w"] = i2w
    modules["lfw_emb"] = modules["w2i"][cfg.W_UNK]
    modules["eos_emb"] = modules["w2i"][cfg.W_EOS]
    consts["pad_token_idx"] = modules["w2i"][cfg.W_PAD]

    return modules, consts, options

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
    return padded_x, padded_x_ext, padded_y, padded_y_ext, x_mask, y_mask, sorted_len_x, max_ext_len



class CustomDataset(Dataset):
    def __init__(self, data_path, modules, consts, options):
        super().__init__()
        self.files = glob.glob(data_path)
        self.modules = modules
        self.consts = consts
        self.options = options

    def __getitem__(self, index):
        file = self.files[index]
        with open(file, 'r', encoding='utf-8') as f:
            new = f.read()
        fs = new.split("[sep]")
        abstract, contents = fs[0].strip(), fs[1].strip()


        w2i = self.modules['w2i']
        i2w = self.modules['i2w']
        dict_size = len(w2i)

        xi_oovs = []
        x_ext = []
        x = []
        contents_words = list(contents)
        for w in contents_words[:self.consts['len_x']] + ["<eos>"]:
            if w not in w2i:
                if w not in xi_oovs:
                    xi_oovs.append(w)
                x_ext.append(dict_size + xi_oovs.index(w))
                x.append(self.modules['lfw_emb'])
            else:
                x_ext.append(w2i[w])
                x.append(w2i[w])

        y_ext = []
        y = []
        abstract_words = list(abstract) + ["<eos>"]
        for w in abstract_words[:self.consts['len_y']]:
            if w not in w2i:
                if w in xi_oovs:
                    y_ext.append(dict_size + xi_oovs.index(w))
                else:
                    y_ext.append(self.modules['lfw_emb'])
                y.append(self.modules['lfw_emb'])
            else:
                y_ext.append(w2i[w])
                y.append(w2i[w])

        return torch.tensor(x), torch.tensor(x_ext), torch.tensor(y), torch.tensor(y_ext), len(xi_oovs)


    def __len__(self):
        return len(self.files)

def evaluate(model, modules, consts, options):
    valid_dataset = CustomDataset(consts['valid_data_path'], modules, consts, options)
    valid_data_loader = DataLoader(valid_dataset, batch_size=consts['batch_size'], collate_fn=padding, num_workers=4, pin_memory=True)

    model.eval()
    total_loss, total_loss_c, total_words = 0, 0, 0
    with torch.no_grad():
        for batch in valid_data_loader:
            x, x_ext, y, y_ext, x_mask, y_mask, len_x, max_ext_len = batch
            y_pred, cost, cost_c = model(torch.LongTensor(x).to(options["device"]),
                                         torch.LongTensor(len_x).to(options["device"]), \
                                         torch.LongTensor(y).to(options["device"]),
                                         torch.FloatTensor(x_mask).to(options["device"]), \
                                         torch.FloatTensor(y_mask).to(options["device"]),
                                         torch.LongTensor(x_ext).to(options["device"]), \
                                         torch.LongTensor(y_ext).to(options["device"]), \
                                         max_ext_len)
            if cost_c is None:
                loss = cost
            else:
                loss = cost + cost_c

            iteration_words = torch.sum(y_mask).item()
            total_loss += cost.item() * iteration_words
            total_loss_c += cost_c.item() * iteration_words if cost_c else 0
            total_words += iteration_words

        return total_loss/total_words, total_loss_c/total_words



def train():
    modules, consts, options = init_modules()
    model = Model(modules, consts, options).to(device)

    total_num = sum(p.numel() for p in model.parameters())
    print('参数量:{}'.format(total_num))

    optimizer = torch.optim.Adam(model.parameters())
    # model, optimizer = load_model('./outputs/models/' + 'epoch_v5_test_6', model, optimizer) #todo
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_function = torch.nn.NLLLoss(ignore_index=0)

    # train_dataset = CustomDataset(consts['train_data_path'], modules, consts, options)
    # train_data_loader = DataLoader(train_dataset, batch_size=consts['batch_size'], shuffle=True, collate_fn=padding, num_workers=4, pin_memory=True)

    # debug
    train_dataset = CustomDataset(consts['valid_data_path'], modules, consts, options)
    train_data_loader = DataLoader(train_dataset, batch_size=consts['batch_size'], collate_fn=padding)


    for epoch in range(0, consts['max_epoch']):

        total_loss, total_loss_c, total_words = 0, 0, 0
        iteration = 0
        for batch in tqdm.tqdm(train_data_loader):
            model.train()
            model.zero_grad()

            x, x_ext, y, y_ext, x_mask, y_mask, len_x, max_ext_len = batch

            y_pred, cost, cost_c = model(torch.LongTensor(x).to(options["device"]),
                                         torch.LongTensor(len_x).to(options["device"]), \
                                         torch.LongTensor(y).to(options["device"]),
                                         torch.FloatTensor(x_mask).to(options["device"]), \
                                         torch.FloatTensor(y_mask).to(options["device"]),
                                         torch.LongTensor(x_ext).to(options["device"]), \
                                         torch.LongTensor(y_ext).to(options["device"]), \
                                         max_ext_len)

            if cost_c is None:
                loss = cost
            else:
                loss = cost + cost_c #todo
                # loss = cost


            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), consts["norm_clip"])
            optimizer.step()

            iteration += 1
            iteration_words = torch.sum(y_mask).item()
            total_loss += cost.item() * iteration_words
            total_loss_c += cost_c.item() * iteration_words if cost_c else 0
            total_words += iteration_words

            if iteration % 100 == 0:
                eval_loss, eval_loss_c = evaluate(model, modules, consts, options)
                print()
                print('epoch: {} | iterations: {}'.format(epoch, iteration))
                print('|-> total_train_loss: {} | total_train_loss_c: {}'.format(total_loss / total_words,
                                                                                 total_loss_c / total_words))
                print('|-> total_evalu_loss: {} | total_evalu_loss_c: {}'.format(eval_loss, eval_loss_c))
                f = './outputs/models/' + 'epoch_v5_test1_' + str(epoch)
                torch.save({"model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict()}, f)


        eval_loss, eval_loss_c = evaluate(model, modules, consts, options)
        print('epoch: {} | iterations: {}'.format(epoch, iteration))
        print(' total_train_loss: {} | total_train_loss_c: {}'.format(total_loss / total_words,
                                                                      total_loss_c / total_words))
        print(' total_evalu_loss: {} | total_evalu_loss_c: {}'.format(eval_loss, eval_loss_c))

        f = './outputs/models/' + 'epoch_v5_test1_' + str(epoch)
        torch.save({"model_state_dict" : model.state_dict(), "optimizer_state_dict" : optimizer.state_dict()}, f)











if __name__ == '__main__':
    train()
