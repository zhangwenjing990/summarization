import os
import torch
import pickle


class Configs(object):
    def __init__(self):
        # 数据和模型路径
        self.root_path = os.getcwd()
        self.train_data_path = self.root_path + '/splited_data/train/*'
        self.valid_data_path = self.root_path + '/splited_data/valid/*'
        # self.test_data_path = self.root_path + '/splited_data/test/*'
        self.test_data_path = self.root_path + '/lcsts/test_set/'

        self.vocab_path = self.root_path + '/splited_data/'
        self.save_model_path = self.root_path + '/outputs/models/'
        self.rouge_path = self.root_path + '/outputs/'
        self.beam_decode_path = self.root_path + '/outputs/beam_decode/'
        self.ground_truth_path = self.root_path + '/outputs/ground_truth/'


        # 重新生成字典 todo
        [_, dic, hfw, w2i, i2w, w2w] = pickle.load(open(self.vocab_path + "dic.pkl", "rb"))
        self.dict_size = len(dic)
        self.dic = dic
        self.w2i = w2i
        self.i2w = i2w
        self.lfw_emb = self.w2i['<unk>']
        self.eos_emb = self.w2i['<eos>']
        self.pad_token_idx = self.w2i['<pad>']

        # 模型基本配置
        self.norm_clip = 2
        self.embedding_size = 150
        self.hidden_size = 150




        # 训练/预测模式配置和设备
        self.is_predicting = False
        self.coverage = False
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.max_epoch = 50
        self.batch_size = 32 * 4
        self.lr = 0.001
        self.len_x = 120
        self.len_y = 50
        self.has_y = True
        self.beam_size = 3
        self.min_len_predict = 10
        self.max_len_predict = 20





