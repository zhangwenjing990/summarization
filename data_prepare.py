# 1，将I,II,III中的多条新闻转换为单条新闻，分别保存到文件中。
# 2，过滤掉II和III中的噪声数据。
import os
import re
import glob
import tqdm
import configs
import pickle
import operator


def wash_string(string):
    # 去除英文标点符号
    new_string = re.subn(r"""[!"#$%&'()*+,-./:;<=>?@\[\\\]^_`{|}~]""", '', string)[0]
    # 去除中文标点符号(保留，。！)
    new_string = re.subn(r'[【】、·：『』「」“”《》……￥#（）‘’]+', '', new_string)[0]
    # 去除数字，空格
    new_string = re.subn(r'[\d\s]+', '', new_string)[0]
    return new_string.lower()

def separate_to_single_new(original_data_path, new_data_path):

    if not os.path.exists(new_data_path):
        os.makedirs(new_data_path)

    with open(original_data_path, 'r', encoding='utf-8') as f:

        news_index = 0
        noise_data_num = 0
        while True:
            abstract, contents = '', ''
            line = f.readline().strip()
            if line == '':
                break

            if line.startswith('<doc'):
                doc_id = line.split('=')[-1][:-1]

                line = f.readline().strip()
                if line.startswith('<human_label>'):
                    human_label = int(line[13])
                    line = f.readline().strip()
                else:
                    human_label = None

                if line.startswith('<summary>'):
                    while True:
                        line = f.readline().strip()
                        if line.startswith('</summary>'):
                            break
                        else:
                            abstract +=line
                    line = f.readline().strip()
                else:
                    raise RuntimeError('summary illegal', doc_id, line)

                if line.startswith('<short_text>'):
                    while True:
                        line = f.readline().strip()
                        if line.startswith('</short_text>'):
                            break
                        contents += line
                    line = f.readline().strip()
                else:
                    raise RuntimeError('short_text illegal', doc_id, line)

                if not line.startswith('</doc>'):
                    raise RuntimeError('doc end illegal', doc_id, line)
            else:
                raise RuntimeError('doc begin illegal', doc_id, line)

            new_abstract, new_contents = wash_string(abstract), wash_string(contents)

            if not new_abstract or not new_contents or (human_label and human_label < 3):
                noise_data_num += 1
                continue

            with open(new_data_path + str(news_index), 'w', encoding='utf-8') as new_file:
                new_file.write(new_abstract + '[sep]' + new_contents)

            news_index += 1

            if news_index % 5000 == 0:
                print('{} news are generated'.format(news_index))

        print('all news in {} are trasformed, there are {} items'.format(original_data_path, news_index))
        print('{} item noises are filted'.format(noise_data_num))

def generate_vocab(data_path):
    train_files = glob.glob(data_path + 'train/*')
    valid_files = glob.glob(data_path + 'valid/*')

    cfgs = configs.Configs()

    all_dic = {}
    files = train_files + valid_files
    i = 0
    for file in tqdm.tqdm(files):
        with open(file, 'r', encoding='utf-8') as f:
            new = f.read()
        for w in new:
            all_dic[w] = all_dic.get(w, 0) + 1
        i += 1
        # if i == 2000:
        #     break
    all_dic = {i: j for i, j in all_dic.items() if j >= 100}

    dic = {}
    w2i = {}
    i2w = {}
    w2w = {}

    for w in [cfgs.W_PAD, cfgs.W_UNK, cfgs.W_EOS]:
        w2i[w] = len(dic)
        i2w[w2i[w]] = w
        dic[w] = 10000
        w2w[w] = w

    for w, tf in all_dic.items():
        if w in dic:
            continue
        w2i[w] = len(dic)
        i2w[w2i[w]] = w
        dic[w] = tf
        w2w[w] = w

    hfw = []
    sorted_x = sorted(dic.items(), key=operator.itemgetter(1), reverse=True)
    # print(sorted_x)
    for w in sorted_x:
        hfw.append(w[0])

    assert len(hfw) == len(dic)
    assert len(w2i) == len(dic)
    print("dump dict...")
    print(len(w2i))
    # pickle.dump([all_dic, dic, hfw, w2i, i2w, w2w], open(cfgs.cc.TRAINING_DATA_PATH + "dic.pkl", "wb"),
    #             protocol=pickle.HIGHEST_PROTOCOL)





if __name__ == '__main__':
    original_data_path = './LCSTS/DATA/'
    new_data_path = './splited_data/'
    separate_to_single_new(original_data_path + 'PART_I.txt', new_data_path + 'train/')
    separate_to_single_new(original_data_path + 'PART_II.txt', new_data_path + 'valid/')
    separate_to_single_new(original_data_path + 'PART_III.txt', new_data_path + 'test/')

    generate_vocab(new_data_path)