import re
import unicodedata
import torch
import torch.nn as nn
import torch.optim as optim
import time
from tqdm import tqdm
import random
import matplotlib.pyplot as plt
import os
import sys

def get_resource_path(relative_path):
    """获取资源的绝对路径，兼容开发环境和打包环境"""
    try:
        # PyInstaller创建临时文件夹，将路径存储在_MEIPASS中
        base_path = sys._MEIPASS
    except AttributeError:
        base_path = os.path.abspath(".")
    
    return os.path.join(base_path, relative_path)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# 起始标志 
SOS_token = 0
# 结束标志
EOS_token = 1
# 最大长度
MAX_LENGTH = 10



eng_word2idx = {"SOS": SOS_token, "EOS": EOS_token}
eng_idx2word = {SOS_token: "SOS", EOS_token: "EOS"}
fra_word2idx = {"SOS": SOS_token, "EOS": EOS_token}
fra_idx2word = {SOS_token: "SOS", EOS_token: "EOS"}
teacher_forcing_ratio = 0.5

def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn' or c in 'àâäéèêëîïôöùûüçÀÂÄÉÈÊËÎÏÔÖÙÛÜÇ'
    )

def normalizeEng(s):
    s = unicodeToAscii(s.lower().strip())
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    return s

def normalizeFra(s):
    s = unicodeToAscii(s.strip())  # 法语不全部小写，保留首字母大写
    s = re.sub(r"([.!?])", r" \1", s)
    # 法语保留字母、法语特殊字符和标点
    s = re.sub(r"[^a-zA-ZàâäéèêëîïôöùûüçÀÂÄÉÈÊËÎÏÔÖÙÛÜÇ.!?,;:]+", r" ", s)
    # 处理法语特殊字符的常见变体
    french_replacements = [
        (r"['`´]", "'"),  # 统一引号类型
        (r"\"", ""),      # 移除英文引号
        (r"\s+", " "),    # 合并多个空格
    ]
    
    for pattern, replacement in french_replacements:
        s = re.sub(pattern, replacement, s)
    
    return s


def get_data():
    with open(get_resource_path('eng-fra.txt'), 'r', encoding='utf-8') as f:
        lines = f.read().strip().split('\n')
    pairs = []
    for l in lines[:16000]:
        parts = l.split('\t')
        if len(parts) >= 2:
            eng = normalizeEng(parts[0])
            fra = normalizeFra(parts[1])
            pairs.append([eng, fra])
    
    for idx in range(len(pairs)):
        for word in pairs[idx][0].split(' '):
            if word not in eng_word2idx:
                index = len(eng_word2idx)
                eng_word2idx[word] = index
                eng_idx2word[index] = word
        for word in pairs[idx][1].split(' '):
            if word not in fra_word2idx:
                index = len(fra_word2idx)
                fra_word2idx[word] = index
                fra_idx2word[index] = word
    
    common_french_words = ["j'", "l'", "d'", "qu'", "m'", "t'", "s'", "n'", "c'"]
    for word in common_french_words:
        if word not in fra_word2idx:
            index = len(fra_word2idx)
            fra_word2idx[word] = index
            fra_idx2word[index] = word
    # print(f"{list(eng_word2idx.items())[:5]}, {list(eng_idx2word.items())[:5]}")
    # print(f"{list(fra_word2idx.items())[:5]}, {list(fra_idx2word.items())[:5]}")
    
    return pairs

pairs = get_data()
    
class DataSet(torch.utils.data.Dataset):
    def __init__(self, pairs):
        self.pairs = pairs
        self.sample_len = len(self.pairs)
        
    def __len__(self):
        return self.sample_len
    
    def __getitem__(self, index):
        eng_sentence = self.pairs[index][0]
        fra_sentence = self.pairs[index][1]
        
        eng_idx = [eng_word2idx[word] for word in eng_sentence.split(' ')]
        eng_idx.append(EOS_token)
        tensor_eng = torch.tensor(eng_idx, dtype=torch.long).to(device)
        
        fra_idx = [fra_word2idx[word] for word in fra_sentence.split(' ')]
        fra_idx.append(EOS_token)
        tensor_fra = torch.tensor(fra_idx, dtype=torch.long).to(device)
        
        return tensor_eng, tensor_fra
        
class Encoder(nn.Module):
    def __init__(self, eng_vocab_size, hidden_size= 256):
        super().__init__()
        self.eng_vocab_size = eng_vocab_size
        self.hidden_size = hidden_size
        self.embed = nn.Embedding(eng_vocab_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, batch_first= True)
        self.softmax = nn.LogSoftmax(dim= -1)

    def forward(self, x):
        embedding = self.embed(x)
        h0 = torch.zeros(1, 1, self.hidden_size).to(device)
        outputs, hn = self.gru(embedding, h0)
        return self.softmax(outputs[0]), hn
    
class Decoder(nn.Module):
    def __init__(self, fra_vocab_size, hidden_size= 256):
        super().__init__()
        self.embed = nn.Embedding(fra_vocab_size, hidden_size)
        self.dropout = nn.Dropout(0.1)
        self.fc1 = nn.Linear(hidden_size * 2, MAX_LENGTH)
        self.fc2 = nn.Linear(hidden_size * 2, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, batch_first= True)
        self.out = nn.Linear(hidden_size, fra_vocab_size)
        self.softmax = nn.LogSoftmax(dim= -1)
        self.relu = nn.ReLU()
        
    def forward(self, Q, K, V):
        embedding = self.embed(Q)
        embedding = self.dropout(embedding)
        attn_W = self.softmax(self.fc1(torch.cat((embedding, K), dim= -1)))
        tmp = torch.bmm(attn_W, V.unsqueeze(0))
        attn = self.relu(self.fc2(torch.cat((embedding, tmp), dim= -1)))
        outputs, hn = self.gru(attn, K)
        return self.softmax(self.out(outputs[0])), hn

def test_decoder(train_loader):
    encoder = Encoder(len(eng_word2idx)).to(device)
    decoder = Decoder(len(fra_word2idx)).to(device)
    for x, y in train_loader:
        encoder_output, encoder_hn = encoder(x.to(device))
        print(encoder_output.shape, encoder_hn.shape)
        decoder_hn = encoder_hn
        encoder_output_c = torch.zeros(MAX_LENGTH, encoder.hidden_size, device= device)
        for idx in range(min(encoder_output.shape[1], MAX_LENGTH)):
            encoder_output_c[idx] = encoder_output[0][idx]
        for idx in range(y.shape[1]):
            temp = y[0][idx].view(1, -1)
            decoder_output, decoder_hn = decoder(temp.to(device), decoder_hn.to(device), encoder_output_c.to(device))
            print(decoder_output.shape)
            break
        break


def train_iter(x, y, encoder, decoder, encoder_optimizer, decoder_optimizer, loss):
    encoder_output, encoder_hn = encoder(x.to(device))
    encoder_output_c = torch.zeros(MAX_LENGTH, encoder.hidden_size, device= device)

    for idx in range(min(encoder_output.shape[1], MAX_LENGTH)):
        encoder_output_c[idx] = encoder_output[0][idx]
    
    decoder_hn = encoder_hn
    input_y = torch.tensor([[SOS_token]], device= device)
    total_loss = 0
    using_teacher_forcing = True if random.random() < teacher_forcing_ratio else False
    
    if using_teacher_forcing:
        for idx in range(y.shape[1]):
            outputs, decoder_hn = decoder(input_y.to(device), decoder_hn.to(device), encoder_output_c.to(device))
            target_y = y[0][idx].view(1)
            total_loss += loss(outputs, target_y.to(device))
            input_y = y[0][idx].view(1, -1)
    else:
        for idx in range(y.shape[1]):
            outputs, decoder_hn = decoder(input_y.to(device), decoder_hn.to(device), encoder_output_c.to(device))
            target_y = y[0][idx].view(1)
            total_loss += loss(outputs, target_y.to(device))
            topv, topi = outputs.topk(1)
            if topi.item() == EOS_token:
                break
            input_y = topi.detach()
    
    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()
    
    total_loss.backward()
    encoder_optimizer.step()
    decoder_optimizer.step()
    
    return total_loss.item() / y.shape[1]


def train_model(train_loader, epochs= 10, lr= 1e-4):
    encoder = Encoder(len(eng_word2idx)).to(device)
    decoder = Decoder(len(fra_word2idx)).to(device)
    
    if os.path.exists("encoder_model.pth") and os.path.exists("decoder_model.pth"):
        print("检测到存在旧模型，基于旧模型继续训练")
        encoder.load_state_dict(torch.load("encoder_model.pth"))
        decoder.load_state_dict(torch.load("decoder_model.pth"))
        # 更改旧模型名称防止被新模型替换
        os.rename("encoder_model.pth", f"encoder_model_old_{time.strftime('%Y-%m-%d-%H-%M')}.pth")
        os.rename("decoder_model.pth", f"decoder_model_old_{time.strftime('%Y-%m-%d-%H-%M')}.pth")
    
    encoder_optimizer = optim.Adam(encoder.parameters(), lr= lr)
    decoder_optimizer = optim.Adam(decoder.parameters(), lr= lr)
    Loss = nn.NLLLoss()
    min_loss = 1000
    plot_loss = []
    for epoch in range(epochs):
        loss_total = 0
        num = 0
        start_time = time.time()
        for x, y in tqdm(train_loader):
            loss = train_iter(x, y, encoder, decoder, encoder_optimizer, decoder_optimizer, Loss)
            loss_total += loss
            num += 1
            if num % 1000 == 0:
                plot_loss.append(loss_total / num)
        print(f"epoch: [{epoch + 1}/{epochs}], loss: {loss_total / num:.6f}, time: {time.time() - start_time:.2f}s")
        if plot_loss[-1] < min_loss:
            torch.save(encoder.state_dict(), "encoder_model.pth")
            torch.save(decoder.state_dict(), "decoder_model.pth")
            print(f"已保存最优模型 Loss : before {min_loss:.4f} after {plot_loss[-1]:.4f}")
            min_loss = plot_loss[-1]
    
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
    plt.plot(plot_loss)
    plt.xlabel("iter(k)")
    plt.ylabel("loss")
    plt.title("损失曲线")
    plt.savefig("loss.png")
    plt.show()
    
def format_french_sentence(word_list):
    if not word_list:
        return ""
    
    # 过滤掉特殊标记
    filtered_words = [w for w in word_list if w not in ['<EOS>', '<UNK>', 'SOS', 'EOS']]
    
    if not filtered_words:
        return ""
    
    # 连接单词
    sentence = " ".join(filtered_words)
    
    # 法语标点格式化规则
    french_punctuation_rules = [
        # 移除标点前的空格
        (r'\s+([.,!?;:])', r'\1'),
        # 处理法语省略号
        (r"\s+'", "'"),
        (r"'\s+", "'"),
        # 处理法语引号
        (r'\s+»', '»'),
        (r'«\s+', '«'),
        # 处理连字符
        (r'\s+-\s+', '-'),
        # 处理括号
        (r'\(\s+', '('),
        (r'\s+\)', ')'),
        # 合并多个空格
        (r'\s+', ' '),
        # 确保标点后有一个空格（除非是句子结尾）
        (r'([.,!?;:])([a-zA-ZàâäéèêëîïôöùûüçÀÂÄÉÈÊËÎÏÔÖÙÛÜÇ])', r'\1 \2'),
    ]
    
    for pattern, replacement in french_punctuation_rules:
        sentence = re.sub(pattern, replacement, sentence)
    
    # 首字母大写（如果句子是完整的）
    if sentence and sentence[0].isalpha():
        sentence = sentence[0].upper() + sentence[1:]
    
    # 确保以标点结尾（如果需要）
    if sentence and sentence[-1].isalnum():
        # 检查是否有疑问词开头
        question_words = ['pourquoi', 'comment', 'quand', 'où', 'qui', 'que', 'quel', 'quelle']
        if any(sentence.lower().startswith(word) for word in question_words):
            sentence += '?'
        else:
            sentence += '.'
    
    # 修复常见的法语错误
    french_corrections = [
        (r'c est', "c'est"),
        (r'j ai', "j'ai"),
        (r'l ecole', "l'école"),
        (r'd un', "d'un"),
        (r'qu il', "qu'il"),
        (r's il', "s'il"),
        (r'n est', "n'est"),
    ]
    
    for wrong, correct in french_corrections:
        sentence = re.sub(r'\b' + wrong + r'\b', correct, sentence, flags=re.IGNORECASE)
    
    return sentence.strip()
    
def eval_seq2seq(x, encoder, decoder):
    decoder_list = []
    with torch.no_grad():
        encoder_output, encoder_hn = encoder(x.to(device))
        encoder_output_c = torch.zeros(MAX_LENGTH, encoder.hidden_size, device=device)
        for idx in range(min(encoder_output.shape[1], MAX_LENGTH)):
            encoder_output_c[idx] = encoder_output[0][idx]
        decoder_hn = encoder_hn
        input_y = torch.tensor([[SOS_token]], device=device)
        for idx in range(MAX_LENGTH):
            outputs, decoder_hn = decoder(input_y, decoder_hn.to(device), encoder_output_c)
            topv, topi = outputs.topk(1)
            if topi.item() == EOS_token:
                decoder_list.append('<EOS>')
                break
            else:
                # 添加安全检查
                word_idx = topi.item()
                if word_idx in fra_idx2word:
                    decoder_list.append(fra_idx2word[word_idx])
                else:
                    decoder_list.append('<UNK>')
            input_y = topi.detach().to(device)
    return format_french_sentence(decoder_list)


def test_use_seq2seq():
    encoder = Encoder(len(eng_word2idx)).to(device)
    decoder = Decoder(len(fra_word2idx)).to(device)
    encoder.load_state_dict(torch.load("encoder_model.pth"))
    decoder.load_state_dict(torch.load("decoder_model.pth"))
    for i in range(3):
        x = pairs[6000 + i][0]
        y = pairs[6000 + i][1]
        tmp_x = [eng_word2idx[word] for word in x.split(' ')]
        tmp_x.append(EOS_token)
        tensor_x = torch.tensor(tmp_x, dtype= torch.long, device= device).view(1, -1)
        print(f"Input: {x}")
        print(f"Target: {y}")
        print(f"Predict: {eval_seq2seq(tensor_x, encoder, decoder)}")

def use_seq2seq(x):
    encoder = Encoder(len(eng_word2idx)).to(device)
    decoder = Decoder(len(fra_word2idx)).to(device)
    encoder.load_state_dict(torch.load(get_resource_path("encoder_model.pth")))
    decoder.load_state_dict(torch.load(get_resource_path("decoder_model.pth")))
    tmp_x = [eng_word2idx[word] for word in x.split(' ')]
    tmp_x.append(EOS_token)
    tensor_x = torch.tensor(tmp_x, dtype= torch.long, device= device).view(1, -1)
    return eval_seq2seq(tensor_x, encoder, decoder)


if __name__ == '__main__':
    dataset = DataSet(pairs)
    train_loader = torch.utils.data.DataLoader(dataset, batch_size= 1, shuffle=True)
    # test_decoder(train_loader)
    # train_model(train_loader, epochs= 10)
    # train_model(train_loader, epochs= 30)
    # test_use_seq2seq()
    # print(normalizeFra("Arrête-toi !"))