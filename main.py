import eng2fra
import time

print(f"{time.strftime('%Y-%m-%d-%H-%M')}")
inputs = input("请输入(English Only): ")

# 处理inputs
inputs = eng2fra.normalizeEng(inputs)

print(eng2fra.use_seq2seq(inputs))