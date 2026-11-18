from collections import Counter
import numpy as np
import math

with open("/Users/mymorningstar/edge-data/got.txt", "r" ) as f:
    text = f.read()



counts = Counter(text)



# print("Unique characters:", len(counts))
# for char, n in counts.most_common():    # sorted from most to least frequent
#     print(repr(char), n)


kept = {c for c, _ in counts.most_common()[:-15]} 
# Character level Vocabulary 
chars = sorted(set(kept))      
stoi = {c: i for i, c in enumerate(chars)} 
ctoi = {i: c for c, i in stoi.items()}


def encode(text):
    encoded_text = []
    for char in text: 
        if char in kept:
            encoded_text.append(stoi[f'{char}'])
        else:
            pass

    return encoded_text

def decode(text):
    decoded_text = []
    for id in text: 
        decoded_text.append(ctoi[id])

    return ''.join(decoded_text)

encoded_text = encode(text)
total = len(encoded_text)
train_set = encoded_text[:-math.floor(total*0.1)]
test_set = encoded_text[-math.floor(total*0.1):]


def batch(b, t, data):
    x_batch = np.zeros((b,t), dtype=int)
    y_batch = np.zeros((b,t), dtype=int) 

    count = 0
    while count <b:
        rng = np.random.default_rng()
        i = rng.integers(0,len(data)-t-1)
        x_batch[count] = data[i: i+t]
        y_batch[count] = data[i+1: i+t+1]
        count+= 1

    return x_batch, y_batch




if __name__ =="__main__":

    print(decode(encoded_text[1000:2000]))
    
    
    x, y = batch(32, 256, train_set)
