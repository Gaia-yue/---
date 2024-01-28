# Model achitecture

## word piece

语言模型token构建在某个vocab上，要让计算机理解人类语言，需要将其转换成计算机能识别的语言。
- 基于空格
- bite pair encodeing
- unigram mdoel(sentence peice)


### bite pair encoding

```python

input:
I = [['the car','the cat','the rat']]

def BPE(input:I):
    ### step 1 ###
    v1 = text.split('')
    ### step 2  ###
    ###使用贪心算法来计算字母对出现的次数，最终计算词元的个数

```

### unigram model(sentence piece)

不同于BPE使用规则来捕捉词元，更有意思的方法是定义一个目标函数来捕捉词元。
假设T是分词器，那么
$$
T是 p(x_{1:L})= \prod _{(i,j) \in T} p(x_{i:j})的一个集合
$$

例子：
1. T(x), x:ababc ，V={ab,c}
2. T = (1,2),(3,4),(5)
3. 似然值： 2/3 * 2/3 * 1/3 = 4/27

高的似然值表明分词的效果越好

```python
def Piece(input):
    # 给定V 使用em算法优化p(x)和T
    # 计算每个词汇x \in{v} 的loss, 衡量将x移除似然值会不会减少
    # 按照loss进行排序. 并保留v中排名靠前的80%的词汇

    return V
```

## 模型架构


语言模型始终遵循$p(x_1,... . .x_L)$, 要实现语言模型. 
其中, 上下文向量表征(contextual embedding)作为先决条件:
$$
[the, mouse, ate, the, cheese] \implies word embeddings
$$

- $V_{L}\to R^{d\times L}$, 根据词表来生成词向量嵌入
- 对于$x_{1:L} = [x_1, x_2, x_3,.... .  . x_L]$, 生成上下文向量表征.

### 语言模型种类
- encoder-only
- decoder-only
- encoder-decoder

### encoder-omly
著名的编码器架构有bert roberta等等, 当然这些编码器架构只能生成上下文向量表征, 经常外接一个linear分类器用来做分类任务. 

例如:

    情感分析:[[cls],佳能,感动,常在]-> 正面情绪(1)


转换形式:$x_{1:L}\to {x_1^D,x_2^D,.. . .. x_L^D}$

缺点:

    不能独立自然生成文本, 需要更多预训练, 如mlm等等


### decoder-only

著名案例: GPT系列 bart等等. 输入$x_{1:L}$, 可以生成上下文表征向量,可以对于下一个token生成概率分布. $p(x_{x+1}|x_{1:I})$.

生成形式:
    
$$
    [[cls],佳能,感动] \implies [常在]
$$


有简单的最大似然的训练目标, 但是对于每个$x_i$, 上下文向量表征只能单向的依赖于左侧上下文$x_{1:i-1}$


### base achitecture

#### Embedding

将词元序列转换成向量

$$ [x_{1:L}] $$

```python
class Embeddings(nn.Module):
    def __init__(self, d_model, vocab):
        super(Embeddings, self).__init__()
        self.lut = nn.Embedding(vocab, d_model)
        self.d_model = d_model

    def forward(self, x):
        return self.lut(x) * math.sqrt(self.d_model)
```

#### embeddings 

input :  $x_{1:L}$
- 首先将我们的输入padding到$sequence-length$的长度得到$x_{1:sequencelength}$。
- 接着我们根据$V^L$和$d model$来定义embedding的大小。
- 输入$x_{sequencelength}$得到序列$S^{d\times sl}$ ， 最后我们将$\sqrt {model^d} \times S^{d\times sl}$得到输入进整个模型的输入。

通常我们会在两个嵌入层以及pre-softmax和linear之间共享参数.  

函数解释 : ```nn.embedding(vocab, d_model)```$R^L\to R^{d\times L}$

```python
class Embeddings(nn.Module):
    def __init__(self, d_model, vocab):
        super(Embeddings, self).__init__()
        self.lut = nn.Embedding(vocab, d_model)
        self.d_model = d_model

    def forward(self, x):
        return self.lut(x) * math.sqrt(self.d_model)
```

#### Sequence model


$$
def \space Senquencemodel(S^{d \times sl})\to  [h_{1:sl}]
$$  

- 针对每个$x_i$进行处理, 考虑其他元素

有多种实现方式: $Feedforwoard \space RNN \space Transformer$
**针对最简单的feedforword**:

由条件概率:
$$
P(A|B) = \frac{P(AB)}{P(B)} 
$$
$$
P(x_{1:i}) = P(x_1)P(x_2|x_1)P(x_3|x_1,x_2)\cdots P(x_L|x_{1:L})
$$
对于类似n-gram模型来说
$$
P(x_{i-n+1:i}) = p(x_{i-n+1}) P(x_{i-n+2}|x_{i-n+1}) \cdots    P(x_{i-1}|x_{i-n+1:i-1})
$$
1. 对于每个$x_{i}$类似于ngram考虑前n个元素.
2. 所以对于每个$x_i$, 都计算$h_i = Feedforword(x_{i-n+1}, \ldots x_{i})$
3. 返回$[h_1, h_2 ,h_3\cdots,h_L]$

#### 递归神经网络序列模型

**包含rnn lstm gru**

$$
def \space sequenceRNN(x:R^{d\times SL})\to R^{d\times SL} 
$$

- 由于得天独厚的物理位置, rnn从左往右开始处理数据. 对于每个$x_i$,记录以往的信息,都计算$h_i = RNN(h_{i-1}, x_i)$
- 最终返回$[h_1,h_2\cdots.h_{L}]$

###### rnn实现方式
1. 简单RNN
$$
def \space simpleRNN(x_{i}:R^{d})\to R^d
$$
- 通过$x_i \space and \space h_{i-1}$来更新$h_i$隐藏状态.
- return $\sigma (z) = \sigma(Uh, Vx, b) $ or $\sigma(z) = \sigma(\max(0,z))$

1. 双向RNN:bidrectionalRNN
$$
def \space bidirectional(x:R^d)\to R^{2d}
$$
- 同时从左往右处理数据, 相当于与左右各一个RNN
- 返回$[h_{1\to },h_{ \leftarrow 1}, \ldots h_{L\to },h_{ \leftarrow L} ]$

#### 注意

- simple rnn很难训练
- 结构更简单的LSTM和GRU变得更简单更容易训练
- 对于超长文本, RNN系列模型无能为力

自LSTM, 深度学习真正进入了NLP领域.

#### Transformer

##### (self)注意力机制
 

$$
At tention(Q,K,V) = \operatorname{softmax}(\frac{QK^T}{\sqrt{d{}}})V
$$

每个key通过与整个query相乘而后进行归一化来得到, 整个序列中的token对$x_i$的注意力分数. 最后将value与$\alpha$相乘得到输出.
$$
scores = \operatorname{softmax}(\frac{QK^T}{\sqrt{d{}}})\implies[\alpha_1,\alpha_2, \ldots \alpha_L]
$$

$$
key = W_{k}x_{i}\space ,where\space W\to R^{L\times d}
$$

$$
values = W_{v}x_{i}
$$

$$
query = W_{q}y
$$
抽象实现:
$$
def \space selfattention(x_{1:L}:R^{d\times L})\to R^{d\times L}
$$

```python
def attention(query, key, value, mask=None, dropout=None):
    "Compute 'Scaled Dot Product Attention'"
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) \
             / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = F.softmax(scores, dim = -1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn
```
##### 多头自注意力机制


$$
attention(x) = \operatorname{softmax}({\frac{Q K^T}{\sqrt d} }) V
$$

$$
Multiheadattention(Q, K ,V) = concat(head_1,head_2, \ldots head_n)W^o, \space     
$$

$$
where \space the \space head_i =attention(W^qQ,W^vV,W^kK) 
$$

$$
(QKV) \space is\space R^{d\times L}, \space W^q \space is \space R^{d\times \frac{L}{h_{number}}}
$$
```python
class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        "Take in model size and number of heads."
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)
        
    def forward(self, query, key, value, mask=None):
        "Implements Figure 2"
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)
        
        ##第一个维度 sequence length
        nbatches = query.size(0)
        
        # 1) Do all the linear projections in batch from d_model => h x d_k 


        query, key, value = \
            [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
             for l, x in zip(self.linears, (query, key, value))]
        
        # 2) Apply attention on all the projected vectors in batch. 
        x, self.attn = attention(query, key, value, mask=mask, 
                                 dropout=self.dropout)
        
        # 3) "Concat" using a view and apply a final linear. 
        x = x.transpose(1, 2).contiguous() \
             .view(nbatches, -1, self.h * self.d_k)
        return self.linears[-1](x)
```

尤其的在decoder中为防止注意力生成时关注到句子的每一个角落,添加mask
```python
def subsequent_mask(size):
    "Mask out subsequent positions."
    attn_shape = (1, size, size)
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    return torch.from_numpy(subsequent_mask) == 0
```

##### Feedforward networks

```def feedforwoard(x)``` $x_{1:L}:R^{d\times L} \to R^{d\times L}$

- $y_i = W_2\max(W_1x_i+b_1,0)+b_2$

实际实现添加dropout, 防止过拟合.

```python
class PositionwiseFeedForward(nn.Module):
    "Implements FFN equation."
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))
```

##### resnet

```python
class SublayerConnection(nn.Module):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """
    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        "Apply residual connection to any sublayer with the same size."
        return x + self.dropout(sublayer(self.norm(x)))
```


##### LayerNorm
代码很简单, 层归一化的目的防止$x_i$过大或者过小.
```python
class LayerNorm(nn.Module):
    "Construct a layernorm module (See citation for details)."
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2
```

##### position Embedding
为了让模型理解位置信息, 加入position信息.

```def position embeding()```
- 偶数信息:$p_{i,2j} = \sin(\frac{i}{10000^{\frac{2j}{d_model}}})$
- 奇数信息:$p_{i,2j+1} = \cos(\frac{i}{10000^{\frac{2j}{d_model}}})$
- 返回$[x_i+p_i\cdots]$

i表示句子中的词元位置, j表示句子中的向量维度位置.


##### 补充
```python
def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])
```
## 总结
- 将transfomerblock堆叠N次就得到了一个超大的模型.