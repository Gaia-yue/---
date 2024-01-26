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

