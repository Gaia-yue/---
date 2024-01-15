#  Prompt

## 1. LLM intro
- 基础LLM
- 指令微调LLM

    基础LLM->指令微调->RLHF-->具备chat能力的大模型

## 2. 提示原则

**清晰而且具体**

### 2.1 善用分隔符

你可以选择用```，"""，< >，<tag> </tag>等等来分割
```python
text = f"""
您应该提供尽可能清晰、具体的指示，以表达您希望模型执行的任务。\
这将引导模型朝向所需的输出，并降低收到无关或不正确响应的可能性。\
不要将写清晰的提示词与写简短的提示词混淆。\
在许多情况下，更长的提示词可以为模型提供更多的清晰度和上下文信息，从而导致更详细和相关的输出。
"""
# 需要总结的文本内容
prompt = f"""
把用三个反引号括起来的文本总结成一句话。
1.2 寻求结构化的输出
有时候我们需要语言模型给我们一些结构化的输出，而不仅仅是连续的文本。
什么是结构化输出呢？就是按照某种格式组织的内容，例如JSON、HTML等。这种输出非常适合在代码
中进一步解析和处理。例如，您可以在 Python 中将其读入字典或列表中。
在以下示例中，我们要求 GPT 生成三本书的标题、作者和类别，并要求 GPT 以 JSON 的格式返回给我
们，为便于解析，我们指定了 Json 的键。
```{text}```
"""
# 指令内容，使用 ``` 来分隔指令和待总结的内容
response = get_completion(prompt)
```
### 2.2 指定结构化输出

```python
prompt = f"""
请生成包括书名、作者和类别的三本虚构的、非真实存在的中文书籍清单，\
并以 JSON 格式提供，其中包含以下键:book_id、title、author、genre。
"""
response = get_completion(prompt)
print(response)

```
### 2.3 提供示例
```python
prompt = f"""
您的任务是以一致的风格回答问题。
<孩子>: 请教我何为耐心。
<祖父母>: 挖出最深峡谷的河流源于一处不起眼的泉眼；最宏伟的交响乐从单一的音符开始；最复杂的挂毯以
一根孤独的线开始编织。
<孩子>: 请教我何为韧性。
"""
response = get_completion(prompt)

<祖父母>: 韧性是一种坚持不懈的品质，就像一棵顽强的树在风雨中屹立不倒。它是面对困难和挑战时不屈不
挠的精神，能够适应变化和克服逆境。韧性是一种内在的力量，让我们能够坚持追求目标，即使面临困难和挫折
也能坚持不懈地努力。
```

### 2.4 要求模型检查是否符合条件

```python

```

### 2.5 给模型时间去思考-指定完成任务所需的步骤

**错误的**
```python
text = f"""
在一个迷人的村庄里，兄妹杰克和吉尔出发去一个山顶井里打水。\
他们一边唱着欢乐的歌，一边往上爬，\
然而不幸降临——杰克绊了一块石头，从山上滚了下来，吉尔紧随其后。\
虽然略有些摔伤，但他们还是回到了温馨的家中。\
尽管出了这样的意外，他们的冒险精神依然没有减弱，继续充满愉悦地探索。
"""
# example 1
prompt_1 = f"""
执行以下操作：
1-用一句话概括下面用三个反引号括起来的文本。
2-将摘要翻译成英语。
3-在英语摘要中列出每个人名。
4-输出一个 JSON 对象，其中包含以下键：english_summary，num_names。
请用换行符分隔您的答案。
Text:
```{text}```
"""
```

```prompt 1:
1-两个兄妹在山上打水时发生意外，但最终平安回家。
2-In a charming village, siblings Jack and Jill set off to fetch water from a
well on top of a hill. While singing joyfully, they climbed up, but
unfortunately, Jack tripped on a stone and rolled down the hill, with Jill
following closely behind. Despite some minor injuries, they made it back to their
cozy home. Despite the mishap, their adventurous spirit remained undiminished as
they continued to explore with delight.
3-Jack, Jill
4-{"english_summary": "In a charming village, siblings Jack and Jill set off to
fetch water from a well on top of a hill. While singing joyfully, they climbed
up, but unfortunately, Jack tripped on a stone and rolled down the hill, with
Jill following closely behind. Despite some minor injuries, they made it back to
their cozy home. Despite the mishap, their adventurous spirit remained
undiminished as they continued to explore with delight.", "num_names": 2}

```
**正确的**


```python
prompt_2 = f"""
1-用一句话概括下面用<>括起来的文本。
2-将摘要翻译成英语。
3-在英语摘要中列出每个名称。
4-输出一个 JSON 对象，其中包含以下键：English_summary，num_names。
请使用以下格式：
文本：<要总结的文本>
摘要：<摘要>
翻译：<摘要的翻译>
名称：<英语摘要中的名称列表>
输出 JSON：<带有 English_summary 和 num_names 的 JSON>
Text: <{text}>
"""

```

```
prompt 2:
Summary: 在一个迷人的村庄里，兄妹杰克和吉尔在山顶井里打水时发生了意外，但他们的冒险精神依然没有
减弱，继续充满愉悦地探索。
Translation: In a charming village, siblings Jack and Jill set off to fetch water
from a well on top of a hill. Unfortunately, Jack tripped on a rock and tumbled
down the hill, with Jill following closely behind. Despite some minor injuries,
they made it back home safely. Despite the mishap, their adventurous spirit
remained strong as they continued to explore joyfully.
Names: Jack, Jill
JSON Output: {"English_summary": "In a charming village, siblings Jack and Jill
set off to fetch water from a well on top of a hill. Unfortunately, Jack tripped
on a rock and tumbled down the hill, with Jill following closely behind. Despite
some minor injuries, they made it back home safely. Despite the mishap, their
adventurous spirit remained strong as they continued to explore joyfully.",
"num_names": 2}

```
### 2.6 指导模型在下结论之前找出一个自己的解法
我们可以在 Prompt 中先要求语言模型自己尝试解决这个问题，思考出自己的解法，然后再与提供的解答进行对比，判断正确性.

```
prompt = f"""
请判断学生的解决方案是否正确，请通过如下步骤解决这个问题：
步骤：
首先，自己解决问题。
然后将您的解决方案与学生的解决方案进行比较，对比计算得到的总费用与学生计算的总费用是否一致，
并评估学生的解决方案是否正确。
在自己完成问题之前，请勿决定学生的解决方案是否正确。
使用以下格式：
问题：问题文本
学生的解决方案：学生的解决方案文本
实际解决方案和步骤：实际解决方案和步骤文本
学生计算的总费用：学生计算得到的总费用
实际计算的总费用：实际计算出的总费用
学生计算的费用和实际计算的费用是否相同：是或否
学生的解决方案和实际解决方案是否相同：是或否
学生的成绩：正确或不正确
问题：
我正在建造一个太阳能发电站，需要帮助计算财务。
- 土地费用为每平方英尺100美元
- 我可以以每平方英尺250美元的价格购买太阳能电池板
- 我已经谈判好了维护合同，每年需要支付固定的10万美元，并额外支付每平方英尺10美元;
作为平方英尺数的函数，首年运营的总费用是多少。
学生的解决方案：
设x为发电站的大小，单位为平方英尺。
费用：
1. 土地费用：100x美元
2. 太阳能电池板费用：250x美元
3. 维护费用：100,000+100x=10万美元+10x美元
总费用：100x美元+250x美元+10万美元+100x美元=450x+10万美元
实际解决方案和步骤：
"""

A:
实际解决方案和步骤：
1. 土地费用：每平方英尺100美元，所以总费用为100x美元。
2. 太阳能电池板费用：每平方英尺250美元，所以总费用为250x美元。
3. 维护费用：固定费用为10万美元，额外费用为每平方英尺10美元，所以总费用为10万美元+10x美
元。
4. 总费用：将上述三项费用相加，得到总费用为100x美元+250x美元+10万美元+10x美元=360x+10万
美元。
学生计算的总费用：450x+10万美元
实际计算的总费用：360x+10万美元
学生计算的费用和实际计算的费用是否相同：否
学生的解决方案和实际解决方案是否相同：否
学生的成绩：不正确
```

### LLM局限性

大模型会出现幻觉!

## 3.迭代优化

没什么好说的,根据prompt的completion质量 来调整prompt

## 4567. 文本概括 推断 文本扩展 聊天机器人 

应用没什么好说的:[详细解释](https://datawhalechina.github.io/prompt-engineering-for-developers/#/C1/6.%20%E6%96%87%E6%9C%AC%E8%BD%AC%E6%8D%A2%20Transforming)

总之来说,单纯跟大模型对话,他是不会记录你的上下文的,写代码时你需要将你的上下文传入进入:
```python
message = [
    {'role':'system', 'content':"对话文本"},
    {'role':'user', 'content':"对话文本"},
    {'role':'', 'content':"对话文本"},
    
]
```

跟大模型对话都有个数据模版,分三个角色:
- system
- assistant
- user
