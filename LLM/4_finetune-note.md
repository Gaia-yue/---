# XTuner 实战

## 1.funetune

### 1.1 增量预训练 
理解为domain pretrain
![Alt text](image-24.png)
### 1.2 指令跟随微调

![Alt text](image-23.png)

instructed gpt 有监督训练
构建模版

#### 1.2.1 LORA & QLORA

![Alt text](image-25.png)
![Alt text](image-26.png)
- 量化4bitwow微调

## 2. Xtuner
    数据格式处理 对话模版 packdataset 加速训练
![Alt text](image-27.png)

### 数据格式处理
目标格式

![Alt text](image-28.png)
```json
[{
    "conversation":[
        {
            "system": "xxx",
            "input": "xxx",
            "output": "xxx"
        }
    ]
},
{
    "conversation":[
        {
            "system": "xxx",
            "input": "xxx",
            "output": "xxx"
        }
    ]
}]
```

### 对话模版
![Alt text](image-29.png)

### 加速训练
- deepspeed-zero
- falsh-attention

### 多数据拼接
![Alt text](image-31.png)

## 快速上手 

![Alt text](image-30.png)

![Alt text](image-32.png)

## 实战 
[文档](https://github.com/InternLM/tutorial/blob/main/xtuner/README.md)