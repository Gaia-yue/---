# Demo
    任务：
        1.InternLM-Chat-7B 智能对话
        2.Lagent工具调用解简单数学题
        3.浦语·灵笔多模态图文创作和理解

##  InternLM-Chat-7B 智能对话

Q ：帮我写一个使用bart基座模型mlm预训练代码

A：
~~~python 
import torch
import torch.nn as nn
import torch.optim as optim

# 加载BART预训练模型
model = MyBARTModel()

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 准备训练数据
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)

# 训练模型
for epoch in range(10):
    running_loss = 0.0
    for i, data in enumerate(train_loader, 0):
        inputs, labels = data
        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if i % 100 == 99:
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 100))
            running_loss = 0.0
~~~

Q: 请你告诉我如何创建一个django 网站
A:
安装Python和Django：首先你需要安装Python和Django。Django是一个Python的Web框架，可以让你快速构建Web应用程序。你可以使用pip命令来安装Django。在命令行中执行以下命令即可安装Django：
```
pip install Django
```

创建Django项目：在命令行中执行以下命令来创建一个Django项目：
```
django-admin startproject projectname
```

其中，projectname是你要创建的项目名称。

创建Django应用：在命令行中执行以下命令来创建一个Django应用：
```
python manage.py startapp appname
```
其中，appname是你要创建的Django应用名称。

配置数据库：Django默认使用SQLite数据库，但你也可以使用其他类型的数据库。在settings.py文件中进行数据库配置，例如：
```DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.sqlite3',
        'NAME': BASE_DIR / 'db.sqlite3',
    }
}```

创建视图：在Django应用中创建视图函数，用于处理HTTP请求并返回响应。例如：
```from django.http import HttpResponse
from app.models import MyModel

def my_view(request):
    my_model = MyModel.objects.all()
    return HttpResponse(str(my_model))
```

创建模板：在Django应用中创建一个HTML模板，用于显示视图函数返回的内容。例如：
```
<!DOCTYPE html>
<html>
<head>
    <title>{% block title %}My App{% endblock %}</title>
</head>
<body>
    <h1>{% block content %}My App{% endblock %}</h1>
</body>
</html>
```


配置URL路由：在Django应用的urls.py文件中配置URL路由，以便将URL映射到视图函数。例如：
```
from django.urls import path
from. import views

urlpatterns = [
    path('my-view/', views.my_view, name='my_view'),
]

```

运行开发服务器：在命令行中执行以下命令来启动Django开发服务器：
python manage.py runserver

这样就可以在浏览器中访问你的博客网站了。

需要注意的是，这只是一个简单的例子，实际的博客网站可能需要更多的功能和配置。

感受：不错。

## Lagent工具调用解简单数学题

1. 抽象数学题

![Alt text](image-5.png)
![Alt text](image-6.png)

2.leecode两数之和
![Alt text](image-7.png)
![Alt text](image-8.png)
![Alt text](image-9.png)
![Alt text](image-10.png)
![Alt text](image-11.png)


## 浦语·灵笔多模态图文创作和理解

### 创作
![Alt text](image-13.png)

### 对话
![Alt text](image-12.png)