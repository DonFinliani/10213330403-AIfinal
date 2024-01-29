


@[TOC](目录)

---

# 当代人工智能lab5
`10213330403 新闻双学位张珈鸣


---


# 一、环境
numpy==1.25.0
opencv_python==4.9.0.80
pandas==2.0.3
Pillow==10.0.0
Pillow==10.2.0
scikit_learn==1.3.1
torch==2.1.2+cu121
torchvision==0.16.2+cu121
torchvision==0.16.2
transformers==4.36.1
~orch==2.0.1
~orch==2.1.2

# 二、实验文件
```c
│   dataloader.py	数据加载与预处理
│   lr_select.py	学习率搜索训练
│   main.py			multi、text、image三种模式的训练以及验证比对
│   model.py		模型构建
│   predict.py		使用multi模型生成预测文件
│   requirements.txt	
│   train.py		具体训练方法代码
│   val.py			具体验证方法代码
│   10213330403张珈鸣lab5实验报告.md
│   README.md
│   
├───Data
│   │   predict.txt
│   │   result.txt
│   │   test_without_label.txt
│   │   train.txt
│   │   
│   ├───data     
  
├───model
│       image_7e-05_dropout0.6_batch_full.pth
│       multi_7e-05_dropout0.6_batch_full.pth
│       text_7e-05_dropout0.6_batch_full.pth
```
# 三、代码运行
main.py文件使用argparse包封装，参数如下
```c
parser=argparse.ArgumentParser(description="")
parser.add_argument('--mode',type=str,help='choose a mode',default='all')
parser.add_argument('--lr',type=float,help="choose learning_rate",default=7e-5)
parser.add_argument('--epochs',type=int,help="epochs num",default=5)
parser.add_argument('--dropout',type=float,help='set arg of dropout',default=0.6)
args=parser.parse_args()
```
可使用默认值运行，进行三种模式的训练与验证比对,也可输入multi、text、image进行单一模型的训练与验证。
```c
 python main.py // python --mode all --lr 7e-5 --epochs 5 --dropout 0.6
  ```
---
# 四、参考库
使用torch、torchvision、PIL、pandas、numpy、transformers、sklearn库实现
