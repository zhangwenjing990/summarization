# summarization
## 任务描述

​	摘要是对一段文本重要内容的总结。目前基于NLP的文本摘要任务主要有两种方式：抽取式和生成式。

​	抽取式主要是通过从文本中抽取出最重要的几句话来完成。

​	生成式摘要是模型根据对文本的理解，生成新的序列。本文基于生成式实现自动摘要生成。

## 数据描述

本项目采用LCSTS数据集。[A Large-Scale Chinese Short Text Summarization Dataset](http://icrc.hitsz.edu.cn/Article/show/139.html)

本文中，将PART-I部分作为训练集，筛选出数据2400545条；PART-II部分作为验证集，筛选出数据8685条；PART-III部分作为测试集，筛选出数据725条。

## 评价指标

本文项目采用ROUGE和BLUE方法评估模型结果。



## 模型结果

baseline模型的rouge分数如下：

​        ![img](https://uploader.shimo.im/f/TjdZlYsS5Jawf4MW.png!thumbnail)
 

pointer-generator模型的rouge分数如下：

​        ![img](https://uploader.shimo.im/f/uGxV3X2dTxmHxFY2.png!thumbnail)
  

unilm模型的rouge分数如下：

​        ![img](https://uploader.shimo.im/f/bkIPov61ZA973N5b.png!thumbnail)
​      