# summarization

本项目详细的文档总结地址：https://shimo.im/docs/QDQ9c3pccYC3H3q6/ 《自动摘要项目实战》

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

baseline模型的ROUGE和BLUE分数如下：

```
---------------------------------------------
C ROUGE-1 Average_R: 0.29999 (95%-conf.int. 0.28621 - 0.31362)
C ROUGE-1 Average_P: 0.34839 (95%-conf.int. 0.33265 - 0.36404)
C ROUGE-1 Average_F: 0.31352 (95%-conf.int. 0.30032 - 0.32694)
---------------------------------------------
C ROUGE-2 Average_R: 0.18354 (95%-conf.int. 0.17160 - 0.19544)
C ROUGE-2 Average_P: 0.21595 (95%-conf.int. 0.20183 - 0.23081)
C ROUGE-2 Average_F: 0.19211 (95%-conf.int. 0.18038 - 0.20418)
---------------------------------------------
C ROUGE-3 Average_R: 0.10441 (95%-conf.int. 0.09376 - 0.11476)
C ROUGE-3 Average_P: 0.12377 (95%-conf.int. 0.11081 - 0.13763)
C ROUGE-3 Average_F: 0.10906 (95%-conf.int. 0.09819 - 0.12027)
---------------------------------------------
C ROUGE-4 Average_R: 0.06478 (95%-conf.int. 0.05632 - 0.07375)
C ROUGE-4 Average_P: 0.07823 (95%-conf.int. 0.06647 - 0.09075)
C ROUGE-4 Average_F: 0.06777 (95%-conf.int. 0.05854 - 0.07744)
---------------------------------------------
C ROUGE-L Average_R: 0.27605 (95%-conf.int. 0.26321 - 0.28968)
C ROUGE-L Average_P: 0.32101 (95%-conf.int. 0.30519 - 0.33750)
C ROUGE-L Average_F: 0.28858 (95%-conf.int. 0.27572 - 0.30193)
---------------------------------------------
C ROUGE-W-1.2 Average_R: 0.14713 (95%-conf.int. 0.13999 - 0.15459)
C ROUGE-W-1.2 Average_P: 0.29850 (95%-conf.int. 0.28416 - 0.31342)
C ROUGE-W-1.2 Average_F: 0.19107 (95%-conf.int. 0.18236 - 0.19999)
---------------------------------------------
C ROUGE-SU4 Average_R: 0.15021 (95%-conf.int. 0.14023 - 0.16112)
C ROUGE-SU4 Average_P: 0.18096 (95%-conf.int. 0.16815 - 0.19433)
C ROUGE-SU4 Average_F: 0.15740 (95%-conf.int. 0.14716 - 0.16851)
BLEU =  0.1376334260922834
BLEU1 =  0.33730819005361434
BLEU2 =  0.20806499554146438
BLEU3 =  0.11998292058070026
BLEU4 =  0.07578387134096957
ratio =  0.8741818181818182
```


pointer-generator模型的ROUGE和BLUE分数如下：

​ 

```
---------------------------------------------
C ROUGE-1 Average_R: 0.34390 (95%-conf.int. 0.33003 - 0.35788)
C ROUGE-1 Average_P: 0.36084 (95%-conf.int. 0.34593 - 0.37624)
C ROUGE-1 Average_F: 0.34581 (95%-conf.int. 0.33175 - 0.36055)
---------------------------------------------
C ROUGE-2 Average_R: 0.21694 (95%-conf.int. 0.20491 - 0.22981)
C ROUGE-2 Average_P: 0.23039 (95%-conf.int. 0.21705 - 0.24446)
C ROUGE-2 Average_F: 0.21881 (95%-conf.int. 0.20647 - 0.23169)
---------------------------------------------
C ROUGE-3 Average_R: 0.12922 (95%-conf.int. 0.11841 - 0.14054)
C ROUGE-3 Average_P: 0.13889 (95%-conf.int. 0.12671 - 0.15180)
C ROUGE-3 Average_F: 0.13055 (95%-conf.int. 0.11953 - 0.14210)
---------------------------------------------
C ROUGE-4 Average_R: 0.08241 (95%-conf.int. 0.07325 - 0.09208)
C ROUGE-4 Average_P: 0.08960 (95%-conf.int. 0.07918 - 0.10143)
C ROUGE-4 Average_F: 0.08339 (95%-conf.int. 0.07412 - 0.09336)
---------------------------------------------
C ROUGE-L Average_R: 0.31482 (95%-conf.int. 0.30187 - 0.32802)
C ROUGE-L Average_P: 0.33107 (95%-conf.int. 0.31705 - 0.34603)
C ROUGE-L Average_F: 0.31686 (95%-conf.int. 0.30386 - 0.33020)
---------------------------------------------
C ROUGE-W-1.2 Average_R: 0.16729 (95%-conf.int. 0.16026 - 0.17443)
C ROUGE-W-1.2 Average_P: 0.30668 (95%-conf.int. 0.29398 - 0.32065)
C ROUGE-W-1.2 Average_F: 0.21172 (95%-conf.int. 0.20289 - 0.22050)
---------------------------------------------
C ROUGE-SU4 Average_R: 0.17964 (95%-conf.int. 0.16869 - 0.19068)
C ROUGE-SU4 Average_P: 0.19243 (95%-conf.int. 0.18006 - 0.20561)
C ROUGE-SU4 Average_F: 0.18090 (95%-conf.int. 0.16970 - 0.19237)
BLEU =  0.1658349533581727
BLEU1 =  0.3532617952648612
BLEU2 =  0.2233511748162932
BLEU3 =  0.13354037267080746
BLEU4 =  0.08607198748043818
ratio =  0.9565784114052953
```


unilm模型的ROUGE和BLUE分数如下：

​ 

```
---------------------------------------------
C ROUGE-1 Average_R: 0.38167 (95%-conf.int. 0.36632 - 0.39655)
C ROUGE-1 Average_P: 0.44924 (95%-conf.int. 0.43160 - 0.46684)
C ROUGE-1 Average_F: 0.40277 (95%-conf.int. 0.38721 - 0.41770)
---------------------------------------------
C ROUGE-2 Average_R: 0.24827 (95%-conf.int. 0.23445 - 0.26198)
C ROUGE-2 Average_P: 0.29750 (95%-conf.int. 0.28150 - 0.31351)
C ROUGE-2 Average_F: 0.26313 (95%-conf.int. 0.24934 - 0.27698)
---------------------------------------------
C ROUGE-3 Average_R: 0.15213 (95%-conf.int. 0.13984 - 0.16451)
C ROUGE-3 Average_P: 0.18439 (95%-conf.int. 0.16969 - 0.19936)
C ROUGE-3 Average_F: 0.16123 (95%-conf.int. 0.14811 - 0.17388)
---------------------------------------------
C ROUGE-4 Average_R: 0.09810 (95%-conf.int. 0.08754 - 0.10878)
C ROUGE-4 Average_P: 0.12029 (95%-conf.int. 0.10744 - 0.13367)
C ROUGE-4 Average_F: 0.10402 (95%-conf.int. 0.09302 - 0.11529)
---------------------------------------------
C ROUGE-L Average_R: 0.34777 (95%-conf.int. 0.33300 - 0.36263)
C ROUGE-L Average_P: 0.40996 (95%-conf.int. 0.39293 - 0.42726)
C ROUGE-L Average_F: 0.36707 (95%-conf.int. 0.35218 - 0.38138)
---------------------------------------------
C ROUGE-W-1.2 Average_R: 0.18180 (95%-conf.int. 0.17371 - 0.18967)
C ROUGE-W-1.2 Average_P: 0.37444 (95%-conf.int. 0.35849 - 0.38981)
C ROUGE-W-1.2 Average_F: 0.23826 (95%-conf.int. 0.22859 - 0.24818)
---------------------------------------------
C ROUGE-SU4 Average_R: 0.20824 (95%-conf.int. 0.19580 - 0.22100)
C ROUGE-SU4 Average_P: 0.25573 (95%-conf.int. 0.23994 - 0.27071)
C ROUGE-SU4 Average_F: 0.22123 (95%-conf.int. 0.20810 - 0.23426)
BLEU =  0.19278854316990632
BLEU1 =  0.438580015026296
BLEU2 =  0.288320064496624
BLEU3 =  0.17851706892802782
BLEU4 =  0.11707777646642276
ratio =  0.8604444444444445
```


​      