### AutoNLP
## 1.1 代码目录结构
```
    |--app/: 微调模型功能目录
    |    |--run_classification.py: 文本分类任务模型训练预测
    |
    |--conf/: 配置文件目录
    |    |--classification_config.json/: 文本分类任务参数输入配置文件
    |
    |--datasets/: 下游任务数据集存储目录
    |
    |--model/: 模型存储目录
    |    |--chn/: 任务名为chn的模型存储目录，自定义
    |    |--pretrain_model/: 预训练模型缓存目录，自定义
    |      
    |--utils/: 本项目所依赖的工具类和函数目录
    |    |--convertutils/: 模型之间转换工具脚本目录
    |        |--convert_tf_checkpoint_to_pytorch.py: 将模型从tf转为pytorch
    |    |--datautils/: 数据处理工具脚本目录
    |        |--classifier_data_processor.py: 文本分类任务相关数据处理类和函数
    |    |--common_utils.py: 通用工具类和函数
    |    |--file_utils.py: 模型网络获取相关文件操作工具
    |    |--metric_utils.py: 任务评测指标函数
    |    |--model_utils.py: 模型操作相关函数
    |
    |--run.sh: 任务执行脚本
```

## 1.2 数据集定义

样本集目录结构
- 训练集：train.txt
- 评测集：dev.txt

### 1.2.1 文本分类

数据集中一行为一个训练样本，编码为utf-8。

- Pointwise任务数据集格式为

```bash
分类类目\t文本内容
```

- 示例
```bash
1	小本子不错，1699元买的，觉得很实惠。 做工不错，和同事一共买了3台，粉色最好看。
0	太差了，空调的噪音很大，设施也不齐全，携程怎么会选择这样的合作伙伴
1	最超值的是6芯电池 华硕的质量有保障， 键盘打字舒服。没有缩小的感觉。
0	钢琴烤漆就是一个指纹收集器，还是喜欢以前的磨砂面，电脑不带系统盘，重新分区费了很长时间，而且感觉该机器不是很结实。
```

- Pairwise任务数据集格式为

```bash
分类类目\t文本内容\t文本内容
```

- 示例
```bash
1	一盒香烟不拆开能存放多久？	一条没拆封的香烟能存放多久。
0	什么是智能手环	智能手环有什么用
1	您好.麻烦您截图全屏辛苦您了.	麻烦您截图大一点辛苦您了.最好可以全屏.
0	苏州达方电子有限公司联系方式	苏州达方电子操作工干什么
```

## 1.3 任务json文件配置说明

### 1.3.1 文本分类

配置文本分类任务的json文件，其模板位于./conf/classification_config.json, 每个配置项具体说明如下：
```bash
{
    "local_rank": -1,                                 # 用于gpu的分布式训练的局部秩
    "max_seq_length": 512,                            # 最大序列长度
    "learning_rate": 5e-5,                            # 学习率
    "do_lower_case": 1,                               # 转换为小写
    "train_batch_size": 32,                           # 训练的batch size
    "eval_batch_size": 8,                             # 评价的batch size
    "test_batch_size": 8,                             # 测试的batch size
    "num_labels": 2,                                  # 数据集类别数量
    "use_cuda" : 1,                                   # 是否使用cuda
    "gpu_device": "0,3",                              # 使用的GPU
    "epochs" : 3,                                     # 模型迭代次数
    "num_workers" : 4,                                # 分布式读数据的worker数量
    "gradient_accumulation_steps" : 1,                # 梯度累积步数
    "warmup_proportion" : 0.1,                        # 热启动比例
    "seed": 42,                                       # 随机器种子
    "cache_dir": "./AutoNLP/model/pretrain_model/",   # 预训练存放的目录
    "output_dir": "./AutoNLP/model/"                  # 模型输出的目录
}
```

## 1.4 评测机制

### 1.4.1 文本分类

- 准确率(Accuarcy): 

调用示例：
```bash
from utils.metric_utils import simple_accuracy
result = simple_accuracy(preds, y_trues)
```
- F1值:

公式如下: 

$$ F1=2\cdot \frac{accuarcy\cdot recall}{accuarcy+recall} $$

调用示例：
```bash
from utils.metric_utils import acc_and_f1
result = acc_and_f1(preds, y_trues)
```

- 皮尔森相关性系数 (Pearson)

公式如下:

$$\rho = \frac{cov(X,Y)}{\sigma_{X}\sigma_{Y}} = \frac{E((X-\mu_{X})(Y-\mu_{Y}))}{\sigma_{X}\sigma_{Y}} = \frac{E(XY)-E(X)E(Y)}{\sqrt{E(X^{2})-E^{2}(X))}\sqrt{E(Y^{2})-E^{2}(Y))}}$$

X，Y两个变量的协方差与两个变量的标准差之积的比值。所以X，Y两个变量的标准差不能为零。皮尔森相关系数受异常值的影响比较大。

评测的标准是预测的序列值与真实标签序列值之间的相关性。

其值范围为-1到+1，0表示两个变量不相关，正值表示正相关，负值表示负相关，值越大表示相关性越强。

调用示例：
```bash
from utils.metric_utils import pearson_and_spearman
result = pearson_and_spearman(preds, y_trues)
```

- 斯皮尔曼相关性系数 (Spearman)

斯皮尔曼相关性系数，通常也叫斯皮尔曼秩相关系数。“秩”，可以理解成就是一种顺序或者排序，那么它就是根据原始数据的排序位置进行求解，这种表征形式就没有了求皮尔森相关性系数时那些限制。

其公式如下:

$$\rho_{s} = 1 - \frac{6\sum d^{2}}{n(n^{2}-1)}$$

计算过程就是：

对两个变量$(X,Y)$的数据进行排序（统一用升序或降序），每个变量在排序之后的位置即为其秩次$(X',Y')$，原始位置相同的$X$，$Y$的秩次$X'$, $Y'$的差值即为$d_{i}$。$n$是变量的个数（或者对数）
Spearman是根据变量的大小顺序所确定的，所以一个异常值不会对Spearman相关系数的计算造成很大影响。

其值范围为-1到+1，0表示两个变量不相关，正值表示正相关，负值表示负相关，值越大表示相关性越强。

调用示例：
```bash
from utils.metric_utils import pearson_and_spearman
result = pearson_and_spearman(preds, y_trues)
```
