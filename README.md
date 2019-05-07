# IJCAI_Keras_Defense
Solution for IJCAI Competition (Defense), coded by Keras &amp; Tensorflow

## Dependency
    python 3.5.4
    Keras == 2.2.4
    imgaug == 0.2.6
    tensorflow-gpu == 1.12.0

## Instruction
### 代码结构
    |Projects - |models - 存放训练好的模型（需要自己创建）
                       
                |pretrained——model - 存放预训练模型（需要自己创建，模型可从release中下载）
                        
                |utils  - data_generator.py  数据读取、生成
                        - iaa_process.py  数据预处理、数据
                        - denseMoE.py   Mixtures of Experts
                        
                |defense.py  预测结果脚本
                
                |train_simplified.py   训练单模型脚本
                
                |train_ensemble.py   训练融合模型脚本
                
                |validation.py  验证单模型脚本
                
                |validation_ensemble.py  验证融合模型脚本
                
 ### 使用说明
 （1） 首先在训练脚本中配置好使用的csv文件和input dir路径;csv需要指定train和val，可自行拆分input dir路径需要指定到保存数据的上一层
 
 （2） 工程可以使用单GPU或多GPU训练，在multi gpus中设置即可;GPU使用数量需要到代码里自行修改
 
 （3） 训练时可以加载imagenet预训练模型，这时将finetune dir设置为None;如果需要用模型finetune，就指定到模型路径的名字
 
 （4） model type需要与加载的预训练模型匹配上，否则会出现错误
 
 （5） 训练时开启augmentation会将训练速度减低到一半左右，但会提升5% - 8%个百分点的准确率

