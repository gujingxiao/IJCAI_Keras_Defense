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
                        - process_labels.py  label的编解码
                        
                |ensemble.py   模型融合代码
                
                |train.py   训练脚本
                
                |val_inference.py  验证及生成提交数据脚本
                
                |results_correction.py  修正预测值位置
