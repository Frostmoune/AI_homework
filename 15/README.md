## 文件说明
- Utils.py: 一些辅助函数，如im2col
- Optimizer.py: 定义了adam和动量sgd两个优化器
- DataHandler.py: 数据处理方法  
- MyLayers.py: 定义各个层（如卷积层，池化层）
- CNN.py: 定义了一个简单的CNN网络  
- main.py: 训练和测试  
**运行前请保证上述文件在同一目录**  
- log.txt: 一次运行的结果记录  

## 运行说明  
- 运行环境：python3.6 64位（64位的Windows和Linux均可）
- 依赖包：numpy
- 运行方法：
    1. 从http://www.cs.toronto.edu/~kriz/cifar.html下载**cifar-10-python.tar.gz**  
    2. 建立**datasets**文件夹，将解压得到的**cifar-10-batches-py**放到**datasets**中
    3. 运行命令`python main.py`  
    4. 可通过修改main.py中的配置自行修改网络和其他超参数

**正确率最高能到55%左右，但训练时间较长，一个epoch需要训练约45分钟**  