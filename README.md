# forwarder.backend.dl4j
## 简介
该项目是[Forwarder](https://github.com/onnx4j/forwarder)的Backend项目，其基于Deeplearning4j（Nd4j）实现所有ONNX的Operator的运算。用户若希望使用基于Nd4j完成forward/inference操作，可依赖此项目。

## 使用
### 运行要求
* Linux/MacOS/Windows
* Maven
* Oracle JRE 1.8
* 符合ONNX规范的模型文件（当前最高支持v9的指令集）
> 注意：对于操作系统与JRE的要求视Nd4j的支持而定。

### 依赖
* nd4j 1.0.0-beta6

### 快速开始
* 导入Maven依赖包
```
<dependency>
  <groupId>org</groupId>
  <artifactId>forwarder.backend.dl4j</artifactId>
  <version>0.0.1</version>
</dependency>
```
> 备注：由于forwarder.backend.dl4j项目还没有上传至Maven中心仓库，请开发者先自行浏览我们的github，checkout所有必须的项目。
* 样例模型

我们构建一个简易的ONNX模型作用演示使用，以下是该模型的结构图：

![model.onnx](https://raw.githubusercontent.com/onnx4j/onnx4j/master/docs/images/simple_model.png?raw=true "model.onnx")

开发者可以点击 [这里](https://github.com/onnx4j/forwarder.demo/tree/master/src/test/resources/simple) 下载该模型。

* 加载与执行ONNX模型
```
String modelPath = "./model.onnx";
Model model = Forwarder.load(modelPath, Config
        .builder()
        .setDebug(true)
        // 内存存储顺序
        .setMemoryByteOrder(ByteOrder.LITTLE_ENDIAN)
        // 使用off-heap内存
        .setMemoryAllocationMode(AllocationMode.DIRECT)
        // RecursionExecutor.class:递归式图遍历执行器，RayExecutor:非递归式图遍历执行器
        .setExecutor(RecursionExecutor.class)
        .build()
    );

try (Backend<?> backend = this.model.backend(backendName)) {
    Tensor x2_0;
    Tensor y0;
    try (Session<?> session = backend.newSession()) {
        x2_0 = TensorBuilder
            .builder(
               DataType.FLOAT, 
               Shape.create(2L, 1L), 
               Tensor.options()
            )
            .name("x2:0")
            .putFloat(3f)
            .putFloat(2f)
            .build();
        y0 = session.feed(x2_0).forward().getOutput("y:0");

        //
        // Dump outputs data
        //
        logger.debug("Output: {}", y0.toString());
    }
}
```

### 高性能计算支持
项目中，在依赖org.onnx4j.backend.dl4j的基础上，通过添加额外的Nd4j依赖包，Nd4j会自动优先加载更高优先级的运行负载，以此可支持AVX2、AVX512、GPU进行更快速的运算。

* Windows平台下的AVX2指令集支持
```
<dependency>
    <groupId>org.nd4j</groupId>
    <artifactId>nd4j-native</artifactId>
    <classifier>windows-x86_64-avx2</classifier>
</dependency>
```

* Linux平台下的AVX512指令集支持
```
<dependency>
    <groupId>org.nd4j</groupId>
    <artifactId>nd4j-native</artifactId>
    <classifier>linux-x86_64-avx512</classifier>
</dependency>
```

* GPU支持
```
<dependency>
    <groupId>org.nd4j</groupId>
    <artifactId>nd4j-cuda-10.0</artifactId>
</dependency>
```

> Nd4j利用CUDA进行GPU计算加速，开发者需自行配置好CUDA与CuDNN，并根据CUDA的版本依赖对应的Nd4j-CUDA包。上述例子假设系统中配置的CUDA版本为10.0。

## Operator支持
### ai.onnx Operators
|Operator|Opset1|Opset2|Opset3|Opset4|Opset5|Opset6|Opset7|Opset8|Opset9|Opset10|Opset11|Opset12|
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
|Abs|1|1|1|1|1|1|1|1|1|1|1|1|
|Add|1|1|1|1|1|1|1|1|1|1|1|1|
|ArgMax|1|1|1|1|1|1|1|1|1|1|1|1|
|AveragePool|1|1|1|1|1|1|1|1|1|1|1|1|
|BatchNormalization|1|1|1|1|1|1|1|1|1|1|1|1|
|Cast|1|1|1|1|1|6|6|6|9|9|9|9|
|Concat|1|1|1|4|4|4|4|4|4|4|4|4|
|Constant|1|1|1|1|1|1|1|1|1|1|1|1|
|Conv|1|1|1|1|1|1|1|1|1|1|1|1|
|Div|1|1|1|1|1|1|1|1|1|1|1|1|
|Dropout|1|1|1|1|1|6|6|6|6|6|6|6|
|Gather|1|1|1|1|1|1|1|1|1|1|1|1|
|Identity|1|1|1|1|1|1|1|1|1|1|1|1|
|ImageScaler|1|1|1|1|1|1|1|1|1|1|1|1|
|LeakyRelu|1|1|1|1|1|1|1|1|1|1|1|1|
|MatMul|1|1|1|1|1|1|1|1|1|1|1|1|
|MaxPool|1|1|1|1|1|1|1|1|1|1|1|1|
|Mul|1|1|1|1|1|6|6|6|6|6|6|6|
|Pad|1|1|1|1|1|1|1|1|1|1|1|1|
|ReduceMax|1|1|1|1|1|1|1|1|1|1|11|12|
|Relu|1|1|1|1|1|1|1|1|1|1|1|1|
|Reshape|1|1|1|1|5|5|5|5|5|5|5|5|
|Shape|1|1|1|1|1|1|1|1|1|1|1|1|
|Sigmoid|1|1|1|1|1|6|6|6|6|6|6|6|
|Softmax|1|1|1|1|1|1|1|1|1|1|11|11|
|Squeeze|1|1|1|1|1|1|1|1|1|1|11|11|
|Sub|1|1|1|1|1|6|7|7|7|7|7|7|
|Sum|1|1|1|1|1|6|6|8|8|8|8|8|
|Transpose|1|1|1|1|1|1|1|1|1|1|1|1|
|Unsqueeze|1|1|1|1|1|1|1|1|1|1|11|11|

### ai.onnx.ml Operators
暂不支持。
 
## 更多
如需要获取更详细的使用方法，可浏览我们所提供的[forwarder.demo](https://github.com/onnx4j/forwarder.demo)项目。
