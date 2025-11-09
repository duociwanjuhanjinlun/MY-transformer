# MY-transformer

A transformer with multi-head self-attention、position-wise FFN、LayerNorm and position coding

一个带有缩放点积注意力、多头注意力、位置无关前馈网络、残差连接与LayerNorm、位置编码的Transformer模型。

运行本项目时，建议先创建虚拟环境：

```bash
conda create -n transformer python=3.10 -y
pip install -r requirements.txt
```

运行训练（示例）:

```bash
python scripts/train.py
```

或者:

```bash
bash scripts/run.sh
```

训练结果将存放在results/
