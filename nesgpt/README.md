# nesgpt — nanoGPT 风格的 NES-MDB 训练骨架

让 AI 自己写 8-bit 音乐的最小实现：用 nanoGPT 风格的 decoder-only
Transformer，在 [NES Music Database](https://github.com/chrisdonahue/nesmdb)
的 TX1 事件序列上做语言建模（预测下一个音乐事件）。

依赖只有 **PyTorch**（模型/训练）+ **numpy**（数据管线），不再需要
LakhNES 那套 2019 年的 PyTorch 1.0 + Python 2 的 nesmdb 合成环境。

## 数据格式

- 每个 `.tx1.txt` 文件是一首歌，每行一个事件：`WT_N`（等待 N 个采样点，
  44.1 kHz）、`P1/P2/TR/NO` 的 `NOTEON_*` / `NOTEOFF`。
- 词表固定 **631 个 token**：`<S>`(id 0) + 630 个符号。`<S>` 既是曲子开头
  也是曲子之间的分隔符。
- 词表文件就是仓库里现成的 `LakhNES/data/tx1_vocab.txt`，`nesgpt.vocab`
  直接复用它（不重新生成）。

## 快速开始

```bash
# 0) 依赖（numpy 已随 chiptunify 装好，torch 需要另装）
pip install torch            # Python 3.13 会装 torch>=2.6

# 1) 把 TX1 数据编码成训练用的 token 二进制
python -m nesgpt.prepare     # 产出 nesgpt/data/{train,valid,test}.bin

# 2) 训练（CPU 起步的小模型，~2.6M 参数）
python -m nesgpt.train --device cpu --max_iters 2000

# 3) 生成
python -m nesgpt.sample --ckpt nesgpt/out/ckpt.pt \
    --out_dir generated --num 1

# 4) 试听（TX1 -> WAV，复用 chiptunify 的 NES APU 合成器）
python -m nesgpt.tx1_render generated/0.tx1.txt -o generated/0.wav
```

## 各模块

| 文件 | 作用 |
| --- | --- |
| `vocab.py` | TX1 词表：符号 ↔ id，文件编码 |
| `prepare.py` | 把 train/valid/test 的 TX1 文件拼接成 `*.bin` |
| `model.py` | 最小 GPT（nanoGPT 风格，含权重绑定、GPT-2 初始化、cosine LR、AdamW） |
| `train.py` | 训练循环：梯度累积、cosine 学习率、验证、checkpoint |
| `sample.py` | 自回归采样生成 `.tx1.txt` |
| `tx1_render.py` | TX1 → WAV（翻译成 `chiptunify` 的 `NesArrangement` 再合成） |

## 默认模型规模

`GPTConfig(block_size=512, n_layer=6, n_head=6, n_embd=384)`，约 **2.6M**
参数，CPU 也能跑起来验证整条链路。真要训出像样的曲子，往上调
`n_layer/n_head/n_embd` 并换 GPU 即可（nanoGPT 的规模经验同样适用）。

## 与 LakhNES 的关系

LakhNES（仓库里的 `LakhNES/`）是原论文的 Transformer-XL 完整实现，但依赖
旧版 PyTorch 和 Python 2 的 `nesmdb`。本目录用现代 PyTorch 重写了最核心的
"事件序列语言模型"这一层；词表和事件编码方式与 LakhNES 保持一致，因此
`sample.py` 输出的 `.tx1.txt` 可以直接用 `nesgpt.tx1_render` 或任何
LakhNES 兼容工具转成音频。
