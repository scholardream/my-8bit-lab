# my-8bit-lab

8-bit 音乐实验室：把任何音乐变成红白机（NES / Famicom）的声音。

## chiptunify — 任意 MIDI → NES 8-bit 音乐

纯 Python 实现，除 numpy 外零依赖：自己解析 SMF（Standard MIDI File），
自己用 numpy 合成 NES APU 的四个声道（两路脉冲波、三角波、LFSR 噪声）。

```bash
python -m venv venv
.\venv\Scripts\Activate.ps1        # 若被拦截：Set-ExecutionPolicy -Scope Process RemoteSigned
pip install numpy

# 内置演示曲（Korobeiniki / 俄罗斯方块主题曲）
python -m chiptunify --demo -o tetris_demo.wav

# 转换自己的 MIDI
python -m chiptunify song.mid -o song_8bit.wav
```

参数：

- `--demo`：渲染内置演示曲，无需输入文件
- `-o PATH`：输出 WAV 路径
- `--no-arp`：关闭琶音（和弦改为直接裁断而不是 60Hz 轮询）
- `--duty-p1` / `--duty-p2`：脉冲 1（主奏）/ 脉冲 2（和声）的占空比，
  可选 0.125 / 0.25 / 0.5 / 0.75，默认 0.5 / 0.25
- `--no-quantize`：关闭 NES 定时器周期量化（音高更准但少了点"红白机味"）
- `--rate`：采样率，默认 44100

## bytebeat — 单公式音乐（Lua）

`bytebeat/bytebeat.lua`，纯 Lua 5.3+（原生位运算符，不用 bit32），零依赖。
逐样本计算 f(t)，取低 8 位写成 8-bit unsigned mono WAV：

```bash
lua bytebeat/bytebeat.lua "t*(t>>10&42)" out.wav --seconds 30 --rate 8000
lua bytebeat/bytebeat.lua --preset 1 out.wav      # 内置 5 个经典公式
```

## 技术亮点

- **纯 Python SMF 解析**：格式 0/1、PPQN 时基、running status、tempo map，
  不碰 pretty_midi / mido
- **NES 定时器周期量化**：音高按 2A03 的 11-bit 定时器周期取整，
  还原真实芯片那种微微跑调的音色
- **LFSR 噪声**：与硬件相同的 15-bit 线性反馈移位寄存器；
  预生成一个 32767 步完整周期后平铺，长歌曲也不会卡在 Python 循环上
- **60Hz 琶音**：NES 只有一个声道发一个音，真实作曲家用帧率级轮询
  假装和弦——这里自动做同样的事

## 路线图

下一步：nanoGPT 在 NES-MDB 数据集上训练，让 AI 自己写 8-bit 音乐。
（`LakhNES/` 是 chrisdonahue/LakhNES 的镜像——在 NES-MDB 上用 Transformer
生成任天堂音乐的先驱工作，致敬。）
