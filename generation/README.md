# UAV-CondSelect generation module

**Source → Mask preparation → Background editing → UAV integration → Final.**

源图与标注经掩膜准备和输入预处理，进行一次 FLUX.2-dev 背景编辑，再融合源 UAV，得到 `final`。

每个源图—条件对只调用一次背景编辑模型。五条件共用同一入口与计算路径：`rain`、`snow`、`fog`、`dust`、`low_light`。融合图就是最终输出，保存后不再处理图像内容。

## 安装与模型

使用 Python 3.12。环境安装及模型获取见 [ENVIRONMENT.md](ENVIRONMENT.md)。CPU 验证不需要权重；生成需要本地 FLUX.2-dev 模型和可用的 CUDA 环境。输入已有保护掩膜时无需加载 SAM。

## 输入

JSONL 每行一个源图，路径相对于该 JSONL 所在目录解析：

```json
{"source_id":"A_0106837","image":"input/source.jpg","annotation":"input/source.xml","core_mask":"input/core.png","protection_mask":"input/protection.png"}
```

`source_id` 必须唯一，只包含字母、数字、下划线或连字符。标注为 VOC-style XML，包含原坐标网格尺寸及像素 XYXY 框。图像与保护掩膜尺寸必须一致。`core_mask` 是可选记录；生成实际读取 `protection_mask`。目录与坐标要求见 [DATA.md](DATA.md)。

## 唯一入口：generate.py

从本模块目录运行（仓库中的 `generation/`）。先预检示例（不加载模型、不产生输出）：

```bash
python generate.py generate --manifest examples/sources.jsonl \
  --model models/FLUX.2-dev --output outputs/example --dry-run
```

五条件正式生成：

```bash
python generate.py generate --manifest examples/sources.jsonl \
  --model models/FLUX.2-dev --output outputs/example
```

仅生成 fog 可增加 `--conditions fog`。自定义配置与提示词分别用 `--config`、`--prompts`，默认是 `configs/generation.json` 和 `prompts/background.json`。修改提示词后须同步更新其 UTF-8 文本 SHA-256。输入清单支持多源，自动遍历所选条件，模型在批次中只加载一次。

若尚无保护掩膜，输入清单只需 `source_id/image/annotation`，先运行：

```bash
python generate.py prepare-masks --manifest data/sources.jsonl \
  --sam-checkpoint models/sam_vit_b_01ec64.pth --output outputs/masks
python generate.py generate --manifest outputs/masks/sources.jsonl \
  --model models/FLUX.2-dev --output outputs/example
```

准备阶段保存选中的 SAM 二值候选、core、protection 和框提示/候选选择元数据。SAM 参数在 `configs/sam.json` 中显式指定。多个目标框取外接并集后提示 SAM。

所有输出目录必须尚不存在。已有目录会被拒绝；执行失败会留下已完成文件和部分清单，请另选新目录处理需要重新运行的输入，不会自动覆盖或默默跳过文件。

## 输出

```text
outputs/example/
  final/<source_id>_<condition>.jpg
  annotations/<source_id>_<condition>.xml
  working_sources/  masks/  editor_inputs/  backgrounds/
  config.json  prompts.json  manifest.jsonl
```

源文件是 PNG 时图像输出为 PNG；JPEG 源保持 JPEG 扩展名、quality=95。`manifest.jsonl` 的最终图字段为 `files.final_image`，记录尺寸、prompt hash、seed、输入和各输出文件 SHA-256。路径相对于结果目录解析，输入引用可能含 `..`；不是重新复制一套源数据。

## 计算设置

- 源 RGB：最长边 1024，比例不放大，各边 round 后向下取 16 的倍数；LANCZOS。示例从 3840×2160 到 1024×576。
- 已存保护掩膜：NEAREST 到工作网格，`>127` 二值化；此时额外膨胀 0；Pillow GaussianBlur radius=28 得 soft mask，无峰值归一化。
- 输入：源 RGB GaussianBlur radius=60，再以同一 soft mask 与未模糊源 RGB 合成。
- 背景：FLUX.2-dev、BF16、model CPU offload、28 inference steps、guidance 3.0、seed 43。每次调用重新建立并播种 `torch.Generator`，不会沿批次推进同一生成器。未传 negative prompt。
- 最终融合：`final = Image.composite(source, background, soft_mask).convert("RGB")`。源 RGB 与背景均使用内存对象；soft mask 为同一 L/uint8 对象。结果尺寸不同时先 LANCZOS 对齐背景。
- 背景可保存为 JPEG 用于检查，但**不会重新读取该 JPEG 参与融合**。工作源和编辑输入的保存同样不会替换内存对象。RGB 直接以 8-bit 通道合成，不转线性光空间。
- 标注按 XML 声明尺寸到最终图尺寸分别缩放 x/y，`int(float())` 读取，`round` 写整数，不作 ±1 原点调整、不裁框。尺寸或缩放后框非法时拒绝执行。

Pillow 的 blur radius 是高斯标准差参数，内部使用近似实现，不能解释成 OpenCV 核半径。SAM 保护掩膜由源尺度 core 经半径4的9×9椭圆核膨胀一次；上述“额外膨胀0”是在已有保护掩膜之后。

## 示例

[完整 fog 示例](examples/README.md)包括源图、标注、掩膜、输入、背景和 final；各独立文件由 [流程清单](examples/flow_manifest.json)对应。

![Fog workflow](examples/workflow.png)

以下是选取的雨、雪、雾、尘定性示例，不代表全体样本的统计表现。源图、final、对应标注与配置引用见 [gallery manifest](examples/gallery/manifest.json)。

![Selected qualitative examples: Source and Final](examples/gallery/overview.png)

## CPU 验证工具

```bash
python tools/validate.py
sha256sum -c checksums.sha256
```

工具检查预处理、掩膜、融合、标注和文件对应关系。五条件接口检查使用模型替身，验证每对输入只调用一次背景编辑及融合使用内存背景。保存的 JPEG 背景有损，以它重建融合不能要求与 final 逐字节相同。

重新绘制示例拼图：`python tools/render_examples.py`。拼图仅缩放排版，独立图像保留原尺寸。示例数据的再分发许可需由权利人确认；第三方声明见 [ENVIRONMENT.md](ENVIRONMENT.md)。
