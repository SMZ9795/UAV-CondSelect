# 数据、标注与掩膜

建议目录：

```text
data/
  images/<source_id>.jpg
  annotations/<source_id>.xml
  masks/<source_id>_protection.png
  sources.jsonl
```

sources.jsonl 中分别引用 images、annotations 和 masks 的相对路径。source_id 独立于文件名，避免不同场景同名文件冲突。不要从 XML 的 filename/path 字段寻找输入；清单配对具有优先级。输出 XML 会写与最终图对应的相对路径。

输入是原图坐标系的 XML，不能给原图搭配已经缩放的坐标但仍宣称原尺寸。转换按 XML 实际声明宽高执行，没有额外的 VOC 一基到零基变换。生成掩膜时会将 XML 坐标映射到实际源图尺寸；最终标注映射到最终尺寸。源图处理是全幅缩放，没有裁剪、填充或旋转。标注用于目标提示和输出几何，不声称生成模型一定保持所有物体外观。

SAM 准备默认：vit_b，context crop 扩展比例0.8、upscale4、max_crop_side1024，实际scale下限1；裁图INTER_CUBIC，box同步乘scale。采用box-only、多候选输出，用 score + 0.3*inside_ratio − penalty 选一个；area_ratio<0.08、>0.95或inside_ratio<0.80分别罚0.5。选中二值候选用INTER_NEAREST还原至crop后放回全图，8连通域面积小于5的碎块移除。配置constrain_to_bbox默认false；开启时先按框限制再清理连通域。保护掩膜由core椭圆9×9膨胀1次。没有重新对天气条件做分割。

examples/input包含一个源图、XML、core和protection；examples/reference是同源fog的工作网格输入、掩膜、背景及final。reference_manifest.json中的路径相对examples解析。该final确为源RGB与背景通过soft mask的融合结果；背景保存副本是JPEG，而实际融合使用的是保存前内存背景。示例XML只清理路径元信息并提供对应尺寸标注。

core是清理后的目标掩膜，不应当称为网络原始候选；prepare-masks另存raw_selected。已存保护掩膜可以直接用于确定性输入准备。新制SAM掩膜依赖指定权重和显式配置，不同环境下的模型数值差异可能影响掩膜边界。

完整流程文件对应关系见 [examples/flow_manifest.json](examples/flow_manifest.json)，雨、雪、雾、尘展示见 [examples/gallery/manifest.json](examples/gallery/manifest.json)。各清单的路径相对于清单所在目录。展示标注与相应图像尺寸一致；源标注保留坐标，仅规范化文件路径元信息。
