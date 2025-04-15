## 待办事项
<details>
<summary> 查看详情 </summary>

- [x] 训练
- [x] 评估
- [x] 导出 onnx
- [x] 上传源代码
- [x] 上传从 paddle 转换的权重，参见 [*链接*](https://github.com/lyuwenyu/RT-DETR/issues/42)
- [x] 与 [*paddle 版本*](../rtdetr_paddle/) 的训练细节对齐
- [x] 基于 [*预训练权重*](https://github.com/lyuwenyu/RT-DETR/issues/42) 微调 rtdetr

</details>

## 模型库

| 模型 | 数据集 | 输入尺寸 | AP<sup>val</sup> | AP<sub>50</sub><sup>val</sup> | 参数量(M) | FPS |  检查点 |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
rtdetr_r18vd | COCO | 640 | 46.4 | 63.7 | 20 | 217 | [链接<sup>*</sup>](https://github.com/lyuwenyu/storage/releases/download/v0.1/rtdetr_r18vd_dec3_6x_coco_from_paddle.pth)
rtdetr_r34vd | COCO | 640 | 48.9 | 66.8 | 31 | 161 | [链接<sup>*</sup>](https://github.com/lyuwenyu/storage/releases/download/v0.1/rtdetr_r34vd_dec4_6x_coco_from_paddle.pth)
rtdetr_r50vd_m | COCO | 640 | 51.3 | 69.5 | 36 | 145 | [链接<sup>*</sup>](https://github.com/lyuwenyu/storage/releases/download/v0.1/rtdetr_r50vd_m_6x_coco_from_paddle.pth)
rtdetr_r50vd | COCO | 640 | 53.1 | 71.2| 42 | 108 | [链接<sup>*</sup>](https://github.com/lyuwenyu/storage/releases/download/v0.1/rtdetr_r50vd_6x_coco_from_paddle.pth)
rtdetr_r101vd | COCO | 640 | 54.3 | 72.8 | 76 | 74 | [链接<sup>*</sup>](https://github.com/lyuwenyu/storage/releases/download/v0.1/rtdetr_r101vd_6x_coco_from_paddle.pth)
rtdetr_18vd | COCO+Objects365 | 640 | 49.0 | 66.5 | 20 | 217 | [链接<sup>*</sup>](https://github.com/lyuwenyu/storage/releases/download/v0.1/rtdetr_r18vd_5x_coco_objects365_from_paddle.pth)
rtdetr_r50vd | COCO+Objects365 | 640 | 55.2 | 73.4 | 42 | 108 | [链接<sup>*</sup>](https://github.com/lyuwenyu/storage/releases/download/v0.1/rtdetr_r50vd_2x_coco_objects365_from_paddle.pth)
rtdetr_r101vd | COCO+Objects365 | 640 | 56.2 | 74.5 | 76 | 74 | [链接<sup>*</sup>](https://github.com/lyuwenyu/storage/releases/download/v0.1/rtdetr_r101vd_2x_coco_objects365_from_paddle.pth)
rtdetr_regnet | COCO | 640 | 51.6 | 69.6 | 38 | 67 | [链接<sup>*</sup>](https://drive.google.com/file/d/1K2EXJgnaEUJcZCLULHrZ492EF4PdgVp9/view?usp=sharing)
rtdetr_dla34 | COCO | 640 | 49.6 | 67.4  | 34 | 83 | [链接<sup>*</sup>](https://drive.google.com/file/d/1_rVpl-jIelwy2LDT3E4vdM4KCLBcOtzZ/view?usp=sharing)

说明
- 表格中的 `COCO + Objects365` 表示使用在 `Objects365` 上预训练的权重在 `COCO` 上微调的模型。
- `链接`<sup>`*`</sup> 是从 paddle 模型转换的预训练权重的链接，以节省能源。*可能与论文中的表格有细微差异*
<!-- - `FPS` 是在单个 T4 GPU 上评估的，$batch\\_size = 1$ 和 $tensorrt\\_fp16$ 模式 -->

## 快速开始

<details>
<summary>安装</summary>

```bash
pip install -r requirements.txt
```

</details>

<details>
<summary>数据</summary>

- 下载并解压 COCO 2017 训练集和验证集图像。
```
path/to/coco/
  annotations/  # 标注 json 文件
  train2017/    # 训练图像
  val2017/      # 验证图像
```
- 修改配置文件中的 [`img_folder`, `ann_file`](configs/dataset/coco_detection.yml)
</details>

<details>
<summary>训练和评估</summary>

- 单 GPU 训练：

```shell
# 单 GPU 训练
export CUDA_VISIBLE_DEVICES=0
python tools/train.py -c configs/rtdetr/rtdetr_r50vd_6x_coco.yml
```

- 多 GPU 训练：

```shell
# 多 GPU 训练
export CUDA_VISIBLE_DEVICES=0,1,2,3
torchrun --nproc_per_node=4 tools/train.py -c configs/rtdetr/rtdetr_r50vd_6x_coco.yml
```

- 多 GPU 评估：

```shell
# 多 GPU 评估
export CUDA_VISIBLE_DEVICES=0,1,2,3
torchrun --nproc_per_node=4 tools/train.py -c configs/rtdetr/rtdetr_r50vd_6x_coco.yml -r path/to/checkpoint --test-only
```

</details>

<details>
<summary>导出</summary>

```shell
python tools/export_onnx.py -c configs/rtdetr/rtdetr_r18vd_6x_coco.yml -r path/to/checkpoint --check
```
</details>

<details open>
<summary>训练自定义数据</summary>

1. 设置 `remap_mscoco_category: False`。此变量仅适用于 ms-coco 数据集。如果要在自己的数据集上使用 `remap_mscoco_category` 逻辑，请根据数据集修改变量 [`mscoco_category2name`](https://github.com/lyuwenyu/RT-DETR/blob/main/rtdetr_pytorch/src/data/coco/coco_dataset.py#L154)。

2. 添加 `-t path/to/checkpoint`（可选）以基于预训练检查点微调 rtdetr。参见 [训练脚本详情](./tools/README.md)。
</details> 