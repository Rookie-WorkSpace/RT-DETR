## 快速开始

<details>
<summary>环境配置</summary>

```shell
pip install -r requirements.txt
```

以下是相应的 `torch` 和 `torchvision` 版本对应关系。
`rtdetr` | `torch` | `torchvision`
|---|---|---|
| `-` | `2.4` | `0.19` |
| `-` | `2.2` | `0.17` |
| `-` | `2.1` | `0.16` |
| `-` | `2.0` | `0.15` |

</details>

<details open>
<summary>示意图</summary>

<div align="center">
<img width="500" alt="image" src="https://github.com/user-attachments/assets/437877e9-1d4f-4d30-85e8-aafacfa0ec56">
</div>

</details>

## 模型库

### 基础模型

| 模型 | 数据集 | 输入尺寸 | AP<sup>val</sup> | AP<sub>50</sub><sup>val</sup> | 参数量(M) | FPS | 配置文件 | 模型权重 | 
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |:---: |
**RT-DETRv2-S** | COCO | 640 | **48.1** <font color=green>(+1.6)</font> | **65.1** | 20 | 217 | [配置文件](./configs/rtdetrv2/rtdetrv2_r18vd_120e_coco.yml) | [下载链接](https://github.com/lyuwenyu/storage/releases/download/v0.2/rtdetrv2_r18vd_120e_coco_rerun_48.1.pth) |
**RT-DETRv2-M**<sup>*<sup> | COCO | 640 | **49.9** <font color=green>(+1.0)</font> | **67.5** | 31 | 161 | [配置文件](./configs/rtdetrv2/rtdetrv2_r34vd_120e_coco.yml) | [下载链接](https://github.com/lyuwenyu/storage/releases/download/v0.1/rtdetrv2_r34vd_120e_coco_ema.pth)
**RT-DETRv2-M** | COCO | 640 | **51.9** <font color=green>(+0.6)</font> | **69.9** | 36 | 145 | [配置文件](./configs/rtdetrv2/rtdetrv2_r50vd_m_7x_coco.yml) | [下载链接](https://github.com/lyuwenyu/storage/releases/download/v0.1/rtdetrv2_r50vd_m_7x_coco_ema.pth)
**RT-DETRv2-L** | COCO | 640 | **53.4** <font color=green>(+0.3)</font> | **71.6** | 42 | 108 | [配置文件](./configs/rtdetrv2/rtdetrv2_r50vd_6x_coco.yml) | [下载链接](https://github.com/lyuwenyu/storage/releases/download/v0.1/rtdetrv2_r50vd_6x_coco_ema.pth)
**RT-DETRv2-X** | COCO | 640 | 54.3 | **72.8** <font color=green>(+0.1)</font> | 76 | 74 | [配置文件](./configs/rtdetrv2/rtdetrv2_r101vd_6x_coco.yml) | [下载链接](https://github.com/lyuwenyu/storage/releases/download/v0.1/rtdetrv2_r101vd_6x_coco_from_paddle.pth)

**说明：**
- `AP` 是在 *MSCOCO val2017* 数据集上评估的。
- `FPS` 是在单个 T4 GPU 上评估的，使用 $batch\\_size = 1$，$fp16$ 和 $TensorRT>=8.5.1$。
- 表格中的 `COCO + Objects365` 表示使用在 `Objects365` 上预训练的权重在 `COCO` 上微调的模型。

### 离散采样模型

| 模型 | 采样方法 | AP<sup>val</sup> | AP<sub>50</sub><sup>val</sup> | 配置文件 | 模型权重 
| :---: | :---: | :---: | :---: | :---: | :---: |
**RT-DETRv2-S_dsp** | discrete_sampling | 47.4 | 64.8 <font color=red>(-0.1)</font> | [配置文件](./configs/rtdetrv2/rtdetrv2_r18vd_dsp_3x_coco.yml) | [下载链接](https://github.com/lyuwenyu/storage/releases/download/v0.1/rtdetrv2_r18vd_dsp_3x_coco.pth)
**RT-DETRv2-M**<sup>*</sup>**_dsp** | discrete_sampling | 49.2 | 67.1 <font color=red>(-0.4)</font> | [配置文件](./configs/rtdetrv2/rtdetrv2_r34vd_dsp_1x_coco.yml) | [下载链接](https://github.com/lyuwenyu/storage/releases/download/v0.1/rrtdetrv2_r34vd_dsp_1x_coco.pth)
**RT-DETRv2-M_dsp** | discrete_sampling | 51.4 | 69.7 <font color=red>(-0.2)</font> | [配置文件](./configs/rtdetrv2/rtdetrv2_r50vd_m_dsp_3x_coco.yml) | [下载链接](https://github.com/lyuwenyu/storage/releases/download/v0.1/rtdetrv2_r50vd_m_dsp_3x_coco.pth)
**RT-DETRv2-L_dsp** | discrete_sampling | 52.9 | 71.3 <font color=red>(-0.3)</font> |[配置文件](./configs/rtdetrv2/rtdetrv2_r50vd_dsp_1x_coco.yml)| [下载链接](https://github.com/lyuwenyu/storage/releases/download/v0.1/rtdetrv2_r50vd_dsp_1x_coco.pth)

**说明：**
- 推理速度的影响与具体设备和软件相关。
- `*_dsp*` 模型继承了 `*_sp*` 模型的知识并适应了 `discrete_sampling` 策略。**您可以使用 TensorRT 8.4（或更早版本）来推理这些模型**

### 采样点消融实验

| 模型 | 采样方法 | 采样点数 | AP<sup>val</sup> | AP<sub>50</sub><sup>val</sup> | 模型权重 
| :---: | :---: | :---: | :---: | :---: | :---: |
**rtdetrv2_r18vd_sp1** | grid_sampling | 21,600 | 47.3 | 64.3 <font color=red>(-0.6) | [下载链接](https://github.com/lyuwenyu/storage/releases/download/v0.1/rtdetrv2_r18vd_sp1_120e_coco.pth)
**rtdetrv2_r18vd_sp2** | grid_sampling | 43,200 | 47.7 | 64.7 <font color=red>(-0.2) | [下载链接](https://github.com/lyuwenyu/storage/releases/download/v0.1/rtdetrv2_r18vd_sp2_120e_coco.pth)
**rtdetrv2_r18vd_sp3** | grid_sampling | 64,800 | 47.8 | 64.8 <font color=red>(-0.1) | [下载链接](https://github.com/lyuwenyu/storage/releases/download/v0.1/rtdetrv2_r18vd_sp3_120e_coco.pth)
rtdetrv2_r18vd(_sp4)| grid_sampling | 86,400 | 47.9 | 64.9 | [下载链接](https://github.com/lyuwenyu/storage/releases/download/v0.1/rtdetrv2_r18vd_120e_coco.pth) 

**说明：**
- 推理速度的影响与具体设备和软件相关。
- `采样点数` 表示每张图像推理时解码器中采样的总点数。

## 使用方法
<details>
<summary> 详细说明 </summary>

1. 训练
```shell
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --master_port=9909 --nproc_per_node=4 tools/train.py -c path/to/config --use-amp --seed=0 &> log.txt 2>&1 &
```

2. 测试
```shell
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --master_port=9909 --nproc_per_node=4 tools/train.py -c path/to/config -r path/to/checkpoint --test-only
```

3. 微调
```shell
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --master_port=9909 --nproc_per_node=4 tools/train.py -c path/to/config -t path/to/checkpoint --use-amp --seed=0 &> log.txt 2>&1 &
```

4. 导出 onnx
```shell
python tools/export_onnx.py -c path/to/config -r path/to/checkpoint --check
```

5. 推理

支持 torch、onnxruntime、tensorrt 和 openvino，详见 *references/deploy*
```shell
python references/deploy/rtdetrv2_onnxruntime.py --onnx-file=model.onnx --im-file=xxxx
python references/deploy/rtdetrv2_tensorrt.py --trt-file=model.trt --im-file=xxxx
python references/deploy/rtdetrv2_torch.py -c path/to/config -r path/to/checkpoint --im-file=xxx --device=cuda:0
```
</details>

## 引用
如果您使用了 `RTDETR` 或 `RTDETRv2`，请使用以下 BibTeX 条目：

<details>
<summary> bibtex </summary>

```latex
@misc{lv2023detrs,
      title={DETRs Beat YOLOs on Real-time Object Detection},
      author={Wenyu Lv and Shangliang Xu and Yian Zhao and Guanzhong Wang and Jinman Wei and Cheng Cui and Yuning Du and Qingqing Dang and Yi Liu},
      year={2023},
      eprint={2304.08069},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}

@misc{lv2024rtdetrv2improvedbaselinebagoffreebies,
      title={RT-DETRv2: Improved Baseline with Bag-of-Freebies for Real-Time Detection Transformer}, 
      author={Wenyu Lv and Yian Zhao and Qinyao Chang and Kui Huang and Guanzhong Wang and Yi Liu},
      year={2024},
      eprint={2407.17140},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2407.17140}, 
}
```
</details> 