"""作者: lyuwenyu
"""

import os 
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

import argparse
import numpy as np 

from src.core import YAMLConfig

import torch
import torch.nn as nn 


def main(args, ):
    """主函数
    """
    cfg = YAMLConfig(args.config, resume=args.resume)

    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu') 
        if 'ema' in checkpoint:
            state = checkpoint['ema']['module']
        else:
            state = checkpoint['model']
    else:
        raise AttributeError('目前仅支持从检查点加载模型状态字典')

    # 注意: 加载训练模式状态 -> 转换为部署模式
    cfg.model.load_state_dict(state)

    class Model(nn.Module):
        def __init__(self, ) -> None:
            super().__init__()
            self.model = cfg.model.deploy()
            self.postprocessor = cfg.postprocessor.deploy()
            print(self.postprocessor.deploy_mode)
            
        def forward(self, images, orig_target_sizes):
            outputs = self.model(images)
            return self.postprocessor(outputs, orig_target_sizes)
    

    model = Model()

    dynamic_axes = {
        'images': {0: 'N', },
        'orig_target_sizes': {0: 'N'}
    }

    data = torch.rand(1, 3, 640, 640)
    size = torch.tensor([[640, 640]])

    torch.onnx.export(
        model, 
        (data, size), 
        args.file_name,
        input_names=['images', 'orig_target_sizes'],
        output_names=['labels', 'boxes', 'scores'],
        dynamic_axes=dynamic_axes,
        opset_version=16, 
        verbose=False
    )


    if args.check:
        import onnx
        onnx_model = onnx.load(args.file_name)
        onnx.checker.check_model(onnx_model)
        print('检查导出的ONNX模型完成...')


    if args.simplify:
        import onnxsim
        dynamic = True 
        input_shapes = {'images': data.shape, 'orig_target_sizes': size.shape} if dynamic else None
        onnx_model_simplify, check = onnxsim.simplify(args.file_name, input_shapes=input_shapes, dynamic_input_shape=dynamic)
        onnx.save(onnx_model_simplify, args.file_name)
        print(f'简化ONNX模型 {check}...')


    # import onnxruntime as ort 
    # from PIL import Image, ImageDraw, ImageFont
    # from torchvision.transforms import ToTensor
    # from src.data.coco.coco_dataset import mscoco_category2name, mscoco_category2label, mscoco_label2category

    # # print(onnx.helper.printable_graph(mm.graph))

    # # 加载原始图像,不调整大小
    # original_im = Image.open('./hongkong.jpg').convert('RGB')
    # original_size = original_im.size

    # # 调整图像大小用于模型输入
    # im = original_im.resize((640, 640))
    # im_data = ToTensor()(im)[None]
    # print(im_data.shape)

    # sess = ort.InferenceSession(args.file_name)
    # output = sess.run(
    #     # output_names=['labels', 'boxes', 'scores'],
    #     output_names=None,
    #     input_feed={'images': im_data.data.numpy(), "orig_target_sizes": size.data.numpy()}
    # )

    # # print(type(output))
    # # print([out.shape for out in output])

    # labels, boxes, scores = output

    # draw = ImageDraw.Draw(original_im)  # 在原始图像上绘制
    # thrh = 0.6

    # for i in range(im_data.shape[0]):

    #     scr = scores[i]
    #     lab = labels[i][scr > thrh]
    #     box = boxes[i][scr > thrh]

    #     print(i, sum(scr > thrh))

    #     for b, l in zip(box, lab):
    #         # 将边界框坐标缩放回原始图像大小
    #         b = [coord * original_size[j % 2] / 640 for j, coord in enumerate(b)]
    #         # 从标签获取类别名称
    #         category_name = mscoco_category2name[mscoco_label2category[l]]
    #         draw.rectangle(list(b), outline='red', width=2)
    #         font = ImageFont.truetype("Arial.ttf", 15)
    #         draw.text((b[0], b[1]), text=category_name, fill='yellow', font=font)

    # # 保存带有边界框的原始图像
    # original_im.save('test.jpg')


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', '-c', type=str, )
    parser.add_argument('--resume', '-r', type=str, )
    parser.add_argument('--file-name', '-f', type=str, default='model.onnx')
    parser.add_argument('--check',  action='store_true', default=False,)
    parser.add_argument('--simplify',  action='store_true', default=False,)

    args = parser.parse_args()

    main(args)
