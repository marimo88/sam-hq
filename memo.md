# メモ

## uvのtorchインストール方法

`project.toml`にcuda版torchをインストールするには以下のように記述

```
dependencies = [
    "torch==2.5.0+cu124",
    "torchvision==0.20.0+cu124"
]
```

### 依存関係の表

[torchとtorchvisionの対応表](https://github.com/pytorch/vision#installation)
[torchとCUDAの対応表](https://github.com/pytorch/pytorch/blob/main/RELEASE.md#release-compatibility-matrix)

torchとtorchvisionの対応ができていないと`ModuleNotFoundError: No module named 'torchvision.ops'`とエラー

### エラーの対処法

[WindowsのuvでCUDA 12.4のPyTorchをインストールする](https://zenn.dev/yashikota/articles/45b4892d6acb10)

## train.pyのコマンド

必要なパッケージ

`gdown matplotlib numpy onnx onnxruntime pycocotools timm opencv-python segment-anything-hq`

```
python -m torch.distributed.launch --nproc_per_node=1 ./train/train.py --checkpoint ./pretrained_checkpoint --model-type vit_b --output ./output
```

## CUDAの切替方法

[CUDA Toolkit Archive](https://developer.nvidia.com/cuda-toolkit-archive)

**システムの環境変数**から変更

※ユーザー環境変数ではない
