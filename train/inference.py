import os
import torch
import numpy as np
from PIL import Image
from segment_anything import sam_model_registry, SamPredictor
import torchvision.transforms as transforms

def inference(image_path, sam_checkpoint_path, output_path, device='cpu'):
    """
    学習済みモデルで推論を実行し、結果を保存する。

    Args:
        image_path (str): 入力画像へのパス。
        sam_checkpoint_path (str): SAMチェックポイントへのパス。
        output_path (str): 出力マスクの保存先パス。
        device (str): 使用するデバイス（'cpu'または'cuda'）。
    """

    # SAMモデルを初期化
    model_type = "vit_b"
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint_path)
    sam.to(device)
    sam.eval()

    # SAM Predictorを初期化
    predictor = SamPredictor(sam)

    # 画像をロードして前処理
    image = Image.open(image_path).convert('RGB')
    image_np = np.array(image)
    predictor.set_image(image_np)

    # ポイントプロンプト（画像の中心）
    h, w = image_np.shape[:2]
    point_coords = np.array([[w // 2, h // 2]])
    point_labels = np.array([1])  # Foreground point

    # マスクを生成
    masks, scores, logits = predictor.predict(
        point_coords=point_coords,
        point_labels=point_labels,
        multimask_output=True
    )

    # 最もスコアの高いマスクを選択
    best_mask_idx = np.argmax(scores)
    mask = masks[best_mask_idx]

    # マスクを画像として保存
    mask_image = Image.fromarray((mask * 255).astype(np.uint8))
    mask_image.save(output_path)

    print(f"推論完了。マスクは {output_path} に保存されました。")


def main():
    # 設定
    image_path = "path/to/your/image.jpg"  # 推論に使用する画像のパス
    sam_checkpoint_path = "sam_finetuned_models/sam_finetuned_best.pth"  # SAMチェックポイントへのパス
    output_path = "prediction.png"  # 出力マスクの保存先パス
    device = "cpu"  # 使用するデバイス（"cpu"または"cuda"）

    # 推論の実行
    inference(image_path, sam_checkpoint_path, output_path, device)


if __name__ == "__main__":
    main()