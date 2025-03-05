import io
import json
import os

import numpy as np
import PIL.Image
import skimage.draw

def json_to_mask(json_path, image_path, output_path):
    """
    labelmeで作成したJSONファイルからマスク画像を生成する。

    Args:
        json_path (str): JSONファイルのパス
        image_path (str): 元画像のパス
        output_path (str): マスク画像の出力パス
    """

    with open(json_path, "r") as f:
        data = json.load(f)

    # 元画像の読み込み
    image = np.array(PIL.Image.open(image_path))
    height, width = image.shape[:2]

    # マスク画像の生成
    mask = np.zeros((height, width), dtype=np.uint8)
    for shape in data["shapes"]:
        points = shape["points"]
        label = shape["label"]
        shape_type = shape["shape_type"]

        # ポリゴンの描画
        if shape_type == "polygon":
            rr, cc = skimage.draw.polygon(
                [p[1] for p in points], [p[0] for p in points], (height, width)
            )
            mask[rr, cc] = 255  # マスク領域を255で塗りつぶす

    # マスク画像の保存
    PIL.Image.fromarray(mask).save(output_path)

if __name__ == "__main__":
    # JSONファイル、元画像、出力マスク画像のパスを指定
    json_path = "path/to/your/annotation.json"
    image_path = "path/to/your/image.jpg"
    output_path = "path/to/your/mask.png"

    # マスク画像の生成
    json_to_mask(json_path, image_path, output_path)