import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from segment_anything import sam_model_registry, SamPredictor
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
from tqdm import tqdm


class CustomSegmentationDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        """
        セグメンテーションのファインチューニング用のカスタムデータセット

        Args:
            data_dir (str): 画像/マスクペアを含むディレクトリ
            transform: 適用するオプションの変換
        """
        self.data_dir = data_dir
        self.transform = transform
        self.image_dir = os.path.join(data_dir, 'images')
        self.mask_dir = os.path.join(data_dir, 'masks')
        self.image_files = sorted([f for f in os.listdir(
            self.image_dir) if f.endswith(('.jpg', '.png', '.jpeg'))])

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        img_path = os.path.join(self.image_dir, img_name)
        mask_path = os.path.join(self.mask_dir, img_name.replace(
            '.jpg', '.png').replace('.jpeg', '.png'))

        image = Image.open(img_path).convert('RGB')
        mask = Image.open(mask_path).convert('L')

        if self.transform:
            image = self.transform(image)
            mask = transforms.ToTensor()(mask)

        return {'image': image, 'mask': mask.squeeze().long()}


def prepare_sam_for_training(checkpoint_path, device='cuda:0'):
    """
    ファインチューニング用にSAMモデルを準備する

    Args:
        checkpoint_path (str): SAMチェックポイントへのパス
        device (str): 使用するデバイス

    Returns:
        model: 準備されたSAMモデル
    """
    model_type = "vit_h"  # オプション: vit_b, vit_l, vit_h
    sam = sam_model_registry[model_type](checkpoint=checkpoint_path)
    sam.to(device)

    # 学習を高速化し、過学習を防ぐためにエンコーダパラメータを凍結する
    for param in sam.image_encoder.parameters():
        param.requires_grad = False

    # マスクデコーダのみをファインチューニングする
    for param in sam.mask_decoder.parameters():
        param.requires_grad = True

    return sam


class SAMFineTuner:
    def __init__(self, sam_checkpoint_path, data_dir, output_dir, device='cuda:0'):
        """
        Segment Anything Modelをファインチューニングする

        Args:
            sam_checkpoint_path (str): SAMチェックポイントへのパス
            data_dir (str): 学習データを含むディレクトリ
            output_dir (str): ファインチューニングされたモデルを保存するディレクトリ
            device (str): 学習に使用するデバイス
        """
        self.device = device
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        # SAMモデルを初期化する
        self.model = prepare_sam_for_training(sam_checkpoint_path, device)

        # データセットとデータローダを準備する
        transform = transforms.Compose([
            transforms.Resize((1024, 1024)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        self.dataset = CustomSegmentationDataset(data_dir, transform=transform)
        self.dataloader = DataLoader(
            self.dataset, batch_size=2, shuffle=True, num_workers=4)

        # 損失関数とオプティマイザを定義する
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(
            [p for p in self.model.parameters() if p.requires_grad],
            lr=1e-5
        )
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=5)

    def train(self, num_epochs=10):
        """
        SAMモデルを学習する

        Args:
            num_epochs (int): 学習するエポック数
        """
        best_loss = float('inf')

        for epoch in range(num_epochs):
            self.model.train()
            epoch_loss = 0.0

            progress_bar = tqdm(
                self.dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")

            for batch in progress_bar:
                images = batch['image'].to(self.device)
                masks = batch['mask'].to(self.device)

                # 勾配をリセットする
                self.optimizer.zero_grad()

                # 順伝播
                with torch.no_grad():
                    image_embeddings = self.model.image_encoder(images)

                # ポイントプロンプトを作成する（画像の中心）
                batch_size = images.shape[0]
                points = torch.tensor(
                    [[[512, 512]]] * batch_size, device=self.device)
                labels = torch.tensor([[1]] * batch_size, device=self.device)

                # 低解像度マスクを取得する
                sparse_embeddings, dense_embeddings = self.model.prompt_encoder(
                    points=points,
                    labels=labels,
                    boxes=None,
                    masks=None
                )

                # マスクを予測する
                mask_predictions, _ = self.model.mask_decoder(
                    image_embeddings=image_embeddings,
                    image_pe=self.model.prompt_encoder.get_dense_pe(),
                    sparse_prompt_embeddings=sparse_embeddings,
                    dense_prompt_embeddings=dense_embeddings,
                    multimask_output=False
                )

                # マスクを元のサイズにアップスケールする
                upscaled_masks = nn.functional.interpolate(
                    mask_predictions,
                    size=(1024, 1024),
                    mode='bilinear',
                    align_corners=False
                )

                # 損失を計算する
                loss = self.criterion(upscaled_masks, masks)

                # 逆伝播と最適化
                loss.backward()
                self.optimizer.step()

                epoch_loss += loss.item()
                progress_bar.set_postfix({"Loss": loss.item()})

            avg_epoch_loss = epoch_loss / len(self.dataloader)
            print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_epoch_loss:.4f}")

            # 学習率を調整する
            self.scheduler.step(avg_epoch_loss)

            # 最良のモデルを保存する
            if avg_epoch_loss < best_loss:
                best_loss = avg_epoch_loss
                torch.save(self.model.state_dict(), os.path.join(
                    self.output_dir, 'sam_finetuned_best.pth'))

        # 最終モデルを保存する
        torch.save(self.model.state_dict(), os.path.join(
            self.output_dir, 'sam_finetuned_final.pth'))
        print(f"Training completed. Models saved to {self.output_dir}")

    def inference(self, image_path, point_coords=None):
        """
        ファインチューニングされたモデルで推論を実行する

        Args:
            image_path (str): 入力画像へのパス
            point_coords (np.ndarray, optional): ポイントプロンプト

        Returns:
            mask: 予測されたマスク
        """
        self.model.eval()
        predictor = SamPredictor(self.model)

        # 画像をロードして前処理する
        image = np.array(Image.open(image_path).convert('RGB'))
        predictor.set_image(image)

        # 座標が提供されていない場合は中心点をデフォルトとする
        if point_coords is None:
            h, w = image.shape[:2]
            point_coords = np.array([[w//2, h//2]])

        point_labels = np.array([1])  # Foreground point

        # マスクを生成
        masks, scores, logits = predictor.predict(
            point_coords=point_coords,
            point_labels=point_labels,
            multimask_output=True
        )

        # 最もスコアの高いマスクを返す
        best_mask_idx = np.argmax(scores)
        return masks[best_mask_idx]


def main():
    # 使用例
    sam_checkpoint_path = "./pretrained_checkpoint/sam_vit_h_4b8939.pth"  # Meta AIからダウンロード
    data_dir = "./train/data"
    output_dir = "./sam_finetuned_models"

    # GPUが利用可能か確認
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"使用デバイス: {device}")

    # ファインチューナーを初期化
    fine_tuner = SAMFineTuner(
        sam_checkpoint_path=sam_checkpoint_path,
        data_dir=data_dir,
        output_dir=output_dir,
        device=device
    )

    # モデルを学習
    fine_tuner.train(num_epochs=10)

    # 推論の例
    test_image = "path/to/test/image.jpg"
    mask = fine_tuner.inference(test_image)

    # 結果を保存
    Image.fromarray(mask.astype(np.uint8) * 255).save("prediction.png")


if __name__ == "__main__":
    main()
