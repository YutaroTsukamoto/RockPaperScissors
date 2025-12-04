# Rock Paper Scissors - Hand Landmark Detection

MediaPipe Handを使用して、手の動画から3×21次元の点群を抽出するアプリケーションです。

## 機能

- 動画ファイルから手のランドマークを検出
- カメラからリアルタイムで手のランドマークを検出
- **複数の手を同時に検出可能**（デフォルト: 最大2つ）
- 21個の手のランドマークポイント（各3次元: x, y, z）を抽出
- グーチョキパーの自動判定
- 左右の手を自動識別
- 結果をNumPy配列形式で保存

## インストール

```bash
pip install -e .
```

## 使用方法

### 動画ファイルから検出

```bash
python main.py --video <動画ファイルのパス> --output <出力ファイル名>
```

例:
```bash
python main.py --video hand_video.mp4 --output landmarks.npy
```

### カメラからリアルタイム検出

```bash
python main.py --camera --output <出力ファイル名> [--max-hands <数>]
```

例:
```bash
# デフォルト（最大2つの手を検出）
python main.py --camera --output camera_landmarks.npy

# 最大3つの手を検出
python main.py --camera --output camera_landmarks.npy --max-hands 3
```

## 出力形式

- 各フレームで検出された手のランドマークは、21個のポイント × 3次元（x, y, z）の配列として保存されます
- 出力ファイルはNumPy配列形式（.npy）で保存されます
- 形状: `(フレーム数, 21, 3)`

## 手のランドマークポイント

MediaPipe Handは21個の手のランドマークポイントを検出します：
- 手首: 1ポイント
- 親指: 4ポイント
- 人差し指: 4ポイント
- 中指: 4ポイント
- 薬指: 4ポイント
- 小指: 4ポイント

各ポイントは正規化された座標（x, y, z）を持ちます。

## 複数手の検出

MediaPipe Handは複数の手を同時に検出できます。デフォルトでは最大2つの手を検出しますが、`--max-hands`オプションで変更可能です。

- 各手に対して個別にグーチョキパーを判定
- 左右の手を自動識別（左手/右手として表示）
- 各手の判定結果を手首の位置付近に表示

例: 両手で同時にグーチョキパーを判定する場合
```bash
python main.py --camera --max-hands 2
```

