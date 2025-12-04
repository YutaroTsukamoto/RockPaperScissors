import cv2
import mediapipe as mp
import os
import csv
import glob
from tqdm import tqdm


def extract_coordinates(dataset_path, output_file):
    mp_hands = mp.solutions.hands

    # MediaPipe Handsの初期化
    # static_image_mode=True: 静止画モード
    # max_num_hands=1: 検出する手の最大数
    # min_detection_confidence=0.5: 検出の信頼度閾値
    with mp_hands.Hands(
        static_image_mode=True, max_num_hands=1, min_detection_confidence=0.5
    ) as hands:
        # 出力ファイルの準備
        with open(output_file, mode="w", newline="") as f:
            writer = csv.writer(f)

            # ヘッダー作成: label, x0, y0, z0, ... x20, y20, z20
            header = ["label"]
            for i in range(21):
                header.extend([f"x{i}", f"y{i}", f"z{i}"])
            writer.writerow(header)

            classes = ["rock", "paper", "scissors"]

            for class_name in classes:
                class_path = os.path.join(dataset_path, class_name)
                if not os.path.exists(class_path):
                    print(f"Warning: {class_path} does not exist.")
                    continue

                # 画像ファイルの取得 (jpg, jpeg, png)
                image_files = []
                for ext in ["*.jpg", "*.jpeg", "*.png"]:
                    image_files.extend(glob.glob(os.path.join(class_path, ext)))

                print(f"Processing {class_name}: {len(image_files)} images found.")

                for image_file in tqdm(image_files, desc=f"Processing {class_name}"):
                    image = cv2.imread(image_file)
                    if image is None:
                        print(f"Failed to read {image_file}")
                        continue

                    # BGRからRGBに変換
                    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

                    # ランドマーク検出
                    results = hands.process(image_rgb)

                    if results.multi_hand_landmarks:
                        for hand_landmarks in results.multi_hand_landmarks:
                            row = [class_name]
                            for landmark in hand_landmarks.landmark:
                                # 座標をリストに追加
                                row.extend([landmark.x, landmark.y, landmark.z])

                            writer.writerow(row)
                            # 1枚の画像から最初に見つかった手だけを使用
                            break


if __name__ == "__main__":
    # データセットのパスと出力ファイル名
    dataset_train_path = os.path.join("dataset", "all")
    output_csv = "all_coordinates.csv"

    print("Starting coordinate extraction...")
    extract_coordinates(dataset_train_path, output_csv)
    print(f"Coordinates extracted to {output_csv}")
