import cv2
import mediapipe as mp
import pickle
import numpy as np
import subprocess
import json
import platform

# モデルのロード
model_path = "svm_model.pkl"
try:
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    print(f"Model loaded from {model_path}")
except FileNotFoundError:
    print(f"Error: {model_path} not found. Please train the model first.")
    exit()

# MediaPipeの初期化
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=20,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)


def preprocess_landmarks(landmarks):
    """
    ランドマーク座標を前処理（正規化・回転）する関数
    preprocessing.py のロジックに基づく
    """
    # (1, 21, 3) の形状にする
    coordinates = np.array(landmarks).reshape(1, 21, 3)

    # 手首を原点にする
    wrist_coords = coordinates[:, 0, :]
    relative_coordinates = coordinates - wrist_coords[:, np.newaxis, :]

    # スケーリング: 手首から人差し指の付け根(5)までの距離
    index_finger_mcp = relative_coordinates[:, 5, :]
    scale = np.linalg.norm(index_finger_mcp, axis=1, keepdims=True)
    scale = scale[:, :, np.newaxis]

    # ゼロ除算を防ぐための小さな値
    scale = np.maximum(scale, 1e-6)

    normalized_coordinates = relative_coordinates / scale

    # 座標系の構築
    # Z軸: 手首 -> 人差し指の付け根
    z_axis = normalized_coordinates[:, 5, :]
    z_axis /= np.linalg.norm(z_axis, axis=1, keepdims=True)

    # 補助ベクトル: 手首 -> 小指の付け根(17)
    v_pinky = normalized_coordinates[:, 17, :]
    v_pinky /= np.linalg.norm(v_pinky, axis=1, keepdims=True)

    # X軸: Z軸と小指ベクトルの外積 (手のひらの法線方向)
    x_axis = np.cross(v_pinky, z_axis)
    x_axis /= np.linalg.norm(x_axis, axis=1, keepdims=True)

    # Y軸: Z軸とX軸の外積 (直交座標系を完成させる)
    y_axis = np.cross(z_axis, x_axis)
    y_axis /= np.linalg.norm(y_axis, axis=1, keepdims=True)

    # 回転行列の構築
    rotation_matrices = np.stack([x_axis, y_axis, z_axis], axis=1)

    # 座標の回転
    rotated_coordinates = np.einsum(
        "nij,nkj->nik", normalized_coordinates, rotation_matrices
    )

    # 平坦化 (1, 63)
    X_final = rotated_coordinates.reshape(rotated_coordinates.shape[0], -1)
    return X_final


cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

print("Press 'q' to exit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to capture image.")
        break

    # 画像を左右反転（鏡のように表示するため）
    frame = cv2.flip(frame, 1)

    # BGRからRGBに変換
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # 手の検出
    results = hands.process(image_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # ランドマークの描画
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # ランドマーク座標の抽出
            landmarks = []
            for landmark in hand_landmarks.landmark:
                landmarks.append([landmark.x, landmark.y, landmark.z])

            # 前処理
            try:
                processed_features = preprocess_landmarks(landmarks)

                # 予測 (decision_functionを使ってスコア判定)
                scores = model.decision_function(processed_features)[0]
                max_score_index = np.argmax(scores)
                max_score = scores[max_score_index]

                # 閾値判定 (0.0が境界線。正の値が大きいほど確信度が高い)
                # この値を調整することでunknownの判定基準を変えられます
                threshold = 0.5
                if max_score < threshold:
                    prediction = "unknown"
                else:
                    prediction = model.classes_[max_score_index]

                # 画面に表示
                # 手首の座標を取得して、その近くにテキストを表示
                wrist = hand_landmarks.landmark[0]
                h, w, _ = frame.shape
                cx, cy = int(wrist.x * w), int(wrist.y * h)

                cv2.putText(
                    frame,
                    prediction,
                    (cx, cy - 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    2,
                    (0, 255, 0),
                    2,
                    cv2.LINE_AA,
                )

            except Exception as e:
                print(f"Prediction error: {e}")

    # 画面表示
    cv2.imshow("Hand Gesture Recognition", frame)

    # 'q'キーで終了
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
