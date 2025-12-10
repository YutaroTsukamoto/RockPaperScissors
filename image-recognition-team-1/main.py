import cv2
import mediapipe as mp
import numpy as np
from typing import List, Tuple, Optional, Dict
import argparse
from PIL import Image, ImageDraw, ImageFont
import platform


class HandLandmarkExtractor:
    """MediaPipe Handを使用して手の動画から3×21次元の点群を抽出するクラス"""
    
    def __init__(self, max_num_hands: int = 2):
        """
        初期化
        
        Args:
            max_num_hands: 同時に検出する最大の手の数（デフォルト: 2）
        """
        self.max_num_hands = max_num_hands
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=max_num_hands,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.mp_drawing = mp.solutions.drawing_utils
        self._japanese_font = None  # フォントキャッシュ
    
    def extract_landmarks_from_frame(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """
        1フレームから手のランドマークを抽出（最初の1つの手のみ）
        
        Args:
            frame: 入力画像フレーム (BGR形式)
            
        Returns:
            3×21次元の点群配列 (21個のポイント、各3次元: x, y, z)
            検出されない場合はNone
        """
        # BGRからRGBに変換
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # 手の検出
        results = self.hands.process(rgb_frame)
        
        if results.multi_hand_landmarks:
            # 最初に検出された手を使用
            hand_landmarks = results.multi_hand_landmarks[0]
            
            # 21個のランドマークポイントを抽出 (各ポイントはx, y, z座標を持つ)
            landmarks = np.zeros((21, 3))
            
            for idx, landmark in enumerate(hand_landmarks.landmark):
                landmarks[idx] = [landmark.x, landmark.y, landmark.z]
            
            return landmarks
        
        return None
    
    def extract_all_landmarks_from_frame(self, frame: np.ndarray) -> List[Tuple[np.ndarray, str]]:
        """
        1フレームからすべての手のランドマークを抽出
        
        Args:
            frame: 入力画像フレーム (BGR形式)
            
        Returns:
            (ランドマーク配列, 手のラベル)のタプルのリスト
        """
        # BGRからRGBに変換
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # 手の検出
        results = self.hands.process(rgb_frame)
        
        all_landmarks = []
        
        if results.multi_hand_landmarks and results.multi_handedness:
            for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                # 手のラベル（Left/Right）を取得
                hand_label = handedness.classification[0].label
                
                # 21個のランドマークポイントを抽出
                landmarks = np.zeros((21, 3))
                for idx, landmark in enumerate(hand_landmarks.landmark):
                    landmarks[idx] = [landmark.x, landmark.y, landmark.z]
                
                all_landmarks.append((landmarks, hand_label))
        
        return all_landmarks
    
    def extract_landmarks_from_video(self, video_path: str) -> List[np.ndarray]:
        """
        動画ファイルから全フレームの手のランドマークを抽出
        
        Args:
            video_path: 動画ファイルのパス
            
        Returns:
            各フレームの3×21次元点群のリスト
        """
        cap = cv2.VideoCapture(video_path)
        all_landmarks = []
        
        if not cap.isOpened():
            raise ValueError(f"動画ファイルを開けませんでした: {video_path}")
        
        frame_count = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            landmarks = self.extract_landmarks_from_frame(frame)
            if landmarks is not None:
                all_landmarks.append(landmarks)
                print(f"フレーム {frame_count}: 手を検出しました")
            else:
                print(f"フレーム {frame_count}: 手が検出されませんでした")
            
            frame_count += 1
        
        cap.release()
        return all_landmarks
    
    def extract_landmarks_from_camera(self, output_file: Optional[str] = None) -> List[np.ndarray]:
        """
        カメラからリアルタイムで手のランドマークを抽出し、グーチョキパーを判定（複数手対応）
        
        Args:
            output_file: 結果を保存するファイルパス（オプション）
            
        Returns:
            各フレームの3×21次元点群のリスト
        """
        cap = cv2.VideoCapture(0)
        all_landmarks = []
        
        if not cap.isOpened():
            raise ValueError("カメラを開けませんでした")
        
        print("カメラから手の検出を開始します。'q'キーで終了します。")
        print(f"最大 {self.max_num_hands} つの手を同時に検出できます。")
        frame_count = 0
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # 可視化
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.hands.process(rgb_frame)
            
            # すべての手のランドマークと判定を取得
            hand_data = []
            
            if results.multi_hand_landmarks and results.multi_handedness:
                for idx, (hand_landmarks, handedness) in enumerate(zip(results.multi_hand_landmarks, results.multi_handedness)):
                    # ランドマークを描画
                    self.mp_drawing.draw_landmarks(
                        frame,
                        hand_landmarks,
                        self.mp_hands.HAND_CONNECTIONS
                    )
                    
                    # 手のラベル（Left/Right）を取得
                    hand_label = handedness.classification[0].label
                    confidence = handedness.classification[0].score
                    
                    # ランドマークを抽出
                    landmarks = np.zeros((21, 3))
                    for i, landmark in enumerate(hand_landmarks.landmark):
                        landmarks[i] = [landmark.x, landmark.y, landmark.z]
                    
                    # 各指が伸びているかを判定
                    fingers = self.check_finger_extended(landmarks)
                    
                    # グーチョキパーを判定
                    judgment = self.judge_rock_paper_scissors(fingers)
                    
                    # 手首の位置を取得（表示位置の決定に使用）
                    wrist_pos = (int(hand_landmarks.landmark[0].x * frame.shape[1]),
                                int(hand_landmarks.landmark[0].y * frame.shape[0]))
                    
                    hand_data.append({
                        'landmarks': landmarks,
                        'label': hand_label,
                        'judgment': judgment,
                        'wrist_pos': wrist_pos,
                        'confidence': confidence
                    })
                    
                    all_landmarks.append(landmarks)
                    print(f"フレーム {frame_count}: {hand_label}手を検出 - {judgment} (信頼度: {confidence:.2f})")
            
            # 各手の判定結果を画面に表示
            for idx, data in enumerate(hand_data):
                # 手のラベル（左手/右手）と判定結果を表示
                display_text = f"{data['label']}: {data['judgment']}"
                self._draw_judgment_at_position(
                    frame, 
                    data['judgment'], 
                    data['wrist_pos'][0], 
                    data['wrist_pos'][1] - 50 - (idx * 80),  # 複数の手がある場合は縦に並べて表示
                    data['label']
                )
            
            # フレームを表示
            cv2.imshow('Rock Paper Scissors Detection', frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            
            frame_count += 1
        
        cap.release()
        cv2.destroyAllWindows()
        
        # 結果を保存
        if output_file and all_landmarks:
            self.save_landmarks(all_landmarks, output_file)
            print(f"結果を {output_file} に保存しました")
        
        return all_landmarks
    
    def _get_japanese_font(self, size: int = 60):
        """
        日本語フォントを取得（キャッシュ付き）
        
        Args:
            size: フォントサイズ
            
        Returns:
            ImageFontオブジェクト
        """
        # キャッシュがあればそれを返す
        if self._japanese_font is not None:
            return self._japanese_font
        
        system = platform.system()
        
        # OSに応じたフォントパスを設定
        if system == "Windows":
            # Windowsのデフォルト日本語フォント
            font_paths = [
                "C:/Windows/Fonts/msgothic.ttc",  # MS ゴシック
                "C:/Windows/Fonts/meiryo.ttc",    # メイリオ
                "C:/Windows/Fonts/yuigoth.ttc",   # 游ゴシック
            ]
        elif system == "Darwin":  # macOS
            font_paths = [
                "/System/Library/Fonts/Helvetica.ttc",
                "/Library/Fonts/Arial Unicode.ttf",
            ]
        else:  # Linux
            font_paths = [
                "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
                "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
            ]
        
        # 利用可能なフォントを探す
        for font_path in font_paths:
            try:
                self._japanese_font = ImageFont.truetype(font_path, size)
                return self._japanese_font
            except (OSError, IOError):
                continue
        
        # フォントが見つからない場合はデフォルトフォントを使用
        self._japanese_font = ImageFont.load_default()
        return self._japanese_font
    
    def _draw_judgment(self, frame: np.ndarray, judgment: str):
        """
        判定結果を画面に描画（日本語対応、左上固定位置）
        
        Args:
            frame: 画像フレーム
            judgment: 判定結果（'グー', 'チョキ', 'パー', '判定不能'）
        """
        self._draw_judgment_at_position(frame, judgment, 20, 20)
    
    def _draw_judgment_at_position(self, frame: np.ndarray, judgment: str, x: int, y: int, hand_label: Optional[str] = None):
        """
        判定結果を指定位置に描画（日本語対応）
        
        Args:
            frame: 画像フレーム
            judgment: 判定結果（'グー', 'チョキ', 'パー', '判定不能'）
            x: X座標
            y: Y座標
            hand_label: 手のラベル（'Left'/'Right'、オプション）
        """
        # 判定結果に応じた色を設定（BGR形式）
        color_map = {
            'グー': (0, 255, 0),      # 緑
            'チョキ': (255, 0, 0),    # 青
            'パー': (0, 0, 255),      # 赤
            '判定不能': (128, 128, 128)  # グレー
        }
        
        color_bgr = color_map.get(judgment, (128, 128, 128))
        # PIL用にRGB形式に変換
        color_rgb = (color_bgr[2], color_bgr[1], color_bgr[0])
        
        # 手のラベルを日本語に変換
        label_text = ""
        if hand_label:
            label_text = "左手" if hand_label == "Left" else "右手"
            display_text = f"{label_text}: {judgment}"
        else:
            display_text = judgment
        
        # フォントサイズ
        font_size = 50
        font = self._get_japanese_font(font_size)
        
        # テキストのサイズを取得
        img_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(img_pil)
        
        # テキストのバウンディングボックスを取得
        bbox = draw.textbbox((0, 0), display_text, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        
        # 位置とパディング
        padding = 15
        
        # 画面外に出ないように調整
        frame_height, frame_width = frame.shape[:2]
        if x + text_width + padding > frame_width:
            x = frame_width - text_width - padding
        if y + text_height + padding > frame_height:
            y = frame_height - text_height - padding
        if x < padding:
            x = padding
        if y < padding:
            y = padding
        
        # 背景の矩形を描画（OpenCVで描画）
        cv2.rectangle(
            frame,
            (x - padding, y - padding),
            (x + text_width + padding, y + text_height + padding),
            (0, 0, 0),
            -1
        )
        
        # PILで日本語テキストを描画
        img_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(img_pil)
        draw.text(
            (x, y),
            display_text,
            font=font,
            fill=color_rgb
        )
        
        # PIL画像をOpenCV画像に変換
        frame[:] = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
    
    def save_landmarks(self, landmarks_list: List[np.ndarray], filepath: str):
        """
        ランドマークをファイルに保存
        
        Args:
            landmarks_list: ランドマークのリスト
            filepath: 保存先ファイルパス
        """
        # NumPy配列として保存
        landmarks_array = np.array(landmarks_list)
        np.save(filepath, landmarks_array)
        print(f"ランドマークを保存しました: {filepath}")
        print(f"形状: {landmarks_array.shape} (フレーム数, 21ポイント, 3次元)")
    
    def load_landmarks(self, filepath: str) -> np.ndarray:
        """
        保存されたランドマークを読み込み
        
        Args:
            filepath: ファイルパス
            
        Returns:
            ランドマーク配列
        """
        return np.load(filepath)
    
    def _calculate_distance(self, point1: np.ndarray, point2: np.ndarray) -> float:
        """
        2点間の3次元距離を計算
        
        Args:
            point1: 第1点の座標 (3,)
            point2: 第2点の座標 (3,)
            
        Returns:
            距離
        """
        return np.linalg.norm(point1 - point2)
    
    def _calculate_angle(self, point1: np.ndarray, point2: np.ndarray, point3: np.ndarray) -> float:
        """
        3点からなる角度を計算（point2を頂点とする角度）
        
        Args:
            point1: 第1点の座標 (3,)
            point2: 頂点の座標 (3,)
            point3: 第3点の座標 (3,)
            
        Returns:
            角度（ラジアン）
        """
        vector1 = point1 - point2
        vector2 = point3 - point2
        
        # ベクトルの正規化
        norm1 = np.linalg.norm(vector1)
        norm2 = np.linalg.norm(vector2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        vector1 = vector1 / norm1
        vector2 = vector2 / norm2
        
        # 内積から角度を計算
        dot_product = np.clip(np.dot(vector1, vector2), -1.0, 1.0)
        angle = np.arccos(dot_product)
        
        return angle
    
    def _is_finger_extended(self, landmarks: np.ndarray, finger_points: List[int], angle_threshold: float = 1.0) -> bool:
        """
        指が伸びているかを相対位置に基づいて判定
        
        Args:
            landmarks: ランドマーク配列 (21, 3)
            finger_points: 指のポイントインデックスのリスト [付け根, 第1関節, 第2関節, 先端]
            angle_threshold: 角度の閾値（ラジアン）。この値より小さい角度なら伸びていると判定
            
        Returns:
            指が伸びているかどうか
        """
        if len(finger_points) < 3:
            return False
        
        # 方法1: 関節間の角度を計算
        # 指が伸びている場合、関節間の角度は小さい（ほぼ一直線）
        total_angle = 0.0
        
        # 各関節間の角度を計算
        for i in range(len(finger_points) - 2):
            angle = self._calculate_angle(
                landmarks[finger_points[i]],
                landmarks[finger_points[i + 1]],
                landmarks[finger_points[i + 2]]
            )
            total_angle += angle
        
        # 平均角度が閾値より小さい場合、指が伸びていると判定
        if len(finger_points) - 2 > 0:
            avg_angle = total_angle / (len(finger_points) - 2)
        else:
            avg_angle = 0.0
        
        # 方法2: 付け根から先端までの直線距離と、各関節を経由した距離を比較
        # 指が伸びている場合、直線距離 ≈ 各関節を経由した距離の合計
        straight_distance = self._calculate_distance(
            landmarks[finger_points[0]],
            landmarks[finger_points[-1]]
        )
        
        path_distance = 0.0
        for i in range(len(finger_points) - 1):
            path_distance += self._calculate_distance(
                landmarks[finger_points[i]],
                landmarks[finger_points[i + 1]]
            )
        
        # 直線距離と経路距離の比率
        if path_distance > 0:
            distance_ratio = straight_distance / path_distance
        else:
            distance_ratio = 0.0
        
        # z座標による判定は削除（手の回転に弱いため）
        # 代わりに、角度と距離比の両方を考慮したより堅牢な判定を使用
        
        # 角度と距離比の両方を考慮
        # 指が伸びている場合：
        # - 角度が小さい（ほぼ一直線）
        # - 距離比が大きい（直線距離 ≈ 経路距離）
        angle_ok = avg_angle < angle_threshold
        distance_ok = distance_ratio > 0.80  # 0.85から0.80に緩和（より柔軟に）
        
        # 両方の条件を満たす場合、または距離比が非常に高い場合（0.95以上）は伸びていると判定
        # 距離比が非常に高い場合は、角度が多少大きくても伸びていると判定
        return (angle_ok and distance_ok) or distance_ratio > 0.95
    
    def check_finger_extended(self, landmarks: np.ndarray) -> Dict[str, bool]:
        """
        各指が伸びているかを相対位置に基づいて判定（手の向きにロバスト）
        
        Args:
            landmarks: ランドマーク配列 (21, 3)
            
        Returns:
            各指が伸びているかの辞書 {'thumb': bool, 'index': bool, 'middle': bool, 'ring': bool, 'pinky': bool}
        """
        fingers = {
            'thumb': False,
            'index': False,
            'middle': False,
            'ring': False,
            'pinky': False
        }
        
        # 各指のポイントインデックス
        # 親指: 1(付け根), 2(第1関節), 3(第2関節), 4(先端)
        # 人差し指: 5(付け根), 6(第1関節), 7(第2関節), 8(先端)
        # 中指: 9(付け根), 10(第1関節), 11(第2関節), 12(先端)
        # 薬指: 13(付け根), 14(第1関節), 15(第2関節), 16(先端)
        # 小指: 17(付け根), 18(第1関節), 19(第2関節), 20(先端)
        
        # 親指の判定（親指は手首(0)から始まる）
        # 親指は他の指と異なり、手首(0)から始まる
        # 親指は動きが特殊なため、より緩い条件で判定
        # 親指の付け根(1)から先端(4)までで判定（手首(0)は除外）
        thumb_points = [1, 2, 3, 4]  # 付け根から先端まで
        fingers['thumb'] = self._is_finger_extended(landmarks, thumb_points, angle_threshold=1.2)
        
        # 人差し指の判定
        index_points = [5, 6, 7, 8]
        fingers['index'] = self._is_finger_extended(landmarks, index_points, angle_threshold=1.0)
        
        # 中指の判定
        middle_points = [9, 10, 11, 12]
        fingers['middle'] = self._is_finger_extended(landmarks, middle_points, angle_threshold=1.0)
        
        # 薬指の判定
        ring_points = [13, 14, 15, 16]
        fingers['ring'] = self._is_finger_extended(landmarks, ring_points, angle_threshold=1.0)
        
        # 小指の判定
        pinky_points = [17, 18, 19, 20]
        fingers['pinky'] = self._is_finger_extended(landmarks, pinky_points, angle_threshold=1.0)
        
        return fingers
    
    def judge_rock_paper_scissors(self, fingers: Dict[str, bool]) -> str:
        """
        グーチョキパーを判定
        
        Args:
            fingers: 各指が伸びているかの辞書
            
        Returns:
            'グー', 'チョキ', 'パー', または '判定不能'
        """
        extended_count = sum(fingers.values())
        
        # グー: すべての指が曲がっている
        if extended_count == 0:
            return 'グー'
        
        # チョキ: 人差し指と中指だけが伸びている
        # 親指は動きが特殊なため、チョキの判定では親指を除外
        # 人差し指と中指が伸びていて、薬指と小指が曲がっていることを確認
        if (fingers['index'] and fingers['middle'] and
            not fingers['ring'] and not fingers['pinky']):
            # 親指以外の指で判定（親指は除外）
            non_thumb_count = sum([fingers['index'], fingers['middle'], 
                                   fingers['ring'], fingers['pinky']])
            if non_thumb_count == 2:  # 人差し指と中指のみが伸びている
                return 'チョキ'
        
        # パー: すべての指が伸びている
        if extended_count == 5:
            return 'パー'
        
        # その他の場合は判定不能
        return '判定不能'


def main():
    parser = argparse.ArgumentParser(description='MediaPipe Handを使用して手の動画から3×21次元の点群を抽出')
    parser.add_argument('--video', type=str, help='動画ファイルのパス')
    parser.add_argument('--camera', action='store_true', help='カメラからリアルタイムで検出')
    parser.add_argument('--output', type=str, default='hand_landmarks.npy', help='出力ファイル名（デフォルト: hand_landmarks.npy）')
    parser.add_argument('--max-hands', type=int, default=2, help='同時に検出する最大の手の数（デフォルト: 2）')
    
    args = parser.parse_args()
    
    extractor = HandLandmarkExtractor(max_num_hands=args.max_hands)
    
    if args.camera:
        # カメラから検出
        landmarks_list = extractor.extract_landmarks_from_camera(args.output)
    elif args.video:
        # 動画ファイルから検出
        landmarks_list = extractor.extract_landmarks_from_video(args.video)
        if landmarks_list:
            extractor.save_landmarks(landmarks_list, args.output)
            print(f"\n検出結果:")
            print(f"  検出されたフレーム数: {len(landmarks_list)}")
            print(f"  各フレームの形状: {landmarks_list[0].shape} (21ポイント, 3次元)")
            print(f"  保存先: {args.output}")
    else:
        print("使用方法:")
        print("  動画ファイルから検出: python main.py --video <動画ファイルのパス>")
        print("  カメラから検出: python main.py --camera")
        print("  出力ファイル指定: python main.py --video <動画ファイル> --output <出力ファイル名>")


if __name__ == "__main__":
    main()
