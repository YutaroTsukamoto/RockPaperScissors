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
    
    def check_finger_extended(self, landmarks: np.ndarray) -> Dict[str, bool]:
        """
        各指が伸びているかを判定
        
        Args:
            landmarks: ランドマーク配列 (21, 3)
            
        Returns:
            各指が伸びているかの辞書 {'thumb': bool, 'index': bool, 'middle': bool, 'ring': bool, 'pinky': bool}
        """
        # 手の向きを判定（左手か右手か）
        # 手首(0)と小指の付け根(17)のx座標を比較
        is_left_hand = landmarks[0][0] > landmarks[17][0]
        
        fingers = {
            'thumb': False,
            'index': False,
            'middle': False,
            'ring': False,
            'pinky': False
        }
        
        # 親指の判定（x座標で判定、左右の手で判定が異なる）
        if is_left_hand:
            # 左手の場合: 親指の先端(4)が関節(3)より右側（x座標が大きい）
            fingers['thumb'] = landmarks[4][0] > landmarks[3][0]
        else:
            # 右手の場合: 親指の先端(4)が関節(3)より左側（x座標が小さい）
            fingers['thumb'] = landmarks[4][0] < landmarks[3][0]
        
        # 人差し指の判定（y座標で判定、先端が関節より上）
        fingers['index'] = landmarks[8][1] < landmarks[6][1]
        
        # 中指の判定
        fingers['middle'] = landmarks[12][1] < landmarks[10][1]
        
        # 薬指の判定
        fingers['ring'] = landmarks[16][1] < landmarks[14][1]
        
        # 小指の判定
        fingers['pinky'] = landmarks[20][1] < landmarks[18][1]
        
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
        
        if extended_count == 0:
            return 'グー'
        elif extended_count == 2 and fingers['index'] and fingers['middle']:
            return 'チョキ'
        elif extended_count == 5:
            return 'パー'
        else:
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
