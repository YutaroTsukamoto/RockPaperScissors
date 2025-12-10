import cv2
import asyncio
import base64
from fastapi import FastAPI, WebSocket, WebSocketDisconnect

app = FastAPI()


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    print("Client connected")

    # 0はMacのWebカメラ。
    # ※ System Aの動画ファイルやRTSP URLがある場合はここを書き換えます
    camera = cv2.VideoCapture(0)

    try:
        while True:
            # 1. 映像取得
            success, frame = camera.read()
            if not success:
                break

            # 2. 加工 (Processing)
            # 例: グレースケールにして、システムB用にリサイズ
            frame = cv2.resize(frame, (640, 480))
            processed_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # デモ用にテキスト描画
            cv2.putText(
                processed_frame,
                "Next.js + Python",
                (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 255, 255),
                2,
            )

            # 3. 送信準備 (JPEG化 -> Base64)
            _, buffer = cv2.imencode(
                ".jpg", processed_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 60]
            )
            jpg_as_text = base64.b64encode(buffer).decode("utf-8")

            # 4. WebSocketで送信
            await websocket.send_text(jpg_as_text)

            # フレームレート調整 (約30fps)
            await asyncio.sleep(0.033)

    except WebSocketDisconnect:
        print("Client disconnected")
    finally:
        camera.release()
