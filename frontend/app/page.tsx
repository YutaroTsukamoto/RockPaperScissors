"use client";

import { useEffect, useRef, useState } from "react";

export default function Home() {
  const [isConnected, setIsConnected] = useState(false);
  const imgRef = useRef<HTMLImageElement>(null);
  const wsRef = useRef<WebSocket | null>(null);

  useEffect(() => {
    // Pythonサーバー(localhost:8000)へ接続
    const ws = new WebSocket("ws://localhost:8000/ws");
    wsRef.current = ws;

    ws.onopen = () => {
      console.log("Connected to Python Server");
      setIsConnected(true);
    };

    ws.onmessage = (event) => {
      // 受信したBase64データをimgタグのsrcに直接セット
      if (imgRef.current) {
        imgRef.current.src = `data:image/jpeg;base64,${event.data}`;
      }
    };

    ws.onclose = () => {
      console.log("Disconnected");
      setIsConnected(false);
    };

    // クリーンアップ
    return () => {
      ws.close();
    };
  }, []);

  return (
    <main className="flex min-h-screen flex-col items-center justify-center bg-gray-900 text-white p-4">
      <h1 className="text-3xl font-bold mb-4">Real-time Stream Demo</h1>

      <div className="mb-4">
        Status:
        <span
          className={isConnected ? "text-green-400 ml-2" : "text-red-400 ml-2"}
        >
          {isConnected ? "Connected ●" : "Disconnected ○"}
        </span>
      </div>

      <div className="relative border-4 border-gray-700 rounded-lg overflow-hidden shadow-2xl">
        {/* ここに映像が表示されます */}
        {/* eslint-disable-next-line @next/next/no-img-element */}
        <img
          ref={imgRef}
          width={640}
          height={480}
          alt="Live Stream"
          className="bg-black"
          style={{ minWidth: "640px", minHeight: "480px" }}
        />
      </div>

      <p className="mt-8 text-gray-400 text-sm">
        System A (Camera) → Python (Processing) → Next.js (Display)
      </p>
    </main>
  );
}
