import numpy as np

data = np.loadtxt("all_coordinates.csv", delimiter=",", skiprows=1, dtype=str)

label = data[:, 0]
coordinates = data[:, 1:].astype(float).reshape(-1, 21, 3)

wrist_coords = coordinates[:, 0, :]
relative_coordinates = coordinates - wrist_coords[:, np.newaxis, :]

index_finger_mcp = relative_coordinates[:, 5, :]
scale = np.linalg.norm(index_finger_mcp, axis=1, keepdims=True)  # (N, 1)

# 正規化
scale = scale[:, :, np.newaxis]

normalized_coordinates = relative_coordinates / scale

z_axis = normalized_coordinates[:, 5, :]
z_axis /= np.linalg.norm(z_axis, axis=1, keepdims=True)

# 補助ベクトルを決める: 手首(0) -> 小指(17)
# これを使って手の「平面」を定義します
v_pinky = normalized_coordinates[:, 17, :]
v_pinky /= np.linalg.norm(v_pinky, axis=1, keepdims=True)

# 新しいX軸を決める: Z軸と小指ベクトルの外積
# これにより、手のひらの向き(法線)が求まります
x_axis = np.cross(v_pinky, z_axis)
x_axis /= np.linalg.norm(x_axis, axis=1, keepdims=True)

# 新しいY軸を決める: Z軸とX軸の外積 (直交させるため)
y_axis = np.cross(z_axis, x_axis)
y_axis /= np.linalg.norm(y_axis, axis=1, keepdims=True)

# 回転行列を構築する (N, 3, 3)
# [x_axis, y_axis, z_axis] を行として積み重ねます
# これにより、元の座標と内積を取ると、新しい軸への射影(回転)になります
rotation_matrices = np.stack([x_axis, y_axis, z_axis], axis=1)

# 全座標に回転行列を適用
# np.matmul または @演算子 で行列積を計算
# (N, 21, 3) @ (N, 3, 3).transpose -> 形状合わせのために工夫が必要
# ここでは einsum を使うと直感的に書けます: "nij, nkj -> nik"
# n:サンプル数, i:21関節, j:元のxyz, k:新しいxyz(基底)
rotated_coordinates = np.einsum(
    "nij,nkj->nik", normalized_coordinates, rotation_matrices
)

X_final = rotated_coordinates.reshape(rotated_coordinates.shape[0], -1)
output = np.concatenate([label.reshape(-1, 1), X_final], axis=1)

np.savetxt(
    "coordinates_normalized.csv",
    output,
    delimiter=",",
    fmt="%s",
    comments="",
)
