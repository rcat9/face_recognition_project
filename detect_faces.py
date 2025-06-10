# detect_faces_v3.py - 高速化版（複数人物対応）
import os
import numpy as np
from skimage.io import imread, imsave
from skimage.transform import resize
import sys
import glob
import time
import concurrent.futures


from face_detector import YoloV5FaceDetector

# 画像の最大サイズを制限する関数
def limit_image_size(image, max_width=1500):
    """画像サイズを制限して処理を高速化"""
    height, width = image.shape[:2]
    if width > max_width:
        scale = max_width / width
        new_height = int(height * scale)
        return resize(image, (new_height, max_width), preserve_range=True).astype(np.uint8)
    return image

# 以下の関数をimport文の下、detect_with_timeout関数の前に追加します
# 大体15行目あたりになるはずです

def is_human_face(face_landmarks):
    """顔のランドマークから人間の顔かどうかを判定する簡易的な関数"""
    if face_landmarks.shape[0] != 5:
        return False  # 5つの特徴点（目、鼻、口の両端）が検出されていない場合は除外
    
    # 目の位置関係をチェック（目が水平に近いことを確認）
    eyes_slope = abs((face_landmarks[1, 1] - face_landmarks[0, 1]) / 
                   max(1e-6, abs(face_landmarks[1, 0] - face_landmarks[0, 0])))
    if eyes_slope > 0.3:  # 傾きが大きすぎる場合は除外
        return False
    
    # 顔のアスペクト比をチェック
    face_width = abs(face_landmarks[1, 0] - face_landmarks[0, 0])
    face_height = abs(face_landmarks[4, 1] - face_landmarks[0, 1])
    aspect_ratio = face_height / max(1e-6, face_width)
    if aspect_ratio < 0.8 or aspect_ratio > 1.8:  # 人間の顔の一般的な範囲外
        return False
    
    return True

# タイムアウト付きの顔検出関数
def detect_with_timeout(detector, image, timeout=30, 
                      max_output_size=10, iou_threshold=0.45, score_threshold=0.5):  # ここを0.3から0.5に変更
    """タイムアウト付きで顔検出を実行"""
    result = None
    
    def _detect():
        nonlocal result
        result = detector.detect_in_image(
            image, 
            max_output_size=max_output_size,
            iou_threshold=iou_threshold, 
            score_threshold=score_threshold
        )
    
    # スレッドで実行
    with concurrent.futures.ThreadPoolExecutor() as executor:
        future = executor.submit(_detect)
        try:
            future.result(timeout=timeout)
            return result
        except concurrent.futures.TimeoutError:
            print("検出処理がタイムアウトしました")
            return np.array([]), np.array([]), np.array([]), np.array([])

# メイン処理
def main():
    # 顔検出器を初期化
    detector = YoloV5FaceDetector()

    # 入力画像フォルダのパス
    input_dir = "sample_images"

    # 出力フォルダがなければ作成
    output_dir = "aligned_faces"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 現在の最大の番号を確認
    existing_faces = glob.glob(os.path.join(output_dir, "face_*.jpg"))
    face_index = 0
    if existing_faces:
        # 既存のファイル名から最大の番号を取得
        for face_path in existing_faces:
            filename = os.path.basename(face_path)
            # face_X.jpg から数字部分を抽出
            try:
                number = int(filename.split('_')[1].split('.')[0])
                face_index = max(face_index, number + 1)
            except:
                pass
        print(f"既存の顔画像を検出しました。次のインデックスは {face_index} から始まります。")

    # 処理履歴ファイル
    history_file = os.path.join(output_dir, "processed_history.txt")
    processed_images = set()

    # 処理履歴ファイルが存在する場合は読み込む
    if os.path.exists(history_file):
        with open(history_file, 'r') as f:
            processed_images = set(line.strip() for line in f)
        print(f"{len(processed_images)}個の処理済み画像記録を読み込みました。")

    # 履歴ファイルを追記モードでオープン
    history_log = open(history_file, 'a')

    # フォルダ内の画像ファイル一覧を取得
    image_files = [f for f in os.listdir(input_dir) 
                if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    
    print(f"フォルダ '{input_dir}' 内の {len(image_files)}枚の画像を処理します...")
    processed_count = 0
    face_count = 0
    skipped_count = 0
    
    # 各画像を処理
    for filename in image_files:
        input_path = os.path.join(input_dir, filename)
        
        # すでに処理済みならスキップ
        if input_path in processed_images:
            print(f"スキップ: {filename}")
            continue
        
        try:
            start_time = time.time()
            print(f"画像を読み込んでいます: {input_path}")
            
            # 画像を読み込み
            image = imread(input_path)
            
            # 画像サイズを制限
            original_size = image.shape
            image = limit_image_size(image)
            if original_size != image.shape:
                print(f"  画像をリサイズしました: {original_size[:2]} → {image.shape[:2]}")
            
            # タイムアウト付きで顔検出を実行
            print(f"  顔を検出中...")
            bbs, pps, ccs, aligned_faces = detect_with_timeout(
                detector, image, timeout=60,  # タイムアウト60秒
                max_output_size=10,  # 最大10個の顔を検出
                iou_threshold=0.45,
                score_threshold=0.3
            )
            
            process_time = time.time() - start_time
            
            # 結果を処理
# 結果を処理
            if len(aligned_faces) > 0:
                print(f"  {filename}から{len(aligned_faces)}個の顔を検出しました！(処理時間: {process_time:.1f}秒)")
                
                # 人間の顔だけをフィルタリング
                human_faces = []
                human_indices = []
                
                for i, (face, landmarks) in enumerate(zip(aligned_faces, pps)):
                    if is_human_face(landmarks):
                        human_faces.append(face)
                        human_indices.append(i)
                
                print(f"  うち人間の顔と判断されたのは {len(human_faces)}/{len(aligned_faces)} 個です")
                
                # 人間の顔だけを保存
                for i, face in enumerate(human_faces):
                    face_path = os.path.join(output_dir, f"face_{face_index}.jpg")
                    imsave(face_path, face)
                    print(f"  顔画像を保存しました: {face_path}")
                    face_index += 1
                    face_count += 1
            else:
                if process_time >= 60:
                    print(f"  {filename}の処理がタイムアウトしました。スキップします。")
                    skipped_count += 1
                else:
                    print(f"  {filename}から顔を検出できませんでした。(処理時間: {process_time:.1f}秒)")
            
            # 処理済みとして記録
            processed_images.add(input_path)
            history_log.write(input_path + '\n')
            history_log.flush()  # すぐに書き込みを反映
            
            processed_count += 1
            
        except Exception as e:
            print(f"画像 {filename} の処理中にエラーが発生しました: {str(e)}")

    # 履歴ファイルを閉じる
    history_log.close()

    print(f"\n処理完了！")
    print(f"合計: {processed_count} 枚の画像を処理")
    print(f"検出: {face_count} 個の顔を検出")
    print(f"スキップ: {skipped_count} 枚の画像をタイムアウトでスキップ")
    print(f"次回実行時は face_{face_index}.jpg から処理が始まります。")

if __name__ == "__main__":
    main()