#!/usr/bin/env python3

import os
import sys
import numpy as np
import glob
import shutil
from sklearn.metrics.pairwise import cosine_similarity
from skimage.io import imread, imsave
import argparse
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import matplotlib as mpl
from scipy import stats
from scipy.signal import find_peaks

# モデルタイプのリスト
MODEL_TYPES = ["effv2s", "r100magface", "r100arcface", "r100adaface"]

def parse_arguments():
    """コマンドライン引数を解析する"""
    parser = argparse.ArgumentParser(description='顔埋め込みベクトルに基づいてクラスタリング')
    parser.add_argument('model', type=str,
                        choices=MODEL_TYPES,
                        help=f'使用するモデルを指定します。選択肢: {", ".join(MODEL_TYPES)}')
    parser.add_argument('--threshold', type=float, default=None, 
                        help='顔を同一人物と判定するコサイン類似度の閾値 (0.0〜1.0、未指定時はモデルに応じた値)')
    parser.add_argument('--input_dir', type=str, default='embedded_faces',
                        help='埋め込みベクトルが格納されているディレクトリ')
    parser.add_argument('--faces_dir', type=str, default='aligned_faces',
                        help='顔画像が格納されているディレクトリ')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='クラスタリング結果を保存するディレクトリ（デフォルトはcompared_faces_モデル名）')
    parser.add_argument('--debug', action='store_true',
                        help='デバッグモードを有効にする（詳細な類似度情報を表示）')
    parser.add_argument('--analyze', action='store_true',
                        help='クラスタリングを行わず、類似度の分析のみを実行')
    parser.add_argument('--visualize', action='store_true',
                        help='類似度の分析結果をグラフで表示（analyze または debug モードで有効）')
    parser.add_argument('--check', action='store_true',
                        help='既存のクラスタリングの問題点をチェック')
    parser.add_argument('--merge', action='store_true',
                        help='類似クラスタを統合')
    
    args = parser.parse_args()
    
    # 出力ディレクトリが指定されていない場合はデフォルト値を設定
    if args.output_dir is None:
        args.output_dir = f"compared_faces_{args.model}"
    
    return args

def sync_history_with_folders(output_dir, history_file):
    """履歴ファイルと実際のフォルダ構造を同期させる関数"""
    if not os.path.exists(output_dir):
        # 出力ディレクトリがない場合は履歴も初期化
        if os.path.exists(history_file):
            os.remove(history_file)
        return set()
    
    # 既存の履歴を読み込む
    existing_history = set()
    if os.path.exists(history_file):
        with open(history_file, 'r') as f:
            existing_history = set(line.strip() for line in f)
    
    # 実際にフォルダ内に存在する顔ID
    existing_faces = set()
    person_folders = [d for d in os.listdir(output_dir) if os.path.isdir(os.path.join(output_dir, d)) and d.startswith("person_")]
    
    for folder in person_folders:
        folder_path = os.path.join(output_dir, folder)
        for file in os.listdir(folder_path):
            if file.endswith('.jpg'):
                face_id = os.path.splitext(file)[0]
                existing_faces.add(face_id)
    
    # 履歴を更新
    updated_history = existing_history.intersection(existing_faces)
    
    # 履歴ファイルが変更されていれば、更新する
    if updated_history != existing_history:
        print(f"履歴ファイルとフォルダ構造の同期を取っています...")
        print(f"削除された顔ID: {len(existing_history - updated_history)}個")
        
        # 履歴ファイルを更新
        with open(history_file, 'w') as f:
            for face_id in sorted(updated_history):
                f.write(face_id + '\n')
    
    return updated_history

def get_processed_faces(output_dir, history_file):
    """処理済みの顔IDを取得し、フォルダ構造と同期をとる"""
    return sync_history_with_folders(output_dir, history_file)

def update_history(face_id, history_file):
    """処理履歴を更新"""
    with open(history_file, 'a') as f:
        f.write(face_id + '\n')

def get_face_person_mapping(output_dir):
    """各顔IDがどのpersonフォルダに属しているかのマッピングを取得"""
    face_to_person = {}
    
    if not os.path.exists(output_dir):
        return face_to_person
    
    person_folders = [d for d in os.listdir(output_dir) if os.path.isdir(os.path.join(output_dir, d)) and d.startswith("person_")]
    
    for folder in person_folders:
        folder_path = os.path.join(output_dir, folder)
        for file in os.listdir(folder_path):
            if file.endswith('.jpg'):
                face_id = os.path.splitext(file)[0]
                face_to_person[face_id] = folder
    
    return face_to_person

def load_embeddings(directory, model_type):
    """ディレクトリ内の指定されたモデルの埋め込みベクトルを読み込む"""
    # 指定されたモデルタイプの埋め込みベクトルのみを検索
    embedding_files = glob.glob(os.path.join(directory, f"*_{model_type}.npy"))
    
    embeddings = {}
    
    for file_path in embedding_files:
        # ファイル名からface_IDを抽出
        filename = os.path.basename(file_path)
        # モデルタイプのサフィックスを削除して元のface_idを取得
        face_id = filename.replace(f"_{model_type}.npy", "")
        
        # 埋め込みベクトルを読み込む
        embedding = np.load(file_path)
        embeddings[face_id] = embedding
    
    return embeddings

def calculate_similarity_matrix(embeddings, output_dir):
    """すべての顔の埋め込みベクトル間のコサイン類似度行列を計算"""
    face_ids = list(embeddings.keys())
    n = len(face_ids)
    
    # personフォルダのマッピングを取得
    face_to_person = get_face_person_mapping(output_dir)
    
    # 表示用のラベル作成（person情報を含む）
    labels = []
    for face_id in face_ids:
        if face_id in face_to_person:
            labels.append(f"{face_id} ({face_to_person[face_id]})")
        else:
            labels.append(face_id)
    
    # コサイン類似度行列を初期化
    similarity_matrix = np.zeros((n, n))
    
    # 各ペアのコサイン類似度を計算
    for i in range(n):
        for j in range(n):
            if i == j:
                similarity_matrix[i, j] = 1.0  # 同じ顔との類似度は1.0
            else:
                # 正しいコサイン類似度の計算（ノルムで割る）
                embedding1 = embeddings[face_ids[i]]
                embedding2 = embeddings[face_ids[j]]
                
                dot_product = np.dot(embedding1, embedding2)
                norm1 = np.linalg.norm(embedding1)
                norm2 = np.linalg.norm(embedding2)
                
                # ゼロ除算対策
                if norm1 < 1e-10 or norm2 < 1e-10:
                    print(f"警告: ベクトルのノルムが小さすぎます: {face_ids[i]}={norm1}, {face_ids[j]}={norm2}")
                    similarity_matrix[i, j] = 0.0
                else:
                    similarity_matrix[i, j] = dot_product / (norm1 * norm2)
    
    return face_ids, labels, similarity_matrix

def find_similar_pairs(face_ids, similarity_matrix, threshold=0.7):
    """類似度が閾値以上のペアを見つける"""
    similar_pairs = []
    n = len(face_ids)
    
    for i in range(n):
        for j in range(i+1, n):  # 重複を避けるため、i < j のみを見る
            similarity = similarity_matrix[i, j]
            if similarity >= threshold:
                similar_pairs.append((face_ids[i], face_ids[j], similarity))
    
    # 類似度の高い順にソート
    similar_pairs.sort(key=lambda x: x[2], reverse=True)
    
    return similar_pairs

def suggest_optimal_threshold(all_similarities):
    """最適な閾値を提案する"""
    # 類似度の分布から最適な閾値を推定
    # 簡単な方法: 最頻値から分布の山を特定
    hist, bin_edges = np.histogram(all_similarities, bins=100)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    # ピークを見つける（ローカルマキシマム）
    peaks, _ = find_peaks(hist, height=max(hist) * 0.2)
    
    if len(peaks) >= 2:
        # 複数のピークがある場合、2つの主要なピーク間の谷を探す
        peak_values = [hist[p] for p in peaks]
        sorted_peaks = [p for _, p in sorted(zip(peak_values, peaks), reverse=True)]
        
        if len(sorted_peaks) >= 2:
            peak1, peak2 = sorted_peaks[0], sorted_peaks[1]
            min_index = np.argmin(hist[min(peak1, peak2):max(peak1, peak2)]) + min(peak1, peak2)
            optimal_threshold = bin_centers[min_index]
            return optimal_threshold, "複数のピーク間の谷"
    
    # 単一ピークまたは複雑な分布の場合、中央値を基準に判断
    median_similarity = np.median(all_similarities)
    mean_similarity = np.mean(all_similarities)
    
    # 中央値と平均値の間を閾値とする
    optimal_threshold = (median_similarity + mean_similarity) / 2
    return optimal_threshold, "分布の中央付近"

def visualize_similarity_matrix(face_ids, labels, similarity_matrix, output_dir, show_images=False):
    """コサイン類似度行列を可視化"""
    plt.figure(figsize=(12, 10))
    
    # ヒートマップとして類似度行列を表示
    cmap = plt.cm.coolwarm
    im = plt.imshow(similarity_matrix, cmap=cmap, vmin=0, vmax=1, interpolation='nearest')
    plt.colorbar(im, label='Cosine Similarity')
    
    # 座標軸にラベル（face_id + person情報）を表示
    plt.xticks(range(len(labels)), labels, rotation=90, fontsize=8)
    plt.yticks(range(len(labels)), labels, fontsize=8)
    
    # グリッド線を追加
    plt.grid(False)
    
    # タイトルを追加
    plt.title('Face Similarity Matrix\n(Each cell shows similarity between row and column faces)', fontsize=12)
    plt.figtext(0.5, 0.01, 'Red=High Similarity (Likely Same Person), Blue=Low Similarity (Likely Different Person)', 
                ha='center', fontsize=10, bbox=dict(facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'similarity_matrix.png'), dpi=150)
    
    if show_images:
        plt.show()
    else:
        plt.close()
    
    print(f"類似度行列を {os.path.join(output_dir, 'similarity_matrix.png')} に保存しました")

def visualize_similar_pairs(similar_pairs, faces_dir, output_dir, top_n=10, show_images=False):
    """類似度が高いペアを可視化"""
    n = min(top_n, len(similar_pairs))
    
    if n == 0:
        print("類似度が閾値を超えるペアがありません")
        return
    
    # personフォルダのマッピングを取得
    face_to_person = get_face_person_mapping(output_dir)
    
    for i in range(n):
        face_id1, face_id2, similarity = similar_pairs[i]
        
        img1_path = os.path.join(faces_dir, f"{face_id1}.jpg")
        img2_path = os.path.join(faces_dir, f"{face_id2}.jpg")
        
        if not os.path.exists(img1_path) or not os.path.exists(img2_path):
            continue
        
        img1 = imread(img1_path)
        img2 = imread(img2_path)
        
        # personフォルダの情報を取得
        person1 = face_to_person.get(face_id1, "Unclassified")
        person2 = face_to_person.get(face_id2, "Unclassified")
        
        # 同じpersonフォルダに属しているか判断
        same_person = person1 == person2 and person1 != "Unclassified"
        
        # プロット設定
        fig = plt.figure(figsize=(10, 5))
        
        # 2つの画像を並べて表示
        ax1 = plt.subplot(1, 2, 1)
        ax1.imshow(img1)
        ax1.set_title(f"ID: {face_id1}\nFolder: {person1}", fontsize=10)
        ax1.axis('off')
        
        ax2 = plt.subplot(1, 2, 2)
        ax2.imshow(img2)
        ax2.set_title(f"ID: {face_id2}\nFolder: {person2}", fontsize=10)
        ax2.axis('off')
        
        # 同じpersonフォルダに属しているかを色で表示
        if same_person:
            border_color = 'green'
            status = "Same Cluster"
        elif person1 != "Unclassified" and person2 != "Unclassified":
            border_color = 'red'
            status = "Different Clusters"
        else:
            border_color = 'gray'
            status = "Unclassified Present"
        
        # 境界線を追加
        for ax in [ax1, ax2]:
            for spine in ax.spines.values():
                spine.set_edgecolor(border_color)
                spine.set_linewidth(5)
                spine.set_visible(True)
        
        plt.suptitle(f"Similarity: {similarity:.4f} - {status}", fontsize=12, color=border_color)
        
        # 問題の可能性を警告
        if same_person and similarity < 0.7:
            warning_text = "Warning: Same cluster but low similarity (possible false positive)"
            plt.figtext(0.5, 0.01, warning_text, 
                      ha='center', fontsize=10, color='orange', 
                      bbox=dict(facecolor='yellow', alpha=0.3))
        elif not same_person and similarity > 0.7 and person1 != "Unclassified" and person2 != "Unclassified":
            warning_text = "Warning: Different clusters but high similarity (possible false negative)"
            plt.figtext(0.5, 0.01, warning_text, 
                      ha='center', fontsize=10, color='purple', 
                      bbox=dict(facecolor='pink', alpha=0.3))
        
        plt.tight_layout(rect=[0, 0.03, 1, 0.97])
        
        # ペア画像を保存
        output_file = os.path.join(output_dir, f"pair_{face_id1}_{face_id2}.png")
        plt.savefig(output_file, dpi=150)
        
        if show_images:
            plt.show()
        else:
            plt.close()
    
    print(f"上位{n}ペアの可視化画像を {output_dir} に保存しました")

def analyze_similarities(embeddings, threshold=0.7, output_dir="compared_faces", faces_dir="aligned_faces", show_images=False):
    """すべての顔間の類似度を分析"""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 類似度行列を計算
    face_ids, labels, similarity_matrix = calculate_similarity_matrix(embeddings, output_dir)
    
    # 類似度行列を可視化
    visualize_similarity_matrix(face_ids, labels, similarity_matrix, output_dir, show_images)
    
    # 類似度が閾値以上のペアを見つける
    similar_pairs = find_similar_pairs(face_ids, similarity_matrix, threshold)
    
    # 類似ペアの数を表示
    print(f"\n類似度が {threshold} 以上のペア: {len(similar_pairs)}個")
    
    # 上位10ペアを表示
    if similar_pairs:
        # personフォルダのマッピングを取得
        face_to_person = get_face_person_mapping(output_dir)
        
        print("\n上位類似ペア:")
        for i, (face_id1, face_id2, similarity) in enumerate(similar_pairs[:10]):
            person1 = face_to_person.get(face_id1, "未分類")
            person2 = face_to_person.get(face_id2, "未分類")
            
            print(f"{i+1}. {face_id1} ({person1}) と {face_id2} ({person2}): 類似度 = {similarity:.4f}")
        
        # 類似ペアを可視化
        visualize_similar_pairs(similar_pairs, faces_dir, output_dir, 10, show_images)
    
    # 類似度のヒストグラムとKDE（カーネル密度推定）を作成
    all_similarities = []
    same_person_similarities = []
    diff_person_similarities = []
    
    # personフォルダのマッピングを取得
    face_to_person = get_face_person_mapping(output_dir)
    
    for i in range(len(face_ids)):
        for j in range(i+1, len(face_ids)):
            sim = similarity_matrix[i, j]
            all_similarities.append(sim)
            
            # 同じpersonフォルダに属しているかどうかで分類
            face_id1, face_id2 = face_ids[i], face_ids[j]
            person1 = face_to_person.get(face_id1)
            person2 = face_to_person.get(face_id2)
            
            if person1 is not None and person2 is not None:
                if person1 == person2:
                    same_person_similarities.append(sim)
                else:
                    diff_person_similarities.append(sim)
    
    # 最適な閾値を推定
    optimal_threshold, method = suggest_optimal_threshold(all_similarities)
    
    # ヒストグラム + KDEプロット
    plt.figure(figsize=(10, 8))
    
    # メインのヒストグラム
    n, bins, patches = plt.hist(all_similarities, bins=50, alpha=0.6, color='gray', density=True, label='All Pairs')
    
    # 閾値線
    plt.axvline(x=threshold, color='r', linestyle='--', linewidth=2, label=f'Current Threshold: {threshold:.2f}')
    plt.axvline(x=optimal_threshold, color='g', linestyle='-', linewidth=2, label=f'Suggested Threshold: {optimal_threshold:.2f}')
    
    # 同一人物と異なる人物の類似度を別々に表示（データがある場合）
    if same_person_similarities and diff_person_similarities:
        # カーネル密度推定
        kde_all = stats.gaussian_kde(all_similarities)
        kde_same = stats.gaussian_kde(same_person_similarities)
        kde_diff = stats.gaussian_kde(diff_person_similarities)
        
        x = np.linspace(0, 1, 1000)
        plt.plot(x, kde_all(x), 'k-', linewidth=2, label='All Distribution')
        plt.plot(x, kde_same(x), 'b-', linewidth=2, label='Same Person Pairs')
        plt.plot(x, kde_diff(x), 'r-', linewidth=2, label='Different Person Pairs')
        
        # 小さなヒストグラムも表示
        ax2 = plt.gca().twinx()
        ax2.hist(same_person_similarities, bins=20, alpha=0.3, color='blue', density=True)
        ax2.hist(diff_person_similarities, bins=20, alpha=0.3, color='red', density=True)
        ax2.set_yticks([])
    
    # グラフの装飾
    plt.xlabel('Cosine Similarity', fontsize=12)
    plt.ylabel('Frequency (Density)', fontsize=12)
    
    plt.title('Face Similarity Distribution\nWith Suggested Threshold', fontsize=14)
    interpretation = (
        "Threshold Interpretation:\n"
        f"・Suggested: {optimal_threshold:.2f} based on distribution\n"
        "・Above threshold: Same person\n"
        "・Below threshold: Different person\n\n"
        "Lower threshold:\n"
        "→ Relaxes same-person criteria\n"
        "→ Different people may be grouped together\n\n"
        "Higher threshold:\n"
        "→ Stricter same-person criteria\n"
        "→ Same person may be in different clusters"
    )
    
    plt.xlim(0, 1)
    plt.grid(True, alpha=0.3)
    
    # 解釈のための注釈
    plt.figtext(0.15, 0.02, interpretation, fontsize=10, 
                bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.5'))
    
    # 凡例の配置
    plt.legend(loc='upper right', fontsize=10)
    
    plt.tight_layout(rect=[0, 0.1, 1, 0.97])
    plt.savefig(os.path.join(output_dir, 'similarity_histogram.png'), dpi=150)
    
    if show_images:
        plt.show()
    else:
        plt.close()
    
    print(f"類似度ヒストグラムを {os.path.join(output_dir, 'similarity_histogram.png')} に保存しました")
    print(f"\n推奨閾値: {optimal_threshold:.4f} ({method}に基づく)")
    print(f"現在の閾値: {threshold:.4f}")
    
    if abs(optimal_threshold - threshold) > 0.05:
        if optimal_threshold > threshold:
            print(f"\n提案: 閾値を {optimal_threshold:.2f} に上げることを検討してください。")
            print(f"これにより、異なる人物が同じクラスタに入る可能性が減ります。")
        else:
            print(f"\n提案: 閾値を {optimal_threshold:.2f} に下げることを検討してください。")
            print(f"これにより、同じ人物が別々のクラスタに分かれる可能性が減ります。")
    
    return similar_pairs

def find_misclassifications(output_dir, embeddings):
    """現在のクラスタリングの問題点を検出する"""
    print("\n=== クラスタリングの問題を検出中... ===")
    
    if not embeddings:
        print("埋め込みベクトルが見つかりません。")
        return
    
    # personフォルダのマッピングを取得
    face_to_person = get_face_person_mapping(output_dir)
    if not face_to_person:
        print("クラスタリングされた顔が見つかりません。")
        return
    
    # 類似度行列を計算
    face_ids = list(embeddings.keys())
    n = len(face_ids)
    
    problems = {
        "false_positives": [],  # 同じクラスタだが類似度が低い（偽陽性）
        "false_negatives": []   # 異なるクラスタだが類似度が高い（偽陰性）
    }
    
    # 類似度の低いしきい値と高いしきい値
    low_threshold = 0.5
    high_threshold = 0.7
    
    for i in range(n):
        for j in range(i+1, n):
            face_id1, face_id2 = face_ids[i], face_ids[j]
            
            # 両方の顔がpersonフォルダに存在するかチェック
            if face_id1 not in face_to_person or face_id2 not in face_to_person:
                continue
            
            # 同じクラスタに属しているか
            same_cluster = face_to_person[face_id1] == face_to_person[face_id2]
            
            # 類似度を計算
            similarity = cosine_similarity(
                embeddings[face_id1].reshape(1, -1),
                embeddings[face_id2].reshape(1, -1)
            )[0][0]
            
            # 問題のあるケースを検出
            if same_cluster and similarity < low_threshold:
                problems["false_positives"].append((face_id1, face_id2, similarity))
            elif not same_cluster and similarity > high_threshold:
                problems["false_negatives"].append((face_id1, face_id2, similarity))
    
    # 問題のあるケースをソート
    problems["false_positives"].sort(key=lambda x: x[2])  # 類似度が低い順
    problems["false_negatives"].sort(key=lambda x: x[2], reverse=True)  # 類似度が高い順
    
    # 結果を表示
    if problems["false_positives"]:
        print(f"\n偽陽性（同じクラスタだが類似度が低い）: {len(problems['false_positives'])}件")
        for i, (face_id1, face_id2, similarity) in enumerate(problems["false_positives"][:5]):
            print(f"{i+1}. {face_id1} と {face_id2}: 類似度 = {similarity:.4f}, クラスタ = {face_to_person[face_id1]}")
    else:
        print("偽陽性は検出されませんでした。")
    
    if problems["false_negatives"]:
        print(f"\n偽陰性（異なるクラスタだが類似度が高い）: {len(problems['false_negatives'])}件")
        for i, (face_id1, face_id2, similarity) in enumerate(problems["false_negatives"][:5]):
            print(f"{i+1}. {face_id1}({face_to_person[face_id1]}) と {face_id2}({face_to_person[face_id2]}): 類似度 = {similarity:.4f}")
    else:
        print("偽陰性は検出されませんでした。")
    
    return problems

def merge_clusters(embeddings, output_dir, similarity_threshold=0.7, debug=False):
    """類似したクラスタを統合する"""
    if debug:
        print("\n=== デバッグモード: 統合対象のクラスタを確認中... ===")
    
    # personフォルダのマッピングを取得
    face_to_person = get_face_person_mapping(output_dir)
    if not face_to_person:
        print("統合対象のクラスタが見つかりません。")
        return
    
    # ユニークなクラスタを取得
    unique_clusters = set(face_to_person.values())
    
    # クラスタの代表ベクトル（各クラスタの平均埋め込みベクトル）を計算
    cluster_embeddings = {}
    for cluster in unique_clusters:
        # このクラスタに属する顔IDを取得
        cluster_faces = [face_id for face_id, person in face_to_person.items() if person == cluster]
        if not cluster_faces:
            continue
        
        # クラスタの平均埋め込みベクトルを計算
        cluster_embedding = np.zeros_like(embeddings[cluster_faces[0]])
        for face_id in cluster_faces:
            if face_id in embeddings:
                cluster_embedding += embeddings[face_id]
        cluster_embedding /= len(cluster_faces)
        cluster_embeddings[cluster] = cluster_embedding
    
    # 類似したクラスタを見つける
    clusters_to_merge = []
    
    for cluster1 in unique_clusters:
        if cluster1 not in cluster_embeddings:
            continue
            
        for cluster2 in unique_clusters:
            if cluster1 == cluster2 or cluster2 not in cluster_embeddings:
                continue
                
            # クラスタ代表ベクトル間の類似度を計算
            similarity = cosine_similarity(
                cluster_embeddings[cluster1].reshape(1, -1),
                cluster_embeddings[cluster2].reshape(1, -1)
            )[0][0]
            
            if similarity >= similarity_threshold:
                clusters_to_merge.append((cluster1, cluster2, similarity))
    
    # 類似度の高い順にソート
    clusters_to_merge.sort(key=lambda x: x[2], reverse=True)
    
    if not clusters_to_merge:
        print("統合対象のクラスタは見つかりませんでした。")
        return
    
    print(f"\n統合候補のクラスタペアが {len(clusters_to_merge)} 組見つかりました。")
    
    # 上位のクラスタペアを表示
    for i, (cluster1, cluster2, similarity) in enumerate(clusters_to_merge[:5]):
        print(f"{i+1}. {cluster1} と {cluster2}: 類似度 = {similarity:.4f}")
    
    # ユーザーに確認（デバッグモードでなければ）
    if not debug:
        answer = input("\nこれらの類似クラスタを統合しますか？ (y/n): ")
        if answer.lower() != 'y':
            print("クラスタ統合をキャンセルしました。")
            return
    
    # 統合を実行
    merged_count = 0
    for cluster1, cluster2, _ in clusters_to_merge:
        # どちらかのクラスタが存在しなければスキップ（既に統合済み）
        if not os.path.exists(os.path.join(output_dir, cluster1)) or not os.path.exists(os.path.join(output_dir, cluster2)):
            continue
        
        # 番号の小さい方をターゲットとして使用
        cluster1_num = int(cluster1.split('_')[1])
        cluster2_num = int(cluster2.split('_')[1])
        
        if cluster1_num < cluster2_num:
            source_cluster = cluster2
            target_cluster = cluster1
        else:
            source_cluster = cluster1
            target_cluster = cluster2
        
        # 元クラスタのすべての顔を対象クラスタに移動
        source_path = os.path.join(output_dir, source_cluster)
        target_path = os.path.join(output_dir, target_cluster)
        
        for file in os.listdir(source_path):
            if file.endswith('.jpg'):
                src_file = os.path.join(source_path, file)
                dst_file = os.path.join(target_path, file)
                
                if not os.path.exists(dst_file):
                    shutil.copy2(src_file, dst_file)
                    print(f"{file} を {source_cluster} から {target_cluster} に移動しました")
        
        # 元クラスタフォルダを削除
        shutil.rmtree(source_path)
        print(f"{target_cluster} に統合した後、{source_cluster} を削除しました")
        merged_count += 1
    
    # 履歴ファイルを更新して変更を反映
    history_file = os.path.join(output_dir, "clustering_history.txt")
    sync_history_with_folders(output_dir, history_file)
    
    print(f"\n{merged_count} 組のクラスタを統合しました。")
    return merged_count

def cluster_faces(embeddings, similarity_threshold=0.7, output_dir="compared_faces", faces_dir="aligned_faces", debug=False, visualize=False):
    """顔をクラスタリングする"""
    # デバッグモードの場合は分析を実行
    if debug:
        print("\n=== デバッグモード: 類似度分析 ===")
        similar_pairs = analyze_similarities(embeddings, similarity_threshold, output_dir, faces_dir, visualize)
        print("\n=== クラスタリング処理を開始します ===\n")
    
    # 履歴ファイル
    history_file = os.path.join(output_dir, "clustering_history.txt")
    
    # 処理済みの顔IDを取得
    processed_faces = get_processed_faces(output_dir, history_file)
    
    # 未処理の顔をフィルタリング
    unprocessed_faces = [face_id for face_id in embeddings.keys() if face_id not in processed_faces]
    
    if not unprocessed_faces:
        print("すべての顔がすでに処理済みです。")
        
        # クラスタの統合を確認
        if debug:
            merge_clusters(embeddings, output_dir, similarity_threshold, debug)
            
        return
    
    print(f"{len(unprocessed_faces)}個の未処理の顔を分類します...")
    
    # クラスタリング結果を保存する辞書
    # {クラスタID: [face_id1, face_id2, ...]}
    clusters = {}
    
    # 既にクラスタに割り当てられた顔ID
    clustered_faces = set()
    
    # 次のクラスタID
    next_cluster_id = 0
    
    # 出力ディレクトリを確認
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"出力ディレクトリを作成しました: {output_dir}")
    
    # 既存のクラスタからIDを取得
    existing_clusters = [d for d in os.listdir(output_dir) if os.path.isdir(os.path.join(output_dir, d)) and d.startswith("person_")]
    if existing_clusters:
        max_id = max([int(d.split("_")[1]) for d in existing_clusters])
        next_cluster_id = max_id + 1
        print(f"既存のクラスタが見つかりました。次のクラスタIDは{next_cluster_id}から始まります。")
    
    # 履歴ファイルの確認
    if not os.path.exists(history_file):
        with open(history_file, 'w') as f:
            pass
        print(f"処理履歴ファイルを作成しました: {history_file}")
    
    # 未処理の顔ペアの類似度を計算
    similarity_data = []
    for i, face_id1 in enumerate(unprocessed_faces):
        # 既にクラスタに割り当てられていればスキップ
        if face_id1 in clustered_faces:
            continue
        
        # 他の未処理顔との類似度を計算
        for face_id2 in unprocessed_faces[i+1:]:
            if face_id2 in clustered_faces:
                continue
            
            similarity = cosine_similarity(
                embeddings[face_id1].reshape(1, -1),
                embeddings[face_id2].reshape(1, -1)
            )[0][0]
            
            similarity_data.append((face_id1, face_id2, similarity))
    
    # 類似度の高い順にソート
    similarity_data.sort(key=lambda x: x[2], reverse=True)
    
    # 高い類似度のペアからクラスタを形成
    while similarity_data:
        face_id1, face_id2, similarity = similarity_data.pop(0)
        
        # 両方の顔がまだ未割当か確認
        if face_id1 in clustered_faces or face_id2 in clustered_faces:
            continue
        
        # 閾値チェック
        if similarity >= similarity_threshold:
            # 新しいクラスタを作成
            new_cluster_dir = f"person_{next_cluster_id}"
            new_cluster_path = os.path.join(output_dir, new_cluster_dir)
            
            if not os.path.exists(new_cluster_path):
                os.makedirs(new_cluster_path)
            
            # 顔画像をコピー
            for face_id in [face_id1, face_id2]:
                src_path = os.path.join(faces_dir, f"{face_id}.jpg")
                dst_path = os.path.join(new_cluster_path, f"{face_id}.jpg")
                
                if os.path.exists(src_path) and not os.path.exists(dst_path):
                    shutil.copy2(src_path, dst_path)
                    
                    # 処理履歴を更新
                    update_history(face_id, history_file)
                    clustered_faces.add(face_id)
            
            print(f"新しいクラスタ'{new_cluster_dir}'を作成し、'{face_id1}'と'{face_id2}'を追加しました (類似度: {similarity:.4f})")
            
            next_cluster_id += 1

# 残りの未処理の顔を処理
    for face_id in unprocessed_faces:
        if face_id in clustered_faces:
            continue  # 既にクラスタに割り当て済みならスキップ
        
        if debug:
            print(f"\n--- 顔ID '{face_id}' の処理 ---")
        
        # まず既存クラスタとの比較
        assigned_to_existing = False
        
        # 既存クラスタを確認
        for cluster_dir in existing_clusters:
            cluster_path = os.path.join(output_dir, cluster_dir)
            
            # このクラスタ内の顔IDを取得
            cluster_face_ids = [os.path.splitext(f)[0] for f in os.listdir(cluster_path) if f.endswith('.jpg')]
            
            if not cluster_face_ids:
                continue
            
            # このクラスタの顔の埋め込みベクトルを読み込み
            cluster_embeddings = {}
            for cluster_face_id in cluster_face_ids:
                if cluster_face_id in embeddings:
                    cluster_embeddings[cluster_face_id] = embeddings[cluster_face_id]
            
            # クラスタ内の顔と比較
            max_similarity = 0.0
            most_similar_face = None
            
            for cluster_face_id, cluster_embedding in cluster_embeddings.items():
                similarity = cosine_similarity(
                    embeddings[face_id].reshape(1, -1), 
                    cluster_embedding.reshape(1, -1)
                )[0][0]
                
                if debug:
                    print(f"  クラスタ '{cluster_dir}' の顔ID '{cluster_face_id}' との類似度: {similarity:.4f}")
                
                if similarity > max_similarity:
                    max_similarity = similarity
                    most_similar_face = cluster_face_id
            
            # 最も類似度が高い顔が閾値以上なら、このクラスタに追加
            if max_similarity >= similarity_threshold:
                # 顔をこのクラスタに追加
                src_path = os.path.join(faces_dir, f"{face_id}.jpg")
                dst_path = os.path.join(cluster_path, f"{face_id}.jpg")
                
                if os.path.exists(src_path) and not os.path.exists(dst_path):
                    shutil.copy2(src_path, dst_path)
                    print(f"'{face_id}'をクラスタ'{cluster_dir}'に追加しました (類似度: {max_similarity:.4f}, 最も似ている顔: {most_similar_face})")
                    
                    # 処理履歴を更新
                    update_history(face_id, history_file)
                    clustered_faces.add(face_id)
                    assigned_to_existing = True
                    break
            elif debug:
                print(f"  クラスタ '{cluster_dir}' との最大類似度 {max_similarity:.4f} は閾値 {similarity_threshold} 未満です")
# 既存クラスタに割り当てられなかった場合、新しいクラスタを作成
        if not assigned_to_existing:
            new_cluster_dir = f"person_{next_cluster_id}"
            new_cluster_path = os.path.join(output_dir, new_cluster_dir)
            
            if not os.path.exists(new_cluster_path):
                os.makedirs(new_cluster_path)
            
            # 顔画像をコピー
            src_path = os.path.join(faces_dir, f"{face_id}.jpg")
            dst_path = os.path.join(new_cluster_path, f"{face_id}.jpg")
            
            if os.path.exists(src_path):
                shutil.copy2(src_path, dst_path)
                print(f"新しいクラスタ'{new_cluster_dir}'を作成し、'{face_id}'を追加しました")
                
                # リスト更新
                clusters[new_cluster_dir] = [face_id]
                update_history(face_id, history_file)
                clustered_faces.add(face_id)
                existing_clusters.append(new_cluster_dir)
                next_cluster_id += 1
            
            if debug:
                print(f"  どのクラスタにも類似した顔がなかったため、新しいクラスタ '{new_cluster_dir}' を作成しました")
    
    # 新しく分類されたクラスタを処理
    for cluster_id, face_ids in clusters.items():
        if len(face_ids) > 0:
            print(f"クラスタ'{cluster_id}'に{len(face_ids)}個の顔を分類しました")
    
    # 未処理の顔を数える
    remaining = len(unprocessed_faces) - len(clustered_faces)
    if remaining > 0:
        print(f"警告: {remaining}個の顔が分類されませんでした。閾値を下げて再試行してください。")
    else:
        print("すべての未処理顔を正常に分類しました！")
        
    # クラスタの統合チェック
    if debug:
        merge_clusters(embeddings, output_dir, similarity_threshold, debug)

def main():
        # コマンドライン引数を解析
    args = parse_arguments()
    
    model_type = args.model
    input_dir = args.input_dir
    output_dir = args.output_dir
    
    # モデルタイプに応じた閾値を設定
    if args.threshold is None:  # ユーザーが指定しない場合
        if model_type in ["r100magface", "r100adaface"]:
            threshold = 0.85  # ResNet100系は高めの閾値
        else:
            threshold = 0.7   # EfficientNetV2Sはデフォルト閾値
    else:
        threshold = args.threshold  # ユーザー指定の閾値を使用
    
    print(f"モデル '{model_type}' の閾値: {threshold}")
    
    parser = argparse.ArgumentParser(description='顔埋め込みベクトルに基づいてクラスタリング')
    parser.add_argument('model', type=str,
                        choices=MODEL_TYPES,
                        help=f'使用するモデルを指定します。選択肢: {", ".join(MODEL_TYPES)}')
    parser.add_argument('--threshold', type=float, default=0.7, 
                        help='顔を同一人物と判定するコサイン類似度の閾値 (0.0〜1.0)')
    parser.add_argument('--input_dir', type=str, default='embedded_faces',
                        help='埋め込みベクトルが格納されているディレクトリ')
    parser.add_argument('--faces_dir', type=str, default='aligned_faces',
                        help='顔画像が格納されているディレクトリ')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='クラスタリング結果を保存するディレクトリ（デフォルトはcompared_faces_モデル名）')
    parser.add_argument('--debug', action='store_true',
                        help='デバッグモードを有効にする（詳細な類似度情報を表示）')
    parser.add_argument('--analyze', action='store_true',
                        help='クラスタリングを行わず、類似度の分析のみを実行')
    parser.add_argument('--visualize', action='store_true',
                        help='類似度の分析結果をグラフで表示（analyze または debug モードで有効）')
    parser.add_argument('--check', action='store_true',
                        help='既存のクラスタリングの問題点をチェック')
    parser.add_argument('--merge', action='store_true',
                        help='類似クラスタを統合')
    
    args = parser.parse_args()
    
    # モデルタイプを取得
    model_type = args.model
    
    # 出力ディレクトリが指定されていない場合はデフォルト値を設定
    if args.output_dir is None:
        args.output_dir = f"compared_faces_{model_type}"
    
    # 埋め込みベクトルを読み込む
    print(f"モデル '{model_type}' の埋め込みベクトルを読み込んでいます...")
    embeddings = load_embeddings(args.input_dir, model_type)
    
    if not embeddings:
        print(f"エラー: ディレクトリ {args.input_dir} に {model_type} モデルの埋め込みベクトルが見つかりません。")
        return
    
    print(f"{len(embeddings)}個の埋め込みベクトルを読み込みました。")
    
    # クラスタリングの問題点をチェック
    if args.check:
        find_misclassifications(args.output_dir, embeddings)
        return
    
    # 統合モード
    if args.merge:
        merge_clusters(embeddings, args.output_dir, args.threshold, args.debug)
        return
    
    # 分析モード
    if args.analyze:
        print("類似度分析を実行しています...")
        analyze_similarities(embeddings, args.threshold, args.output_dir, args.faces_dir, args.visualize)
        return
    
    # クラスタリングモード
    cluster_faces(embeddings, args.threshold, args.output_dir, args.faces_dir, args.debug, args.visualize)

if __name__ == "__main__":
    main()