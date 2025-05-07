# face_recognition_v2.py - マルチモデル対応版
import os
import sys
import numpy as np
from skimage.io import imread
import glob
import subprocess
import importlib
import argparse

# Keras_insightfaceへのパスを追加
sys.path.append('./Keras_insightface')

# 必要なモジュールをインポート
import tensorflow as tf
from tensorflow import keras

# 色空間変換用の関数をインポート
import convert_colors

# モデルタイプの定義
# MODEL_TYPES辞書にr100adafaceを追加
MODEL_TYPES = {
    "effv2s": {
        "name": "EfficientNetV2S with MagFace",
        "file": "efficientnetv2_s_magface.h5",
        "needs_keras_cv": True,
        "input_size": (112, 112),
        "preprocess_func": convert_colors.convert_image_to_model_input
    },
    "r100magface": {
        "name": "ResNet100 with MagFace",
        "file": "r100_magface.h5",
        "needs_keras_cv": False,
        "input_size": (112, 112),
        "preprocess_func": convert_colors.convert_image_to_model_input
    },
    "r100arcface": {
        "name": "ResNet100 with ArcFace",
        "file": "r100_arcface.h5",
        "needs_keras_cv": False,
        "input_size": (112, 112),
        "preprocess_func": convert_colors.convert_image_to_model_input
    },
    "r100adaface": {  # 新しく追加
        "name": "ResNet100 with AdaFace",
        "file": "r100_adaface.h5",
        "needs_keras_cv": False,
        "input_size": (112, 112),
        "preprocess_func": convert_colors.convert_image_to_model_input
    }
}

def parse_arguments():
    """コマンドライン引数を解析する関数"""
    parser = argparse.ArgumentParser(description='顔画像から埋め込みベクトルを抽出する')
    parser.add_argument('model', type=str,
                        choices=list(MODEL_TYPES.keys()),
                        help=f'使用するモデルを指定します。選択肢: {", ".join(MODEL_TYPES.keys())}')
    parser.add_argument('--input_dir', type=str, default='aligned_faces',
                        help='顔画像が格納されているディレクトリ')
    parser.add_argument('--output_dir', type=str, default='embedded_faces',
                        help='埋め込みベクトルを保存するディレクトリ')
    parser.add_argument('--verbose', action='store_true',
                        help='詳細な出力を表示します')
    
    return parser.parse_args()

def check_and_install_packages(model_type):
    """必要なパッケージをインストールする関数"""
    required_packages = []
    
    # モデルタイプに応じて必要なパッケージを追加
    if MODEL_TYPES[model_type]["needs_keras_cv"]:
        required_packages.append('keras-cv-attention-models')
    
    for package in required_packages:
        try:
            importlib.import_module(package.replace('-', '_'))
            print(f"{package}は既にインストールされています")
        except ImportError:
            print(f"{package}をインストールします...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])

def build_efficientnetv2s_model(model_path):
    """EfficientNetV2SとMagFaceを組み合わせたモデルを構築する関数"""
    from keras_cv_attention_models import efficientnet
    
    # Keras_insightfaceのmodelsモジュールをインポート
    try:
        from Keras_insightface import models
    except ImportError:
        import models
    
    # EfficientNetV2-Sモデルを構築
    basic_model = efficientnet.EfficientNetV2S(
        input_shape=(112, 112, 3), 
        num_classes=0, 
        pretrained="imagenet",
        first_strides=1  # 入力サイズが小さいため
    )
    
    # 特徴抽出層を追加
    model = models.buildin_models(
        basic_model, 
        dropout=0.2,
        emb_shape=512,  # 512次元の埋め込みベクトル
        output_layer='F',
        bn_momentum=0.9,
        bn_epsilon=1e-5,
        add_pointwise_conv=True,
        pointwise_conv_act="swish",
        scale=True,
        use_bias=False
    )
    
    # L2正則化を追加
    model = models.add_l2_regularizer_2_model(
        model, 
        weight_decay=5e-4, 
        apply_to_batch_normal=False
    )
    
    # モデルのサマリーを表示
    model.summary()
    
    # モデルを保存
    model.save(model_path)
    print(f"モデルを保存しました: {model_path}")
    
    return model

def build_resnet100_model(model_path, model_type):
    """ResNet100とMagFace/ArcFaceを組み合わせたモデルを構築する関数"""
    try:
        from Keras_insightface import models
        
        # モデルタイプに応じたパラメータを設定
        if model_type == "r100magface":
            # MagFaceの設定
            backbone = "r100"
            loss_type = "MagFace"
        elif model_type == "r100arcface":
            # ArcFaceの設定
            backbone = "r100" 
            loss_type = "ArcFace"
        elif model_type == "r100adaface":
            # AdaFaceの設定（必要に応じて追加）
            backbone = "r100"
            loss_type = "AdaFace"
        
        # r100バックボーンのモデルを構築
        model = models.buildin_models(
            backbone,  # "r100"を直接文字列として渡す
            dropout=0.2,
            emb_shape=512,
            output_layer='F',  # 特徴ベクトルを出力するレイヤー
            bn_momentum=0.9,
            bn_epsilon=1e-5,
            add_pointwise_conv=False,
            scale=True
        )
        
        # L2正則化を追加
        model = models.add_l2_regularizer_2_model(
            model, 
            weight_decay=5e-4, 
            apply_to_batch_normal=False
        )
        
        # モデルのサマリーを表示
        model.summary()
        
        # モデルを保存
        model.save(model_path)
        print(f"モデルを保存しました: {model_path}")
        
        return model
    except Exception as e:
        print(f"モデル構築中にエラーが発生しました: {e}")
        return None

def build_model(model_type, model_dir):
    """モデルタイプに応じたモデルを構築/ロードする関数"""
    model_info = MODEL_TYPES[model_type]
    model_path = os.path.join(model_dir, model_info["file"])
    
    print(f"デバッグ: モデル情報: {model_info}")
    print(f"デバッグ: モデルパス: {model_path}")
    
    # モデルがすでに存在する場合はロードして返す
    if os.path.exists(model_path):
        print(f"デバッグ: モデルファイルが存在します")
        print(f"既存のモデル {model_info['name']} をロードします: {model_path}")
        try:
            model = keras.models.load_model(model_path)
            print("モデルのロードが完了しました！")
            return model
        except Exception as e:
            print(f"デバッグ: モデルロードエラー詳細: {str(e)}")
            print(f"モデルのロード中にエラーが発生しました: {str(e)}")
            print("新しいモデルを構築します...")
    else:
        print(f"デバッグ: モデルファイルが見つかりません: {model_path}")
        print(f"モデル {model_info['name']} が見つかりません。新しいモデルを構築します...")
    
    # モデルタイプに応じてモデルを構築
    if model_type == "effv2s":
        print("デバッグ: effv2sモデルを構築します")
        return build_efficientnetv2s_model(model_path)
    elif model_type in ["r100magface", "r100arcface", "r100adaface"]:
        print(f"デバッグ: {model_type}モデルを構築します")
        model = build_resnet100_model(model_path, model_type)
        print(f"デバッグ: {model_type}モデル構築結果: {model is not None}")
        return model
    else:
        raise ValueError(f"未対応のモデルタイプです: {model_type}")

def extract_embedding(model, image_path, model_type):
    """画像から512次元の埋め込みベクトルを抽出する関数"""
    try:
        print(f"デバッグ: 埋め込み抽出開始 - {image_path}")
        # モデル情報を取得
        model_info = MODEL_TYPES[model_type]
        
        # 画像を読み込む
        image = imread(image_path)
        print(f"デバッグ: 画像サイズ: {image.shape}")
        
        # 画像の形状を確認
        if image.shape[:2] != model_info["input_size"]:
            print(f"デバッグ: リサイズが必要 {image.shape[:2]} -> {model_info['input_size']}")
            # リサイズ処理...
        
        # モデルに応じた前処理
        preprocessed_image = model_info["preprocess_func"](image)
        
        # バッチ次元を追加
        input_tensor = np.expand_dims(preprocessed_image, axis=0)
        
        print(f"デバッグ: モデル予測実行直前 - モデルタイプ: {model_type}")
        print(f"デバッグ: モデル: {type(model)}")
        
        # 埋め込みベクトルを抽出
        embedding = model.predict(input_tensor, verbose=0)

        print(f"デバッグ: 埋め込みベクトル抽出成功 - 形状: {embedding.shape}")

        # モデルによって正規化の有無を切り替え
        if model_type in ["effv2s"]:
            # EfficientNet系は正規化が必要
            norm = np.linalg.norm(embedding)
            if norm < 1e-10:
                print(f"警告: 埋め込みベクトルのノルムが非常に小さい: {norm}")
                normalized = np.ones_like(embedding) / np.sqrt(embedding.size)
            else:
                normalized = embedding / norm
            return normalized[0]  # バッチ次元を削除
        else:
            # MagFace / AdaFaceなどは正規化しない
            return embedding[0]  # そのまま返す（バッチ次元だけ削除）
    
    except Exception as e:
        print(f"デバッグ: 埋め込みベクトル抽出エラー詳細: {str(e)}")
        print(f"埋め込みベクトル抽出中にエラーが発生しました: {str(e)}")
        return None

def main():
    # コマンドライン引数を解析
    args = parse_arguments()
    
    model_type = args.model
    input_dir = args.input_dir
    output_dir = args.output_dir
    verbose = args.verbose
    
    # モデルディレクトリの設定
    model_dir = "./models"
    
    # 履歴ファイルのパス
    history_file = os.path.join(output_dir, f"embedding_history_{model_type}.txt")
    
    print(f"顔認識プログラムを開始します... (モデル: {MODEL_TYPES[model_type]['name']})")
    
    # ディレクトリが存在しない場合は作成
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    
    # 処理済みの画像パスを読み込む
    processed_images = set()
    if os.path.exists(history_file):
        with open(history_file, 'r') as f:
            processed_images = set(line.strip() for line in f)
        print(f"{len(processed_images)}個の処理済み画像記録を読み込みました。")
    
    # 処理対象の顔画像ファイルを取得
    face_images = glob.glob(os.path.join(input_dir, "*.jpg"))
    
    if not face_images:
        print(f"ディレクトリ {input_dir} に顔画像が見つかりません。")
        return
    
    # 未処理の画像のみを抽出
    unprocessed_images = []
    for image_path in face_images:
        if image_path not in processed_images:
            unprocessed_images.append(image_path)
    
    print(f"合計 {len(face_images)} 枚の顔画像が見つかりました")
    print(f"うち {len(unprocessed_images)} 枚が未処理です")
    
    if not unprocessed_images:
        print("すべての画像は既に処理済みです！")
        return
    
    # 履歴ファイルを追記モードでオープン
    history_log = open(history_file, 'a')
    
    # 必要なパッケージをインストール
    check_and_install_packages(model_type)
    
    # モデルを構築またはロード
    model = build_model(model_type, model_dir)
    
    print(f"{len(unprocessed_images)}枚の顔画像を処理します...")
    
    # 各顔画像から埋め込みベクトルを抽出して保存
    for i, image_path in enumerate(unprocessed_images):
        filename = os.path.basename(image_path)
        name_without_ext = os.path.splitext(filename)[0]
        
        print(f"[{i+1}/{len(unprocessed_images)}] 処理中: {filename}")
        
        try:
            # 埋め込みベクトルを抽出
            embedding = extract_embedding(model, image_path, model_type)
            
            if embedding is not None:
                # モデルタイプを含むファイル名で保存（同じ顔でも異なるモデルの結果を区別）
                output_path = os.path.join(output_dir, f"{name_without_ext}_{model_type}.npy")
                np.save(output_path, embedding)
                
                if verbose:
                    print(f"埋め込みベクトルを保存しました: {output_path}")
                    print(f"埋め込みベクトルの次元: {embedding.shape}")
                else:
                    print(f"埋め込みベクトルを保存: {name_without_ext}")
                
                # 処理済みとして記録
                processed_images.add(image_path)
                history_log.write(image_path + '\n')
                history_log.flush()  # すぐに書き込みを反映
        except Exception as e:
            print(f"画像 {filename} の処理中にエラーが発生しました: {str(e)}")
    
    # 履歴ファイルを閉じる
    history_log.close()
    
    print("すべての顔画像の処理が完了しました！")
    print(f"埋め込みベクトルは {output_dir} に保存されました。")
    print(f"使用したモデル: {MODEL_TYPES[model_type]['name']}")

if __name__ == "__main__":
    main()