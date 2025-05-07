# 処理の流れ

## 1. 顔検出と切り取り（detect_faces.py）

1. **入力画像の準備**:
   - `sample_images`フォルダ内の画像ファイル（JPG、JPEG、PNG）を対象とする

2. **顔検出の実行**:
   - `Keras_insightface`ライブラリの`face_detector.py`モジュールを読み込む
   - `YoloV5FaceDetector`クラスをインスタンス化（引数なし）
   - 各画像に対して`detector.detect_in_image()`メソッドを呼び出し
   - このメソッドは画像から顔の境界ボックス、顔のランドマーク、信頼度スコアを返す
   - 同時に`face_align_landmarks()`メソッドを使用して顔を正規化（アライメント）

3. **顔画像の保存**:
   - 検出された顔を112×112ピクセルサイズに自動リサイズ（アライメント処理の一部として実行）
   - リサイズされた顔画像を`aligned_faces`フォルダに連番で保存（例：face_0.jpg, face_1.jpg）
   - `aligned_faces/processed_history.txt`に処理済み画像パスを記録

## 2. 特徴ベクトル抽出（face_recognition.py）

1. **モデルの準備**:
   - `keras-cv-attention-models`パッケージから`EfficientNetV2S`モデルをロード
   - 入力サイズを(112, 112, 3)に設定、ImageNet事前学習済み重みを使用
   - `Keras_insightface`の`models.buildin_models()`関数を使用して出力層を追加
   - 512次元の埋め込みベクトルを生成するよう設定
   - モデルを`models/efficientnetv2_s_magface.h5`として保存（初回のみ）

2. **画像の前処理**:
   - `aligned_faces`フォルダから顔画像を読み込み
   - サイズが112×112でない場合は自動的にリサイズ
   - `convert_colors.py`の`convert_image_to_model_input()`関数を使用して色空間変換
   - RGB値を-1～1の範囲に正規化（(pixel - 127.5) * 0.0078125の計算式で変換）
   - バッチ次元を追加してモデル入力形式に変換（[1, 112, 112, 3]）

3. **特徴ベクトル抽出**:
   - 前処理した画像をモデルに入力し、`model.predict()`で512次元ベクトルを取得
   - 出力ベクトルをL2正規化（ベクトルの長さが1になるよう正規化）

4. **結果の保存**:
   - 抽出された512次元ベクトルを`embedded_faces`フォルダにNumPy配列として保存
   - 元の顔画像と同じ名前で拡張子を`.npy`に変更（例：face_0.npy）
   - `embedded_faces/embedding_history.txt`に処理済み画像パスを記録

このパイプラインにより、元の画像から顔を検出し、標準化された特徴ベクトルへと変換する一連の処理が実現されます。各ステップで生成されるファイルは明確に管理され、重複処理を避けるための仕組みも実装されています。