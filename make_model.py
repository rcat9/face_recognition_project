import os
import sys

# Keras_insightface と backbones にパスを通す
sys.path.append("./Keras_insightface")
from Keras_insightface import models

model_dir = "./models"
os.makedirs(model_dir, exist_ok=True)

def make_and_save_model(name, save_as):
    print(f"{save_as}.h5 を構築中...")

    # ベースモデルを構築
    base_model = models.__init_model_from_name__(  # MagFace/AdaFace共通のベース
        name,
        input_shape=(112, 112, 3),
        weights="imagenet"
    )

    # MagFaceやAdaFace用の層を追加
    model = models.buildin_models(
        base_model,
        dropout=0.2,
        emb_shape=512,
        output_layer='E',
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

    # 保存
    model_path = os.path.join(model_dir, f"{save_as}.h5")
    model.save(model_path)
    print(f"✔ {save_as}.h5 を保存しました！")

# モデル構築（名前はベース名、保存名は自由に）
make_and_save_model("r100", "r100magface")
make_and_save_model("r100", "r100adaface")
make_and_save_model("r100", "r10012madaface")
