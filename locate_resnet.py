import sys
sys.path.append('./Keras_insightface')
import Keras_insightface
print(dir(Keras_insightface))  # 使用可能なモジュール一覧

# おそらく正しいインポート方法は以下のいずれかかもしれません
from Keras_insightface.backbones import resnet
print(dir(resnet))  # ResNet100がここにあるか確認