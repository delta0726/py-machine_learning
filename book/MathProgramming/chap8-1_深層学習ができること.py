# ******************************************************************************
# Title       : AI・データサイエンスのための数学プログラミング
# Chapter     : 6 感染症の影響を予測してみよう
# Theme       : 6-1 イメージで理解する感染症モデル
# Creat Date  : 2021/1/10
# Final Update:
# Page        : P255 - P259
# ******************************************************************************


# ＜概要＞
# - 最も単純な微分方程式の1つである｢ねずみ算｣について理解を深める
#   --- 何の制限もない状態で噂が広がっていく様子のシミュレーションなどに用いらえれる


# ＜目次＞
# 0 準備
# 1 画像ファイル読み込み
# 2 モデルによる予測


# 0 準備 ---------------------------------------------------------------------

# ライブラリ
import numpy as np

from PIL import Image
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image


# 画像読み込み
# --- 画像確認のみ
filename = "png/vegi.png"
im = Image.open(filename)

# 表示
im


# 1 画像ファイル読み込み -----------------------------------------------------------

# 画像ファイル読み込み(224x224にリサイズ)
img = image.load_img(filename, target_size=(224, 224))

# 配列に変換
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)


# 2 モデルによる予測 --------------------------------------------------------------

# モデル構築
# --- 学習済モデルのVGG16をロード
model = VGG16(weights='imagenet')

# 予測
# --- 上位5位までのクラス
preds = model.predict(preprocess_input(x))

# 予測値のデコード
# --- 上位5つのみ計算、最初のものを表示
results = decode_predictions(preds, top=5)[0]


for result in results:
    print(result[1], result[2])

#%%
