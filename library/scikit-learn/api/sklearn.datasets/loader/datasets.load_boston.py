# ---------------------------------------------------------------------------------------------------
# Library   : Scikit-Learn
# Category  : datasets
# Function  : load_boston
# Created by: Owner
# Created on: 2021/5/11
# URL       : https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_boston.html#sklearn.datasets.load_boston
# ---------------------------------------------------------------------------------------------------


# ＜概要＞
# - 回帰用データセットの代表例であるBostonデータセットをロードする


# ＜構文＞
# sklearn.datasets.load_boston(*, return_X_y=False)


# ＜引数＞



# ＜目次＞
# 0 準備


# 0 準備 -----------------------------------------------------------------------------------

# ライブラリ
import pandas as pd
from sklearn.datasets import load_boston

# データ準備
boston = load_boston()

# データ概要
boston.keys()
boston.items()

# アイテム一覧
dir(boston)


# 1 データフレームに変換 ---------------------------------------------------------------------

# データフレーム作成
df = pd.DataFrame(boston.data, columns=boston.feature_names)
df['target'] = boston.target
df.head()
