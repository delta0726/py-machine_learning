# ---------------------------------------------------------------------------------------------------
# Library   : Scikit-Learn
# Category  : datasets
# Function  :
# Created by: Owner
# Created on: 2021//
# URL       : https://scikit-learn.org/stable/modules/classes.html
# ---------------------------------------------------------------------------------------------------


# ＜概要＞




# ＜構文＞



# ＜引数＞



# ＜目次＞
# 0 準備
# 1 データフレームに変換


# 0 準備 -----------------------------------------------------------------------------------

# ライブラリ
import pandas as pd
from sklearn.datasets import load_iris

# データ準備
iris = load_iris()

# データ概要
iris.keys()
iris.items()

# アイテム一覧
dir(iris)


# 1 データフレームに変換 ---------------------------------------------------------------------

# データフレーム作成
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['species'] = iris.target_names[iris.target]
df.head()
