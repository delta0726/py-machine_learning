# ******************************************************************************
# Chapter   : 3 次元削減 - PCAから性能テストまで
# Title     : 3-8 次元削減をパイプラインでテストする（Recipe25)
# Created by: Owner
# Created on: 2020/12/25
# Page      : P100 - P102
# ******************************************************************************

# ＜概要＞
# - 次元削減を前処理で使用してSVMで学習器を作成する


# ＜目次＞
# 0 準備
# 1 パイプラインの定義
# 2 パラメータチューニングの実行


# 0 準備 ------------------------------------------------------------------------------------------

from sklearn.datasets import load_iris
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC, LinearSVC
from sklearn.decomposition import PCA, NMF, TruncatedSVD
from sklearn.manifold import Isomap


# データロード
iris = load_iris()


# 1 パイプラインの定義 ------------------------------------------------------------------------------

# パイプラインの定義
# --- 次元削減
# --- SVM分類器
pipe = Pipeline([
    ('reduce_dim', PCA()),
    ('classify', SVC())
])

# パラメータ設定
params_grid = [
    {
        'reduce_dim': [PCA(), NMF(), Isomap(), TruncatedSVD()],
        'reduce_dim__n_components': [2, 3],
        'classify': [SVC(), LinearSVC()],
        'classify__C': [1, 10, 100, 1000]
    }
]

# 確認
print(params_grid)


# 2 パラメータチューニングの実行 -----------------------------------------------------------------------

# ＜ポイント＞
# - グリッドサーチを用いてハイパーパラメータのチューニングを行う


# グリッドの設定
grid = GridSearchCV(pipe, cv=3, n_jobs=-1, param_grid=params_grid)
vars(grid)

# チューニング
grid.fit(iris.data, iris.target)

# ベストチューニング
grid.best_params_

# チューニングスコア
grid.best_score_
