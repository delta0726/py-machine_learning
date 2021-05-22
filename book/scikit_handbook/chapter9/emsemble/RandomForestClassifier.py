# ******************************************************************************
# Title     : Scikit-Learnデータ分析実装ハンドブック
# Chapter   : 9 Scikit-Learn API
# Theme     : RandomForestClassifier
# Created by: Owner
# Created on: 2021/5/23
# Page      : P445 - P447
# ******************************************************************************


# ＜概要＞
# ランダムフォレスト回帰分析器


# ＜参考資料＞
# sklearn.ensemble.RandomForestClassifier
# https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html#sklearn.ensemble.RandomForestClassifier

# パラメータの解説
# https://ichi.pro/randamufuxoresutohaipa-parame-tachu-ningu-no-bigina-zugaido-77596161963319


# ＜目次＞
# 1 書式
# 2 引数
# 3 使用例


# 1 書式 --------------------------------------------------------------------------

# class sklearn.ensemble.RandomForestClassifier(
#    n_estimators=100,
#    *, 
#    criterion='gini', 
#    max_depth=None, 
#    min_samples_split=2, 
#    min_samples_leaf=1, 
#    min_weight_fraction_leaf=0.0, 
#    max_features='auto', 
#    max_leaf_nodes=None, 
#    min_impurity_decrease=0.0, 
#    min_impurity_split=None, 
#    bootstrap=True, 
#    oob_score=False, 
#    n_jobs=None, 
#    random_state=None, 
#    verbose=0, 
#    warm_start=False, 
#    class_weight=None, 
#    ccp_alpha=0.0, 
#    max_samples=None
# )


# 2 引数 --------------------------------------------------------------------------

# n_estimators            ： 森の中の木の数
# criterion               ： 分割の品質絵を測定する関数
# max_depth               ： 木の深さの最大値
# min_samples_split       ： 内部ノードを分割するのに必要なサンプルの最小数
# min_samples_leaf        ： 葉ノードが必要とする最小サンプル数（min_n）
# min_weight_fraction_leaf： 葉ノードが必要とする最小のウエイト加重合計
# max_features            ： 最適なスプリットを探す際に考慮する特徴量の数（mtry）
# max_leaf_nodes          ： 葉の最大ノード数
# min_impurity_decrease   ： 不純物の減少量
# min_impurity_split      ： 木の成長をアーリーストップさせる閾値
# bootstrap               ： 木を構築する際にブートストラップを使用するかどうか
# oob_score               ： 汎化精度をすいていするためにOut-of-bagサンプルを使用するかどうか
# n_jobs                  ： 学習と予測を並行して実行するジョブ数
# random_state            ： 乱数シード値
# verbose                 ： 詳細出力のレベル
# warm_start              ： 前回実行時の解を再利用するか
# ccp_alpha               ：
# class_weight            ：
# max_samples             ：


# 3 使用例 ------------------------------------------------------------------------

# ライブラリ
from pprint import pprint
from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import make_regression

# データセット作成
X, y = make_regression(n_features=4, n_informative=2,
                       random_state=0, shuffle=False)

# モデル構築
model = RandomForestRegressor(max_depth=2, random_state=0, n_estimators=100)

# 学習
# --- 初期値を含めたフォーミュラが出力される
model.fit(X, y)

# 確認
pprint(vars(model))

# 変数重要度
model.feature_importances_

# 予測値
model.predict([[0, 0, 0, 0]])
