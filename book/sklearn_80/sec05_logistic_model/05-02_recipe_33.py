# ******************************************************************************
# Chapter   : 5 ロジスティック回帰
# Title     : 5-2 UCI Machine Learning Repositoryからのデータの読み込み（Recipe33)
# Created by: Owner
# Created on: 2020/12/25
# Page      : P132 - P133
# ******************************************************************************

# ＜概要＞
# - ローカルファイルを読み込む


# ＜目次＞
# 0 準備


# 0 準備 ------------------------------------------------------------------------------------------

import os
import pandas as pd


# パスの取得
current_path = os.getcwd()
file = os.path.sep.join(['', 'csv', 'pima-indians-diabetes.csv'])

# 列名指定
column_names = ['pregnancy_x',
                'plasma_con',
                'blood_pressure',
                'skin_mm',
                'insulin',
                'bmi',
                'pedigree_func',
                'age',
                'target']

# データ取得
all_data = pd.read_csv(current_path + file,  names=column_names)
