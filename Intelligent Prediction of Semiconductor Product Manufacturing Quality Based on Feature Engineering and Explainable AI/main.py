import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
import shap
import os

from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
from boruta import BorutaPy

# 防止中文字体报错（可选，视操作系统而定）
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial']
plt.rcParams['axes.unicode_minus'] = False

OUTPUT_DIR = 'paper_outputs'

# ==========================================
# 1. 数据准备 (Data Loading)
# ==========================================
def load_data(use_synthetic=True):
    if use_synthetic:
        print("正在生成模拟数据 (模拟 SECOM)...")
        np.random.seed(42)
        X = np.random.randn(1567, 591)
        X[:, :50] = np.nan  # 模拟缺失
        y = np.zeros(1567)
        y[np.random.choice(1567, 104, replace=False)] = 1

        # 模拟数据自带名字
        feature_names = [f"Feature_{i}" for i in range(X.shape[1])]
        df_X = pd.DataFrame(X, columns=feature_names)
        df_y = pd.Series(y, name="Pass_Fail")
        return df_X, df_y
    else:
        try:
            print("正在加载真实 SECOM 数据...")
            # 读取数据
            X = pd.read_csv('secom.data', sep=" ", header=None)
            y = pd.read_csv('secom_labels.data', sep=" ", header=None)[0]
            y = y.replace(-1, 0)  # -1转0(Pass), 1转1(Fail)

            # --- 关键修复：重命名列名为字符串 ---
            # 解决 SHAP 报错 "TypeError: can only concatenate str (not 'int') to str"
            # 我们直接把 0, 1, 2... 变成 "Sensor_0", "Sensor_1"...
            X.columns = [f"Sensor_{i}" for i in range(X.shape[1])]

            return X, y
        except FileNotFoundError:
            print("未找到文件，切换回模拟模式...")
            return load_data(True)


# !!! 使用真实数据 !!!
X, y = load_data(use_synthetic=False)

# ==========================================
# 2. 数据清洗 (Paper Sect 3.2)
# ==========================================
print("\n[Step 1] Data Cleaning...")

# 2.1 剔除缺失率 > 50% 的列
missing_ratio = X.isnull().mean()
cols_to_drop = missing_ratio[missing_ratio > 0.5].index
X_clean = X.drop(columns=cols_to_drop)
print(f"剔除高缺失列后: {X_clean.shape[1]} 特征")

# 2.2 剔除常量列
temp_imputer = SimpleImputer(strategy='median')
X_filled_temp = pd.DataFrame(temp_imputer.fit_transform(X_clean), columns=X_clean.columns)
nunique = X_filled_temp.nunique()
cols_to_drop_const = nunique[nunique == 1].index
X_clean = X_clean.drop(columns=cols_to_drop_const)
print(f"剔除常量列后: {X_clean.shape[1]} 特征")

# 2.3 最终填补 (Median)
imputer = SimpleImputer(strategy='median')
X_imputed = pd.DataFrame(imputer.fit_transform(X_clean), columns=X_clean.columns)

# ==========================================
# 3. 特征缩放 (Paper Sect 3.3)
# ==========================================
print("\n[Step 2] Feature Scaling...")
scaler = MinMaxScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X_imputed), columns=X_imputed.columns)

# ==========================================
# 4. 特征选择 Boruta (Paper Sect 3.4)
# ==========================================
print("\n[Step 3] Feature Selection via Boruta...")
print("注意：Boruta 正在计算特征重要性，这可能需要几分钟...")

rf = RandomForestClassifier(n_jobs=-1, class_weight='balanced', max_depth=5, random_state=42)
feat_selector = BorutaPy(rf, n_estimators='auto', verbose=0, random_state=42, perc=90)

# Boruta 需要 numpy array
feat_selector.fit(X_scaled.values, y.values)

# 获取特征
selected_features = X_scaled.columns[feat_selector.support_].tolist()

# 防止没选出特征的兜底逻辑
if len(selected_features) == 0:
    print("Boruta 未选出特征，使用前50个作为备选。")
    selected_features = X_scaled.columns[:50].tolist()

print(f"Boruta 选出的特征数量: {len(selected_features)}")
X_selected = X_scaled[selected_features]

# ==========================================
# 5. 划分与 SMOTE (Paper Sect 3.5)
# ==========================================
print("\n[Step 4] Split & SMOTE...")
X_train, X_test, y_train, y_test = train_test_split(
    X_selected, y, test_size=0.2, random_state=42, stratify=y
)

smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

# ==========================================
# 6. 模型调优 (Paper Sect 3.6)
# ==========================================
print("\n[Step 5] Hyperparameter Tuning...")

xgb_clf = xgb.XGBClassifier(
    objective='binary:logistic',
    eval_metric='logloss',
    n_jobs=-1,
    random_state=42
)

param_dist = {
    'n_estimators': [100, 200, 300],
    'learning_rate': [0.01, 0.05, 0.1, 0.2],
    'max_depth': [3, 4, 5],
    'subsample': [0.7, 0.8, 1.0],
    'colsample_bytree': [0.6, 0.8, 1.0]
}

random_search = RandomizedSearchCV(
    estimator=xgb_clf,
    param_distributions=param_dist,
    n_iter=15,  # 迭代次数，可根据时间调整
    scoring='roc_auc',  # 既然AUC高，我们就针对AUC优化
    cv=3,
    verbose=1,
    random_state=42,
    n_jobs=-1
)

random_search.fit(X_train_res, y_train_res)
best_model = random_search.best_estimator_
print(f"最佳参数: {random_search.best_params_}")

# ==========================================
# 7. 评估与阈值优化 (Critical for Paper)
# ==========================================
print("\n[Step 6] Evaluation & Threshold Optimization...")

# 获取预测概率
y_pred_proba = best_model.predict_proba(X_test)[:, 1]
auc_score = roc_auc_score(y_test, y_pred_proba)

print(f"--- 核心指标: AUC = {auc_score:.4f} ---")

# --- 默认阈值 0.5 ---
print("\n[A] 默认阈值 (0.5) 结果:")
print(classification_report(y_test, (y_pred_proba > 0.5).astype(int), target_names=['Pass', 'Fail']))

# --- 优化阈值 (利用高AUC优势) ---
# 既然 AUC 很高，说明概率排序是对的，只是分割线太高了
# 我们设定阈值为 0.25 (或者根据你的需求调整)
opt_threshold = 0.25
y_pred_opt = (y_pred_proba > opt_threshold).astype(int)

print(f"\n[B] 优化阈值 ({opt_threshold}) 结果 (论文推荐):")
print(classification_report(y_test, y_pred_opt, target_names=['Pass', 'Fail']))

# 绘制优化后的混淆矩阵
plt.figure(figsize=(6, 5))
sns.heatmap(confusion_matrix(y_test, y_pred_opt), annot=True, fmt='d', cmap='Reds')
plt.title(f'Confusion Matrix (Threshold={opt_threshold})')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'Figure1.png'))
plt.show()


# ==========================================
# 9. SHAP 可解释性 (旧版风格 + 保存功能)
# ==========================================
print("\n[Step 7] SHAP Interpretation & Saving Images...")

# 准备工作
explainer = shap.TreeExplainer(best_model)
shap_values = explainer.shap_values(X_test)
feature_names_str = X_test.columns.astype(str).tolist()

# ---------------------------------------------------------
# 图 1: Summary Plot (全局蜂群图)
# ---------------------------------------------------------
print("正在生成并保存：SHAP Summary Plot...")
plt.figure(figsize=(10, 8)) # 设置画布大小，防止保存时显示不全

# 关键 1: 加上 show=False，防止函数自己弹窗后清空画布
shap.summary_plot(shap_values, X_test, feature_names=feature_names_str, show=False)

# 关键 2: 获取当前图像并保存
plt.title("SHAP Summary Plot", fontsize=16)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'Figure2.png'))
print("已保存: Paper_Fig_SHAP_Summary.png")
plt.close() # 关闭画布，释放内存

# ---------------------------------------------------------
# 图 2: Waterfall Plot (局部单样本图)
# ---------------------------------------------------------
print("正在生成并保存：SHAP Waterfall Plot...")

# 智能找一个次品样本（和之前逻辑一样）
fail_indices = np.where(y_test.values == 1)[0]
if len(fail_indices) > 0:
    fail_probs = best_model.predict_proba(X_test)[:, 1][fail_indices]
    target_idx_relative = fail_indices[np.argmax(fail_probs)]
else:
    target_idx_relative = 0

plt.figure(figsize=(8, 6))

# 关键 1: show=False
shap.plots.waterfall(
    shap.Explanation(
        values=shap_values[target_idx_relative],
        base_values=explainer.expected_value,
        data=X_test.iloc[target_idx_relative].values,
        feature_names=feature_names_str
    ),
    max_display=10,
    show=False
)

# 关键 2: 保存
plt.title(f"Local Explanation for Instance {target_idx_relative}", fontsize=14)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'Figure3.png'))
print("已保存: Paper_Fig_SHAP_Waterfall.png")
plt.close()

print("\n所有图片已保存至当前目录！")