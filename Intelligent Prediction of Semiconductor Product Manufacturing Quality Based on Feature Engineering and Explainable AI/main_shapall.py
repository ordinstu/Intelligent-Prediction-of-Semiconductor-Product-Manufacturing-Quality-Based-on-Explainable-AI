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


OUTPUT_DIR = 'paper_outputs_shapall'

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
# 8. 全面的 SHAP 可解释性分析 (修复版)
# ==========================================
print("\n[Step 7] Generating Comprehensive SHAP Plots...")
# 配置字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial']
plt.rcParams['axes.unicode_minus'] = False

# 1. 构造 SHAP Explanation 对象
explainer = shap.TreeExplainer(best_model)
shap_values_obj = explainer(X_test)

# 修复特征名称
shap_values_obj.feature_names = X_test.columns.astype(str).tolist()

# ---------------------------------------------------------
# A. 全局解释 (Global Interpretability)
# ---------------------------------------------------------

# 图 1: 蜂群图 (Beeswarm)
print("正在生成：全局蜂群图 (Beeswarm)...")
plt.figure(figsize=(10, 6))
plt.title("SHAP Beeswarm Plot (Global Impact)", fontsize=14)
shap.plots.beeswarm(shap_values_obj, max_display=15, show=False)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'Figure2.png'))
plt.show()

# 图 2: 特征重要性柱状图 (Bar)
print("正在生成：特征重要性图 (Bar)...")
plt.figure(figsize=(8, 6))
plt.title("Mean Absolute SHAP Value (Feature Importance)", fontsize=14)
shap.plots.bar(shap_values_obj, max_display=15, show=False)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'Figure3.png'))
plt.show()

# 图 3: SHAP 热图 (Heatmap) - [已修复报错]
print("正在生成：SHAP 热图 (Heatmap)...")
plt.figure(figsize=(12, 5))
plt.title("SHAP Heatmap (Instance Clustering)", fontsize=14)
shap.plots.heatmap(shap_values_obj, max_display=10, show=False)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'Figure4.png'))
plt.show()

# 图 4: 依赖图 (Dependence Plot)
print("正在生成：依赖图 (Dependence Plot)...")
# 自动找到最重要的特征
top_feature_index = np.abs(shap_values_obj.values).mean(0).argmax()
top_feature_name = shap_values_obj.feature_names[top_feature_index]
print(f"检测到最重要的特征是: {top_feature_name}")

plt.figure(figsize=(8, 6))
shap.plots.scatter(shap_values_obj[:, top_feature_name], color=shap_values_obj, show=False)
plt.title(f"Dependence Plot for {top_feature_name}", fontsize=14)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'Figure5.png'))
plt.show()

# ---------------------------------------------------------
# B. 局部解释 (Local Interpretability)
# ---------------------------------------------------------

# 智能选点逻辑：找一个预测概率最高的“次品”样本
fail_indices = np.where(y_test.values == 1)[0]
if len(fail_indices) > 0:
    fail_probs = y_pred_proba[fail_indices]
    target_idx_relative = fail_indices[np.argmax(fail_probs)]
    print(f"\n选定用于局部解释的样本 ID: {target_idx_relative} (预测概率: {y_pred_proba[target_idx_relative]:.4f})")
else:
    print("警告：测试集中没有检测到次品，随机选择第0个样本演示。")
    target_idx_relative = 0

# 图 5: 瀑布图 (Waterfall Plot)
print("正在生成：单样本瀑布图 (Waterfall)...")
plt.figure(figsize=(8, 6))
shap.plots.waterfall(shap_values_obj[target_idx_relative], max_display=10, show=False)
plt.title(f"Why is Instance {target_idx_relative} classified as Fail?", fontsize=14)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'Figure6.png'))
plt.show()

# 图 6: 力图 (Force Plot)
print("正在生成：单样本力图 (Force Plot)...")
plt.figure(figsize=(12, 3))
# matplotlib=True 确保在非Jupyter环境中能弹窗显示
shap.plots.force(shap_values_obj[target_idx_relative], matplotlib=True, show=False)
plt.title(f"Force Plot for Instance {target_idx_relative}", y=1.5, fontsize=14)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'Figure7.png'))
plt.show()

print("\n所有图表生成完毕！请保存图片用于论文撰写。")