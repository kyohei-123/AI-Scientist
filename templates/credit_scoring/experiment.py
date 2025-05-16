# %%
# templates/credit_scoring/experiment.py
import argparse
import json
import os
import pathlib

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler

variable_descriptions = {
    "X1": "Current assets - All the assets of a company that are expected to be sold or used as a result of standard business operations over the next year.",
    "X2": "Cost of goods sold - The total amount a company paid as a cost directly related to the sale of products.",
    "X3": "Depreciation and amortization - Depreciation refers to the loss of value of a tangible fixed asset over time (such as property, machinery, buildings, and plant). Amortization refers to the loss of value of intangible assets over time.",
    "X4": "EBITDA - Earnings before interest, taxes, depreciation, and amortization. It is a measure of a company's overall financial performance, serving as an alternative to net income.",
    "X5": "Inventory - The accounting of items and raw materials that a company either uses in production or sells.",
    "X6": "Net Income - The overall profitability of a company after all expenses and costs have been deducted from total revenue.",
    "X7": "Total Receivables - The balance of money due to a firm for goods or services delivered or used but not yet paid for by customers.",
    "X8": "Market value - The price of an asset in a marketplace. In this dataset, it refers to the market capitalization since companies are publicly traded in the stock market.",
    "X9": "Net sales - The sum of a company's gross sales minus its returns, allowances, and discounts.",
    "X10": "Total assets - All the assets, or items of value, a business owns.",
    "X11": "Total Long-term debt - A company's loans and other liabilities that will not become due within one year of the balance sheet date.",
    "X12": "EBIT - Earnings before interest and taxes.",
    "X13": "Gross Profit - The profit a business makes after subtracting all the costs that are related to manufacturing and selling its products or services.",
    "X14": "Total Current Liabilities - The sum of accounts payable, accrued liabilities, and taxes such as Bonds payable at the end of the year, salaries, and commissions remaining.",
    "X15": "Retained Earnings - The amount of profit a company has left over after paying all its direct costs, indirect costs, income taxes, and its dividends to shareholders.",
    "X16": "Total Revenue - The amount of income that a business has made from all sales before subtracting expenses. It may include interest and dividends from investments.",
    "X17": "Total Liabilities - The combined debts and obligations that the company owes to outside parties.",
    "X18": "Total Operating Expenses - The expenses a business incurs through its normal business operations.",
}

expected_signs = {
    "X1": 1,  # Current assets - Higher assets → safer → Positive relationship expected
    "X2": -1,  # Cost of goods sold - Higher costs → lower profit → Negative relationship expected
    "X3": -1,  # Depreciation and amortization - Higher depreciation → aging assets → Negative relationship expected
    "X4": 1,  # EBITDA - Higher earnings → safer → Positive relationship expected
    "X5": 0,  # Inventory - Depends (high inventory could mean unsold stock or readiness to sell) → Ambiguous
    "X6": 1,  # Net Income - Higher profit → safer → Positive relationship expected
    "X7": 1,  # Total Receivables - More receivables might indicate business volume, but risk of default → Ambiguous, assume positive
    "X8": 1,  # Market value - Higher market cap → safer → Positive relationship expected
    "X9": 1,  # Net sales - Higher sales → safer → Positive relationship expected
    "X10": 1,  # Total assets - Higher assets → safer → Positive relationship expected
    "X11": -1,  # Total Long-term debt - Higher debt → riskier → Negative relationship expected
    "X12": 1,  # EBIT - Higher operating earnings → safer → Positive relationship expected
    "X13": 1,  # Gross Profit - Higher gross profit → safer → Positive relationship expected
    "X14": -1,  # Total Current Liabilities - Higher short-term liabilities → riskier → Negative relationship expected
    "X15": 1,  # Retained Earnings - Higher retained earnings → safer → Positive relationship expected
    "X16": 1,  # Total Revenue - Higher revenue → safer → Positive relationship expected
    "X17": -1,  # Total Liabilities - Higher overall liabilities → riskier → Negative relationship expected
    "X18": -1,  # Total Operating Expenses - Higher operating expenses → lower profits → Negative relationship expected
}


def load_data(file_path):
    """データを読み込む"""
    try:
        print(f"Loading data from {file_path}")
        df = pd.read_csv(file_path)
        return df
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        exit(1)


def preprocess_data(df):
    """データの前処理"""
    y = df["status_label"].map({"alive": 0, "failed": 1})
    X = df.drop(columns=["status_label", "year", "company_name"])

    # 欠損値を中央値で埋める
    X = X.fillna(X.median())

    # 標準化
    scaler = StandardScaler()
    X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

    # 高相関列を削除
    corr_matrix = X.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [column for column in upper.columns if any(upper[column] > 0.999)]
    X = X.drop(columns=to_drop)

    return X, y


def split_data(X, y, df):
    """年を基準にデータを分割"""
    # 年の情報を取得
    years = df["year"]

    # 直近2年をテストデータ、それ以前を学習データに分割
    test_years = sorted(years.unique())[-2:]  # 直近2年
    train_mask = years < min(test_years)  # 学習データ: 直近2年より前
    test_mask = years.isin(test_years)  # テストデータ: 直近2年

    # 学習データとテストデータに分割
    X_train, X_test = X[train_mask], X[test_mask]
    y_train, y_test = y[train_mask], y[test_mask]

    return X_train, X_test, y_train, y_test


def train_model(X_train, y_train, model_type="logistic"):
    """
    モデルの学習
    Args:
        X_train (pd.DataFrame): 学習データの特徴量
        y_train (pd.Series): 学習データのラベル
        model_type (str): モデルの種類 ("logistic", "random_forest", etc.)
    Returns:
        model: 学習済みモデル
    """
    if model_type == "logistic":
        # ロジスティック回帰モデルの学習
        X_train_const = sm.add_constant(X_train)  # 定数項を追加
        model = sm.Logit(y_train, X_train_const).fit()
    elif model_type == "random_forest":
        # ランダムフォレストモデルの学習
        from sklearn.ensemble import RandomForestClassifier

        model = RandomForestClassifier(random_state=42, n_estimators=20, max_depth=3)
        model.fit(X_train, y_train)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

    return model


def evaluate_model(model, X_test, y_test, expected_signs=None, variable_descriptions=None):
    """モデルの評価"""
    X_test_const = sm.add_constant(X_test)  # 定数項を追加
    y_prob = model.predict(X_test_const)
    auc = float(roc_auc_score(y_test, y_prob))
    ar = 2 * auc - 1  # AR値の計算

    # 係数の符号をチェック
    if expected_signs and hasattr(model, "params"):
        print("\nChecking coefficient signs:")
        for var, expected_sign in expected_signs.items():
            if var in model.params:
                actual_coef = model.params[var]
                if expected_sign == 1 and actual_coef > 0:
                    print(f"  {var}: OK (expected positive, got {actual_coef:.3f})")
                elif expected_sign == -1 and actual_coef < 0:
                    print(f"  {var}: OK (expected negative, got {actual_coef:.3f})")
                elif expected_sign == 0:
                    print(f"  {var}: Ambiguous sign, skipping check.")
                else:
                    description = variable_descriptions.get(var, "No description available")
                    print(
                        f"  {var}: SIGN MISMATCH! Expected {expected_sign}, got {actual_coef:.3f}. Description: {description}"
                    )
            else:
                print(f"  {var}: Not found in model parameters.")

    return auc, ar


def save_results(all_metrics, out_dir):
    """複数モデルの結果をJSONファイルに保存"""
    os.makedirs(out_dir, exist_ok=True)
    file_path = os.path.join(out_dir, "final_info.json")

    # final_infoの構造に合わせてデータを整形
    final_info = {
        model_name: {
            "means": metrics,  # 平均値としてmetricsをそのまま格納
            "stderrs": {},  # 標準誤差は空の辞書として初期化（必要に応じて計算）
            "raw_values": metrics,  # 生の値をそのまま格納（必要に応じて変更可能）
        }
        for model_name, metrics in all_metrics.items()
    }

    try:
        with open(file_path, "w") as f:
            json.dump(final_info, f, indent=2)
        print(f"Results saved to {file_path}")
    except IOError:
        print(f"Error: Could not save results to {file_path}")


def plot_cumulative_defaults(y_test, y_prob, out_dir, model_name, ar_value):
    """累積デフォルト比率をプロット"""
    # スコアとラベルをデータフレームにまとめる
    df = pd.DataFrame({"score": y_prob, "label": y_test})

    # スコアの降順に並べ替え
    df = df.sort_values(by="score", ascending=False)

    # 累積デフォルト比率を計算
    df["cumulative_defaults"] = df["label"].cumsum() / df["label"].sum()

    # 評価対象数の累積割合を計算
    df["population_ratio"] = np.arange(1, len(df) + 1) / len(df)

    # プロット
    plt.figure(figsize=(8, 6))
    label_name = f"{model_name} (AR={ar_value:.2f})"  # ラベル名をモデル名 + AR値に設定
    plt.plot(df["population_ratio"], df["cumulative_defaults"], label=label_name, color="blue")
    plt.xlabel("Number of Entity (Cumulative Ratio)")  # 横軸ラベル
    plt.ylabel("Cumulative Default Ratio")  # 縦軸ラベル
    plt.title("Cumulative Default Ratio")
    plt.legend()
    plt.grid(True)

    # 保存
    plot_path = os.path.join(f"model_performance_{out_dir}.png")
    plt.savefig(plot_path)
    plt.close()
    print(f"Cumulative default ratio plot saved to {plot_path}")


def plot_cumulative_defaults_multiple(models_results, y_test, out_dir):
    """
    複数モデルの累積デフォルト比率を同じグラフにプロット
    Args:
        models_results (dict): モデル名をキー、予測確率とAR値を値とする辞書
                               例: {"Logistic Regression": (y_prob, ar_value), ...}
        y_test (pd.Series): テストデータのラベル
        out_dir (str): グラフの保存先ディレクトリ
    """
    plt.figure(figsize=(8, 6))

    for model_name, (y_prob, ar_value) in models_results.items():
        # スコアとラベルをデータフレームにまとめる
        df = pd.DataFrame({"score": y_prob, "label": y_test})

        # スコアの降順に並べ替え
        df = df.sort_values(by="score", ascending=False)

        # 累積デフォルト比率を計算
        df["cumulative_defaults"] = df["label"].cumsum() / df["label"].sum()

        # 評価対象数の累積割合を計算
        df["population_ratio"] = np.arange(1, len(df) + 1) / len(df)

        # プロット
        label_name = f"{model_name} (AR={ar_value:.2f})"  # ラベル名をモデル名 + AR値に設定
        plt.plot(df["population_ratio"], df["cumulative_defaults"], label=label_name)

    # グラフの設定
    plt.xlabel("Number of Entity (Cumulative Ratio)")  # 横軸ラベル
    plt.ylabel("Cumulative Default Ratio")  # 縦軸ラベル
    plt.title("Cumulative Default Ratio (Multiple Models)")
    plt.legend()
    plt.grid(True)

    # 保存
    os.makedirs(out_dir, exist_ok=True)
    plot_path = os.path.join(f"model_performance_{out_dir}.png")
    plt.savefig(plot_path)
    plt.close()
    print(f"Cumulative default ratio plot saved to {plot_path}")


def main():
    # 引数の解析
    parser = argparse.ArgumentParser(description="Run experiment")
    parser.add_argument("--out_dir", type=str, default="run_0", help="Output directory")
    parser.add_argument(
        "--data_dir",
        type=str,
        default="../../../data/american_bankruptcy",
        help="Data directory",
    )
    args = parser.parse_args()

    # データの読み込み
    file_path = pathlib.Path(args.data_dir) / "american_bankruptcy.csv"
    df = load_data(file_path)

    # データの前処理
    X, y = preprocess_data(df)

    # データの分割
    X_train, X_test, y_train, y_test = split_data(X, y, df)

    # 複数モデルの結果を格納する辞書
    all_metrics = {}
    models_results = {}

    # モデル1: ロジスティック回帰
    model = train_model(X_train, y_train, model_type="logistic")
    auc, ar = evaluate_model(model, X_test, y_test, expected_signs, variable_descriptions)
    coef = model.params
    pvals = model.pvalues

    # 係数の符号チェックの結果を格納する辞書
    sign_check_results = {}
    if expected_signs and hasattr(model, "params"):
        print("\nChecking coefficient signs:")
        for var, expected_sign in expected_signs.items():
            if var in model.params:
                actual_coef = model.params[var]
                if expected_sign == 1 and actual_coef > 0:
                    result = "OK (expected positive, got {:.3f})".format(actual_coef)
                    print(f"  {var}: {result}")
                elif expected_sign == -1 and actual_coef < 0:
                    result = "OK (expected negative, got {:.3f})".format(actual_coef)
                    print(f"  {var}: {result}")
                elif expected_sign == 0:
                    result = "Ambiguous sign, skipping check."
                    print(f"  {var}: {result}")
                else:
                    description = variable_descriptions.get(var, "No description available")
                    result = "SIGN MISMATCH! Expected {}, got {:.3f}. Description: {}".format(
                        expected_sign, actual_coef, description
                    )
                    print(f"  {var}: {result}")
                sign_check_results[var] = result  # 結果を格納
            else:
                result = "Not found in model parameters."
                print(f"  {var}: {result}")
                sign_check_results[var] = result  # 結果を格納
    else:
        sign_check_results = "Sign check not performed."

    all_metrics["logistic_regression"] = {
        "auc": auc,
        "ar": ar,
        "coeff": {v: {"beta": float(coef[v]), "p": float(pvals[v])} for v in coef.index},
        "sign_check": sign_check_results,  # 符号チェックの結果を保存
    }
    models_results["logistic_regression"] = (model.predict(sm.add_constant(X_test)), ar)

    # モデル2: ランダムフォレスト
    rf_model = train_model(X_train, y_train, model_type="random_forest")
    y_prob_rf = rf_model.predict_proba(X_test)[:, 1]
    auc_rf = roc_auc_score(y_test, y_prob_rf)
    ar_rf = 2 * auc_rf - 1
    all_metrics["random_forest"] = {
        "auc": auc_rf,
        "ar": ar_rf,
        "feature_importance": dict(zip(X_train.columns, rf_model.feature_importances_)),
    }
    models_results["Random Forest"] = (y_prob_rf, ar_rf)

    # 結果の保存
    save_results(all_metrics, args.out_dir)

    # 各モデルの累積デフォルト比率のプロット
    plot_cumulative_defaults_multiple(models_results, y_test, args.out_dir)


# %%
if __name__ == "__main__":
    main()
