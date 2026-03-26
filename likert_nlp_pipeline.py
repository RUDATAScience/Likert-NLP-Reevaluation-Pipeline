import numpy as np
import pandas as pd
from janome.tokenizer import Tokenizer
import matplotlib.pyplot as plt
import seaborn as sns

def generate_dummy_data(n_samples=10):
    """
    ダミーデータの生成 (5段階評価 + 日本語記述回答)
    意図的に「手抜きの3」と「熟慮の3」などを混在させる
    """
    np.random.seed(42)
    data = {
        'user_id': range(1, n_samples + 1),
        'score': [3, 3, 5, 1, 3, 4, 2, 3, 5, 1],
        'text': [
            "特になし", 
            "普通です。", 
            "機能Aの使い勝手が圧倒的に良く、業務効率が劇的に改善したため非常に満足している。", 
            "全く使えない。起動が遅すぎる。", 
            "デザインは良いが、操作性が少し分かりにくい部分があり、トータルではどちらとも言えない。", 
            "概ね満足しているが、価格が少し高い。", 
            "マニュアルが不親切。", 
            "", # 空欄
            "最高！", 
            "ああああ" # スパム的な回答
        ]
    }
    return pd.DataFrame(data)

def analyze_text(text, tokenizer):
    """テキストから「認知的負荷をかけた度合い（熱量）」を抽出する関数"""
    if pd.isna(text) or str(text).strip() == "":
        return 0, 0, True  # 文字数0, 名詞0, 手抜きフラグTrue
    
    # 完全な手抜き（無効回答）の判定
    lazy_words = ["特になし", "なし", "特になし。", "普通", "あ", "あああ", "普通です。", "特にありません"]
    if str(text).strip() in lazy_words:
        return len(str(text)), 0, True
    
    # 形態素解析で名詞の数をカウント（具体性の指標）
    tokens = tokenizer.tokenize(str(text))
    noun_count = sum(1 for token in tokens if token.part_of_speech.startswith('名詞'))
    
    return len(str(text)), noun_count, False

def run_pipeline():
    print("=== 1. データの読み込み ===")
    df_raw = generate_dummy_data()
    print(df_raw)

    print("\n=== 2. データ分解とNLP解析の実行 ===")
    df_score = df_raw[['user_id', 'score']].copy()
    df_text = df_raw[['user_id', 'text']].copy()
    
    tokenizer = Tokenizer()
    
    # 特徴量の抽出
    extracted_features = df_text['text'].apply(lambda x: pd.Series(analyze_text(x, tokenizer)))
    extracted_features.columns = ['text_length', 'noun_count', 'is_lazy']
    df_text = pd.concat([df_text, extracted_features], axis=1)

    # テキスト由来の確信度（Text Confidence）を計算
    df_text['text_confidence'] = np.where(
        df_text['is_lazy'], 
        0.1,  # 手抜き回答へのペナルティ
        np.clip(np.log1p(df_text['text_length']) * 0.5 + np.log1p(df_text['noun_count']) * 0.5, 0.5, 3.0)
    )

    print("\n=== 3. スコアとテキストの再結合および総合評価 ===")
    df_processed = pd.merge(df_score, df_text, on='user_id')

    # マイノリティ保護（極端なスコア 1, 5 へのベースライン重み加算）
    df_processed['score_weight'] = np.where(df_processed['score'].isin([1, 5]), 1.5, 1.0)

    # 最終的な「回答の妥当性（Final Weight）」の算出
    df_processed['final_weight'] = df_processed['score_weight'] * df_processed['text_confidence']

    # 最終ウェイトを正規化
    df_processed['final_weight'] = df_processed['final_weight'] / df_processed['final_weight'].max()

    display_cols = ['user_id', 'score', 'text', 'is_lazy', 'text_confidence', 'final_weight']
    print(df_processed[display_cols].round(3))

    print("\n=== 【分析結果】「3（普通）」と答えた人の中身の違い ===")
    score_3_df = df_processed[df_processed['score'] == 3]
    for _, row in score_3_df.iterrows():
        print(f"User {row['user_id']}: テキスト「{row['text']}」 -> 最終信頼度ウェイト: {row['final_weight']:.3f}")

    print("\n=== 4. グラフの生成と保存 ===")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # 単純なカウント（従来の1人1票）
    sns.countplot(x='score', data=df_processed, ax=axes[0], color='skyblue')
    axes[0].set_title("Conventional Count (1 Person = 1 Vote)\n Illusion of 'Centralization'")
    axes[0].set_ylim(0, 5)

    # 確信度ウェイトによる加重集計（真のシグナル強度）
    weighted_scores = df_processed.groupby('score')['final_weight'].sum().reset_index()
    sns.barplot(x='score', y='final_weight', data=weighted_scores, ax=axes[1], color='salmon')
    axes[1].set_title("Re-evaluated Weight (Factoring in Text Confidence)\n Revealing True Signals")
    axes[1].set_ylim(0, 5)

    plt.tight_layout()
    plt.savefig('reevaluation_result.png')
    print("グラフを 'reevaluation_result.png' として保存しました。")
    plt.show()

if __name__ == "__main__":
    run_pipeline()
