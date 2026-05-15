# Src/frame_detection.py
"""
Frame detection untuk tweet MBG.
Dipanggil dari 03_SentimentAnalysis.ipynb setelah frame_keywords didefinisikan.
"""

import pandas as pd


def assign_frames(text, frame_keywords):
    """
    Assign semua frame yang match ke satu tweet.
    Returns: (frame_utama, all_frames_list)
    - frame_utama: frame dengan match terbanyak (atau 'tidak_terklasifikasi')
    - all_frames_list: list semua frame yang match
    """
    if pd.isna(text) or not isinstance(text, str):
        return 'tidak_terklasifikasi', []

    t = text.lower()
    matched = {}

    for frame, keywords in frame_keywords.items():
        count = sum(1 for kw in keywords if kw in t)
        if count > 0:
            matched[frame] = count

    if not matched:
        return 'tidak_terklasifikasi', []

    # Frame utama = frame dengan keyword match terbanyak
    frame_utama = max(matched, key=matched.get)
    all_frames = sorted(matched.keys())

    return frame_utama, all_frames


def apply_frame_detection(df, frame_keywords, text_col='clean_text'):
    """
    Apply frame detection ke seluruh dataframe.
    Menambahkan kolom: frame_utama, all_frames
    """
    results = df[text_col].apply(lambda x: assign_frames(x, frame_keywords))
    df['frame_utama'] = results.apply(lambda x: x[0])
    df['all_frames'] = results.apply(lambda x: x[1])

    # Stats
    total = len(df)
    classified = (df['frame_utama'] != 'tidak_terklasifikasi').sum()
    coverage = classified / total * 100

    print(f"Frame detection coverage: {classified}/{total} ({coverage:.1f}%)")
    print(f"\nDistribusi frame:")
    print(df['frame_utama'].value_counts())

    return df
