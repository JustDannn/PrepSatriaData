# TASK.md — Satria Data 2026 | Case 2: Analisis Tweet MBG
> Last updated: 2026-05-15 | Status: Fase 4 berjalan, belum selesai

---

## STATUS KESELURUHAN

```
Fase 1 — Preprocessing        ████████████ SELESAI
Fase 2 — EDA                  ████████████ SELESAI
Fase 3 — Sentiment Analysis   ████████░░░░ SELESAI (ada improvement pending)
Fase 4 — Topic Modeling       ████░░░░░░░░ BERJALAN — perlu selesaikan
Fase 5 — Network Analysis     ░░░░░░░░░░░░ BELUM MULAI
Fase 6 — Polarization         ░░░░░░░░░░░░ BELUM MULAI
Fase 7 — Visualisasi Final    ░░░░░░░░░░░░ BELUM MULAI
```

---

## FASE 1 — Preprocessing (`01_PreProcessed.ipynb`)

### ✅ Sudah Selesai
- `clean_tweet()` — hapus URL, mention, expand CamelCase hashtag, lowercase, strip non-alfa
- `normalize_slang()` — Colloquial Indonesian Lexicon + custom dict MBG
- Stemming Sastrawi di-cache ke `../Data/Cache/stemming_dict.json`
- `remove_repetition()` — deduplikasi token dalam satu tweet
- Filter hapax legomena (kata muncul < 2x dibuang)
- Output: `../Data/Cleaned/Case 2 Dataset.csv`

### ⚠️ Improvement Pending

**[P1-01] Parse kolom metadata di preprocessing, bukan di tiap notebook**
Kolom `view_count`, `user_followers_count`, `user_statuses_count`, `retweet_count`,
`favorite_count`, `created_at_dt`, `user_created_at_dt`, `mentions_parsed` harus
di-parse sekali di sini dan disimpan ke cleaned CSV, bukan di-parse ulang di setiap
notebook downstream.

```python
# Tambahkan di akhir 01_PreProcessed.ipynb sebelum save
import ast

df['created_at_dt']        = pd.to_datetime(df['created_at'], errors='coerce', utc=True)
df['user_created_at_dt']   = pd.to_datetime(df['user_created_at'], errors='coerce', utc=True)
df['view_count']           = pd.to_numeric(df['view_count'], errors='coerce').fillna(0)
df['user_followers_count'] = pd.to_numeric(df['user_followers_count'], errors='coerce').fillna(0)
df['user_statuses_count']  = pd.to_numeric(df['user_statuses_count'], errors='coerce').fillna(0)
df['retweet_count']        = pd.to_numeric(df['retweet_count'], errors='coerce').fillna(0)
df['favorite_count']       = pd.to_numeric(df['favorite_count'], errors='coerce').fillna(0)

df['mentions_parsed'] = df['mentions'].apply(
    lambda x: ast.literal_eval(x) if isinstance(x, str) and x.startswith('[') else []
)
df['mention_usernames'] = df['mentions_parsed'].apply(
    lambda lst: [m['username'] for m in lst if isinstance(m, dict)]
)
df['mention_count'] = df['mention_usernames'].apply(len)
```

**[P1-02] Tambahkan `Src/config.py` agar path tidak hardcoded**
```python
# Src/config.py
from pathlib import Path
ROOT           = Path(__file__).parent.parent
RAW_DATA       = ROOT / "Data" / "Raw"       / "Case 2 Dataset.csv"
CLEANED_DATA   = ROOT / "Data" / "Cleaned"   / "Case 2 Dataset.csv"
ANALYZED_DATA  = ROOT / "Data" / "Analyzed"  / "mbg_analyzed.csv"
STEMMING_CACHE = ROOT / "Data" / "Cache"     / "stemming_dict.json"
SLANG_CACHE    = ROOT / "Data" / "Cache"     / "slang_dict.json"
FIG_DIR        = ROOT / "Output" / "Figures"
```

---

## FASE 2 — EDA (`02_Eda.ipynb`)

### ✅ Sudah Selesai
- Distribusi tweet, unique user, verified user, quote tweet
- Distribusi view/retweet/favorite (dengan buang outlier P95)
- Top 10 source device
- Distribusi panjang teks & word count
- Top 30 kata, WordCloud, Bigram, Trigram
- Volume per hari, distribusi jam, heatmap hari × jam

### ⚠️ Improvement Pending

**[P2-01] EDA berbasis rich metadata belum ada sama sekali**
EDA sekarang hanya melihat teks. Padahal data punya banyak sinyal user-level
yang penting untuk lomba.

```python
# Tambahkan section baru di 02_Eda.ipynb

# A. Distribusi akun berdasarkan umur (kapan dibuat)
df['account_age_days'] = (
    pd.Timestamp("2025-06-01", tz='UTC') - df['user_created_at_dt']
).dt.days
plt.hist(df['account_age_days'].clip(0, 5000), bins=50)
plt.title('Distribusi Umur Akun (hari)')
plt.axvline(180, color='red', linestyle='--', label='6 bulan (threshold buzzer)')
plt.legend(); plt.show()

# B. Distribusi follower count (log scale)
plt.hist(df['user_followers_count'].clip(0, 10000), bins=100)
plt.yscale('log')
plt.title('Distribusi Follower Count')
plt.show()

# C. Korelasi view_count vs follower
import seaborn as sns
sns.scatterplot(data=df.sample(2000), x='user_followers_count',
                y='view_count', alpha=0.3)
plt.xscale('log'); plt.yscale('log')
plt.title('Follower vs View Count (log-log)')
plt.show()

# D. Top 20 akun paling sering di-mention
from collections import Counter
all_mentions = [m for lst in df['mention_usernames'] for m in lst]
top_mentions = pd.DataFrame(Counter(all_mentions).most_common(20),
                            columns=['username', 'mention_count'])
print(top_mentions)
# → Siapa yang jadi focal point diskusi?
```

**[P2-02] Spike detection pada time series**
Volume tweet per hari sudah ada, tapi belum ada anotasi event nyata.
```python
# Tandai tanggal-tanggal penting
events = {
    '2025-01-06': 'MBG Resmi Diluncurkan',
    '2025-02-': 'Kasus keracunan pertama (cek tanggal pasti)',
    # tambahkan sesuai temuan data
}
# Plot ulang dengan axvline + anotasi di tiap event
```

---

## FASE 3 — Sentiment Analysis (`03_SentimentAnalysis.ipynb`)

### ✅ Sudah Selesai
- VADER + Indo Lexicon (baseline)
- IndoBERT final label (`sentiment_bert`, `sentiment_bert_score`)
- Aspect/Frame Detection keyword-based (10 frame, coverage 71.4%)
- Heatmap & stacked bar sentiment × frame
- Save ke `mbg_analyzed.csv`

### ⚠️ Improvement Pending (Prioritas Tinggi)

**[P3-01] IndoBERT perlu `return_all_scores=True` — WAJIB untuk Fase 6**
Saat ini hanya label menang yang disimpan. Skor per kelas (pos/neg/netral)
dibutuhkan untuk polarization index di Fase 6.
```python
# Ganti pipeline dengan:
sentiment_model_full = pipeline(
    "text-classification",
    model="ayameRushia/bert-base-indonesian-1.5G-sentiment-analysis-smsa",
    truncation=True, max_length=128, device=device,
    token=os.getenv("TOKEN"),
    return_all_scores=True  # ← KUNCI
)

# Parse hasilnya:
df['score_positif'] = [next((x['score'] for x in p if x['label'].lower()=='positive'), 0) for p in preds]
df['score_negatif'] = [next((x['score'] for x in p if x['label'].lower()=='negative'), 0) for p in preds]
df['score_netral']  = [next((x['score'] for x in p if x['label'].lower()=='neutral'),  0) for p in preds]
```

**[P3-02] View-weighted sentiment — temuan mandiri untuk presentasi**
```python
total_views = df['view_count'].sum()
raw_dist    = df['sentiment_bert'].value_counts(normalize=True).mul(100).round(1)
view_dist   = df.groupby('sentiment_bert')['view_count'].sum().div(total_views).mul(100).round(1)

comparison  = pd.DataFrame({'raw_%': raw_dist, 'view_weighted_%': view_dist})
print(comparison)
# Gap > 5% antara raw dan view-weighted = temuan yang layak masuk highlight
```

**[P3-03] Frame detection — 3 perbaikan struktural**

1. Overlap keyword: `gizi_kesehatan` punya keyword `keracunan`, `basi`, `higienis`
   yang seharusnya eksklusif di `keamanan_pangan`. Bersihkan overlap ini.

2. `sosial_pemerataan` punya keyword `papua` & `daerah`. Padahal BERTopic
   membuktikan Papua adalah cluster sendiri tentang penolakan, bukan pemerataan.
   Pisah jadi frame baru `resistensi_regional`.

3. Tambahkan sub-label granular untuk `keamanan_pangan`:
```python
keamanan_insiden_kw  = ['keracunan', 'racun', 'korban', 'siswa sakit',
                         'kasus racun', 'rumah sakit', 'tewas', 'meninggal']
keamanan_kualitas_kw = ['menu', 'basi', 'busuk', 'kotor', 'serangga',
                         'tidak higienis', 'nasi', 'susu basi']

def detect_keamanan_subframe(text):
    if pd.isna(text): return None
    t = text.lower()
    if any(kw in t for kw in keamanan_insiden_kw):  return 'insiden_keracunan'
    if any(kw in t for kw in keamanan_kualitas_kw): return 'kualitas_menu'
    return None

df['keamanan_subframe'] = df['clean_text'].apply(detect_keamanan_subframe)
```

**[P3-04] User profiling & buzzer detection**
```python
MBG_LAUNCH = pd.Timestamp("2025-01-06", tz='UTC')

def classify_user_type(desc):
    if pd.isna(desc) or str(desc).strip() == "": return "no_bio"
    d = desc.lower()
    if any(k in d for k in ['relawan', 'projo', 'tim pemenangan']): return "relawan_politik"
    if any(k in d for k in ['wartawan', 'jurnalis', 'reporter', 'media']): return "media"
    if any(k in d for k in ['dokter', 'dr.', 'ahli gizi', 'nutrisi']): return "health_professional"
    if any(k in d for k in ['guru', 'dosen', 'mahasiswa', 'peneliti']): return "akademisi"
    if any(k in d for k in ['aktivis', 'ngo', 'lsm']): return "aktivis"
    if any(k in d for k in ['wkwk', 'literally', 'bestie', '💀', '🫠']): return "genz"
    if any(k in d for k in ['pengusaha', 'ceo', 'founder', 'bisnis']): return "bisnis"
    return "general_public"

df['user_type']      = df['user_description'].apply(classify_user_type)
df['is_new_account'] = df['user_created_at_dt'] >= MBG_LAUNCH
df['buzzer_suspect'] = (
    df['is_new_account'] &
    (df['user_followers_count'] < 100) &
    (df['user_statuses_count'] > 500)
)
```

---

## FASE 4 — Topic Modeling (`04_TopicModeling.ipynb`) ← SEDANG BERJALAN

### ✅ Sudah Selesai
- BERTopic fit berhasil: 56 topik → dikurangi jadi 32 topik
- Noise (topic -1): 5.700 docs (41.8%) — terlalu tinggi, perlu di-reduce
- 32 topik tersisa, top 10 sudah diidentifikasi

### 📋 Pemetaan Topik vs Frame Manual (validasi awal)

| Topic | Keywords Dominan | Frame Manual | Status |
|-------|-----------------|--------------|--------|
| -1 | gizi, perintah, kerja, dapur | (noise) | ❌ Perlu reduce_outliers |
| 0 | indonesia, presiden, generasi, gizi | `dukungan_umum` | ✅ Match |
| 1 | anak sekolah, bayar, wkwkwk | `ekonomi` / skeptis | ✅ Match |
| 2 | menu, serangga, basi, nasi, susu | `keamanan_pangan_kualitas` | ✅ Match kuat |
| 3 | bagaimana, cara, benar, racun | `kebijakan_implementasi` | ✅ Match |
| 4 | stunting, cegah, emas, manfaat | `gizi_kesehatan` | ✅ Match |
| 5 | jangan, kasus racun, korban, siswa | `keamanan_pangan_insiden` | ✅ Match kuat |
| 6 | **papua, mbgpapua, tolak, adat** | ❌ **TIDAK ADA di frame manual** | 🆕 Frame baru! |
| 7 | enggak usah, enggak jelas | `political_skeptis` | ✅ Match |
| 8 | vendor, catering, industri, bayar | `anggaran_fiskal` | ✅ Match |

**Konsistensi BERTopic vs frame manual: ~85%** ← ini narasi validasi yang kuat

### 🔧 Yang Perlu Diselesaikan di Cell Terakhir (belum dirun)

**[P4-01] Reduce outliers — PRIORITAS PERTAMA**
```python
new_topics = topic_model.reduce_outliers(
    docs, topics,
    strategy="probabilities",
    threshold=0.05
)
topic_model.update_topics(docs, topics=new_topics)

print(f"Noise sebelum reduce: {topics.count(-1)} ({topics.count(-1)/len(topics)*100:.1f}%)")
print(f"Noise setelah reduce: {new_topics.count(-1)} ({new_topics.count(-1)/len(new_topics)*100:.1f}%)")
```

**[P4-02] Assign topic ke df dan merge dengan kolom lain**
```python
# Buat mapping docs → df (attention: docs sudah di-filter, panjang bisa beda dari df)
doc_indices = df[df['clean_text'].notna() & (df['clean_text'].str.strip().str.len() > 10)].index
topic_series = pd.Series(new_topics, index=doc_indices, name='bert_topic')
df = df.join(topic_series)
df['bert_topic'] = df['bert_topic'].fillna(-1).astype(int)

# Tambahkan nama topik dari topic_info
topic_info = topic_model.get_topic_info().set_index('Topic')['Name'].to_dict()
df['bert_topic_name'] = df['bert_topic'].map(topic_info)
```

**[P4-03] Mapping topic → frame manual**
```python
topic_to_frame_bert = {
    0:  "dukungan_umum",
    1:  "ekonomi_skeptis",
    2:  "keamanan_pangan_kualitas",
    3:  "kebijakan_implementasi",
    4:  "gizi_kesehatan",
    5:  "keamanan_pangan_insiden",
    6:  "resistensi_regional",        # frame baru dari BERTopic!
    7:  "political_skeptis",
    8:  "anggaran_vendor",
    # Lengkapi 9–31 setelah lihat topic_info lengkap
}
df['frame_bert'] = df['bert_topic'].map(topic_to_frame_bert).fillna('unassigned')
```

**[P4-04] Cross-analysis topic × sentiment × view**
```python
# Sentimen per topic (view-weighted)
topic_sentiment = df.groupby('bert_topic').agg(
    count=('tweet_id', 'count'),
    total_views=('view_count', 'sum'),
    pct_negatif=('sentiment_bert', lambda x: (x == 'negatif').mean() * 100),
    pct_positif=('sentiment_bert', lambda x: (x == 'positif').mean() * 100),
    avg_score_neg=('score_negatif', 'mean'),
    topic_name=('bert_topic_name', 'first')
).reset_index().sort_values('total_views', ascending=False)

print(topic_sentiment[topic_sentiment['bert_topic'] != -1].head(15))
```

**[P4-05] Investigasi Topic 0 yang suspiciously besar (3.616 docs)**
```python
# Topic 0 terlalu besar — kemungkinan catch-all
# Sample & baca manual untuk konfirmasi apakah perlu di-split
topic_0_docs = df[df['bert_topic'] == 0]['full_text'].sample(20, random_state=42)
for t in topic_0_docs:
    print(f"- {t[:120]}\n")

# Cek distribusi buzzer_suspect di topic 0 vs topik lain
print(pd.crosstab(df['bert_topic'], df['buzzer_suspect'], normalize='index').round(3).head(10))
# Hipotesis: kalau topic 0 banyak buzzer_suspect → itu bukan dukungan organik
```

**[P4-06] Investigasi Topic 6 (Papua) secara mendalam**
```python
topic_6 = df[df['bert_topic'] == 6]
print(f"Total tweet Papua: {len(topic_6)}")
print(f"Sentimen: {topic_6['sentiment_bert'].value_counts(normalize=True).mul(100).round(1)}")
print(f"Avg views: {topic_6['view_count'].mean():.0f}")

# Lihat siapa yang paling vocal
print(topic_6.groupby('username').agg(
    tweets=('tweet_id', 'count'),
    total_views=('view_count', 'sum'),
    followers=('user_followers_count', 'first')
).sort_values('total_views', ascending=False).head(10))

# Sample tweet mentah
for t in topic_6['full_text'].sample(10, random_state=7):
    print(f"- {t[:150]}\n")
```

**[P4-07] Visualisasi wajib untuk presentasi**
```python
import os
fig_topic_dir = "../Output/Figures/topic"
os.makedirs(fig_topic_dir, exist_ok=True)

# a. 2D scatter semua topik
fig_scatter = topic_model.visualize_topics()
fig_scatter.write_html(f"{fig_topic_dir}/topics_scatter.html")

# b. Dendrogram kedekatan antar topik
fig_hier = topic_model.visualize_hierarchy()
fig_hier.write_html(f"{fig_topic_dir}/topics_hierarchy.html")

# c. Barchart keywords per topik (top 10 topik)
fig_bar = topic_model.visualize_barchart(top_n_topics=10)
fig_bar.write_html(f"{fig_topic_dir}/topics_barchart.html")

# d. Heatmap topic × frame_utama (validasi konsistensi)
cross_topic_frame = pd.crosstab(df['bert_topic'], df['frame_utama'])
cross_topic_frame = cross_topic_frame[cross_topic_frame.index != -1]
import seaborn as sns, matplotlib.pyplot as plt
plt.figure(figsize=(14, 10))
sns.heatmap(cross_topic_frame, annot=True, fmt='d', cmap='Blues', linewidths=0.3)
plt.title('Cross-validation: BERTopic vs Frame Manual')
plt.tight_layout()
plt.savefig(f"{fig_topic_dir}/cross_validation_topic_frame.png", dpi=150)
plt.show()
```

**[P4-08] Temporal topic evolution — topik mana yang naik/turun seiring waktu**
```python
# Merge timestamp ke docs
df['month_year'] = df['created_at_dt'].dt.to_period('M')

topic_time = df[df['bert_topic'] != -1].groupby(
    ['month_year', 'bert_topic']
).size().unstack(fill_value=0)

# Normalisasi per bulan
topic_time_pct = topic_time.div(topic_time.sum(axis=1), axis=0) * 100

# Plot topik yang paling dinamis
top_dynamic = topic_time_pct.std().sort_values(ascending=False).head(6).index
topic_time_pct[top_dynamic].plot(figsize=(14, 6))
plt.title('Evolusi Topik MBG per Bulan (% share)')
plt.xlabel('Bulan')
plt.ylabel('% dari total tweet bulan itu')
plt.legend(title='Topic ID')
plt.tight_layout()
plt.savefig(f"{fig_topic_dir}/topic_evolution_time.png", dpi=150)
plt.show()
```

**[P4-09] Save topic model & update mbg_analyzed.csv**
```python
import pickle

# Simpan model untuk reuse di Fase 5 & 6
with open("../Data/Cache/bertopic_model.pkl", "wb") as f:
    pickle.dump(topic_model, f)

# Update master CSV
df.to_csv("../Data/Analyzed/mbg_analyzed.csv", index=False)
print(f"✅ Tersimpan. Kolom baru: bert_topic, bert_topic_name, frame_bert")
print(f"   Topic -1 (noise): {(df['bert_topic']==-1).sum()} ({(df['bert_topic']==-1).mean()*100:.1f}%)")
```

---

## FASE 5 — Network Analysis (`05_NetworkAnalysis.ipynb`) ← BELUM MULAI

> **Dependensi:** Fase 4 harus selesai dulu (butuh `bert_topic` di df)

### Yang Perlu Dibangun

**[P5-01] Bangun graph dari 3 jenis relasi**
```python
import networkx as nx

# A. Reply graph: user A reply ke user B
G_reply = nx.DiGraph()
for _, row in df[df['in_reply_to_screen_name'].notna()].iterrows():
    G_reply.add_edge(
        row['username'],
        row['in_reply_to_screen_name'],
        weight=row['view_count'],
        sentiment=row['sentiment_bert'],
        frame=row['frame_utama'],
        topic=row['bert_topic']
    )

# B. Mention graph: user A mention user B
for _, row in df[df['mention_count'] > 0].iterrows():
    for mentioned in row['mention_usernames']:
        G_mention.add_edge(row['username'], mentioned, weight=row['view_count'])

# C. Retweet graph (kalau ada kolom retweeted_user — cek dari quoted_username)
```

**[P5-02] Metrics per node (user)**
```python
# Centrality
degree_centrality     = nx.degree_centrality(G_reply)
betweenness           = nx.betweenness_centrality(G_reply, weight='weight')
in_degree             = dict(G_reply.in_degree())   # siapa yang paling banyak di-reply

# Gabungkan ke dataframe user
user_metrics = pd.DataFrame({
    'degree_centrality': degree_centrality,
    'betweenness': betweenness,
    'in_degree': in_degree
}).sort_values('betweenness', ascending=False)
```

**[P5-03] Community detection (Louvain)**
```python
import community as community_louvain  # pip install python-louvain

G_undirected = G_reply.to_undirected()
partition    = community_louvain.best_partition(G_undirected)
nx.set_node_attributes(G_reply, partition, 'community')

print(f"Jumlah komunitas: {len(set(partition.values()))}")

# Profil tiap komunitas: sentimen dominan, frame dominan, ada buzzer?
```

**[P5-04] Buzzer cluster check**
Hipotesis: akun `buzzer_suspect` akan membentuk cluster tersendiri yang lebih
terkoneksi satu sama lain (koordinasi), bukan tersebar acak di seluruh graph.
```python
buzzer_nodes = set(df[df['buzzer_suspect'] == True]['username'])

# Berapa buzzer yang berada di komunitas yang sama vs tersebar?
buzzer_communities = {node: partition[node] for node in buzzer_nodes if node in partition}
from collections import Counter
print("Distribusi buzzer per komunitas:", Counter(buzzer_communities.values()))
```

**[P5-05] Identifikasi influencer per frame**
```python
# Top 5 influencer per frame (berdasarkan betweenness + follower count)
df_with_metrics = df.merge(
    pd.DataFrame(betweenness.items(), columns=['username', 'betweenness']),
    on='username', how='left'
)
df_with_metrics['influence_score'] = (
    df_with_metrics['user_followers_count'].clip(upper=100_000) / 100_000 * 0.4 +
    df_with_metrics['view_count'].clip(upper=50_000) / 50_000 * 0.4 +
    df_with_metrics['betweenness'].fillna(0) * 0.2
)

top_per_frame = (
    df_with_metrics.groupby(['frame_utama', 'username'])
    .agg(avg_influence=('influence_score', 'mean'),
         total_views=('view_count', 'sum'),
         tweet_count=('tweet_id', 'count'),
         sentiment=('sentiment_bert', lambda x: x.mode()[0]))
    .reset_index()
    .sort_values(['frame_utama', 'avg_influence'], ascending=[True, False])
    .groupby('frame_utama').head(5)
)
```

---

## FASE 6 — Polarization Measurement (`06_Polarization.ipynb`) ← BELUM MULAI

> **Dependensi:** Fase 4 (score_positif/score_negatif) + Fase 5 (community structure)

### Yang Perlu Dibangun

**[P6-01] Controversy score per tweet**
```python
# Tweet yang "kontroversial" = model ragu-ragu antara positif & negatif
# Gunakan score_positif & score_negatif dari P3-01
df['controversy_score'] = 1 - abs(df['score_positif'] - df['score_negatif'])
# Score mendekati 1 = sangat kontroversial (model hampir 50:50)
# Score mendekati 0 = tweet jelas positif atau jelas negatif
```

**[P6-02] Frame-level polarization index**
```python
# Polarization index per frame: seberapa terbagi opini di frame itu
def polarization_index(group):
    pos = (group == 'positif').mean()
    neg = (group == 'negatif').mean()
    # 0 = semua satu suara, 1 = tepat 50:50
    return 1 - abs(pos - neg)

frame_polarization = df.groupby('frame_utama')['sentiment_bert'].apply(
    polarization_index
).sort_values(ascending=False)
print("Frame paling terpolarisasi:")
print(frame_polarization)
```

**[P6-03] Network-based polarization (RWC — Random Walk Controversy)**
Pendekatan yang lebih sophisticated: kalau dua komunitas di graph hampir tidak
pernah berinteraksi → polarized. Perlu hasil community detection dari Fase 5.
```python
# Hitung cross-community edge ratio
community_A = {n for n, c in partition.items() if c == 0}
community_B = {n for n, c in partition.items() if c == 1}

cross_edges   = sum(1 for u, v in G_reply.edges() if
                    (u in community_A and v in community_B) or
                    (u in community_B and v in community_A))
total_edges   = G_reply.number_of_edges()
isolation_idx = 1 - (cross_edges / total_edges)
print(f"Isolation index: {isolation_idx:.3f}")
# Mendekati 1 = komunitas tidak saling bicara = sangat terpolarisasi
```

**[P6-04] Korelasi polarization dengan topic & sentiment**

---

## FASE 7 — Visualisasi & Storytelling (`07_Visualization.ipynb`) ← BELUM MULAI

> **Dependensi:** Semua fase sebelumnya harus selesai

### Yang Perlu Dibangun

**[P7-01] Dashboard HTML interaktif (Plotly)**
```python
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Layout: 4 panel utama
# Panel 1: Sentiment donut (raw vs view-weighted) — "Siapa yang didengar?"
# Panel 2: Frame bubble chart (x=sentiment_negatif%, y=total_views, size=count)
# Panel 3: Topic evolution time series — "Narasi yang bergeser"
# Panel 4: Network graph teaser — "Siapa yang menggerakkan"
```

**[P7-02] Network visualization (gephi-style atau pyvis)**
```python
from pyvis.network import Network  # pip install pyvis

net = Network(height="600px", width="100%", directed=True)
# Warnai node berdasarkan komunitas
# Ukuran node = betweenness centrality
# Warna edge = sentimen tweet
```

**[P7-03] Narasi akhir — struktur cerita untuk slide presentasi**

| Slide | Pesan Utama | Data Pendukung |
|-------|-------------|----------------|
| 1 | "MBG didominasi percakapan negatif, tapi yang paling didengar berbeda" | View-weighted vs raw sentiment |
| 2 | "Dua alarm merah: keracunan dan korupsi anggaran" | Frame keamanan_pangan (73% negatif), anggaran (58.8% negatif) |
| 3 | "Ada pendukung genuinе, tapi publik tahu ada buzzer" | Buzzer detection + frame dukungan_umum |
| 4 | "Papua: resistensi lokal yang tak tertangkap survey nasional" | Topic 6 BERTopic |
| 5 | "Diskusi terpolarisasi: dua komunitas nyaris tidak berinteraksi" | RWC score Fase 6 |
| 6 | "Rekomendasi: komunikasi krisis keamanan pangan harus jadi prioritas" | Actionable insight |

---

## DEPENDENCY GRAPH

```
01_Preprocessing
    └──► 02_EDA
    └──► 03_Sentiment  ──► [score_positif/negatif, buzzer_suspect, user_type]
              └──────────► 04_TopicModeling  ──► [bert_topic, frame_bert]
                                    └───────────► 05_NetworkAnalysis  ──► [community, betweenness]
                                                          └────────────► 06_Polarization
                                                                               └──────► 07_Visualization
```

---

## KOLOM FINAL DI `mbg_analyzed.csv`

Setelah semua fase selesai, kolom yang harus ada:

| Kolom | Dibuat di | Keterangan |
|-------|-----------|------------|
| `clean_text` | Fase 1 | Teks sudah preprocessing |
| `created_at_dt` | Fase 1 | Datetime tweet |
| `user_created_at_dt` | Fase 1 | Datetime akun dibuat |
| `view_count` | Fase 1 | Jumlah tayangan tweet |
| `mention_usernames` | Fase 1 | List username yang di-mention |
| `mention_count` | Fase 1 | Jumlah mention |
| `sentiment_vader` | Fase 3 | Label VADER (baseline) |
| `sentiment_bert` | Fase 3 | Label IndoBERT (final) |
| `sentiment_bert_score` | Fase 3 | Confidence label menang |
| `score_positif` | Fase 3 🆕 | Skor kelas positif IndoBERT |
| `score_negatif` | Fase 3 🆕 | Skor kelas negatif IndoBERT |
| `score_netral` | Fase 3 🆕 | Skor kelas netral IndoBERT |
| `frame_utama` | Fase 3 | Frame keyword-based |
| `all_frames` | Fase 3 | Semua frame yang relevan |
| `keamanan_subframe` | Fase 3 🆕 | insiden_keracunan / kualitas_menu |
| `user_type` | Fase 3 🆕 | Profil user dari bio |
| `buzzer_suspect` | Fase 3 🆕 | Boolean flag |
| `controversy_score` | Fase 6 🆕 | Tingkat kontroversi per tweet |
| `bert_topic` | Fase 4 🆕 | Topic ID dari BERTopic |
| `bert_topic_name` | Fase 4 🆕 | Nama topik dari BERTopic |
| `frame_bert` | Fase 4 🆕 | Mapping topic → frame |
| `community` | Fase 5 🆕 | Komunitas dari Louvain |
| `betweenness` | Fase 5 🆕 | Betweenness centrality user |
| `influence_score` | Fase 5 🆕 | Composite influence score |

---

## IMMEDIATE NEXT ACTION

```
HARI INI:
  1. [ ] Selesaikan cell terakhir 04_TopicModeling.ipynb (P4-01 s/d P4-09)
  2. [ ] Jalankan P3-01 (return_all_scores) — rerun IndoBERT, simpan ulang mbg_analyzed.csv
  3. [ ] Jalankan P3-04 (buzzer_suspect + user_type) — kolom ini dibutuhkan di P4-05

BESOK:
  4. [ ] Buat 05_NetworkAnalysis.ipynb, mulai dari P5-01 (bangun graph)
  5. [ ] Selesaikan P4-08 (temporal topic evolution) — bagus untuk slide presentasi

LUSA:
  6. [ ] Buat 06_Polarization.ipynb (P6-01 s/d P6-03)
  7. [ ] Mulai P7-03 — outline narasi slide dulu sebelum koding visualisasi
```
