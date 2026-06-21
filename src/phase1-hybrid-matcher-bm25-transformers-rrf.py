import pandas as pd
import numpy as np
import torch
from sentence_transformers import SentenceTransformer, CrossEncoder
from rank_bm25 import BM25Okapi
import faiss
from typing import List, Dict, Tuple, Any
from tqdm import tqdm
import logging
import sys
import re
from collections import defaultdict

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - [%(levelname)s] - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

class HybridEntityMatcher:
    """
    Modüler Varlık Çözümleme ve Eşleştirme Motoru (RRF + Cross-Encoder Reranking).
    
    Özellikler:
    - Sparse Search: BM25
    - Dense Search: BAAI/bge-m3 (Embedding)
    - Fusion: Reciprocal Rank Fusion (RRF)
    - Reranking: BAAI/bge-reranker-v2-m3 (Batch Inference optimize)
    """
    
    def __init__(self, embedding_model_name: str = "BAAI/bge-m3", 
                 reranker_model_name: str = "BAAI/bge-reranker-v2-m3",
                 use_gpu: bool = True):
        
        self.device = "cpu"
        if use_gpu:
            if torch.cuda.is_available():
                self.device = "cuda"
            elif torch.backends.mps.is_available(): 
                self.device = "mps"
        
        logger.info(f"Hesaplama Cihazı: {self.device}")

        # 1. Embedding Modeli
        logger.info(f"Embedding modeli yükleniyor: {embedding_model_name}")
        self.embedder = SentenceTransformer(embedding_model_name, device=self.device)
        
        # 2. Reranker Modeli
        logger.info(f"Reranker modeli yükleniyor: {reranker_model_name}")
        self.reranker = CrossEncoder(reranker_model_name, device=self.device)
        
        # State Değişkenleri
        self.corpus_df = None
        self.corpus_texts = []
        self.bm25 = None
        self.faiss_index = None
        self.corpus_embeddings = None
        
    def _preprocess(self, text: str) -> str:
        """
        Minimalist temizlik stratejisi (Final Karar #1).
        
        Kurallar:
        1. Lowercase yap.
        2. Tireleri (-) boşluğa çevir (S-23 -> S 23).
        3. Ondalık sayıları (1.5) KORU.
        4. Diğer noktalama işaretlerini sil.
        5. Birimleri (lt, ml) ASLA değiştirme/standartlaştırma.
        """
        if not isinstance(text, str):
            return ""
        
        text = text.lower()
        text = text.replace("-", " ")
        
        # Regex: Harf, Sayı, Boşluk ve Nokta (.) dışındaki her şeyi sil.
        # Noktayı koruyoruz çünkü "1.5" ile "15" farklıdır.
        #text = re.sub(r'[^a-z0-9\.\s]', '', text)
        text = re.sub(r'[^a-zçğıöşü0-9\.\s]', '', text)
        # Çoklu boşlukları temizle
        return " ".join(text.split())

    def _tokenize(self, text: str) -> List[str]:
        """BM25 için tokenization."""
        return self._preprocess(text).split()

    def fit(self, df_corpus: pd.DataFrame, text_column: str, code_column: str):
        """Ürün havuzunu (Corpus) indeksler."""
        logger.info("Corpus indeksleniyor...")
        
        self.corpus_df = df_corpus.copy()
        # Null kontrolü
        self.corpus_df[text_column] = self.corpus_df[text_column].fillna("")
        self.corpus_texts = self.corpus_df[text_column].tolist()
        
        # 1. BM25 İndeksi
        logger.info("BM25 indeksi oluşturuluyor...")
        tokenized_corpus = [self._tokenize(doc) for doc in self.corpus_texts]
        self.bm25 = BM25Okapi(tokenized_corpus)
        
        # 2. FAISS (Vektör) İndeksi
        logger.info("Vektör embeddingleri oluşturuluyor...")
        self.corpus_embeddings = self.embedder.encode(
            self.corpus_texts, 
            batch_size=32, # GPU memoryye göre artırılabilir
            show_progress_bar=True, 
            convert_to_numpy=True,
            normalize_embeddings=True 
        )
        
        d = self.corpus_embeddings.shape[1]
        self.faiss_index = faiss.IndexFlatIP(d) # Inner Product (Cosine Sim. ile aynı çünkü normalize ettik)
        
        if self.device == "cuda":
            res = faiss.StandardGpuResources()
            self.faiss_index = faiss.index_cpu_to_gpu(res, 0, self.faiss_index)
            
        self.faiss_index.add(self.corpus_embeddings)
        logger.info(f"İndeksleme tamamlandı. Ürün sayısı: {len(self.corpus_texts)}")

    def _retrieve_candidates_rrf(self, query: str, top_k: int = 25, rrf_k: int = 60) -> List[int]:
        """
        Tek bir sorgu için BM25 ve FAISS sonuçlarını RRF (Reciprocal Rank Fusion) ile birleştirir.
        (Final Karar #2)
        """
        clean_query = self._preprocess(query)
        
        #  BM25 Search 
        tokenized_query = self._tokenize(clean_query)
        bm25_scores = self.bm25.get_scores(tokenized_query)
        # En yüksek skordan en düşüğe sırala ve indeksleri al
        bm25_indices = np.argsort(bm25_scores)[::-1][:top_k]
        
        #  FAISS Search 
        query_embedding = self.embedder.encode([clean_query], convert_to_numpy=True, normalize_embeddings=True)
        _, faiss_indices = self.faiss_index.search(query_embedding, top_k)
        faiss_indices = faiss_indices[0]
        
        #  RRF Fusion 
        # Formül: Score = sum(1 / (k + rank))
        rrf_scores = defaultdict(float)
        
        # BM25 Ranks
        for rank, idx in enumerate(bm25_indices):
            rrf_scores[idx] += 1 / (rrf_k + rank + 1)
            
        # FAISS Ranks
        for rank, idx in enumerate(faiss_indices):
            rrf_scores[idx] += 1 / (rrf_k + rank + 1)
            
        # RRF skoruna göre sırala ve top_k döndür
        sorted_candidates = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
        final_indices = [idx for idx, score in sorted_candidates[:top_k]]
        
        return final_indices

    def batch_process_optimized(self, query_list: List[str], output_file: str, top_n: int = 3):
        """
        Batch Inference Mimarisi (Final Karar #3).
        
        Adımlar:
        1. Tüm sorgular için RRF ile adayları topla.
        2. Tek bir devasa (Query, Document) listesi oluştur.
        3. Reranker'a batch olarak gönder (GPU utilization maksimizasyonu).
        4. Sonuçları birleştir ve analiz et.
        """
        logger.info(f"Batch işlem başlıyor. Toplam Sorgu: {len(query_list)}")
        
        # --- Adım 1: Tüm Adayları Topla (Retrieval Phase) ---
        # Bu kısımda her sorgu için "Hangi dokümanlarla kıyaslanacak" listesi çıkarılır.
        all_pairs = [] # [[query, doc_text], ...]
        query_mapping = [] # Hangi pair hangi query'ye ait?
        
        # Boş sorguları filtrele
        valid_queries = [(i, q) for i, q in enumerate(query_list) if isinstance(q, str) and q.strip()]
        
        logger.info("Adım 1/3: Adaylar Belirleniyor (Retrieval + RRF)...")
        for q_idx, query_text in tqdm(valid_queries):
            candidate_indices = self._retrieve_candidates_rrf(query_text, top_k=25)
            
            for doc_idx in candidate_indices:
                doc_text = self.corpus_texts[doc_idx]
                all_pairs.append([query_text, doc_text])
                
                # Meta veriyi sakla: (Orijinal Sorgu Indexi, Bulunan Ürün Indexi)
                query_mapping.append({
                    "query_idx": q_idx,
                    "doc_idx": doc_idx,
                    "query_text": query_text
                })
        
        if not all_pairs:
            logger.warning("Hiçbir aday bulunamadı.")
            return

        #  Adım 2: Toplu Reranking (Inference Phase) 
        logger.info(f"Adım 2/3: Reranking (Cross-Encoder) - Toplam {len(all_pairs)} çift puanlanacak...")
        
        # Batch size GPU VRAM'e göre 64, 128, 256 yapılabilir.
        # Sigmoid YOK. Ham Logits dönecek (Final Karar #4).
        scores = self.reranker.predict(
            all_pairs, 
            batch_size=64, 
            show_progress_bar=True,
            activation_fn=None # Sigmoid iptal, Raw Logits istiyoruz.
        )
        
        #  Adım 3: Sonuçları Geri Dağıt (Regrouping) 
        logger.info("Adım 3/3: Sonuçlar derleniyor...")
        
        # Sonuçları sorgu bazında grupla
        # structure: {query_idx: [ (score, doc_idx), ... ]}
        grouped_results = defaultdict(list)
        for i, score in enumerate(scores):
            meta = query_mapping[i]
            q_idx = meta["query_idx"]
            d_idx = meta["doc_idx"]
            grouped_results[q_idx].append((score, d_idx))
            
        # Final DataFrame Hazırlığı
        final_rows = []
        all_top1_scores = [] # Analiz için sakla
        all_gap_scores = []  # Gap analizi için sakla

        for i, query_text in enumerate(query_list):
            row = {"source_text": query_text}
            
            if i in grouped_results:
                # Skoruna göre sırala (Büyükten küçüğe)
                matches = sorted(grouped_results[i], key=lambda x: x[0], reverse=True)[:top_n]
                
                # İstatistik toplama (Analiz için)
                if matches:
                    all_top1_scores.append(matches[0][0])
                    if len(matches) > 1:
                        all_gap_scores.append(matches[0][0] - matches[1][0])
                
                for rank, (score, doc_idx) in enumerate(matches):
                    r_num = rank + 1
                    match_data = self.corpus_df.iloc[doc_idx]
                    
                    row[f"match_{r_num}_text"] = match_data["candidate_text"] 
                    row[f"match_{r_num}_id"] = match_data["entity_id"]
                    # Ham skor yazılıyor (örn: 4.12)
                    row[f"match_{r_num}_score_logit"] = round(float(score), 4)
            
            final_rows.append(row)
            
        # Excel'e kaydet
        df_out = pd.DataFrame(final_rows)
        df_out.to_excel(output_file, index=False)
        logger.info(f"Dosya kaydedildi: {output_file}")
        
        # --- Adım 4: Threshold Analizi (Final Karar #5) ---
        self._print_threshold_analysis(all_top1_scores, all_gap_scores)

    def _print_threshold_analysis(self, top1_scores: List[float], gap_scores: List[float]):
        """
        Kullanıcıya eşik değeri önerisi sunan analiz modülü.
        """
        if not top1_scores:
            return

        logger.info("="*50)
        logger.info("MÜHENDİSLİK ANALİZİ & THRESHOLD ÖNERİSİ")
        logger.info("="*50)
        
        scores = np.array(top1_scores)
        p90 = np.percentile(scores, 90) # En iyi %10
        p50 = np.percentile(scores, 50) # Medyan
        p10 = np.percentile(scores, 10) # En kötü %10 (ama yine de 1. sırada gelenler)
        
        logger.info(f"Toplam Eşleşme Sayısı: {len(scores)}")
        logger.info(f"Skor Dağılımı (Raw Logits):")
        logger.info(f"  - Maksimum Skor (En güvenli): {np.max(scores):.2f}")
        logger.info(f"  - Medyan Skor (Ortalama güven): {p50:.2f}")
        logger.info(f"  - P10 Skoru (Zayıf eşleşmeler sınırı): {p10:.2f}")
        
        logger.info("-" * 30)
        
        # Öneri Mantığı
        # Genelde Reranker'larda 0.0 altı negatif, 3.0 üstü çok pozitiftir.
        suggested_threshold = max(0.0, p10) # En az 0 olsun, veya P10 olsun.
        
        logger.info(f"ÖNERİLEN THRESHOLD STRATEJİSİ:")
        logger.info(f"1. Güvenli Bölge (> 3.0): Otomatik onaylanabilir.")
        logger.info(f"2. İnceleme Bölgesi ({suggested_threshold:.2f} - 3.0): İnsan kontrolü gerekir.")
        logger.info(f"3. Red Bölgesi (< {suggested_threshold:.2f}): Muhtemelen yanlış eşleşme.")
        
        if gap_scores:
            avg_gap = np.mean(gap_scores)
            logger.info(f"Ortalama Confidence Gap (1. ve 2. aday farkı): {avg_gap:.2f}")
            if avg_gap < 1.0:
                logger.warning("UYARI: Model 1. ve 2. ürün arasında kararsız kalıyor (Gap < 1.0). Veri temizliği gerekebilir.")
        
        logger.info("="*50)

# main
if __name__ == "__main__":
    
    # Dosya yolları (Kendi ortamına göre düzenle)
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    INPUT_PATH = os.path.join(BASE_DIR, "data", "source_data.xlsx")
    OUTPUT_PATH = os.path.join(BASE_DIR, "data", "phase1_output.xlsx")
    
    try:
        # Veriyi Oku
        logger.info("Veri okunuyor...")
        df = pd.read_excel(INPUT_PATH)
        
        # Katalog (DB) Hazırlığı
        catalog_df = df[["candidate_text", "entity_id"]].dropna(subset=["candidate_text"]).drop_duplicates(subset=["candidate_text"]).reset_index(drop=True)
        
        # Sorgu Listesi
        queries = df["source_text"].tolist()
        
        if len(catalog_df) == 0:
            logger.error("Katalog boş!")
        else:
            # Sınıfı Başlat
            matcher = HybridEntityMatcher()
            
            # 1. İndeksle
            matcher.fit(catalog_df, text_column="candidate_text", code_column="entity_id")
            
            # 2. Batch İşlem Başlat (Optimize Edilmiş Fonksiyon)
            matcher.batch_process_optimized(queries, OUTPUT_PATH)
            
    except Exception as e:
        logger.error(f"Kritik Hata: {e}")
        import traceback
        traceback.print_exc()