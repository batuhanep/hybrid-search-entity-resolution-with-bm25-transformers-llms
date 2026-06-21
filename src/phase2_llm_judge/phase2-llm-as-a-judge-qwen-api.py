import pandas as pd
import requests
import json
import logging
import sys
from typing import Dict, Any, Optional
from tenacity import retry, stop_after_attempt, wait_exponential
from tqdm import tqdm
import os
from dotenv import load_dotenv
load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - [%(levelname)s] - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

class CloudLLMJudge:
    """
    OpenRouter API üzerinden bulut tabanlı LLM'i kullanarak ürün eşleşmelerini 
    değerlendiren Hakem (Judge) sınıfı.
    """
    def __init__(
        self, 
        model_name: str = "qwen/qwen-2.5-72b-instruct", 
        base_url: str = "https://openrouter.ai/api/v1/chat/completions"
):
        """
        Sınıf başlatıcı. OpenRouter kimlik doğrulama ve model ayarlarını yapılandırır.
        
        Args:
            model_name (str): OpenRouter üzerinde çalışan hedef modelin adı.
            base_url (str): OpenRouter API endpoint adresi.
            api_key (str): OpenRouter API erişim anahtarı.
        """
        self.model_name = model_name
        self.generate_url = base_url
        self.api_key = os.getenv("OPENROUTER_API_KEY")
        if not self.api_key:
            raise ValueError("Kritik Güvenlik Hatası: OPENROUTER_API_KEY bulunamadı. .env dosyasını kontrol edin.")
        
        self.system_prompt = """Sen gürültülü veri setleri üzerinde çalışan bir Veri Eşleştirme (Entity Resolution) sistemisin.
Temel görevin, sunulan metin kayıtları arasındaki 'Yanlış Pozitif' (farklı varlıkları aynı kabul etme) eşleşmeleri kesin olarak engellemektir.
Sana girdi olarak bir 'Kaynak Kayıt (Source Entity)' ve bu kayıtla eşleşme ihtimali olan 3 adet 'Aday Kayıt (Candidate Entities)' verilecektir.

Aday kayıtların, Kaynak Kayıt ile yapısal ve anlamsal olarak 'Birebir Aynı Varlık' (Exact Match) olup olmadığını doğrula. 
Metin içindeki nümerik değerler, kapasite göstergeleri veya birim farklılıkları varlıkların farklı olduğunu gösterir; bu tür durumlarda eşleşmeyi reddet.

YALNIZCA AŞAĞIDAKİ JSON ŞEMASINI DÖNDÜR, BAŞKA HİÇBİR METİN EKLEME:

{
  "reasoning": "Kararın arkasındaki anlamsal gerekçenin kısa özeti.",
  "best_match_index": 1,
  "is_exact_match": true,
  "needs_human_review": false
}

Eğer adayların hiçbiri kaynak kayıt ile birebir aynı varlığı temsil etmiyorsa şemayı şu şekilde doldur:

{
  "reasoning": "Eşleşme bulunamama gerekçesi.",
  "best_match_index": null,
  "is_exact_match": false,
  "needs_human_review": true
}
"""

    def _clean_json_output(self, raw_text: str) -> str:
        """
        LLM'in döneceği metni temizleyerek geçerli bir JSON formatına dönüştürür.
        Modellerin Markdown blokları eklemesine karşı defansif bir kontrol mekanizmasıdır.
        """
        cleaned = raw_text.replace("```json", "").replace("```", "").strip()
        start_idx = cleaned.find("{")
        end_idx = cleaned.rfind("}")
        
        if start_idx != -1 and end_idx != -1 and end_idx >= start_idx:
            return cleaned[start_idx:end_idx+1]
        return cleaned

    @retry(
        stop=stop_after_attempt(3), 
        wait=wait_exponential(multiplier=1, min=2, max=10),
        reraise=False
    )
    def evaluate_candidates(self, main_prod: str, candidates: list) -> Optional[Dict[str, Any]]:
        """
        Ana ürünü ve adayları alıp OpenRouter API'sine OpenAI uyumlu formatta gönderir.
        
        Args:
            main_prod (str): Orijinal ürün metni.
            candidates (list): 3 aday ürünün metinlerinden oluşan liste.
            
        Returns:
            dict: JSON formatında LLM kararı. Hata durumunda None döner.
        """
        prompt = f"Ana Ürün: '{main_prod}'\n\n"
        for i, cand in enumerate(candidates, 1):
            prompt += f"Aday {i}: '{cand}'\n"
        
        prompt += "\nLütfen JSON formatında kararını ver."

        # OpenRouter için OpenAI uyumlu Chat Completions payload yapısı
        payload = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.0,
            "max_tokens": 100,
            "response_format": {"type": "json_object"}  # API seviyesinde JSON çıktısını zorunlu kılar
        }

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        try:
            response = requests.post(self.generate_url, json=payload, headers=headers, timeout=120)
            response.raise_for_status()
            
            # OpenAI standartlarına uygun olarak JSON yanıtının çözülmesi
            response_json = response.json()
            raw_response = response_json.get("choices", [{}])[0].get("message", {}).get("content", "")
            
            clean_json_str = self._clean_json_output(raw_response)
            decision = json.loads(clean_json_str)
            return decision

        except json.JSONDecodeError as e:
            logger.error(f"JSON Parse Hatası. LLM düzgün JSON dönmedi: {e}\nHam Çıktı: {raw_response}")
            return None
        except requests.exceptions.RequestException as e:
            logger.error(f"OpenRouter bağlantı hatası: {e}. Ağ bağlantısını kontrol edin.")
            raise  # Hata durumunda hata fırlatılarak tenacity retry mekanizması tetiklenir

def process_phase_2(input_excel: str, output_excel: str):
    """
    Faz 1'den çıkan veriyi okuyup Faz 2 (LLM) işlemlerini yürütür ve kaydeder.
    """
    logger.info(f"Faz 1 sonuçları okunuyor: {input_excel}")
    try:
        df = pd.read_excel(input_excel)
    except FileNotFoundError:
        logger.error(f"Dosya bulunamadı: {input_excel}")
        return

    judge = CloudLLMJudge()
    results = []

    for row in tqdm(df.itertuples(), total=len(df), desc="LLM Hakem Değerlendiriyor"):
        main_product = str(getattr(row, "source_text", ""))
        
        # adayları topla
        candidates = [
            str(getattr(row, "match_1_text", "")),
            str(getattr(row, "match_2_text", "")),
            str(getattr(row, "match_3_text", ""))
        ]
        
        candidates = [c if c.lower() != "nan" else "Bulunamadı" for c in candidates]

        # LLM'e değerlendirt
        decision = judge.evaluate_candidates(main_product, candidates)
        
        if decision:
            # LLM'in verdiği 1, 2, 3 indeksi, o adayın tam metni ve skoru
            best_idx = decision.get("best_match_index")
            llm_best_product = None
            llm_best_code = None
            
            if best_idx and str(best_idx).isdigit() and 1 <= int(best_idx) <= 3:
                idx = int(best_idx)
                llm_best_product = getattr(row, f"match_{idx}_text", None)
                llm_best_code = getattr(row, f"match_{idx}_id", None)

            results.append({
                "llm_best_match_index": best_idx,
                "llm_best_product_text": llm_best_product,
                "llm_best_product_code": llm_best_code,
                "llm_is_exact_match": decision.get("is_exact_match"),
                "llm_needs_human_review": decision.get("needs_human_review"),
                "llm_reasoning": decision.get("reasoning")
            })
        else:
            # Hata durumu için varsayılan değerler
            results.append({
                "llm_best_match_index": None,
                "llm_best_product_text": None,
                "llm_best_product_code": None,
                "llm_is_exact_match": False,
                "llm_needs_human_review": True,
                "llm_reasoning": "LLM API Hatası veya Parse Sorunu"
            })

    result_df = pd.DataFrame(results)
    final_df = pd.concat([df.reset_index(drop=True), result_df], axis=1)

    logger.info(f"İşlem tamamlandı. Yeni dosya kaydediliyor: {output_excel}")
    final_df.to_excel(output_excel, index=False)

if __name__ == "__main__":
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    INPUT_FILE = os.path.join(BASE_DIR, "data", "phase1_output.xlsx")
    OUTPUT_FILE = os.path.join(BASE_DIR, "data", "final_resolved_api.xlsx")
    
    process_phase_2(INPUT_FILE, OUTPUT_FILE)