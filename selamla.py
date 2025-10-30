from sentence_transformers import SentenceTransformer, util

# --- 1. MODEL VE BİLGİ TABANI ---

# Cümleleri sayısal vektörlere dönüştüren embedding modeli
embed_model = SentenceTransformer("all-MiniLM-L6-v2")

# Küçük bir bilgi tabanı (örnek metinler)
knowledge_base = [
    "İstanbul Türkiye'nin en kalabalık şehridir.",
    "Ay dünyadan yaklaşık 384.000 kilometre uzaktadır.",
    "Yapay zeka, verilerden öğrenebilen bilgisayar sistemlerini tanımlar.",
    "Boğaziçi Köprüsü 15 Temmuz Şehitler Köprüsü olarak da bilinir."
]

# --- 2. KULLANICI SORUSU ---
question = input("Soru gir: ")

# --- 3. EN İLGİLİ BİLGİYİ BUL ---
question_embedding = embed_model.encode(question, convert_to_tensor=True)
knowledge_embeddings = embed_model.encode(knowledge_base, convert_to_tensor=True)

cosine_scores = util.pytorch_cos_sim(question_embedding, knowledge_embeddings)
best_match_idx = cosine_scores.argmax().item()

retrieved_text = knowledge_base[best_match_idx]

print(f"\n🔍 En alakalı bilgi: {retrieved_text}")

# --- 4. BASİT CEVAP ÜRETİCİ ---
# Basit bir mantık: bulunan bilgiyi cümleyle birleştiriyoruz
if "kim" in question.lower():
    answer = f"{retrieved_text} Bu bilgi, sorduğun kişiyle ilgili olabilir."
elif "ne" in question.lower() or "nedir" in question.lower():
    answer = f"{retrieved_text} Bu, sorunun cevabı olabilir."
else:
    answer = f"Soruna göre en alakalı bilgi: {retrieved_text}"

print(f"\n💬 Cevap: {answer}")











pip install numpy
pip install scikit-learn
pip install sentence-transformers
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity # Benzerlik hesaplaması için

# --- 1. Dizin Oluşturma Aşaması ---

# 1.1. Veri Kaynağı (Basitleştirilmiş)
documents = [
    "Retriever-Augmented Generation (RAG) 2020'de tanıtıldı.",
    "Büyük Dil Modelleri (LLM) eğitim verileri dışındaki bilgilere erişmek için RAG kullanır.",
    "Vektör veritabanı, RAG sistemlerinin önemli bir bileşenidir.",
    "İstanbul Türkiye'nin en kalabalık şehridir.",
    "RAG, halüsinasyonları azaltmaya yardımcı olabilir."
]

# 1.2. Gömme Modeli Yükleme
# Gerçek uygulamada daha güçlü modeller kullanılır (Örn: all-MiniLM-L6-v2)
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

# 1.3. Verileri Gömme ve Vektör Veritabanına Depolama
doc_embeddings = embedding_model.encode(documents)
print(f"Toplam {len(doc_embeddings)} belge parçası gömüldü.")

# Vektör veritabanı (Basit NumPy dizisi)
vector_db = {
    "texts": documents,
    "vectors": doc_embeddings
}

# --- 2. Geri Alma ve Üretim Aşaması ---

def get_rag_response(query, vector_db, top_k=2):
    # 2.1. Sorgu Gömme
    query_embedding = embedding_model.encode([query])

    # 2.2. İlgili Bilgiyi Geri Alma (Benzerlik Araması)
    # Kosinüs Benzerliği hesaplama
    similarities = cosine_similarity(query_embedding, vector_db["vectors"])[0]

    # En yüksek benzerliğe sahip k-adet belgeyi bulma
    top_k_indices = np.argsort(similarities)[::-1][:top_k]
    retrieved_chunks = [vector_db["texts"][i] for i in top_k_indices]
    
    # Geri Alınan Bağlamı gösterme
    print("\n--- Geri Alınan Bağlam (Context) ---")
    for chunk in retrieved_chunks:
        print(f"- {chunk}")
    print("----------------------------------\n")

    # 2.3. İstem Artırma
    context = "\n".join(retrieved_chunks)
    
    # LLM'ye gönderilecek nihai istem
    augmented_prompt = f"""Aşağıdaki bağlamı kullanarak soruyu yanıtlayın:
    
    Bağlam: 
    ---
    {context}
    ---
    
    Soru: {query}
    
    Yanıtınızı yalnızca verilen bağlamdaki bilgilere dayanarak oluşturun.
    """

    # 2.4. Yanıt Üretme (Gerçek LLM çağrısı yerine basit bir yer tutucu)
    # Gerçek uygulamada burası OpenAI, Gemini, HuggingFace LLM API çağrısı olacaktır.
    
    if "RAG" in context or "LLM" in context:
        generated_response = f"LLM Yanıtı: (Bağlamı kullanarak üretildi) RAG'ın avantajları, halüsinasyonları azaltması ve LLM'lere güncel veya özel bilgi sağlamasıdır. Örneğin, '{retrieved_chunks[0]}' cümlesi bir örnektir."
    else:
        generated_response = "LLM Yanıtı: (Bağlamda yeterli bilgi yoktu) Üzgünüm, verilen bağlamda bu soruya yanıt verecek yeterli bilgi bulunmamaktadır."
        
    print(f"--- LLM'ye Gönderilen Artırılmış İstem (İlk Kısım) ---\n{augmented_prompt[:250]}...\n----------------------------------")
    return generated_response

# --- Örnek Kullanım ---
user_query = "RAG kullanmanın faydaları nelerdir?"
response = get_rag_response(user_query, vector_db)
print("Nihai Yanıt:", response)

print("\n--- Alakasız Sorgu Örneği ---")
user_query_2 = "Türkiye'nin başkenti neresidir?"
response_2 = get_rag_response(user_query_2, vector_db)
print("Nihai Yanıt:", response_2)


