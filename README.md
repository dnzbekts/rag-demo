# Rag-Demo: LangGraph ile Güçlendirilmiş RAG Uygulaması

Bu proje, yerel LLM (Ollama) ve LangGraph kullanarak geliştirilmiş, doğruluk kontrolü (Hallucination) ve alaka düzeyi (Relevance) denetimleri içeren bir RAG (Retrieval-Augmented Generation) prototipidir.

## Özellikler

- **Yerel LLM Desteği**: Ollama üzerinden `llama3.1` ve `nomic-embed-text` modellerini kullanır.
- **LangGraph İş Akışı**: Klasik zincir yapıları yerine, döngülü ve kontrollü bir graf yapısı sunar.
- **Alaka Düzeyi Kontrolü (Relevance Grader)**: Getirilen belgelerin soruyla alakalı olup olmadığını kontrol eder.
- **Halüsinasyon Denetimi (Hallucination Grader)**: Üretilen yanıtın verilen kaynaklara (bilgi.txt) dayanıp dayanmadığını doğrular.
- **Hata Yönetimi**: Eğer bilgi bulunamazsa veya yanıt kaynaklarla uyuşmazsa "Bu konuda bilgi bulamadım" yanıtını verir.

## Gereksinimler

- Python 3.10+
- [Ollama](https://ollama.com/) (llama3.1 ve nomic-embed-text modelleri indirilmiş olmalıdır)

## Kurulum

1. Projeyi klonlayın:
   ```bash
   git clone https://github.com/dnzbekts/rag-demo.git
   cd rag-demo
   ```

2. Sanal ortam oluşturun ve aktif edin:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

3. Bağımlılıkları yükleyin:
   ```bash
   pip install langgraph langchain langchain-community langchain-ollama chromadb
   ```

4. Ollama modellerini indirin (eğer henüz indirilmediyse):
   ```bash
   ollama pull llama3.1
   ollama pull nomic-embed-text
   ```

## Kullanım

Uygulamayı başlatmak için:

```bash
python3 app.py
```

`bilgi.txt` dosyasına istediğiniz bilgileri ekleyerek RAG sistemini kişiselleştirebilirsiniz.

## Akış Diyagramı

Uygulama şu adımları izler:
1. **Retrieve**: Soruyla ilgili belgeleri vektör veritabanından getirir.
2. **Grade Documents**: Belgelerin soruyla alakasını kontrol eder.
3. **Generate**: Eğer belgeler alakalıysa yanıt üretir.
4. **Check Hallucinations**: Yanıtın gerçeklere dayanıp dayanmadığını kontrol eder.
5. **Output**: Doğrulanmış yanıtı veya "bilgi bulamadım" mesajını döner.
