# Hektor Unattended Architecture

Amaç: kullanıcı motor hesabını bir kez bağlar; bundan sonra eğitim işletimine müdahale etmez.

## Tek kontrol döngüsü

```text
Unattended Supervisor (desired-state reconciler)
├─ Motor Guardian: AutoDriver/Codex canlı mı? Değilse tek instance yeniden bağla
├─ Sentinel: servis, SQLite, eğitim, RAG ve orkestrasyon sağlık sinyalleri
├─ Self-Healing Controller: allow-list runbook ile teşhis → tamir → doğrulama
├─ Auto-LoRA: kalite gate → detached eğitim → eval → güvenli terfi
└─ RAG Learning Loop: eğitim kaynak kilidinde bekle, sonra hafıza döngüsüne dön
```

`UnattendedSupervisor` tek desired-state sahibidir. Uzman ajanlar birbirlerini doğurmaz;
supervisor gerçek durumu her 60 saniyede yeniden değerlendirir. Böylece restart veya motor
oturumunun bitmesi kalıcı bir durma yaratmaz.

## Değişmez güvenlik kuralları

- Aynı anda en fazla bir abonelik motoru çalışır.
- `STOP_ALL` bütün otomatik eylemlerden üstündür.
- Eğitim yetkisini yalnız `UnattendedTrainingPolicy` verir.
- Kalite/regresyon/eval kapıları başarısızsa eğitim veya terfi yapılmaz.
- Tamirler idempotent allow-list runbook'lardır; serbest LLM shell/git tamiri yoktur.
- Başarısız motor başlangıcı exponential backoff ve circuit breaker ile seyreltilir.
- Supervisor, self-heal ve eğitim durumu JSON/SQLite checkpoint'lerinde kalıcıdır.

## Dayanıklılık modeli

Bu tasarım Kubernetes controller reconciliation, health endpoint/self-healing ve circuit
breaker desenlerini izler. Uzun işlerin yan etkileri idempotent aşamalara ayrılır; tamamlanan
orkestrasyon aşamaları checkpoint sayesinde restart sonrasında tekrar çalıştırılmaz.

Kaynaklar:

- https://kubernetes.io/docs/concepts/architecture/controller/
- https://learn.microsoft.com/azure/architecture/guide/design-principles/self-healing
- https://learn.microsoft.com/azure/architecture/patterns/circuit-breaker
- https://learn.microsoft.com/azure/architecture/patterns/health-endpoint-monitoring
- https://langchain-ai.github.io/langgraph/concepts/durable_execution/
