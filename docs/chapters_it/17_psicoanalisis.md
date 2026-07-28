# Capitolo 17: Psicoanalisi del Transformer

## Una metafora (non una diagnosi clinica)

Freud distingueva strati del mentale.
Senza forzare identità, lo stack del transformer
ammette una **lettura per profondità**:

| Istanza | Componente | Funzione approssimata |
|---------|------------|----------------------|
| **Inconscio** | Embedding | Associazioni latenti, "il già noto" senza contesto |
| **Preconscio** | Attenzione + layer medi | Porta in scena relazioni e quadri |
| **Coscienti** | Ultimi layer + logits + sample | Ciò che si dice *adesso*

## Es/Io/Super-Io (lettura libera)

| | Analogia nel modello |
|--|---------------------|
| **Es** | Impulsi di peso grezzo, direzioni semantiche grezze |
| **Io** | Residuale + norme: negozia tra impulsi e forma |
| **Super-Io** | Bias di addestramento / sicurezza / stile "corretto" del baseline |

La perturbazione `mystical` non "libera l'es" in senso freudiano:
**rimischia** l'equilibrio di voci già presenti nei pesi.

## Perché annotare questa metafora

- Aiuta a *parlare* dell'interno senza solo matrici.  
- Collega allo zoom macro↔micro (cap. 6).  
- Non sostituisce le misure: è una **mappa narrativa**.

## Limite

Un LLM non ha inconscio soggettivo.
Ha **statistica compressa**. La metafora è
strumento di esplorazione, non ontologia.

---

*Capitolo successivo: Ciò che Abbiamo Imparato*
