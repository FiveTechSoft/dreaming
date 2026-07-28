# Capitolo 18: Ciò che Abbiamo Imparato

## Risultati principali

1. **TinyLlama è un microcosmo cartografabile**  
   22 layer, 9 tensori/layer, dim reali 2048 / 5632 / GQA 32×4.

2. **Un motore C proprio chiude il cerchio**  
   GGUF F16, BPE, KV-cache, OpenMP, ~6–10 tok/s,
   `--perturb` e `--steer` in runtime.

3. **I pesi contengono prospettive**  
   Non solo fatti: toni e voci. Perturbare con
   gerarchia preservata cambia la voce, non spegne il parlato.

4. **Regola d'Oro geometrica**  
   Attn → accademico; FFN → pratico; Emb → semplice.

5. **Superficie di coerenza**  
   Tangente (amplify) abitabile; normale (rumore forte) vuoto.

6. **Spazio di embedding: isole, non un asse unico**  
   Dodici aree semantiche quasi ortogonali; PCA usa
   centinaia di dimensioni; opposits non antipodali.

7. **Macrocosmo ↔ microcosmo**  
   Il metodo è andata e ritorno: senso ↔ tensore ↔ testo.

8. **Strumenti di viaggio**  
   Mappa HTML su GitHub, script di geometria, llama-cli
   per batterie Q4, motore C per orologeria fine.

## Limitazioni dello studio

- Valutazione della "prospettiva" ancora qualitativa.  
- TinyLlama ≠ modelli all'avanguardia (la superficie può cambiare).  
- Mappa 2D è proiezione, non la geometria vera.  
- Runtime F32 di perturbazione richiede molta RAM.  
- Non tutte le tecniche v10/v11 sono nel motore C.

## Domande aperte

- Dove (quali layer) si accende il clima mistico nel residuale?  
- Le direzioni di prospettiva si trasferiscono tra modelli?  
- Come misurare la prospettiva in modo automatico e affidabile?  
- Cosa succede nella superficie di coerenza a 7B / 70B?

---

*Capitolo successivo: Il Futuro dell'Esplorazione*
