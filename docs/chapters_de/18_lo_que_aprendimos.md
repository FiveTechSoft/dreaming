# Kapitel 18: Was wir gelernt haben

## Hauptbefunde

1. **TinyLlama ist ein kartierbarer Mikrokosmos**  
   22 Schichten, 9 Tensoren/Schicht, tatsächliche Dims 2048 / 5632 / GQA 32×4.

2. **Eine eigene C-Engine schließt den Kreis**  
   GGUF F16, BPE, KV-Cache, OpenMP, ~6–10 tok/s,
   `--perturb` und `--steer` zur Laufzeit.

3. **Gewichte enthalten Perspektiven**  
   Nicht nur Fakten: Töne und Stimmen. Perturbation mit
   bewahrter Hierarchie ändert die Stimme, löscht nicht das Sprechen.

4. **Geometrische Goldene Regel**  
   Attn → akademisch; FFN → praktisch; Emb → einfach.

5. **Kohärenz-Oberfläche**  
   Tangential (amplify) bewohnbar; normal (starkes noise) Leere.

6. **Embedding-Raum: Inseln, nicht eine einzige Achse**  
   Zwölf semantische Bereiche fast orthogonal; PCA nutzt
   Hunderte Dimensionen; Gegensätze nicht antipodal.

7. **Makrokosmos ↔ Mikrokosmos**  
   Die Methode ist Hin und Her: Sinn ↔ Tensor ↔ Text.

8. **Reisewerkzeuge**  
   HTML-Karte auf GitHub, Geometrie-Skripte, llama-cli
   für Q4-Testreisen, C-Engine für feine Uhrmacherkunst.

## Einschränkungen der Studie

- Bewertung von „Perspektive" noch qualitativ.  
- TinyLlama ≠ Grenzmodelle (die Oberfläche kann sich ändern).  
- 2D-Karte ist Projektion, nicht die wahre Geometrie.  
   F32-Laufzeit-Perturbation erfordert viel RAM.  
- Nicht alle v10/v11-Techniken sind in der C-Engine.

## Offene Fragen

- Wo (welche Schichten) entzündet sich das mystische Klima im Residual?  
- Lassen sich Perspektiv-Richtungen zwischen Modellen übertragen?  
- Wie misst man Perspektive automatisch und zuverlässig?  
- Was passiert auf der Kohärenz-Oberfläche bei 7B / 70B?

---

*Nächstes Kapitel: Die Zukunft der Erforschung*