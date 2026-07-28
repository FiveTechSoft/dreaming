# Capitolo 13: I Primi Layer (0–5)

## Il vestibolo del microcosmo

I layer iniziali trasformano l'embedding "a riposo"
in una rappresentazione che già percepisce **vicini** e **sintassi**.

```
Layer 0:   ingresso, schemi molto locali
Layer 1:   sintassi base
Layer 2–5: relazioni tra parole adiacenti
```

(Questa partizione è un'**ipotesi di lavoro** del progetto,
guidata da esperimenti di ablazione e dalla letteratura
su "early = sintassi / late = semantica". Non è un taglio
rigido nel codice.)

## Che forze dominano qui

- **Embedding** pesa ancora molto nel residuale (inerzia della nascita).  
- **Attenzione** inizia ad accoppiare bigrammi e dipendenze corte.  
- **FFN** regola il lessico locale.

## Segnali nel testo

Se una perturbazione anticipata "rompe" il modello, spesso
si vede in **grammatica** e token strani, non solo nel tono.

Se il baseline suona generico e il mystical cambia il clima
senza distruggere la sintassi, i layer anticipi continuano
ad ancorare la lingua.

## Esperimento suggerito

Confrontare generazioni con targeting solo su `blk.0`–`blk.5`
rispetto a solo `blk.13`–`blk.21` (script v11 / tensor tests).
Ipotesi: early → forma; late → voce e decisione.

---

*Capitolo successivo: I Layer Intermedi (6–12)*
