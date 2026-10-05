# Punti difficili

Savestate nei momenti in cui un agente tipicamente si blocca. Per ognuno:

| Campo | Esempio |
|---|---|
| `id` | `red_001_old_man` |
| `gioco` | red |
| `savestate` | `red_001_old_man.state` (non versionato se contiene dati della ROM) |
| `situazione` | Il vecchio blocca la strada a nord di Smeraldopoli finché non si consegna il pacco al Prof. Oak |
| `successo` | L'agente raggiunge la Strada 2 |
| `passi massimi` | 5000 |

Ogni punto diventa anche un test automatico: carica lo stato, fai girare l'agente per N passi, verifica il criterio di successo.
