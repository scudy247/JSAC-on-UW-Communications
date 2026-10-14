# Stima del Doppler assistita da sensori per le comunicazioni acustiche subacquee

*Sintesi per la tutor — 14 ottobre 2026*

## In un minuto
Lavoro sulle comunicazioni acustiche subacquee tra nodi in movimento, come boe e veicoli sottomarini. Le onde e il
moto dei nodi producono un effetto Doppler che cambia *durante* il pacchetto. Oggi ogni ricevitore lo stima solo dal
segnale ricevuto.

L'idea è semplice: ogni nodo conosce già il proprio moto grazie ai suoi sensori di movimento (IMU, GNSS). Il
trasmettitore può quindi inviare il proprio moto misurato *dentro il pacchetto stesso*, con pochi bit. Il ricevitore
combina questa informazione con i propri sensori e con il segnale.

Ho costruito un simulatore completo, guidato da dati reali di moto di boe ondametriche, e ho confrontato l'idea in
modo equo con i migliori ricevitori che usano solo il segnale. Il risultato onesto, finora, è questo:
- **su un collegamento semplice, a cammino singolo,** un buon ricevitore basato solo sul segnale ottiene le stesse
  prestazioni. Qui i sensori servono soprattutto a usare un ricevitore molto più semplice;
- **un vantaggio reale potrebbe esserci** nei canali con cammini multipli, con molti utenti contemporanei e a
  rapporto segnale-rumore molto basso. Sto verificando questi casi ora.

**Obiettivo:** special issue di IEEE JSAC sulle comunicazioni acustiche subacquee di nuova generazione, scadenza
31 marzo 2027.

## Cosa è già solido
1. **I dati reali dicono che il Doppler delle onde non è prevedibile** con 2–3 secondi di anticipo (il tempo di
   andata e ritorno di un messaggio). Quindi l'informazione sul moto deve viaggiare *dentro* il pacchetto, non come
   risposta successiva.
2. **Una formula chiusa per i bit necessari:** circa 15–30 bit per secondo di pacchetto in mare mosso. È verificata
   con le simulazioni: nel 90% dei casi l'errore è sotto 1 dB.
3. **Un progetto pratico dell'informazione inviata:** intestazione del pacchetto robusta (letta nel 100% dei casi da
   0 dB con un solo cammino), parametri concordati per classe di stato del mare, protezione automatica se
   l'intestazione arriva sbagliata.
4. **Un modello validato che dice *quando* i sensori aiutano:** errore mediano di 0,13 dB rispetto alle simulazioni.
   La regola che ne esce: i sensori *sostituiscono la complessità del ricevitore*, non alzano il limite massimo di
   prestazione. Un ricevitore semplice con i sensori vale quanto un ricevitore complesso senza sensori.
5. **Requisiti realistici per i sensori,** basati sui datasheet di dispositivi reali (Kongsberg, Xsens, Bosch): basta
   un sensore MEMS di classe industriale aiutato dal GNSS.

## Cosa ho imparato lungo la strada
I primi risultati sembravano molto migliori (guadagni da +6 a +15 dB). Il motivo era che i ricevitori di confronto
erano troppo deboli e il modello dei sensori era troppo ideale. Correggendo questi punti, il guadagno su un
collegamento semplice è sparito.

Il lavoro ora è più credibile, e risponde anche a un articolo recente (Lee et al., 2026) secondo cui il Doppler non
è il problema principale in questi sistemi: abbiamo una spiegazione di *quando* ha ragione e quando no.

## Prossimi passi
| Quando | Cosa |
|---|---|
| Questa settimana | Risultati sui canali con cammini multipli (simulazioni in corso) e analisi a basso rapporto segnale-rumore |
| Settimana prossima | Scenario con molti utenti contemporanei; ripetizione delle simulazioni finali con statistiche ampie |
| Venerdì 23 ottobre circa | Breve presentazione completa con figure |

## Domande per la tutor
1. **Impostazione dell'articolo:** è meglio un lavoro completo e onesto del tipo *"quando conviene la stima del
   Doppler assistita da sensori, e a quale costo"*, oppure puntare su un solo scenario forte (cammini multipli o
   molti utenti), se le simulazioni lo confermano?
2. **Dati sperimentali:** conosce qualcuno che abbia dati di prove in mare con *sia* registrazioni acustiche *sia*
   dati dei sensori di movimento? È il limite principale che un revisore indicherebbe: oggi il moto è reale, ma il
   canale acustico è simulato.

---
*Dettagli tecnici: `README.md` (panoramica), `code/RESULTS.md` (tutti i risultati), `STUDY_GUIDE.md` (come orientarsi
nel progetto).*
