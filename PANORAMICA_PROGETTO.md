# Stima del Doppler assistita da sensori nelle comunicazioni acustiche subacquee

*Panoramica del progetto per la tutor: cosa sto facendo, perché, quale lacuna della letteratura voglio colmare e a che
punto sono. Linguaggio semplice, nessun numero: i valori dettagliati sono nel registro dei risultati
([`code/RESULTS.md`](code/RESULTS.md)).*

Obiettivo di pubblicazione: lo special issue di IEEE JSAC (Journal on Selected Areas in Communications) dedicato
ai sistemi di comunicazione acustica subacquea di nuova generazione, con scadenza a fine marzo.

---

## In breve

Sott'acqua le comunicazioni usano il suono. Quando i nodi si muovono (boe trascinate dalle onde, veicoli autonomi),
il segnale ricevuto viene "compresso" o "dilatato" nel tempo: è l'effetto Doppler. A differenza della radio, in acqua
il suono è lento e quindi il Doppler è molto forte; inoltre, con le onde, **cambia anche durante un singolo
pacchetto**. Oggi ogni ricevitore stima questo Doppler soltanto dal segnale che riceve.

La mia idea è semplice: **ogni nodo sa già come si sta muovendo**, perché ha sensori di movimento (accelerometri,
giroscopi, GPS quando è in superficie). Allora:

- il **trasmettitore** misura il proprio movimento mentre trasmette e lo inserisce, in pochi bit, **dentro lo stesso
  pacchetto** ("informazione laterale");
- il **ricevitore** combina questa informazione con i propri sensori e con ciò che ricava dal segnale.

Ogni parte del Doppler viene così stimata da chi la conosce meglio, e le parti vengono fuse al ricevitore.

---

## La lacuna nella letteratura

Ho fatto una ricerca sistematica (la mia biblioteca locale di articoli, ricerche sul web e su database bibliografici).
Le note sono nella cartella [`literature/`](literature/). Le lacune che ho trovato sono queste.

**Prima lacuna: nessun ricevitore acustico subacqueo usa i sensori di movimento per stimare il Doppler.**
Tutti i lavori che mettono insieme sensori e Doppler vanno nella direzione opposta: usano il Doppler misurato dal
modem per migliorare la navigazione del veicolo (ad esempio i lavori di Aubry, di Garin e di Choi e colleghi).
L'unico lavoro vicino che ho trovato è un rapporto tecnico non sottoposto a revisione (Oiwa e colleghi) che usa un
accelerometro, e solo da un lato del collegamento. Nel mondo radio esistono analogie, soprattutto brevetti: i
terminali satellitari che pre-compensano il Doppler conoscendo la propria posizione GPS, o droni che usano i sensori
inerziali. Quindi la mia affermazione non è "il primo a usare sensori per il Doppler", ma "il primo a farlo nelle
comunicazioni acustiche subacquee, con il movimento reale delle onde, dimensionando l'informazione da trasmettere e
fondendola al ricevitore".

**Seconda lacuna: nessun trasmettitore comunica al ricevitore il proprio movimento, e nessuno ha studiato quanti
bit servano.** Esistono protocolli (ad esempio DE-Sync e APE-Sync) in cui il Doppler misurato viene rimandato
indietro, ma solo per sincronizzare gli orologi. Le intestazioni dei pacchetti contengono indirizzi, qualità del canale
o schema di modulazione, mai il movimento del trasmettitore.

**Terza lacuna: il compromesso tra accuratezza, costo e complessità della stima del Doppler, quando si dispone di
un'informazione a priori dai sensori, non è stato analizzato.** La letteratura studia stimatori basati solo sul
segnale (ad esempio Li e colleghi, Cho e Kim, vari inseguitori di Kalman), ma non quanto un'informazione a priori
cambi le cose. Un lavoro recente (Lee e colleghi), con dati reali da una boa, sostiene addirittura che in molti casi
il Doppler non sia il vero limite: è la principale obiezione con cui mi confronto.

**Quarta lacuna (estensione): ricevitori con molti utenti.** Quando una stazione riceve contemporaneamente più nodi
in movimento, oggi stima il Doppler di ciascuno dal segnale (lavori di Tu, Duman, Stojanovic e Proakis). Nessuno usa
informazioni a priori sul movimento di ciascun utente.

Ho anche controllato un articolo che dal titolo sembrava molto vicino (correzione del Doppler tramite frequenza di
campionamento regolabile): stima il Doppler dal segnale, quindi non tocca le mie lacune.

---

## Il percorso fatto finora

### Il punto di partenza e il primo cambio di direzione
All'inizio l'idea era diversa: **prevedere** il Doppler del pacchetto successivo e adattare di conseguenza il
collegamento. Prima di costruire qualunque cosa ho voluto verificare se il moto delle onde fosse davvero
prevedibile. Ho scaricato **dati reali di movimento di boe oceaniche** (dalle reti di boe ondametriche CDIP, sulla
costa pacifica degli Stati Uniti, e dagli ormeggi OOI, nell'Oregon e nel mare di Irminger) e ho misurato quanto bene
si possa prevedere il loro moto.

Risultato: **il moto di una boa non si può prevedere sul tempo di andata e ritorno di un segnale acustico**. Quindi
qualunque schema basato su previsioni o su informazioni rimandate indietro è destinato a fallire. Da qui la scelta
chiave: l'informazione sul movimento deve **viaggiare dentro lo stesso pacchetto** che descrive, così non è mai
"vecchia".

### Dalla compensazione alla stima
Per qualche giorno il lavoro si è concentrato su **chi deve compensare** quale parte del Doppler (il trasmettitore, il
ricevitore, entrambi). Poi ho riformulato il contributo in modo più pulito: **i compensatori esistenti funzionano
bene se ricevono una stima corretta; il problema aperto è la stima**. Chi stima quale parte, con quali informazioni,
con quale accuratezza e a quale costo.

### La costruzione del simulatore
Ho costruito una catena di simulazione completa, in Python:

- **movimento reale** delle boe come ingresso (non modelli inventati);
- un pacchetto di comunicazione realistico (modulazione OFDM, la stessa usata dai modem acustici moderni, con
  segnali di sincronizzazione all'inizio e alla fine, un'intestazione e dati codificati);
- un **canale acustico** con effetto Doppler esatto, cammini multipli calcolati con il modello di propagazione
  Bellhop (riflessioni su superficie e fondale, ciascuna con il proprio Doppler, superficie ruvida, profilo della
  velocità del suono), rumore;
- **sensori di movimento realistici**, con errori modellati a partire dalle schede tecniche di dispositivi reali (dal
  sensore economico da smartphone fino ai sensori professionali per uso navale), con e senza GPS;
- **tutti i ricevitori rilevanti**: quelli della letteratura basati solo sul segnale e quelli che propongo io,
  assistiti dai sensori;
- metriche di qualità del segnale ricevuto e di pacchetti persi dopo la decodifica.

Ogni componente ha i suoi test automatici e tutto il codice è sotto controllo di versione.

### La verifica critica: confronti equi
Questa è stata la parte più importante, e la più istruttiva. I primi risultati mostravano un grande vantaggio dei
sensori. Ho però sottoposto il lavoro a una revisione critica, chiedendomi se il confronto fosse davvero equo, e ho
trovato diversi problemi:

- i ricevitori di riferimento "solo segnale" erano più deboli di quanto potessero essere (parametri non ottimizzati,
  una ricerca della frequenza troppo stretta, mancanza di tecniche note come la demodulazione a FFT parziale);
- il modello dei sensori era troppo ottimistico;
- la stima del canale perdeva parte delle prestazioni per tutti i ricevitori;
- i risultati si basavano su troppo pochi pacchetti.

Ho corretto questi punti uno alla volta, rafforzando soprattutto i ricevitori di riferimento. Il risultato è stato
netto: **su un collegamento con un solo cammino, un ricevitore "solo segnale" ben progettato è buono quanto quello
assistito dai sensori**. Il vantaggio iniziale era dovuto a confronti non equi. È un risultato meno spettacolare ma
molto più credibile, e lo considero un risultato in sé.

### Capire perché, e dove invece i sensori servono
Per capire questo risultato ho sviluppato due strumenti teorici:

- un **modello analitico** che, data una qualunque combinazione di sensori e di stime dal segnale, predice la
  qualità del segnale ricevuto, validato contro le simulazioni;
- un **limite teorico** (bound di Cramér–Rao bayesiano) su quanto bene si possa stimare il Doppler, con e senza i
  sensori.

Entrambi dicono la stessa cosa: **nel caso semplice il segnale contiene già abbastanza informazione sul Doppler**, e
un buon ricevitore riesce a estrarla. I sensori non alzano il "soffitto" delle prestazioni: **sostituiscono la
complessità del ricevitore**. Un ricevitore molto semplice più i sensori raggiunge quello che, senza sensori,
richiede un inseguitore sofisticato.

Da qui la domanda di questi giorni: **in quali condizioni il segnale porta poca informazione sul Doppler, o è costoso
estrarla?** È lì che i sensori possono dare un vantaggio reale.

---

## I contributi, oggi

**Risultati solidi:**

1. **Un limite misurato sulla prevedibilità del Doppler**, da dati reali di boe: il moto delle onde non si prevede sul
   tempo di andata e ritorno, quindi l'informazione sul movimento deve viaggiare nel pacchetto.
2. **Quanti bit servono**: una formula chiusa che lega il numero di bit dell'informazione laterale all'accuratezza
   del Doppler, verificata con le simulazioni. Il costo è piccolo rispetto alla dimensione del pacchetto.
3. **Il progetto pratico dell'informazione laterale**: come quantizzarla in modo che trasmettitore e ricevitore si
   accordino in anticipo (per classe di stato del mare e per tipo di piattaforma), un'intestazione robusta anche con
   cammini multipli, una protezione che rende innocua un'intestazione ricevuta male. Inoltre, a differenza della
   pre-compensazione fatta dal trasmettitore, l'informazione laterale serve a qualunque ricevitore, in qualunque
   direzione.
4. **Una formula che lega l'errore di stima del Doppler all'interferenza tra sottoportanti**, e un modello
   predittivo validato contro le simulazioni con buona accuratezza.
5. **Quando i sensori servono e quando no**: sostituiscono la complessità del ricevitore ma non alzano il soffitto
   delle prestazioni nel caso semplice; la regola vale per diverse frequenze portanti.
6. **Un limite teorico** che spiega perché: in media il segnale basta, il guadagno dei sensori riguarda i pacchetti
   difficili, e l'informazione a priori può sostituire il tracciamento basato sui piloti.
7. **Requisiti realistici sui sensori**: sensori industriali aiutati dal GPS sono sufficienti; la combinazione tra
   sensori inerziali e GPS va regolata in base alla qualità della misura di assetto.

**Risultati di supporto:** un minor costo di calcolo del ricevitore nei pacchetti brevi e nei cammini multipli, e un
piccolo vantaggio nell'aggancio iniziale del pacchetto in condizioni di segnale molto debole.

**Ciò che non è un contributo, e sappiamo perché:** un guadagno di prestazioni su un collegamento semplice contro un
buon ricevitore "solo segnale". Questo risponde in modo onesto all'obiezione del lavoro di Lee e colleghi.

---

## Dove si cerca ancora un vantaggio reale (lavori in corso)

| Scenario | Perché il segnale da solo è in difficoltà | Stato |
|---|---|---|
| **Cammini multipli** con Doppler diversi (ad esempio una riflessione sul fondale che "sente" il movimento verticale delle boe in modo diverso dal cammino diretto) | Bisogna stimare il Doppler di ciascun cammino separatamente; i sensori, con la geometria, lo predicono | Ricevitori riprogettati per un canale realistico; simulazioni in corso, primi segnali incoraggianti |
| **Molti utenti contemporanei** verso una stazione | Il Doppler di ciascuno va stimato da segnali sovrapposti | Prossima settimana |
| **Segnale molto debole** e tempi di integrazione lunghi (collegamenti a lunga distanza) | Il tracciamento basato sul segnale crolla sotto una soglia; i sensori non dipendono dalla qualità del segnale | Prossimi giorni |
| **Meno piloti** nel pacchetto | Se il Doppler viene dai sensori, parte dei simboli di riferimento può essere usata per i dati | Da fare |

A seconda di quali di questi scenari confermeranno un vantaggio, l'articolo potrà avere due forme:

- **centrato su uno scenario forte** (cammini multipli o molti utenti), con il resto come supporto;
- oppure un articolo del tipo **"stima del Doppler assistita da sensori: quando conviene, a quale costo e come
  progettarla"**, che mette insieme tutti i contributi solidi elencati sopra.

---

## Limiti attuali e domande aperte

- Il **movimento è reale**, ma il canale acustico e gli errori dei sensori sono simulati. Un revisore chiederà dati
  reali: l'ideale sarebbe una prova in mare, o un insieme di dati che contenga **sia le registrazioni acustiche sia i
  dati dei sensori di movimento** dei nodi.
- Nei cammini multipli il vantaggio dipende dall'ambiente: con certi profili della velocità del suono rimane un solo
  cammino importante e il problema sparisce.
- Il ricevitore deve conoscere con buona precisione la geometria dei cammini (gli angoli di partenza e di arrivo).

**Domande per la tutor:**

1. Quale impostazione dell'articolo le sembra più forte: un articolo onesto su "quando conviene" oppure puntare su uno
   scenario specifico?
2. Conosce gruppi o progetti con **dati reali di prove in mare** che includano sia l'acustica sia i sensori di
   movimento?
3. Ha senso includere l'estensione a molti utenti nella prima sottomissione, o lasciarla per la revisione?

---

## Dove trovare il resto

- [`README.md`](README.md): il piano del progetto (in inglese).
- [`STUDY_GUIDE.md`](STUDY_GUIDE.md): l'ordine in cui leggere documenti e codice.
- [`code/PLAN.md`](code/PLAN.md): le fasi del lavoro e il loro stato.
- [`code/RESULTS.md`](code/RESULTS.md): il registro dettagliato di tutti i risultati e di tutte le correzioni,
  con i numeri.
- [`literature/`](literature/): le note sulla ricerca bibliografica.
