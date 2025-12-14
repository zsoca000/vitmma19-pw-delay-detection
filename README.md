# Delay Prediction GNN – README

## 1. Koncepció

A projekt célja, hogy a városi buszhálózat késéseit előre jelezze grafikus neurális hálózat (GNN) segítségével. Az alapötlet a következő:

- **Node-ok:** a buszmegállók.  
- **Edge-ek:** ha egy busz egy útvonalon keresztülhalad a két megálló között, az élt hoz létre.  
- **Dinamikus node feature-ok:** megszabott intervallumokra megvizsgáltam, hogy az egyes stopokon áthaladó trip-ek késéseinek **átlaga, szórása, minimuma, maximuma** valamint az **áthaladó trip-ek száma** mennyi. (Ezek számolása a `src/data_processing.py` fájlban találhatók.)  
- **Statikus node feature-ok:** a földrajzi koordináták (latitude, longitude).  
- **Edge feature-ok:** az adott élre vonatkozó **távolság** és **átlagos menetidő**.  

A modell **bemenete**: a nap, az óra és a trip. A nap-óra alapján kiválasztok egy előre kreált gráfot, majd azon aggregálok több NNConv rétegen keresztül. Ezután a trip node-jai alapján egy **average pooling**-ot készítek, amelyhez hozzákonkatenálom a **day és hour** feature-okat, majd egy **MLP**-be adom, amely az **end-of-trip delay**-t prediktálja.

---

## 2. Jelenlegi állapot

**Állapot:** félkész

- **Adat:** statikus és dinamikus node feature-ok elkészítve, edge feature-ok is rendelkezésre állnak.  
- **Modell:** a GNN struktúrája kész, NNConv rétegekkel implementálva.  
- **Hiányzó lépések:** adatskálázás (node, edge, delay, day, hour), majd a modell tanítása.  
- **Tesztelés:** a modell már tesztelve van dimenziók és méretek szempontjából. Jelenleg **batch processing nincs**, mivel a batch-ek a gráfok mentén **flattenálódnak**, ezért a hagyományos batch mechanizmus nem használható.

**Megjegyzés:** a projekt jelenleg nem futtatható, mivel a skálázás és a tanítás még hiányzik. Néhány napon belül tervezzük a működő pipeline commitolását.


---

## 3. Baseline modell

Van egy egyszerű baseline modellem, ami **route-onként és időpontonként átlagol** a delay-ekből.  
- Teljesítmény: kb **120 másodperc MAE**.  
- Cél: a GNN modellnek ennél jobban kellene teljesítenie, mivel a gráf aggregáció és a node/edge feature-ok részletesebb információt adnak.

---

## 4. Nehézségek és tanulságok

Mivel nem volt lehetőségünk a projektünkkel külön órán foglalkozni, és a kurzus során sok esetben mások adatainak feldolgozásával kellett foglalkoznunk, a pipeline implementálása különösen nehézkes volt. A fejlesztés során azonban rengeteget tanultam.

Kérem a megértésetket, hogy jelenleg csak **félkész állapot** került leadásra, a teljes működő megoldás hamarosan következik.
