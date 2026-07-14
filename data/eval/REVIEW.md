# Eval set — review sheet

> Human-readable rendering of `data/eval/*.jsonl` for verification. **Not** read by the eval itself — the pipeline always reads the `.jsonl` files. Edit the `.jsonl`, not this sheet, then re-run `python src/validate_eval_set.py` and regenerate with `python scripts/make_review_sheet.py`.

| domain | count |
|---|---|
| medical | 21 |
| business | 15 |
| technology | 20 |
| **total** | **56** |

## medical (21)

### medical #1 — Rubophen
🔗 [source](https://rubophen.hu/betegtajekoztato-rubophen-500mg)

**Document:**

> Rubophen 500 mg tabletta
>
> A készítmény hatóanyaga a paracetamol.
>
> Milyen betegségek kezelésére alkalmazható? Lázcsillapításra, valamint különböző eredetű fájdalmak – fejfájás, fogfájás, reumatikus és izomfájdalmak, menstruációs fájdalmak – csillapítására.
>
> Adagolás: Az ajánlott egyszeri adag 500-1000 mg (1-2 db 500 mg-os tabletta). Legfeljebb naponta négyszer ismételhető 4 óra elteltével. Maximális napi adag: 8 tabletta (4000 mg).
>
> Lehetséges mellékhatások: angioödéma, anafilaxiás sokk, toxikus epidermális nekrolízis, Stevens–Johnson-szindróma, pusztulózis, gyomortáji fájdalom, hányinger, hányás, bőrpír, csalánkiütés, bőrkiütés, asztma, orrnyálkahártya-duzzanat, vérképzőszervi rendellenességek, sárgaság.
>
> Ne szedje a Rubophent, ha allergiás a paracetamolra vagy a gyógyszer egyéb összetevőjére, ha súlyos vese- és/vagy májkárosodásban szenved, ha heveny májbetegsége vagy vírusos eredetű májgyulladása van, ha bizonyos veleszületett enzimrendellenessége van, illetve 6 éves kor alatt.

**Gold:**

| field | value |
|---|---|
| drug_name | Rubophen |
| active_ingredient | paracetamol |
| indication | lázcsillapítás és különböző eredetű fájdalmak (fejfájás, fogfájás, reumatikus és izomfájdalmak, menstruációs fájdalmak) csillapítása |
| dosage | egyszeri adag 500-1000 mg (1-2 tabletta), naponta legfeljebb négyszer 4 óránként; maximális napi adag 8 tabletta (4000 mg) |
| side_effects | • angioödéma<br>• anafilaxiás sokk<br>• toxikus epidermális nekrolízis<br>• Stevens–Johnson-szindróma<br>• pusztulózis<br>• gyomortáji fájdalom<br>• hányinger<br>• hányás<br>• bőrpír<br>• csalánkiütés<br>• bőrkiütés<br>• asztma<br>• orrnyálkahártya-duzzanat<br>• vérképzőszervi rendellenességek<br>• sárgaság |
| contraindications | • allergia a paracetamolra<br>• súlyos vese- és/vagy májkárosodás<br>• heveny májbetegség vagy vírusos májgyulladás<br>• veleszületett enzimrendellenesség<br>• 6 éves kor alatt |

---

### medical #2 — No-Spa Forte Neo
🔗 [source](https://www.patikaradar.hu/betegtajekoztatok/no-spa-forte-neo-gorcsoldo-filmtabletta-f06cf2a2872a/)

**Document:**

> No-Spa Forte Neo görcsoldó filmtabletta
>
> Hatóanyag: drotaverin-hidroklorid.
>
> Javallatok: Simaizomgörcsök epe eredetű megbetegedésekben: epekő, epehólyag-gyulladás, epeutak gyulladása. Húgyúti eredetű simaizomgörcsök: vesekő, húgyvezetékkő, vesemedence-gyulladás, húgyhólyag-gyulladás, húgyhólyaggörcs.
>
> Adagolás: A készítmény ajánlott adagja felnőtteknek naponta 120-240 mg, 2-3 részletben.
>
> Lehetséges mellékhatások (ritka): allergiás reakciók (csalánkiütés, bőrkiütés, viszketés), fejfájás, szédülés, álmatlanság, szívdobogásérzés, vérnyomásesés, hányinger, székrekedés.
>
> Ne szedje, ha allergiás a drotaverinre vagy a gyógyszer egyéb összetevőjére, illetve ha súlyos máj-, vese- vagy szívbetegsége van.

**Gold:**

| field | value |
|---|---|
| drug_name | No-Spa Forte Neo |
| active_ingredient | drotaverin-hidroklorid |
| indication | epe eredetű (epekő, epehólyag-gyulladás, epeutak gyulladása) és húgyúti eredetű (vesekő, húgyvezetékkő, vesemedence-gyulladás, húgyhólyag-gyulladás) simaizomgörcsök |
| dosage | felnőtteknek naponta 120-240 mg, 2-3 részletben |
| side_effects | • csalánkiütés<br>• bőrkiütés<br>• viszketés<br>• fejfájás<br>• szédülés<br>• álmatlanság<br>• szívdobogásérzés<br>• vérnyomásesés<br>• hányinger<br>• székrekedés |
| contraindications | • allergia a drotaverinre<br>• súlyos máj-, vese- vagy szívbetegség |

---

### medical #3 — Cataflam-V
🔗 [source](https://ogyei.gov.hu/kiseroirat/bh/bh_0000015470_20251210153252.doc)

**Document:**

> Cataflam-V 50 mg tabletta
>
> A Cataflam-V 50 mg tabletta hatóanyaga a diklofenák, mely a nem-szteroid gyulladásgátló gyógyszerek csoportjába tartozik.
>
> Javallatok: Műtétet (például fogorvosi, illetve ortopéd sebészi beavatkozást) követő gyulladás és fájdalom; fájdalmas, gyulladásos izom-ínsérülések; reumatikus ízületi fájdalom fellobbanása; köszvényes roham; fájdalmas hát-nyak szindróma, teniszkönyök és a lágyrész-reumatizmus egyéb formái.
>
> Adagolás: A kezelés kezdetén a javasolt napi adag általában 100-150 mg, 2-3 részre elosztva. Enyhébb esetekben általában napi 50 mg (1 tabletta) elegendő. Ne vegyen be többet, mint napi 3 Cataflam-V tabletta.
>
> Lehetséges mellékhatások: fejfájás, szédülés, forgó jellegű szédülés, hányinger, hányás, hasmenés, allergiás reakció, gyomorfájdalom.
>
> Ne szedje, ha allergiás a diklofenákra, ha aktív gyomorfekély vagy bélfekély, vérzés vagy a tápcsatorna falának átfúródása áll fenn, ha a terhesség utolsó három hónapjában van, illetve ha diagnosztizált szívbetegségben és/vagy agyi érkatasztrófában szenved.

**Gold:**

| field | value |
|---|---|
| drug_name | Cataflam-V |
| active_ingredient | diklofenák |
| indication | műtétet követő gyulladás és fájdalom, fájdalmas gyulladásos izom-ínsérülések, reumatikus ízületi fájdalom fellobbanása, köszvényes roham és a lágyrész-reumatizmus egyéb formái |
| dosage | kezdetben napi 100-150 mg 2-3 részre elosztva, enyhébb esetekben napi 50 mg (1 tabletta); legfeljebb napi 3 tabletta |
| side_effects | • fejfájás<br>• szédülés<br>• forgó jellegű szédülés<br>• hányinger<br>• hányás<br>• hasmenés<br>• allergiás reakció<br>• gyomorfájdalom |
| contraindications | • allergia a diklofenákra<br>• aktív gyomor- vagy bélfekély, vérzés vagy a tápcsatorna átfúródása<br>• terhesség utolsó három hónapja<br>• diagnosztizált szívbetegség és/vagy agyi érkatasztrófa |

---

### medical #4 — Kalmopyrin
🔗 [source](https://www.patikaradar.hu/betegtajekoztatok/kalmopyrin-500-mg-tabletta-5b2f0349de2d/)

**Document:**

> Kalmopyrin 500 mg tabletta
>
> Hatóanyag: acetilszalicilsav.
>
> Javallatok: enyhe és középerős fájdalmak, pl. fejfájás, fogfájás, hátfájás, ízületi fájdalmak és menstruációs fájdalom csillapítására; továbbá lázas, hűléses és reumás megbetegedések tüneteinek enyhítésére.
>
> Adagolás: A készítmény ajánlott napi adagja felnőttek részére: 1-3-szor 1-2 db tabletta (1-3-szor 500-1000 mg). A maximális napi adag 4000 mg (8 db tabletta).
>
> Lehetséges mellékhatások: hányinger, hányás, hasmenés, gyomor-bélrendszeri vérzés, vérszegénység, asztmás roham, viszkető csalánkiütés, súlyos bőrreakciók, gyomor-bélrendszeri fekélyek, alacsony vércukorszint, szédülés, fülzúgás, Reye-szindróma.
>
> Ne szedje: allergia az acetilszalicilsavra vagy egyéb összetevőre; vérzékenységre való hajlam; gyomor- vagy nyombélfekély; gyermekek bárányhimlő- és influenzavírus-fertőzése; terhesség utolsó 3 hónapja; 12 éves kor alatti gyermekek.

**Gold:**

| field | value |
|---|---|
| drug_name | Kalmopyrin |
| active_ingredient | acetilszalicilsav |
| indication | enyhe és középerős fájdalmak (fejfájás, fogfájás, hátfájás, ízületi és menstruációs fájdalom) csillapítása, valamint lázas, hűléses és reumás megbetegedések tüneteinek enyhítése |
| dosage | felnőtteknek 1-3-szor 1-2 tabletta (1-3-szor 500-1000 mg); maximális napi adag 4000 mg (8 tabletta) |
| side_effects | • hányinger<br>• hányás<br>• hasmenés<br>• gyomor-bélrendszeri vérzés<br>• vérszegénység<br>• asztmás roham<br>• viszkető csalánkiütés<br>• súlyos bőrreakciók<br>• gyomor-bélrendszeri fekélyek<br>• alacsony vércukorszint<br>• szédülés<br>• fülzúgás<br>• Reye-szindróma |
| contraindications | • allergia az acetilszalicilsavra<br>• vérzékenységre való hajlam<br>• gyomor- vagy nyombélfekély<br>• gyermekek bárányhimlő- és influenzavírus-fertőzése<br>• terhesség utolsó 3 hónapja<br>• 12 éves kor alatt |

---

### medical #5 — ACC
🔗 [source](https://www.patikaradar.hu/betegtajekoztatok/acc-200-granulatum-e599e31be66f/)

**Document:**

> ACC 200 mg granulátum
>
> Hatóanyag: acetilcisztein.
>
> Javallatok: hurutos meghűlés, krónikus arcüreggyulladás, középfülgyulladás, mukoviszcidózis, valamint a hörgők akut és krónikus gyulladása, krónikus obstruktív tüdőbetegség, hörgőtágulat és tüdőgyulladás.
>
> Adagolás: Felnőttek, valamint 14 éven felüli serdülők és gyermekek: naponta 2-3-szor 1 tasak granulátum.
>
> Lehetséges mellékhatások: szájnyálkahártya-gyulladás, fejfájás, fülzúgás vagy fülcsengés, láz, allergiás reakciók, gyomorégés, hányinger, hányás, hasmenés, nehézlégzés, hörgőgörcs, vérzés.
>
> Ne alkalmazza, ha allergiás az acetilciszteinre vagy a gyógyszer egyéb összetevőjére, ha terhes és/vagy szoptat, illetve 6 éves kor alatti gyermekek kezelésére.

**Gold:**

| field | value |
|---|---|
| drug_name | ACC |
| active_ingredient | acetilcisztein |
| indication | hurutos meghűlés, krónikus arcüreggyulladás, középfülgyulladás, mukoviszcidózis, valamint a hörgők akut és krónikus gyulladása, krónikus obstruktív tüdőbetegség, hörgőtágulat és tüdőgyulladás |
| dosage | felnőtteknek és 14 éven felülieknek naponta 2-3-szor 1 tasak granulátum |
| side_effects | • szájnyálkahártya-gyulladás<br>• fejfájás<br>• fülzúgás<br>• fülcsengés<br>• láz<br>• allergiás reakciók<br>• gyomorégés<br>• hányinger<br>• hányás<br>• hasmenés<br>• nehézlégzés<br>• hörgőgörcs<br>• vérzés |
| contraindications | • allergia az acetilciszteinre<br>• terhesség és/vagy szoptatás<br>• 6 éves kor alatt |

---

### medical #6 — Espumisan
🔗 [source](https://www.patikaradar.hu/betegtajekoztatok/espumisan-40-mg-ml-belsoleges-emulzios-cseppek-55af697e91ba/)

**Document:**

> Espumisan 40 mg/ml belsőleges emulziós cseppek
>
> Hatóanyag: szimetikon.
>
> Javallatok: A gyomor és a belek bélgázzal összefüggő panaszainak tüneti kezelésére, mint például gázfelhalmozódás (meteorizmus), puffadás, teltségérzés és csecsemőkori hasfájás (három hónapos kólika); továbbá hasi képalkotó diagnosztikus vizsgálatok előkészítésére.
>
> Adagolás: 14 éves kor feletti serdülők és felnőttek: 50 csepp (2 ml) 3-5 alkalommal naponta.
>
> Lehetséges mellékhatások: túlérzékenységi reakciók, köztük csalánkiütés, bőrkiütés, bőrpír, viszketés és allergiás bőrgyulladás.
>
> Ne alkalmazza, ha allergiás a hatóanyagra vagy a gyógyszer egyéb összetevőjére.

**Gold:**

| field | value |
|---|---|
| drug_name | Espumisan |
| active_ingredient | szimetikon |
| indication | a gyomor és a belek bélgázzal összefüggő panaszainak (gázfelhalmozódás, puffadás, teltségérzés, csecsemőkori hasfájás) tüneti kezelése, valamint hasi képalkotó vizsgálatok előkészítése |
| dosage | 14 éves kor feletti serdülők és felnőttek: 50 csepp (2 ml) naponta 3-5 alkalommal |
| side_effects | • csalánkiütés<br>• bőrkiütés<br>• bőrpír<br>• viszketés<br>• allergiás bőrgyulladás |
| contraindications | • allergia a hatóanyagra vagy a gyógyszer egyéb összetevőjére |

---

### medical #7 — Coldrex MaxGrip
🔗 [source](https://www.patikaradar.hu/betegtajekoztatok/coldrex-maxgrip-citrom-por-belsoleges-oldathoz-6833c8756a3c/)

**Document:**

> Coldrex MaxGrip citrom por belsőleges oldathoz
>
> Hatóanyagok: paracetamol, fenilefrin-hidroklorid, aszkorbinsav (C-vitamin).
>
> Javallatok: a megfázás és az influenza tüneteinek, mint például láz, fejfájás, torokfájás, végtagfájdalom, orrdugulás, valamint az orrmelléküreg-gyulladás (szinuszitisz) és a vele járó fájdalom enyhítésére.
>
> Adagolás: 4-6 óránként 1 tasak. 24 órán belül 4 tasaknál többet ne alkalmazzon. Két adag alkalmazása között minimum 4 órának el kell telnie.
>
> Lehetséges mellékhatások: fejfájás, szédülés, idegesség, álmatlanság, vérnyomás-emelkedés, hányinger, hányás.
>
> Ne szedje, ha allergiás a paracetamolra, a fenilefrin-hidrokloridra vagy az aszkorbinsavra; ha triciklusos antidepresszánsokat, béta-blokkolókat vagy egyéb vérnyomáscsökkentő szereket szed; ha monoamin-oxidáz-gátló (MAO-gátló) gyógyszereket szed; ha pajzsmirigy-túlműködése van; ha súlyos szívbetegségben szenved; ha magas a vérnyomása; ha cukorbeteg; ha zárt zugú glaukómája van.

**Gold:**

| field | value |
|---|---|
| drug_name | Coldrex MaxGrip |
| active_ingredient | paracetamol, fenilefrin-hidroklorid, aszkorbinsav |
| indication | a megfázás és az influenza tüneteinek (láz, fejfájás, torokfájás, végtagfájdalom, orrdugulás, orrmelléküreg-gyulladás) enyhítése |
| dosage | 4-6 óránként 1 tasak, 24 órán belül legfeljebb 4 tasak, két adag között minimum 4 óra |
| side_effects | • fejfájás<br>• szédülés<br>• idegesség<br>• álmatlanság<br>• vérnyomás-emelkedés<br>• hányinger<br>• hányás |
| contraindications | • allergia a paracetamolra, a fenilefrin-hidrokloridra vagy az aszkorbinsavra<br>• triciklusos antidepresszánsok, béta-blokkolók vagy egyéb vérnyomáscsökkentők szedése<br>• MAO-gátló gyógyszerek szedése<br>• pajzsmirigy-túlműködés<br>• súlyos szívbetegség<br>• magas vérnyomás<br>• cukorbetegség<br>• zárt zugú glaukóma |

---

### medical #8 — Milgamma neuro
🔗 [source](https://www.patikaradar.hu/betegtajekoztatok/milgamma-neuro-100-100-mg-bevont-tabletta-f054c261d605/)

**Document:**

> Milgamma neuro 100/100 mg bevont tabletta
>
> Hatóanyagok: benfotiamin (zsíroldékony B1-vitamin) és piridoxin-hidroklorid (B6-vitamin).
>
> Javallatok: idegbántalmak (úgynevezett neuropátia) kezelésére való gyógyszer. Bizonyítottan B1- és B6-vitaminhiány okozta idegrendszeri betegségek kezelésére alkalmazható.
>
> Adagolás: A készítmény ajánlott adagja felnőtteknek napi 1 darab bevont tabletta. Heveny (akut) esetekben az adag legfeljebb napi 3-szor 1 darab bevont tablettára emelhető.
>
> Lehetséges mellékhatások: bőrreakciók, csalánkiütés, bőrkiütések, sokkos állapot, hányinger, felfúvódás, hasmenés, székrekedés, hasi fájdalom.
>
> Ne szedje, ha allergiás a tiaminra, a benfotiaminra, a piridoxin-hidrokloridra vagy a gyógyszer egyéb összetevőjére; ha Ön terhes vagy szoptat.

**Gold:**

| field | value |
|---|---|
| drug_name | Milgamma neuro |
| active_ingredient | benfotiamin, piridoxin-hidroklorid |
| indication | idegbántalmak (neuropátia) kezelése, bizonyítottan B1- és B6-vitaminhiány okozta idegrendszeri betegségekben |
| dosage | felnőtteknek napi 1 bevont tabletta; heveny esetekben legfeljebb napi 3-szor 1 tabletta |
| side_effects | • bőrreakciók<br>• csalánkiütés<br>• bőrkiütések<br>• sokkos állapot<br>• hányinger<br>• felfúvódás<br>• hasmenés<br>• székrekedés<br>• hasi fájdalom |
| contraindications | • allergia a tiaminra, a benfotiaminra vagy a piridoxin-hidrokloridra<br>• terhesség vagy szoptatás |

---

### medical #9 — Teva-Ambrobene
🔗 [source](https://www.patikaradar.hu/betegtajekoztatok/teva-ambrobene-30-mg-tabletta-615e47c5e0fc/)

**Document:**

> Teva-Ambrobene 30 mg tabletta
>
> Hatóanyag: ambroxol-hidroklorid.
>
> Javallatok: A hörgők és a tüdő sűrű váladékképződéssel járó heveny és idült betegségeinek kezelésére, nyákoldásra.
>
> Adagolás: Felnőtteknek az első 2-3 napban 3-szor 1 tabletta (3-szor 30 mg ambroxol-hidroklorid), majd naponta 2-szer 1 tabletta.
>
> Lehetséges mellékhatások: émelygés, hányás, hasmenés, bőrkiütés, csalánkiütés, anafilaxiás sokk, Stevens-Johnson-szindróma.
>
> Ne szedje, ha allergiás az ambroxol-hidrokloridra vagy a gyógyszer egyéb összetevőjére, illetve gyomor-, bélfekélyek esetén.

**Gold:**

| field | value |
|---|---|
| drug_name | Teva-Ambrobene |
| active_ingredient | ambroxol-hidroklorid |
| indication | a hörgők és a tüdő sűrű váladékképződéssel járó heveny és idült betegségeinek kezelése, nyákoldás |
| dosage | az első 2-3 napban 3-szor 1 tabletta, majd naponta 2-szer 1 tabletta |
| side_effects | • émelygés<br>• hányás<br>• hasmenés<br>• bőrkiütés<br>• csalánkiütés<br>• anafilaxiás sokk<br>• Stevens-Johnson-szindróma |
| contraindications | • allergia az ambroxol-hidrokloridra<br>• gyomor-, bélfekélyek |

---

### medical #10 — Strepsils Plus
🔗 [source](https://www.patikaradar.hu/betegtajekoztatok/strepsils-plus-szopogato-tabletta-18a855aea241/)

**Document:**

> Strepsils Plus szopogató tabletta
>
> Hatóanyagok: amilmetakrezol, 2,4-diklór-benzil-alkohol, lidokain-klorid.
>
> Javallatok: torok-, garat-, mandulagyulladás, fogínygyulladás és szájnyálkahártya-gyulladás kezelésére.
>
> Adagolás: 1 szopogató tabletta 2 órás időközönként, közvetlenül étkezés után. Naponta legfeljebb 8 tablettát szabad bevenni.
>
> Lehetséges mellékhatások: allergiás reakciók, az arc, ajkak, száj, nyelv vagy torok megduzzadása, csalánkiütés, hörgőgörcs, hasi fájdalom, émelygés, szájnyálkahártya irritáció, methemoglobinémia.
>
> Ne alkalmazza, ha allergiás a lidokainra vagy a gyógyszer egyéb összetevőjére; lidokain adását követő rosszullét (görcsroham) esetén; súlyos májműködési zavar esetén; kifejezetten alacsony pulzusszám esetén; 12 éves életkor alatt; methemoglobinémia esetén; ha asztmában vagy bronchospazmusban szenved.

**Gold:**

| field | value |
|---|---|
| drug_name | Strepsils Plus |
| active_ingredient | amilmetakrezol, 2,4-diklór-benzil-alkohol, lidokain-klorid |
| indication | torok-, garat-, mandulagyulladás, fogínygyulladás és szájnyálkahártya-gyulladás kezelése |
| dosage | 1 szopogató tabletta 2 óránként, közvetlenül étkezés után; naponta legfeljebb 8 tabletta |
| side_effects | • allergiás reakciók<br>• az arc, ajkak, száj, nyelv vagy torok megduzzadása<br>• csalánkiütés<br>• hörgőgörcs<br>• hasi fájdalom<br>• émelygés<br>• szájnyálkahártya irritáció<br>• methemoglobinémia |
| contraindications | • allergia a lidokainra<br>• lidokain adását követő rosszullét (görcsroham)<br>• súlyos májműködési zavar<br>• kifejezetten alacsony pulzusszám<br>• 12 éves kor alatt<br>• methemoglobinémia<br>• asztma vagy bronchospazmus |

---

### medical #11 — Voltaren Dolo
🔗 [source](https://www.webbeteg.hu/gyogyszerkereso/voltaren-dolo-25/36667/betegtajekoztato)

**Document:**

> VOLTAREN DOLO 25 mg lágy kapszula
>
> Hatóanyag: diklofenák (diklofenák-kálium, 25 mg kapszulánként).
>
> Javallatok: enyhíti az izom- és ízületi fájdalmakat, a hátfájást, fejfájást, fogfájást és a menstruáció következtében fellépő fájdalmakat, valamint csökkenti a megfázás, illetve influenza tüneteit.
>
> Adagolás: A panaszok megjelenése esetén kezdő adagként 1 db kapszula. Amennyiben szükséges, a továbbiakban 4-6 óránként 1 db kapszula. 24 óra alatt legfeljebb 3 db kapszula (75 mg) vehető be.
>
> Lehetséges mellékhatások: fejfájás, szédülés, hányinger, hányás, hasmenés, bőrkiütés, csalánkiütés, aluszékonyság, hallászavarok, székrekedés.
>
> Ne szedje: allergia a diklofenákra vagy egyéb összetevőkre; korábbi allergiás reakció gyulladásgátló szerekre; szívbetegség vagy agyérbetegség; perifériás artéria-betegség; aktív gyomor- vagy bélfekély; emésztőrendszeri vérzés; súlyos vese- vagy májelégtelenség; terhesség utolsó három hónapja.

**Gold:**

| field | value |
|---|---|
| drug_name | Voltaren Dolo |
| active_ingredient | diklofenák |
| indication | izom- és ízületi fájdalmak, hátfájás, fejfájás, fogfájás és menstruációs fájdalmak enyhítése, valamint a megfázás és influenza tüneteinek csökkentése |
| dosage | kezdő adag 1 kapszula, majd szükség esetén 4-6 óránként 1 kapszula; naponta legfeljebb 3 kapszula (75 mg) |
| side_effects | • fejfájás<br>• szédülés<br>• hányinger<br>• hányás<br>• hasmenés<br>• bőrkiütés<br>• csalánkiütés<br>• aluszékonyság<br>• hallászavarok<br>• székrekedés |
| contraindications | • allergia a diklofenákra<br>• korábbi allergiás reakció gyulladásgátló szerekre<br>• szívbetegség vagy agyérbetegség<br>• perifériás artéria-betegség<br>• aktív gyomor- vagy bélfekély<br>• emésztőrendszeri vérzés<br>• súlyos vese- vagy májelégtelenség<br>• terhesség utolsó három hónapja |

---

### medical #12 — Fenistil
🔗 [source](https://www.patikaradar.hu/betegtajekoztatok/fenistil-1-mg-g-gel-6cb7eebe183f/)

**Document:**

> Fenistil 1 mg/g gél
>
> Hatóanyag: dimetindén-maleát.
>
> Javallatok: különböző bőrreakciókat (pl. bőrkiütések, csalánkiütés, rovarcsípés, napégés és felületes égési sérülés) kísérő viszketés enyhítésére.
>
> Lehetséges mellékhatások: bőrszárazság, bőrön jelentkező égő érzés, viszkető kiütés.
>
> Ne alkalmazza, ha allergiás a dimetindén-maleátra vagy a gyógyszer egyéb összetevőjére.

**Gold:**

| field | value |
|---|---|
| drug_name | Fenistil |
| active_ingredient | dimetindén-maleát |
| indication | különböző bőrreakciókat (bőrkiütések, csalánkiütés, rovarcsípés, napégés, felületes égési sérülés) kísérő viszketés enyhítése |
| dosage | _null_ |
| side_effects | • bőrszárazság<br>• bőrön jelentkező égő érzés<br>• viszkető kiütés |
| contraindications | • allergia a dimetindén-maleátra |

---

### medical #13 — Rennie
🔗 [source](https://www.patikaradar.hu/betegtajekoztatok/rennie-cukormentes-ragotabletta-38a3c5e0cabc/)

**Document:**

> Rennie cukormentes rágótabletta
>
> Hatóanyagok: kalcium-karbonát, magnézium-karbonát.
>
> Javallatok: gyomorégés, savas felböfögés, emésztési zavarok, terhességi emésztési zavarok, valamint a teltségérzet kezelésére.
>
> Adagolás: 1 vagy 2 db rágótablettát javasolt elrágni vagy elszopogatni, lehetőleg étkezés után 1 órával.
>
> Lehetséges mellékhatások: hányinger, hányás, székrekedés, izomgyengeség, bőrkiütések, csalánkiütés, bőrviszketés, angioödéma, hasmenés.
>
> Ne szedje, ha allergiás a hatóanyagokra vagy egyéb összetevőjére; ha magas a vér kalciumszintje vagy olyan betegségben szenved, amely eredményezheti a vér kalciumszintjének növekedését; ha kalcium tartalmú vesekövei vannak; ha súlyos vesebetegségben szenved; ha alacsony a vér foszfátszintje.

**Gold:**

| field | value |
|---|---|
| drug_name | Rennie |
| active_ingredient | kalcium-karbonát, magnézium-karbonát |
| indication | gyomorégés, savas felböfögés, emésztési zavarok, terhességi emésztési zavarok és teltségérzet kezelése |
| dosage | 1 vagy 2 rágótabletta elrágása vagy elszopogatása, lehetőleg étkezés után 1 órával |
| side_effects | • hányinger<br>• hányás<br>• székrekedés<br>• izomgyengeség<br>• bőrkiütések<br>• csalánkiütés<br>• bőrviszketés<br>• angioödéma<br>• hasmenés |
| contraindications | • allergia a hatóanyagokra<br>• magas vér kalciumszint vagy azt eredményező betegség<br>• kalcium tartalmú vesekövek<br>• súlyos vesebetegség<br>• alacsony vér foszfátszint |

---

### medical #14 — Nasivin Classic
🔗 [source](https://www.patikaradar.hu/betegtajekoztatok/nasivin-classic-0-5-mg-ml-oldatos-orrspray-7433fb6ea573/)

**Document:**

> Nasivin Classic 0,5 mg/ml oldatos orrspray
>
> Hatóanyag: oximetazolin-hidroklorid.
>
> Javallatok: nátha (rinitisz akuta), allergiás nátha (rinitisz allergika), rohamokban fellépő szénanátha (rinitisz vazomotorika), a váladékürülés megkönnyítésére az orr- és/vagy melléküregek gyulladása esetén, valamint náthával együttjáró fülkürt- és középfül-gyulladás.
>
> Adagolás: 6 éves kortól mind a két orrnyílásba naponta 2-3 alkalommal 1 befújás. A készítmény egyszeri adagját legfeljebb naponta háromszor lehet alkalmazni.
>
> Lehetséges mellékhatások: nyugtalanság, álmatlanság, fáradtság, fejfájás, hallucináció, görcsrohamok, szívdobogásérzet, szapora szívműködés, vérnyomás emelkedése, égő vagy száraz orrnyálkahártya, tüsszögés, orrdugulás, túlérzékenységi reakciók.
>
> Ne alkalmazza, ha allergiás az oximetazolinra vagy a gyógyszer egyéb összetevőjére, ha Ön rinitisz szikka betegségben szenved, illetve benzalkónium-klorid-túlérzékenység esetén.

**Gold:**

| field | value |
|---|---|
| drug_name | Nasivin Classic |
| active_ingredient | oximetazolin-hidroklorid |
| indication | nátha, allergiás nátha, rohamokban fellépő szénanátha, az orr- és melléküregek gyulladása esetén a váladékürülés megkönnyítése, valamint náthával együtt járó fülkürt- és középfülgyulladás |
| dosage | 6 éves kortól mindkét orrnyílásba naponta 2-3 alkalommal 1 befújás, legfeljebb naponta háromszor |
| side_effects | • nyugtalanság<br>• álmatlanság<br>• fáradtság<br>• fejfájás<br>• hallucináció<br>• görcsrohamok<br>• szívdobogásérzet<br>• szapora szívműködés<br>• vérnyomás emelkedése<br>• égő vagy száraz orrnyálkahártya<br>• tüsszögés<br>• orrdugulás<br>• túlérzékenységi reakciók |
| contraindications | • allergia az oximetazolinra<br>• rinitisz szikka<br>• benzalkónium-klorid-túlérzékenység |

---

### medical #15 — Panadol Rapid Extra
🔗 [source](https://www.patikaradar.hu/betegtajekoztatok/panadol-rapid-extra-500mg-65mg-filmtabletta-58208f9e5f92/)

**Document:**

> Panadol Rapid Extra 500 mg/65 mg filmtabletta
>
> Hatóanyagok: paracetamol, koffein.
>
> Javallatok: fejfájást és migrént, fogfájást, fogászati beavatkozást/foghúzást követő fájdalmat, védőoltásokat követő lázat és fájdalmat, fülfájást, reumatikus és izomfájdalmakat, oszteoartritisszel társult fájdalmat, menstruációs görccsel járó fájdalmat, valamint légúti eredetű fertőzések (megfázás, influenza) következtében fellépő lázat és fájdalmat.
>
> Adagolás: 65 kg feletti serdülők és felnőttek: 4-6 óránként 1-2 tabletta. 24 órán belül 8 tablettánál többet ne vegyen be.
>
> Lehetséges mellékhatások: anafilaxiás sokk, bőrkiütés, hámlás, viszketés, Stevens–Johnson-szindróma, toxikus epidermális nekrolízis, hörgőgörcs, idegesség, szédülés.
>
> Ne szedje, ha allergiás a paracetamolra, a koffeinre vagy a gyógyszer egyéb összetevőjére; ha egy bizonyos enzimrendellenességben szenved (a glükóz-6-foszfát-dehidrogenáz enzim hiánya esetén); túlzott, illetve krónikus alkoholfogyasztás esetén; 12 évesnél fiatalabb beteg esetén.

**Gold:**

| field | value |
|---|---|
| drug_name | Panadol Rapid Extra |
| active_ingredient | paracetamol, koffein |
| indication | fejfájás és migrén, fogfájás, védőoltást követő láz és fájdalom, fülfájás, reumatikus és izomfájdalmak, oszteoartritisszel társult fájdalom, menstruációs görcs, valamint megfázás és influenza okozta láz és fájdalom kezelése |
| dosage | 65 kg feletti serdülők és felnőttek: 4-6 óránként 1-2 tabletta; 24 órán belül legfeljebb 8 tabletta |
| side_effects | • anafilaxiás sokk<br>• bőrkiütés<br>• hámlás<br>• viszketés<br>• Stevens–Johnson-szindróma<br>• toxikus epidermális nekrolízis<br>• hörgőgörcs<br>• idegesség<br>• szédülés |
| contraindications | • allergia a paracetamolra vagy a koffeinre<br>• glükóz-6-foszfát-dehidrogenáz enzim hiánya<br>• túlzott vagy krónikus alkoholfogyasztás<br>• 12 évesnél fiatalabb életkor |

---

### medical #16 — Mebucain Mint
🔗 [source](https://www.patikaradar.hu/betegtajekoztatok/mebucain-mint-2mg-1mg-szopogato-tabletta-bb35f61ec90d/)

**Document:**

> Mebucain Mint 2 mg/1 mg szopogató tabletta
>
> Hatóanyagok: cetilpiridinium-klorid, lidokain-hidroklorid.
>
> Javallatok: a torokfájás és a szájüreg enyhe fertőzéseinek kezelésére; csökkenti a száraz köhögés, megfázás és influenza során kialakuló torokfájdalmat.
>
> Adagolás: Komoly heveny gyulladás: 1 tabletta 1-2 óránként. Enyhébb gyulladás: 1 tabletta 2-3 óránként. Naponta legfeljebb 6 szopogató tablettát lehet bevenni.
>
> Lehetséges mellékhatások: émelygés, irritáció a szájüregben, irritáció a torokban, túlérzékenység, bőrkiütés.
>
> Ne alkalmazza, ha allergiás a cetilpiridinium-kloridra, a lidokain-hidrokloridra, egyéb amidok csoportjába tartozó helyi érzéstelenítőre vagy a gyógyszer egyéb összetevőjére; 6 évesnél fiatalabb gyermekeknek nem adható; ne alkalmazza, ha Ön terhes vagy szoptat.

**Gold:**

| field | value |
|---|---|
| drug_name | Mebucain Mint |
| active_ingredient | cetilpiridinium-klorid, lidokain-hidroklorid |
| indication | a torokfájás és a szájüreg enyhe fertőzéseinek kezelése, valamint a száraz köhögés, megfázás és influenza során kialakuló torokfájdalom csökkentése |
| dosage | heveny gyulladásnál 1 tabletta 1-2 óránként, enyhébb gyulladásnál 1 tabletta 2-3 óránként; naponta legfeljebb 6 tabletta |
| side_effects | • émelygés<br>• irritáció a szájüregben<br>• irritáció a torokban<br>• túlérzékenység<br>• bőrkiütés |
| contraindications | • allergia a cetilpiridinium-kloridra vagy a lidokain-hidrokloridra<br>• 6 éves kor alatt<br>• terhesség vagy szoptatás |

---

### medical #17 — Canesten
🔗 [source](https://www.patikaradar.hu/betegtajekoztatok/canesten-10-mg-g-gomba-elleni-krem-6b99d91d3c70/)

**Document:**

> Canesten 10 mg/g krém
>
> Hatóanyag: klotrimazol.
>
> Javallatok: a bőr, illetve a külső nemi szervek gombás fertőzésének kezelésére alkalmas.
>
> Lehetséges mellékhatások: angioödéma, anafilaxiás reakció, alacsony vérnyomás, ájulás, hólyagok, kontakt dermatitisz, bőrvörösség, bőrhámlás, viszketés, bőrkiütés, csalánkiütés, szúró/égő érzés, irritáció.
>
> Ne alkalmazza, ha allergiás a klotrimazolra vagy a gyógyszer egyéb összetevőjére.

**Gold:**

| field | value |
|---|---|
| drug_name | Canesten |
| active_ingredient | klotrimazol |
| indication | a bőr, illetve a külső nemi szervek gombás fertőzésének kezelése |
| dosage | _null_ |
| side_effects | • angioödéma<br>• anafilaxiás reakció<br>• alacsony vérnyomás<br>• ájulás<br>• hólyagok<br>• kontakt dermatitisz<br>• bőrvörösség<br>• bőrhámlás<br>• viszketés<br>• bőrkiütés<br>• csalánkiütés<br>• szúró/égő érzés<br>• irritáció |
| contraindications | • allergia a klotrimazolra |

---

### medical #18 — Neo Citran
🔗 [source](https://www.patikaradar.hu/betegtajekoztatok/neo-citran-belsoleges-por-felnotteknek-247bb69b82b6/)

**Document:**

> Neo Citran belsőleges por felnőtteknek
>
> Hatóanyagok: fenilefrin-hidroklorid, feniramin-maleát, aszkorbinsav, paracetamol.
>
> Javallatok: a megfázás és az influenza tüneteinek enyhítésére alkalmazandó, mint például a láz és a hidegrázás átmeneti csökkentése, fejfájás, végtagok elnehezedése, fájdalma, orrdugulás, orrfolyás.
>
> Adagolás: 1 tasak tartalmát 1 pohár (kb. 2,5 dl) forró, de nem lobogó vízben kell feloldani és meginni, amint elfogadható hőmérsékletűre hűlt. Szükség esetén ez az adag 3-4 óra elteltével megismételhető, de 24 óra alatt összesen legfeljebb 3 tasak tartalmát szabad bevenni.
>
> Lehetséges mellékhatások: allergiás reakciók, zihálás, légszomj, bőrkiütés, csalánkiütés, szájszárazság, fáradtság, fejfájás, alvászavar, vizeletürítési zavar, vérnyomásváltozás, methemoglobinémia.
>
> Ne szedje: allergia a fenilefrin-hidrokloridra, feniramin-maleátra, aszkorbinsavra, paracetamolra; súlyos máj- és veseelégtelenség; alacsony fehérvérsejtszám; kezeletlen magas vérnyomás; pajzsmirigy-túlműködés; súlyos szív- és érrendszeri betegség; zárt zugú glaukóma; feokromocitóma; MAO-gátlók szedése; triciklusos antidepresszánsok; béta-blokkolók; 14 évesnél fiatalabb gyermekek.

**Gold:**

| field | value |
|---|---|
| drug_name | Neo Citran |
| active_ingredient | fenilefrin-hidroklorid, feniramin-maleát, aszkorbinsav, paracetamol |
| indication | a megfázás és az influenza tüneteinek (láz, hidegrázás, fejfájás, végtagfájdalom, orrdugulás, orrfolyás) enyhítése |
| dosage | 1 tasak tartalmát 1 pohár forró vízben feloldva; szükség esetén 3-4 óránként megismételhető, 24 óra alatt legfeljebb 3 tasak |
| side_effects | • allergiás reakciók<br>• zihálás<br>• légszomj<br>• bőrkiütés<br>• csalánkiütés<br>• szájszárazság<br>• fáradtság<br>• fejfájás<br>• alvászavar<br>• vizeletürítési zavar<br>• vérnyomásváltozás<br>• methemoglobinémia |
| contraindications | • allergia a fenilefrin-hidrokloridra, feniramin-maleátra, aszkorbinsavra vagy paracetamolra<br>• súlyos máj- és veseelégtelenség<br>• alacsony fehérvérsejtszám<br>• kezeletlen magas vérnyomás<br>• pajzsmirigy-túlműködés<br>• súlyos szív- és érrendszeri betegség<br>• zárt zugú glaukóma<br>• feokromocitóma<br>• MAO-gátlók szedése<br>• triciklusos antidepresszánsok<br>• béta-blokkolók<br>• 14 évesnél fiatalabb gyermekek |

---

### medical #19 — Aspirin Protect
🔗 [source](https://www.patikaradar.hu/betegtajekoztatok/aspirin-protect-100-mg-gyomornedv-ellenallo-bevont-tabletta-1c19ac5f6e73/)

**Document:**

> Aspirin Protect 100 mg gyomornedv ellenálló bevont tabletta
>
> Hatóanyag: acetilszalicilsav.
>
> Javallatok: heveny szívizomelhalás (miokardiális infarktus) esetén; egy újabb szívinfarktus kialakulásának megelőzésére; átmeneti agyi keringési zavar (úgynevezett TIA – tranziens iszkémiás attack) és az agyi keringészavar következtében kialakuló szövetelhalás (agyi infarktus) megelőzésére.
>
> Adagolás: Szívinfarktus gyanúja esetén azonnal 100-300 mg Aspirin Protect tabletta bevétele javasolt. Az infarktust követően 30 napon át naponta 100-300 mg Aspirin Protect bevétele javasolt.
>
> Lehetséges mellékhatások: szédülés, fülzúgás, orrvérzés, hasi fájdalom, kiütés, vashiányos vérszegénység, agranulocitózis, anafilaxiás reakciók.
>
> Ne szedje, ha allergiás (túlérzékeny) a szalicilátokra vagy az Aspirin Protect egyéb összetevőjére; ha heveny gyomor-, bélrendszeri fekélyben szenved; ha fokozott vérzékenységben szenved; ha súlyos vesekárosodásban szenved; ha súlyos májkárosodásban szenved; ha súlyos szívelégtelenségben szenved.

**Gold:**

| field | value |
|---|---|
| drug_name | Aspirin Protect |
| active_ingredient | acetilszalicilsav |
| indication | heveny szívinfarktus esetén, újabb szívinfarktus megelőzésére, valamint átmeneti agyi keringési zavar (TIA) és agyi infarktus megelőzésére |
| dosage | szívinfarktus gyanúja esetén azonnal 100-300 mg; az infarktust követően 30 napon át naponta 100-300 mg |
| side_effects | • szédülés<br>• fülzúgás<br>• orrvérzés<br>• hasi fájdalom<br>• kiütés<br>• vashiányos vérszegénység<br>• agranulocitózis<br>• anafilaxiás reakciók |
| contraindications | • allergia a szalicilátokra<br>• heveny gyomor-bélrendszeri fekély<br>• fokozott vérzékenység<br>• súlyos vesekárosodás<br>• súlyos májkárosodás<br>• súlyos szívelégtelenség |

---

### medical #20 — Bepanthen
🔗 [source](https://www.webbeteg.hu/gyogyszerkereso/bepanthen-krem-30/511)

**Document:**

> BEPANTHEN krém
>
> Hatóanyag: dexpanthenol.
>
> Javallatok: sebgyógyulás és hámosodás elősegítésére kisebb sérülések (banális égések és horzsolások) esetén; bőrirritáció (például radioterápia, fototerápia vagy UV fény expozíció után); krónikus fekélyesedések és felfekvési seb; a bőr szárazságának, berepedezettségének megelőzésére; szoptatás alatt a mell rendszeres ápolására és a sebes vagy berepedezett mellbimbó kezelésére.
>
> Lehetséges mellékhatások: átmeneti allergiaszerű bőrtünetek.
>
> Ne alkalmazza, ha túlérzékeny a dexpantenolra vagy a gyógyszer egyéb összetevőjére.

**Gold:**

| field | value |
|---|---|
| drug_name | Bepanthen |
| active_ingredient | dexpanthenol |
| indication | sebgyógyulás és hámosodás elősegítése kisebb sérülések (égések, horzsolások) esetén, bőrirritáció, krónikus fekélyesedések és felfekvési sebek kezelése, valamint a bőr szárazságának és berepedezettségének megelőzése |
| dosage | _null_ |
| side_effects | • allergiaszerű bőrtünetek |
| contraindications | • túlérzékenység a dexpantenolra |

---

### medical #21 — Algoflex Gyorsan Ható
🔗 [source](https://www.patikaradar.hu/betegtajekoztatok/algoflex-gyorsan-hato-400-mg-filmtabletta-8c8889e8f981/)

**Document:**

> Algoflex Gyorsan Ható 400 mg filmtabletta
>
> Hatóanyag: ibuprofén (ibuprofén-lizinát formájában).
>
> Milyen betegségek esetén alkalmazható: különböző eredetű, enyhe, illetve közepesen erős akut fájdalmak tüneti kezelésére, például fejfájás (beleértve a tenziós fejfájást és a migrént), fogfájás, foghúzást követő fájdalom. A láz tüneti kezelésére is alkalmas.
>
> Adagolás: A kezdő adag egy tabletta Algoflex Gyorsan Ható (400 mg ibuprofén), majd szükség szerint további egy tabletta (400 mg ibuprofén) 6 óránként. Egy nap alatt összesen legfeljebb 3 tabletta szedhető.
>
> Lehetséges mellékhatások: hányinger, hányás, gyomorégés, hasmenés, szorulás, puffadás; fejfájás, szédülés, álmatlanság; ritkán súlyos reakciók, például szívroham és májelégtelenség.
>
> Ne szedje, ha: allergiás az ibuprofénre; gyomor- vagy nyombélvérzése vagy fekélye van; súlyos szív-, vese- vagy májelégtelenségben szenved; terhessége utolsó harmadában van.

**Gold:**

| field | value |
|---|---|
| drug_name | Algoflex Gyorsan Ható |
| active_ingredient | ibuprofén |
| indication | különböző eredetű, enyhe, illetve közepesen erős akut fájdalmak és láz tüneti kezelése |
| dosage | kezdő adag egy tabletta (400 mg ibuprofén), majd szükség szerint további egy tabletta 6 óránként; naponta legfeljebb 3 tabletta |
| side_effects | • hányinger<br>• hányás<br>• gyomorégés<br>• hasmenés<br>• szorulás<br>• puffadás<br>• fejfájás<br>• szédülés<br>• álmatlanság<br>• szívroham<br>• májelégtelenség |
| contraindications | • allergia az ibuprofénre<br>• gyomor- vagy nyombélvérzés vagy fekély<br>• súlyos szív-, vese- vagy májelégtelenség<br>• terhesség utolsó harmada |

---

## business (15)

### business #1 — Merkantil Bank
🔗 [source](https://www.portfolio.hu/bank/20260706/felvasarlas-a-magyar-lizingpiacon-bovul-a-merkantil-bank-847750)

**Document:**

> A Merkantil Bank megvásárolja a Business Lease Hungary Kft.-t, amivel az OTP Csoporthoz tartozó pénzintézet az operatív lízing és a flottakezelési szolgáltatások piacán bővíti a portfólióját. A százszázalékos üzletrész megvásárlásáról szóló megállapodást a felek már aláírták, a tranzakció pedig a hatósági jóváhagyások megszerzését követően, várhatóan szeptemberben zárul le a közlemény szerint.
>
> A nemzetközi lízingpiacon egyre nagyobb az igény a komplex, szolgáltatásalapú megoldásokra, azon belül is kifejezetten az operatív lízinghez kapcsolódó flottakezelésre. Mivel ez a szegmens Magyarországon a nyugat-európai országokhoz képest még kevésbé elterjedt, a terület jelentős növekedési potenciállal bír.

**Gold:**

| field | value |
|---|---|
| company | Merkantil Bank |
| event_type | felvásárlás |
| amount | _null_ |
| currency | _null_ |
| date | _null_ |
| involved_parties | • Merkantil Bank<br>• Business Lease Hungary Kft.<br>• OTP Csoport |

---

### business #2 — AutoWallis
🔗 [source](https://www.portfolio.hu/uzlet/20260227/porogtek-az-autoeladasok-megis-visszaesett-a-profit-az-autowallisnal-820874)

**Document:**

> Vegyes eredményeket szállított 2025-ben az AutoWallis, ugyan az eladások és a bevétel is közel 20 százalékot ugrott, a növekedés ára rövid távon megjelent az eredménysoron. Az EBITDA csökkent, az EPS visszaesett, részben a marzsokra nehezedő versenynyomás, részben a növekvő költségek miatt. A menedzsment viszont továbbra is optimista, és hisz a hosszú távú stratégiai célok megvalósíthatóságában.
>
> Az AutoWallis 2025-ös évét a látványos növekedés és a romló jövedelmezőség kettőssége jellemezte. Az értékesített gépjárművek száma 11,7 százalékkal, 54 ezer darab fölé ugrott tavaly, míg a negyedik negyedévben 21,6 százalékos növekedést követően 14 640 autót adott el a cég.

**Gold:**

| field | value |
|---|---|
| company | AutoWallis |
| event_type | éves eredmények (2025) |
| amount | _null_ |
| currency | _null_ |
| date | _null_ |
| involved_parties | _[]_ |

---

### business #3 — Mol-csoport
🔗 [source](https://www.portfolio.hu/uzlet/20260616/bejelentette-a-mol-megvan-a-reszvenyesi-megallapodas-a-nis-iranyitasarol-843330)

**Document:**

> A Mol-csoport Részvényesi Megállapodást írt alá a szerb kormánnyal a Naftna Industrija Srbije (NIS) jövőbeli irányításáról, miután a Gazpromnyefttel folytatott tárgyalások keretében a vállalat 56,15 százalékos részesedésének megvásárlására készül. A tranzakcióval a magyar olajcég Szerbia egyetlen nagy olajfinomítójának, a pancsovai létesítménynek, valamint egy több mint 400 töltőállomásból álló hálózatnak a többségi tulajdonosává válna. Az ügylet hátterében az áll, hogy az orosz tulajdonosi háttér miatt a NIS amerikai szankciós nyomás alá került, ami kényszerhelyzetet teremtett mind Szerbia, mind a Gazpromnyeft számára. A vételár korábbi becslések szerint 900 millió és 1,41 milliárd dollár között alakulhat, a tranzakció lezárásához azonban még további hatósági engedélyek szükségesek.

**Gold:**

| field | value |
|---|---|
| company | Mol-csoport |
| event_type | részesedés-felvásárlás |
| amount | _null_ |
| currency | _null_ |
| date | _null_ |
| involved_parties | • Mol-csoport<br>• szerb kormány<br>• Gazpromnyeft<br>• Naftna Industrija Srbije |

---

### business #4 — MBH Bank
🔗 [source](https://www.vg.hu/penz-es-tokepiac/2026/07/mbh-bank-eurokotveny-kibocsatas-500-millio-euro)

**Document:**

> Jelentős befektetői érdeklődés mellett hajtott végre újabb sikeres nemzetközi kötvénykibocsátást az MBH Bank. A kiemelt befektetői érdeklődést jól mutatja, hogy közel két és félszeres túljegyzés érkezett a hitelintézet MREL-képes kötvényeire. A magyar nagybank az ajánlatok beérkezése után végül 500 millió euró össznévértékű kötvénykibocsátást hajt végre. A mostani tranzakcióval együtt a bankcsoport az idei évben már összesen 1,5 milliárd euró értékben hajtott végre kötvénykibocsátásokat a nemzetközi piacokon.
>
> Az MBH Bank nemzetközi kötvényprogramjának következő állomásaként újabb 500 millió euró értékű MREL-képes kötvények kibocsátásáról döntött a társaság. A befektetői érdeklődés a mostani kibocsátás során is rendkívül magas volt, és közel két és félszeres túljegyzés mellett, a hitelintézet végül 500 millió euró össznévértékű kötvénykibocsátásról döntött.

**Gold:**

| field | value |
|---|---|
| company | MBH Bank |
| event_type | kötvénykibocsátás |
| amount | `500000000` |
| currency | EUR |
| date | _null_ |
| involved_parties | _[]_ |

---

### business #5 — DH Group
🔗 [source](https://www.portfolio.hu/ingatlan/20260227/eros-evet-zart-a-dh-group-kiderult-mennyi-lehet-az-osztalek-820858)

**Document:**

> Fordulópontot hozott a 2025-ös év a DH Group számára: a vállalat nemcsak rekordokat döntött, hanem egy érettebb, regionálisan diverzifikált pénzügyi platformmá fejlődött. A csoport minden várakozást felülmúló eredményekkel zárta az évet, túlteljesítve az ötéves stratégiai terv mérföldköveit, miközben a 2026-os előrejelzések további stabil, organikus növekedést vetítenek előre.
>
> A tavalyi év utolsó negyedéve kiemelkedően sikerült a DH Groupnál: a konszolidált árbevétel 28 százalékos emelkedéssel 14,5 milliárd forintra nőtt, míg az EBITDA 75 százalékos ugrással elérte a 2,9 milliárd forintot.

**Gold:**

| field | value |
|---|---|
| company | DH Group |
| event_type | éves eredmények (2025) |
| amount | _null_ |
| currency | _null_ |
| date | _null_ |
| involved_parties | _[]_ |

---

### business #6 — Frasers Group
🔗 [source](https://www.vg.hu/kereskedelem/2026/07/hervis-frasres-aruhaz-sports-direct)

**Document:**

> A brit kiskereskedelmi és befektetési vállalat, a Frasers Group felvásárolta a Hervis romániai és magyarországi sportszer-kiskereskedelmi tevékenységét, ami újabb lépést jelent a cég közép- és kelet-európai terjeszkedésében. Az ügylet összesen 78 üzletet érint: Romániában 49, Magyarországon pedig 29 Hervis-áruház kerül az új tulajdonoshoz. A tranzakció 2026 júniusában zárult le, miután a vevő megszerezte a szükséges hatósági jóváhagyásokat. A Hervis üzletek még idén bezárnak, és a felvásárló új márkanév alatt nyitja újra azokat.
>
> A felvásárlás befejezése azt is jelenti, hogy a magyarországi és romániai Hervis áruházak megszűnnek. A brit cég ugyanis 2026 folyamán fokozatosan Sports Direct arculatra alakítja át ezeket az egységeket. A Sports Direct az Egyesült Királyság piacvezető sportáru-kereskedelmi márkája, melynek hétszáz üzlete van hazájában valamint Európában.

**Gold:**

| field | value |
|---|---|
| company | Frasers Group |
| event_type | felvásárlás |
| amount | _null_ |
| currency | _null_ |
| date | 2026 júniusában |
| involved_parties | • Frasers Group<br>• Hervis<br>• Sports Direct |

---

### business #7 — 4iG
🔗 [source](https://www.vg.hu/penz-es-tokepiac/2026/06/4ig-milliardos-tranzakciok-a-lathataron-a-cseh-ceggel)

**Document:**

> A 4iG többségi közvetett leányvállalatai, az N7 Defence Holding és az ARZENÁL Fegyvergyár, valamint közvetett kisebbségi leányvállalata, a Colt CZ Hungary kötelező erejű term sheetet (szándéknyilatkozat) írtak alá a Colt CZ Group Internationallel - közölte a magyar tőzsdei cég kedden kora reggel, még a tőzsdenyitás előtt.
>
> A term sheet a Colt CZ Hungary tőkehelyzetének rendezésére, a társaság tulajdonosi szerkezetének tervezett átalakítására, valamint a felek közötti jövőbeni gyártási és kereskedelmi együttműködés fő feltételeire vonatkozó keretrendszert rögzíti.

**Gold:**

| field | value |
|---|---|
| company | 4iG |
| event_type | szándéknyilatkozat (term sheet) |
| amount | _null_ |
| currency | _null_ |
| date | _null_ |
| involved_parties | • 4iG<br>• N7 Defence Holding<br>• ARZENÁL Fegyvergyár<br>• Colt CZ Hungary<br>• Colt CZ Group International |

---

### business #8 — Magyar Telekom
🔗 [source](https://www.vg.hu/penz-es-tokepiac/2026/06/magyar-telekom-ipari-meretekben-veszi-sajat-reszvenyeit)

**Document:**

> Több mint 3,3 millió, egészen pontosan 3 369 912 darab saját részvényt vásárolt kedden a Magyar Telekom a Budapesti Értéktőzsdén fix ügyletek keretében az Erste Befektetési Zrt. mint befektetési szolgáltató közreműködésével - jelentette be szerda reggel a távközlési társaság.
>
> A tranzakciókra a február 25-én és május 13-án közzétett rendkívüli tájékoztatásaival összhangban, a részvényesek javadalmazása céljából került sor.
>
> A részvények átlagára 2768,15 forint volt, az ügyleteket követően a társaság tulajdonában lévő saját részvények száma 25 495 694 darabra nőtt.

**Gold:**

| field | value |
|---|---|
| company | Magyar Telekom |
| event_type | sajátrészvény-vásárlás |
| amount | _null_ |
| currency | _null_ |
| date | _null_ |
| involved_parties | • Magyar Telekom<br>• Erste Befektetési Zrt.<br>• Budapesti Értéktőzsde |

---

### business #9 — MagNet Bank
🔗 [source](https://www.portfolio.hu/bank/20250610/bejelentettek-felvasarolja-egy-magyar-bank-a-masikat-767005)

**Document:**

> A 30 éves MagNet Bank megállapodott a Polgári Bank részvények döntő többségének felvásárlásáról. Az akvizícióval a MagNet Bank tovább növeli országos jelenlétét, különös tekintettel a kelet-magyarországi régiókra - közölték.
>
> A tranzakció véglegesítéséhez a Magyar Nemzeti Bank és a Gazdasági Versenyhivatal jóváhagyása megérkezett, így a Polgári Bank hamarosan a 100%-ban magyar magántulajdonú MagNet Bank leányvállalatává válik.

**Gold:**

| field | value |
|---|---|
| company | MagNet Bank |
| event_type | felvásárlás |
| amount | _null_ |
| currency | _null_ |
| date | _null_ |
| involved_parties | • MagNet Bank<br>• Polgári Bank<br>• Magyar Nemzeti Bank<br>• Gazdasági Versenyhivatal |

---

### business #10 — Turbine
🔗 [source](https://startuponline.hu/25-millio-dollaros-befektetest-szerzett-a-magyar-gyogyszerszerkutato-turbine)

**Document:**

> A Turbine 2026. február végén bejelentette: 25 millió dollár (körülbelül 8 milliárd forint) értékben sikerrel vont be tőkét Series B körében.
>
> Az ügyletet az Interactive Venture Partners amerikai befektetési alap vezeti, akihez új befektetőként csatlakozott az innovatív bőrápolás és -kutatás német úttörőjeként a Beiersdorf Venture Capital.
>
> A friss tőkét a Turbine arra tervezi fordítani, hogy bővítse sejtszimulációs platformját, és egy top kategóriás gyógyszeripari vállalattal együttműködésben az immunológia területére terjeszkedjen.
>
> A Turbine neve nem ismeretlen a gyógyszerkutatásban: a cégnek olyan partnerei vannak, mint az MSD (Merck & Co.), az AstraZeneca és a Bayer.

**Gold:**

| field | value |
|---|---|
| company | Turbine |
| event_type | finanszírozási kör (B sorozat) |
| amount | `25000000` |
| currency | USD |
| date | 2026. február végén |
| involved_parties | • Turbine<br>• Interactive Venture Partners<br>• Beiersdorf Venture Capital |

---

### business #11 — Indotek Group
🔗 [source](https://www.portfolio.hu/gazdasag/20260626/teljesen-magyar-kezbe-kerult-az-auchan-845894)

**Document:**

> Az Indotek Group megvásárolta az Auchan Retail Internationaltől az Auchan Magyarország Kft. fennmaradt 53 százalékos tulajdonrészét.

**Gold:**

| field | value |
|---|---|
| company | Indotek Group |
| event_type | felvásárlás |
| amount | _null_ |
| currency | _null_ |
| date | _null_ |
| involved_parties | • Indotek Group<br>• Auchan Retail International<br>• Auchan Magyarország Kft. |

---

### business #12 — Witorp Kft.
🔗 [source](https://www.vg.hu/vilaggazdasag-magyar-gazdasag/2026/05/leggazdagabb-magyar-veres-tibor-osztrak-multi-vaci-ut)

**Document:**

> Újabb jelentős ingatlan kerül magyar kézbe Budapesten, miután az osztrák CA Immo megállapodott a Capital Square irodaházat birtokló társaság értékesítéséről. Az üzlet vevői oldalán Veres Tibor érdekeltsége áll, aki közel 480 milliárd forintos vagyonával Magyarország leggazdagabb üzletemberei közé tartozik.
>
> Tulajdonost vált Budapest egyik legismertebb irodaháza: a Váci úti üzleti negyed meghatározó épületének számító Capital Square magyar kézbe kerül, miután a Wing csoporthoz tartozó Witorp Kft. megállapodott az osztrák CA Immobilien Anlagen AG-val az ingatlant birtokló társaság megvásárlásáról. Az ügylet révén az egyik legértékesebb budapesti irodaingatlan felett szerezhet ellenőrzést Veres Tibor érdekeltsége, miközben az osztrák ingatlancsoport egy újabb magyarországi eszközétől válik meg.

**Gold:**

| field | value |
|---|---|
| company | Witorp Kft. |
| event_type | felvásárlás (ingatlantranzakció) |
| amount | _null_ |
| currency | _null_ |
| date | _null_ |
| involved_parties | • Witorp Kft.<br>• Wing<br>• CA Immobilien Anlagen AG<br>• Veres Tibor |

---

### business #13 — 4iG
🔗 [source](https://index.hu/gazdasag/2025/12/01/4ig-netcom-telecom-uzlet-vasarlas-tranzakcio-gvh-jovahagyas/)

**Document:**

> A 4iG megvásárolta a Netfone Telecom virtuális mobilszolgáltató 99 százalékos tulajdonrészét. Utóbbi vállalat 106 ezer ügyféllel rendelkezik. A tranzakciót a Gazdasági Versenyhivatal 2025 októberében jóváhagyta.
>
> A Budapesti Értéktőzsde honlapján közzétett közlemény szerint lezárult a Netfone Telecom Távközlési és Szolgáltató Kft. megvásárlására vonatkozó tranzakció, amelynek eredményeként a 4iG Távközlési Holding Zrt. megszerezte az országos virtuális mobilszolgáltató 99 százalékos tulajdonrészét.

**Gold:**

| field | value |
|---|---|
| company | 4iG |
| event_type | felvásárlás |
| amount | _null_ |
| currency | _null_ |
| date | _null_ |
| involved_parties | • 4iG<br>• Netfone Telecom<br>• 4iG Távközlési Holding Zrt.<br>• Gazdasági Versenyhivatal |

---

### business #14 — PastPay
🔗 [source](https://fintech.hu/a-pastpay-12-millio-euros-finanszirozasi-kort-zart-ez-az-eddigi-legnagyobb-tokebevonas-a-kozep-kelet-europai-b2b-bnpl-piacon/)

**Document:**

> A magyar PastPay 12 millió eurós finanszírozási kört zárt – rekordot döntött Közép-Kelet-Európában
>
> A kockázati-tőke befektetési szakértő Platina Capital által vezetett, közel 5 milliárd forintos finanszírozási kör részben a PastPay üzletfejlesztésének felgyorsítására, részben pedig a B2B BNPL finanszírozásra lesz fordítva. A Platina Capital által vezetett befektetési kör számos vezető pénzügyi intézmény és neves magánbefektető közreműködésével zajlott. A PastPay küldetése, hogy innovatív B2B finanszírozási megoldásával rugalmas fizetési feltételeket biztosítson az üzleti tranzakciók során.

**Gold:**

| field | value |
|---|---|
| company | PastPay |
| event_type | finanszírozási kör (A sorozat) |
| amount | `12000000` |
| currency | EUR |
| date | _null_ |
| involved_parties | • PastPay<br>• Platina Capital |

---

### business #15 — Waberer's
🔗 [source](https://www.portfolio.hu/uzlet/20250526/a-waberers-tobbsegi-reszesedest-szerzett-a-pannon-busz-rentben-763781)

**Document:**

> A Waberer's bejelentette, hogy a 2025. február 25-én aláírt adás-vételi szerződésnek megfelelően a társaság ma lezárta a közúti személyszállítási szolgáltatást nyújtó Pannon-Busz-Rent 51%-os részesedésének megvásárlását.

**Gold:**

| field | value |
|---|---|
| company | Waberer's |
| event_type | felvásárlás (részesedésszerzés) |
| amount | _null_ |
| currency | _null_ |
| date | 2025-02-25 |
| involved_parties | • Waberer's<br>• Pannon-Busz-Rent |

---

## technology (20)

### technology #1 — Redmi Note 15 Pro 5G
🔗 [source](https://mobilarena.hu/teszt/xiaomi_redmi_note_15_pro_5g_teszt_velemeny/bevezeto_doboz_tartalma.html)

**Document:**

> Idén is öt modellel startolt el január harmadik hetével a Redmi Note 15-ös sorozat. Az összes Note 15 Pro 8/256 GB-os kiszerelésben kerül hivatalos magyarországi forgalomba. 139 990 forintért kapható a Redmi Note 15 Pro 5G. A telefon mellé csak egy szürke szilikon védőtokot, egy SIM-tálcát nyitó tűt, egy USB-A végű Type-C kábelt kap. A 45 wattos töltő tehát kimarad az EU-ban forgalomba kerülő modellek mellől.

**Gold:**

| field | value |
|---|---|
| product | Redmi Note 15 Pro 5G |
| manufacturer | Xiaomi |
| version | _null_ |
| key_specs | • 8/256 GB-os kiszerelés<br>• 45 wattos töltő<br>• 5G |
| release_date | _null_ |
| price | `139990` |

---

### technology #2 — LG C4
🔗 [source](https://prohardver.hu/teszt/lg_c4_2024_oled_teve/kepminoseg.html)

**Document:**

> Az LG C4 esetében a képminőségért az LG Display által fejlesztett és gyártott OLED evo panel, és a házon belül készülő, új, 7. generációs Alpha 9 képfeldolgozó chip felel. A kijelző 3840x2160 képpont megjelenítését szavatolja, tehát egy 4K-s televízióról van szó. Ez pedig azt jelenti, hogy az általunk tesztelt 65 hüvelykes képátló esetében a pixelsűrűség 67,6 ppi. A maximális fényerő terén komoly változás nem tapasztalható, hiszen 1174,3 cd/m² értéket mértünk. A modell maximális natív képfrissítési rátája 144 Hz, ami előrelépés a tavalyi év 120 hertzéhez képest. Az input lag a Leo Bodnar eszközzel végzett méréseink alapján csak 10,3 ms.

**Gold:**

| field | value |
|---|---|
| product | LG C4 |
| manufacturer | LG |
| version | _null_ |
| key_specs | • OLED evo panel<br>• 7. generációs Alpha 9 képfeldolgozó chip<br>• 3840x2160 képpont (4K)<br>• 65 hüvelykes képátló<br>• 67,6 ppi pixelsűrűség<br>• 1174,3 cd/m² maximális fényerő<br>• 144 Hz maximális natív képfrissítési ráta<br>• 10,3 ms input lag |
| release_date | _null_ |
| price | _null_ |

---

### technology #3 — LG G3
🔗 [source](https://prohardver.hu/teszt/lg_g3_oled_teve_teszt/lg_g3_teve.html)

**Document:**

> Az LG korábbi jó szokását megtartva idén is minden vetélytársánál több OLED televíziót küld csatába. Az LG G3 televíziót, hasonlóan a „G”-széria korábbi darabjaihoz alapvetően arra szánták, hogy a felhasználó a falra akassza. Végre megérkezett a hazai boltokba is, így nekünk is lehetőségünk nyílt rá, hogy néhány napon át tesztelhessük 55 hüvelykes változatát. Panel: 10 bites Micro Lens OLED Evo panel (40-120 Hz). Képernyőméret: 55 hüvelyk (139 cm). Maximális felbontás: Ultra High Definition (3840x2160 pixel). Alpha9 képfeldolgozó processzor. Tömeg: 17,8 kg. Fogyasztói ár: kb. 940 000 forint.

**Gold:**

| field | value |
|---|---|
| product | LG G3 |
| manufacturer | LG |
| version | _null_ |
| key_specs | • 10 bites Micro Lens OLED Evo panel (40-120 Hz)<br>• 55 hüvelyk (139 cm)<br>• Ultra High Definition (3840x2160 pixel)<br>• Alpha9 képfeldolgozó processzor<br>• 17,8 kg |
| release_date | _null_ |
| price | `940000` |

---

### technology #4 — iPhone 15
🔗 [source](https://www.penzcentrum.hu/tech/20230912/itt-vannak-az-iphone-15-es-iphone-15-pro-arai-bemutattak-az-apple-uj-csucsmobiljait-1141151)

**Document:**

> Az iPhone 15-nél a legnagyobb újdonság, hogy már az alap, nem Pro telefonok is megkapják a dynamic islandet, ami most csak a 14 Pro-kban elérhető. A 15 és 15 Plus minden eddiginél Ceramic Shield bevonatot kapott, 2000 nites a kijelző fényereje, és javarészt újrahasznosított alumíniumból készül.
> A 15-ösnél ugyanúgy 6,1 hüvelykes a „sima”, és 6,7 hüvelykes a Plus változat, mint az eggyel ezelőtti szériában, a telefon éleit valamelyest lekerekítették. A főkamera 48 megapixeles, jobb mint a 14-ben volt, és megkapja a legfejlettebb szoftveres rásegítéseket, az ultraszéles kamera 12 megapixel felbontású.
> Az iPhone 15 az USA-ban 799 dolláros áron kerül bevezetésre, ami forintban 286 000, a Plus pedig kint 899 dollár lesz, ami 322 ezer forint átváltva. A magyar árakról egyelőre nincs információ, amint azok is kijönnek, közöljük majd őket.

**Gold:**

| field | value |
|---|---|
| product | iPhone 15 |
| manufacturer | Apple |
| version | _null_ |
| key_specs | • Dynamic Island<br>• Ceramic Shield bevonat<br>• 2000 nites kijelző fényereje<br>• 6,1 hüvelykes kijelző<br>• 48 megapixeles főkamera<br>• 12 megapixel ultraszéles kamera |
| release_date | _null_ |
| price | _null_ |

---

### technology #5 — Galaxy S24 FE
🔗 [source](https://www.origo.hu/techbazis/2024/12/samsung-galaxy-s24fe-teszt)

**Document:**

> Az S24 FE dizájnja letisztult, modern, és határozottan prémium érzetet kelt. A vékony kávák, az alumínium keret és a Gorilla Glass Victus 2 védelem mind a minőségi megmunkálásról tanúskodnak. A 6,4 hüvelykes Dynamic AMOLED 2X kijelző pedig lenyűgöző látványt nyújt. A színek élénkek és pontosak, a kontrasztarány kiváló, a 120Hz-es adaptív frissítési frekvencia pedig gördülékeny görgetést és játékélményt biztosít.
> Az S24 FE a Samsung saját Exynos 2400 processzorát használja, ami a Geekbench tesztek alapján felveszi a versenyt a Snapdragon 8 Gen 3-mal. A 8GB RAM bőven elegendő a több alkalmazás egyidejű futtatásához, a játékok és a nehéz feladatok elvégzéséhez. A telefon akkumulátora 4500mAh kapacitású, ami kényelmesen kibír egy teljes napot normál használat mellett.

**Gold:**

| field | value |
|---|---|
| product | Galaxy S24 FE |
| manufacturer | Samsung |
| version | FE |
| key_specs | • 6,4 hüvelykes Dynamic AMOLED 2X kijelző<br>• Gorilla Glass Victus 2 védelem<br>• 120Hz-es adaptív frissítési frekvencia<br>• Exynos 2400 processzor<br>• 8GB RAM<br>• 4500mAh akkumulátor |
| release_date | _null_ |
| price | _null_ |

---

### technology #6 — WH-1000XM5
🔗 [source](https://mobilarena.hu/teszt/sony_wh-1000xm5_teszt/bevezeto_doboz.html)

**Document:**

> Nem lehet azt mondani, hogy a Sony a semmiből robbant be az aktív zajszűrős Bluetooth fejhallgatók piacára, hat évvel ezelőtt érkezett az MDR-1000X, rá egy évre gyors névváltással a WH-1000XM2, aztán a Sony ezt a formulát csiszolta tovább 2018-ban a WH-1000XM3-mal, 2020-ban pedig a WH-1000XM4-gyel. Eljött 2022, s a Sony úgy érezte, fel kell rúgnia az eddigi gyakorlatot és nagyobbat kell ugrania, mint az elmúlt években és két részre kell bontania az aktív zajszűrős prémium fejhallgatók vonalát.
> De azért tegyük hozzá, hogy egyik eszköz sem olcsó: az újonc WH-1000XM5 hazai induló ára 155 ezer forint, a 2020-as WH-1000XM4 pedig 130 ezer forint a Sony hivatalos magyar oldalán.

**Gold:**

| field | value |
|---|---|
| product | WH-1000XM5 |
| manufacturer | Sony |
| version | _null_ |
| key_specs | • aktív zajszűrő<br>• Bluetooth fejhallgató<br>• prémium kategória |
| release_date | 2022 |
| price | `155000` |

---

### technology #7 — Pixel 8a
🔗 [source](https://mobilarena.hu/teszt/google_pixel_8a_teszt_velemeny/bevezeto_doboz_kulso.html)

**Document:**

> Idén májusban mutatta be a Google a Pixel 8a-t, a Pixel 8-as modellek jóárasított változatát. A telefon hozzánk jelentős késéssel jutott el, de végül közvetlenül a Google hazai képviselete küldte el a modellt, mert szeptemberrel megindult a Pixel készülékek hazai forgalmazása. A Pixel 9 Pro XL után érkezett hozzánk meg a Pixel 8a, és mivel ez még régebbi készülék, a dobozában az USB-C kábel mellett találunk egy USB-A átalakítót is a telefon mellett. Tok vagy fali adapter nincsen, az ár pedig 235 ezer forint a 8/128 GB-os modellért a hivatalos értékesítési csatornákon.
> A 6,1 hüvelykes OLED kijelző 1080 x 2400 pixeles felbontást (420 ppi) kínál 9:20-as képarányon. Megvan a HDR10, a HLG és a HDR10+ támogatás, illetve az akár 120 Hz-es frissítés.

**Gold:**

| field | value |
|---|---|
| product | Pixel 8a |
| manufacturer | Google |
| version | _null_ |
| key_specs | • 6,1 hüvelykes OLED kijelző<br>• 1080 x 2400 pixeles felbontás (420 ppi)<br>• HDR10, HLG és HDR10+ támogatás<br>• 120 Hz-es frissítés<br>• 8/128 GB |
| release_date | _null_ |
| price | `235000` |

---

### technology #8 — Galaxy S24
🔗 [source](https://mobilarena.hu/teszt/samsung_galaxy_s24_teszt_velemeny/kijelzo_teljesitmeny_akku.html)

**Document:**

> A doboz tartalmában nem fogunk különbséget találni, a Samsung a kisebb modellhez sem ad semmit a töltőkábelen kívül, viszont kompenzál valamennyit érte az IP68-as vízállósággal és a jó összeszerelési minőséggel. A kijelzőt Gorilla Glass Victus 2 védi itt is, ám a panel bár ugyanaz a Dynamic LTPO AMOLED 2X típus, mint a nagytestvérben, csak 8-bites és 6,2 hüvelyken itt csak 1080 x 2340 pixel (416 ppi) terül szét.
> Az Exynos 2400 a kisebb és alacsonyabb felbontású kijelzővel jobb eredményeket ért el a grafikailag intenzívebb teszteken, ám általánosságban véve ugyanazt a szintet hozta, mint középső modell.

**Gold:**

| field | value |
|---|---|
| product | Galaxy S24 |
| manufacturer | Samsung |
| version | _null_ |
| key_specs | • IP68-as vízállóság<br>• Gorilla Glass Victus 2<br>• 6,2 hüvelykes Dynamic LTPO AMOLED 2X kijelző<br>• 1080 x 2340 pixel (416 ppi)<br>• Exynos 2400 |
| release_date | _null_ |
| price | _null_ |

---

### technology #9 — LG C3
🔗 [source](https://prohardver.hu/teszt/lg_c3_oled_teve_teszt/lg_c3.html)

**Document:**

> A káva keskeny, kikapcsolt állapotban láthatatlan, a talp diszkrét, az összhatás ízléses. A harmadik generációnál tartó széria a belépőszintű, ám olcsóbb B3 és a nagyobb tudású, de értelemszerűen drágább LG G3 között helyezkedik el. Panel: 10 bites OLED Evo panel (40-120 Hz). Képernyőméret: 55 hüvelyk (139 cm). Maximális felbontás: Ultra High Definition (3840x2160 pixel). Alpha9 képfeldolgozó processzor. Fogyasztói ár: kb. 760 000 forint.

**Gold:**

| field | value |
|---|---|
| product | LG C3 |
| manufacturer | LG |
| version | _null_ |
| key_specs | • 10 bites OLED Evo panel (40-120 Hz)<br>• 55 hüvelyk (139 cm)<br>• Ultra High Definition (3840x2160 pixel)<br>• Alpha9 képfeldolgozó processzor |
| release_date | _null_ |
| price | `760000` |

---

### technology #10 — Nothing Phone (2)
🔗 [source](https://mobilarena.hu/teszt/nothing_phone_2_teszt_kritika_velemeny/bevezeto_kulso_doboz.html)

**Document:**

> Az újonc 8/128 GB-os kiszereléssel indít 246 ezer forintos áron hazánkban, ez jelentősen magasabb az elődnél, cserébe megvan a Snapdragon 8+ Gen 1-es csúcskategóriás lapkakészlet, valamivel gyorsabb lett a töltés és nagyobb az aksi, javult a víz- és porállóság és megkapta az IP54-es minősítést, valamint a nagyobb házban átrendezték kicsit a hátlapi LED fényeket.
> A hátlapi Glyph interfész logikája és alapvető felhasználási módja nem változott a tavalyihoz képest, pár szoftveres extra került bele ugyanakkor, ilyen például a Glyph időzítő is.

**Gold:**

| field | value |
|---|---|
| product | Nothing Phone (2) |
| manufacturer | Nothing |
| version | _null_ |
| key_specs | • 8/128 GB-os kiszerelés<br>• Snapdragon 8+ Gen 1 lapkakészlet<br>• IP54-es minősítés<br>• hátlapi Glyph interfész |
| release_date | _null_ |
| price | `246000` |

---

### technology #11 — Galaxy Watch6 Classic
🔗 [source](https://mobilarena.hu/teszt/samsung_galaxy_watch6_classic_teszt/kulso_kijelzo_doboz.html)

**Document:**

> A Galaxy Watch6 két kiadásban, négy méretben és nyolc változatban érkezett meg, és valahol örömhír, hogy a 40 mm-es alapmodell 130 ezerért elvihető, mert mindent tud, amit a drágabb kiszerelések - a 44 mm-es ára kapásból 150 ezer forint. A legendás lünettát visszahozó, 43 mm-es Watch6 Classic 170 rugós nyitással sokkol, ami indokolatlanul nagy felár.
> Hozzám a 43 mm-es Classic került, és azonnal rátekertem a tavaly kihagyott lünettára, precíz lépdelést tapasztalva, de a két gomb és a kitűnő érintésérzékelés is rendben van. A keskeny gyűrű önmagában extra védelmet ad az 1,4"-re nőtt képátlónak és a zafírüvegnek, de megvan az IP68 és az 5 ATM minősítés is a sekélyvizes úszáshoz, valamint a MIL-STD-810H jelző.

**Gold:**

| field | value |
|---|---|
| product | Galaxy Watch6 Classic |
| manufacturer | Samsung |
| version | Classic 43 mm |
| key_specs | • forgatható lünetta<br>• 1,4" képátló<br>• zafírüveg<br>• IP68 és 5 ATM minősítés<br>• MIL-STD-810H |
| release_date | _null_ |
| price | _null_ |

---

### technology #12 — iPad Air 13"
🔗 [source](https://mobilarena.hu/teszt/apple_ipad_air_13_m2_2024_a_meret_a_lenyeg/bevezeto_doboz.html)

**Document:**

> A tesztünk alanyául szolgáló eddigi legnagyobb iPad Air 13 hüvelykes kijelzővel 399 990 forintos árral indul (474 990 forint, ha kell az 5G-s modem is), míg az ugyanekkora méretű Pro tablet 649 990 forint minimum (749 990 forint 5G modemmel).
> A dobozt kiürítette az Apple, idén már nem jár fali adapter a tablet mellé, csak egy mindkét végén USB-C csatlakozós kábelt adnak, az legalább szövet borítású, talán strapabíróbb. Az új iPad Airek szürke, kék, lila és csillagfény (fehér) színekben érhetők el.

**Gold:**

| field | value |
|---|---|
| product | iPad Air 13" |
| manufacturer | Apple |
| version | M2 (2024) |
| key_specs | • 13 hüvelykes kijelző<br>• opcionális 5G-s modem<br>• USB-C csatlakozó |
| release_date | _null_ |
| price | `399990` |

---

### technology #13 — PlayStation 5 Pro
🔗 [source](https://www.pcwplus.hu/pcwlite/megvan-a-playstation-5-pro-magyar-ara-es-hat-igen-draga-lesz-359899.html)

**Document:**

> Már előrendelhető a konzol, ami a megjelenéskor várhatóan limitált példányszámban érkezik, de még idén lesz újabb szállítmány.
> A PlatinumShop oldalán két csomag volt elérhető: az egyik az alapgép 329 990 Ft-ért, a másikhoz pedig lemezmeghajtó is jár, ez 379 990 Ft-ba kerül.

**Gold:**

| field | value |
|---|---|
| product | PlayStation 5 Pro |
| manufacturer | Sony |
| version | _null_ |
| key_specs | • limitált példányszámú megjelenés<br>• opcionális lemezmeghajtó |
| release_date | _null_ |
| price | `329990` |

---

### technology #14 — ROG Strix Scar 18
🔗 [source](https://index.hu/tech/godmode/2025/05/02/asus-rog-strix-scar-18-g835lx-teszt/)

**Document:**

> A motorháztető alatti elemeket ugyancsak nem aprózta el az Asus, mind a processzor, mind a videokártya tekintetében az abszolút csúcskategóriát kapjuk. Előbbi egy 24 maggal operáló Intel Core Ultra 9 275HX, míg utóbbi egy Nvidia GeForce RTX 5090 - persze nem a „rendes”, hanem a laptopokba szánt verzió. Egy 18 hüvelykes, 2560 x 1600 képpontos, 16:10 képarányú, 240 hertzes, matt felületű mini LED-kijelzőt kapunk.
> Az Asus ROG Strix Scar 18 G835LX április végén/május elején kerül a boltok polcaira, viszont már most előrendelhető az Asus estore-jából potom 2 millió 589 ezer forintos ajánlott fogyasztói áron.

**Gold:**

| field | value |
|---|---|
| product | ROG Strix Scar 18 |
| manufacturer | Asus |
| version | G835LX |
| key_specs | • Intel Core Ultra 9 275HX (24 mag)<br>• Nvidia GeForce RTX 5090 laptop<br>• 18 hüvelykes, 2560 x 1600 képpontos kijelző<br>• 16:10 képarány<br>• 240 hertzes mini LED-kijelző |
| release_date | _null_ |
| price | `2589000` |

---

### technology #15 — Samsung QN95C
🔗 [source](https://prohardver.hu/teszt/samsung_qn95c_teve_teszt/samsung_qn95c.html)

**Document:**

> Az elmúlt években megszokott irányvonaltól ezúttal sem tért el a vállalat, azaz a Samsung QN95C egy meglehetősen konzervatív külsejű televízió, amelyről hiányoznak a látványos, meghökkentő díszítőelemek. A dél-koreai cégnek akadnak olyan televíziói, amelynél a ki- és bemeneteket egy külső dobozon helyezik el, azonban a QN95C nem tartozik ezek közé, a csatlakozók a hátlapra kerültek. Található itt négy darab HDMI 2.1, két USB 2.0, egy Ethernet és egy optikai digitális audió csatlakozó is. Samsung QN95C (QE65QN95CATXXH). Panel: Samsung Neo QLED (10 bit), 144 Hz, FreeSync. Háttérvilágítás: Mini LED. Képernyőméret: 65 hüvelyk (165 cm). Felbontás: 4K (3840x2160 pixel). Fogyasztói ár: 1 250 000 forint.

**Gold:**

| field | value |
|---|---|
| product | Samsung QN95C |
| manufacturer | Samsung |
| version | QE65QN95CATXXH |
| key_specs | • Neo QLED (10 bit) panel, 144 Hz, FreeSync<br>• Mini LED háttérvilágítás<br>• 65 hüvelyk (165 cm)<br>• 4K (3840x2160 pixel)<br>• négy HDMI 2.1 |
| release_date | _null_ |
| price | `1250000` |

---

### technology #16 — Roomba i7+
🔗 [source](https://tesztplussz.hu/irobot-roomba-i7-robotporszivo-teszt/)

**Document:**

> Az iRobot jelenleg kétféle típusban kínálja a vásárlók számára a Roomba i7-et: mi az i7+ modellt vettük górcső alá, ami abban különbözik az alapkiadástól, hogy a robotporszívó mellé jár egy Clean Base névre keresztelt automatikus szennyeződésgyűjtő rendszer. Ez arra szolgál, hogy a dokkolóhoz tartozó eszköz képes kiszippantani a robotporszívó belső tartályából egy külső porzsákba az összegyűjtött koszt és port, ami kifejezetten kényelmes megoldás.
> A Roomba i7 és i7+ közti különbség azonban az automatikus szennyeződésgyűjtő képesség mellett abban is megmutatkozik, hogy itthon például 300 ezer forint körüli összegért lehet beszerezni az automatikus ürítőfunkcióra alkalmas, tehát portartályos dokkolóval ellátott típust, ami sokak számára nagyon magas összeg.

**Gold:**

| field | value |
|---|---|
| product | Roomba i7+ |
| manufacturer | iRobot |
| version | i7+ |
| key_specs | • Clean Base automatikus szennyeződésgyűjtő rendszer<br>• portartályos dokkoló<br>• automatikus ürítőfunkció |
| release_date | _null_ |
| price | `300000` |

---

### technology #17 — OnePlus 12
🔗 [source](https://mobilarena.hu/hir/oneplus_12_es_12r_europai_ar.html)

**Document:**

> A csúcsmodell is régiónként eltérő áron indít zöld és fekete színű házzal, és annyi különbséggel a kínai OnePlus 12 modellekhez képest, hogy a globális verzióból nem készül 24 GB / 1 TB kiszerelésű változat, 16/512 GB az elérhető legnagyobb opció. Ennek az ára hazánkból 1169 euró, előrendelési akcióban 1069 euró, a kisebb modell 12/256 GB-os kiszerelésben a magyar webshopban 999 euró. Itt is LPDDR5x típusú RAM és UFS 4.0-s tárhely van. A kijelző 6,82 hüvelykes LTPO4 AMOLED kijelző 120 Hz-es frissítéssel és 1440 x 3168 pixeles felbontással.

**Gold:**

| field | value |
|---|---|
| product | OnePlus 12 |
| manufacturer | OnePlus |
| version | 12/256 GB |
| key_specs | • 6,82 hüvelykes LTPO4 AMOLED kijelző<br>• 120 Hz-es frissítés<br>• 1440 x 3168 pixeles felbontás<br>• LPDDR5x RAM<br>• UFS 4.0-s tárhely |
| release_date | _null_ |
| price | `999` |

---

### technology #18 — Motorola Edge 50 Ultra
🔗 [source](https://mobilarena.hu/teszt/motorola_edge_50_ultra_teszt_velemeny_kritika/bevezeto_doboz_kulso.html)

**Document:**

> 2024 különlegesebb telefonjai közé sorolható a Motorola Edge 50 Ultra, amit főleg megjelenésével ér el. Pont tíz évvel ezelőtt a Moto X hátlapján már kísérleteztek fával, most aztán ahogy a márka többször anyacéget váltott, úgy halt el az egyedi hátlap. Idén az Edge 50 Ultrán visszatért, az aktuális csúcsmodell elérhető műbőr borítású hátoldal mellett egy speciális fa változat is. Árban nincs eltérés a borítás okán, itthon 16 GB RAM-mal és 1 TB tárhellyel kapható a telefon 399 999 forintért.
> A Nordic Wood fantázianévre hallgató fa változat kerete ugyanúgy alumínium, az előlapot ugyanúgy Gorilla Glass Victus edzett üveg védi, de ellentétben a bőrhatású szilikon-polimer hátlaptól, itt valódi farostrétegekből álló hátoldalt kapunk.

**Gold:**

| field | value |
|---|---|
| product | Motorola Edge 50 Ultra |
| manufacturer | Motorola |
| version | _null_ |
| key_specs | • 16 GB RAM<br>• 1 TB tárhely<br>• Gorilla Glass Victus edzett üveg<br>• műbőr vagy fa (Nordic Wood) hátlap |
| release_date | 2024 |
| price | `399999` |

---

### technology #19 — HERO12 Black
🔗 [source](https://www.guidelee.hu/teszt/akciokamera/gopro-hero12/)

**Document:**

> A GoPro HERO12 Black a következő tagja az akciókamerák családjának, amely már régóta uralja a kompakt, strapabíró sportkamerák szegmensét. A 12. generáció számos újítást hoz, például Bluetooth hangtámogatást, továbbfejlesztett HyperSmooth stabilizációt, hosszabb akkumulátor-üzemidőt és HDR videófelvétel támogatást.
> A műszaki specifikációkat tekintve a 12. generáció szinte semmi változást nem hoz az elődjéhez képest. A kevés változás egyike, amely néhány felhasználó számára érdekes lehet, a GPS-kapcsolat hiánya. Ennek következtében a GoPro HERO12 nem tudja rögzíteni a felvétel helyét, sebességét, magasságát és egyéb paramétereket.

**Gold:**

| field | value |
|---|---|
| product | HERO12 Black |
| manufacturer | GoPro |
| version | Black |
| key_specs | • Bluetooth hangtámogatás<br>• HyperSmooth stabilizáció<br>• HDR videófelvétel támogatás<br>• nincs GPS-kapcsolat |
| release_date | _null_ |
| price | _null_ |

---

### technology #20 — Fenix 8 Pro
🔗 [source](https://mobilarena.hu/teszt/fenix_8_pro_teszt/kulso_kezeles_uzemido.html)

**Document:**

> Máskülönben a Fenix 8 Pro egy Fenix 8, csak vastagabb: egy tökéletesen egyberakott, 10 ATM védelemmel erősített, 40 méteres búvárkodást támogató, szálerősített polimer vázat kínáló monstrum titán hátlappal és kerettel, zafírüveg védelemmel és 22 mm-en cserélhető, jól szellőző, kényelmes, erős és kellően hosszú szilikonpánttal. Fenix 8 Pro 47 mm AMOLED: 47 x 47 x 16 mm, 77 g, 1,4" 454 x 454 pixel, 497 900 forint.

**Gold:**

| field | value |
|---|---|
| product | Fenix 8 Pro |
| manufacturer | Garmin |
| version | 47 mm AMOLED |
| key_specs | • 10 ATM védelem<br>• 40 méteres búvárkodás<br>• titán hátlap és keret<br>• zafírüveg védelem<br>• 1,4" 454 x 454 pixel AMOLED<br>• 77 g |
| release_date | _null_ |
| price | `497900` |

---
