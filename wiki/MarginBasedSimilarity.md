# Margin-based scoring do filtrowania dwujęzycznych par zdań

Margin-based scoring służy do oceny, czy dwa zdania w różnych językach są rzeczywiście swoimi odpowiednikami znaczeniowymi. Metoda może korzystać z embeddingów LaBSE, ale nie jest specyficzna dla tego modelu — działa z dowolnym modelem tworzącym wspólną, wielojęzyczną przestrzeń embeddingów.

Metodę zaproponowali Artetxe i Schwenk jako alternatywę dla filtrowania par za pomocą prostego, globalnego progu podobieństwa cosinusowego.

---

## Problem z samym cosine similarity

Najprostszy sposób filtrowania par wygląda następująco:

$$
\cos(x,y) \geq t
$$

gdzie:

- $x$ i $y$ to embeddingi zdań w dwóch językach,
- $t$ to ustalony próg.

Problem polega na tym, że wartości cosine similarity nie mają jednakowej „skali pewności" dla wszystkich zdań.

Przykładowo:

- para A ma cosine $0{,}82$, ale inne podobne zdania osiągają $0{,}81$, $0{,}80$ i $0{,}79$;
- para B ma cosine $0{,}75$, ale następni kandydaci osiągają tylko $0{,}52$, $0{,}49$ i $0{,}47$.

Mimo niższego cosine para B jest prawdopodobnie bardziej jednoznacznym dopasowaniem. Para A może zawierać zdania ogólne, boilerplate (standardowy, powtarzalny fragment tekstu) albo teksty mające wiele semantycznie podobnych odpowiedników.

Margin ocenia więc nie tylko bezwzględne podobieństwo pary, lecz również to, jak bardzo dana para wyróżnia się na tle najbliższych konkurentów.

---

## Definicja

Dla zdania $x$ w języku źródłowym i zdania $y$ w języku docelowym definiujemy lokalny poziom podobieństwa:

$$
b(x,y)=\frac{1}{2k}\left(\sum_{z \in NN_k(x)} \cos(x,z) + \sum_{z \in NN_k(y)} \cos(y,z)\right)
$$

gdzie:

- $NN_k(x)$ oznacza $k$ najbliższych sąsiadów zdania $x$ w korpusie drugiego języka;
- $NN_k(y)$ oznacza $k$ najbliższych sąsiadów zdania $y$ w korpusie pierwszego języka;
- $b(x,y)$ jest średnim podobieństwem w lokalnym otoczeniu obu zdań.

Oryginalna praca stosowała domyślnie $k=4$.

Autorzy rozpatrywali trzy warianty wyniku.

### 1. Absolute

$$
score_{\text{absolute}}(x,y)=\cos(x,y)
$$

Jest to zwykłe cosine similarity, bez wykorzystania marginesu.

### 2. Distance margin

$$
score_{\text{distance}}(x,y) = \cos(x,y)-b(x,y)
$$

Wynik mówi, o ile podobieństwo badanej pary przewyższa średnie podobieństwo jej lokalnych konkurentów.

### 3. Ratio margin

$$
score_{\text{ratio}}(x,y) = \frac{\cos(x,y)}{b(x,y)}
$$

Wynik mówi, ile razy podobieństwo badanej pary jest większe od lokalnego poziomu odniesienia.

W praktyce najczęściej stosuje się wariant **ratio**.

---

## Intuicyjna interpretacja ratio margin

Dla wyniku:

$$
r(x,y)=\frac{\cos(x,y)}{b(x,y)}
$$

można przyjąć następującą intuicję:

| Wartość $r$ | Interpretacja |
|---|---|
| $r \approx 1$ | para nie wyróżnia się względem innych kandydatów |
| $r > 1$ | para jest lepsza od przeciętnego lokalnego kandydata; im wyższy wynik, tym bardziej jednoznaczne dopasowanie |
| $r < 1$ | nominalna para jest słabsza niż jej lokalne otoczenie i jest prawdopodobnie błędna |

Nie oznacza to jednak, że „1.0" jest uniwersalnym progiem akceptacji.

---

## Jak zastosować metodę do już sparowanego korpusu

Załóżmy, że korpus zawiera nominalne pary: `source[i] ↔ target[i]`.

Procedura jest następująca:

1. Obliczyć embeddingi LaBSE dla wszystkich zdań po obu stronach korpusu.
2. Znormalizować embeddingi do długości 1, aby iloczyn skalarny odpowiadał cosine similarity.
3. Dla każdego zdania źródłowego znaleźć $k$ najbliższych zdań docelowych.
4. Dla każdego zdania docelowego znaleźć $k$ najbliższych zdań źródłowych.
5. Dla każdej nominalnej pary $(source[i], target[i])$ obliczyć:
   - cosine similarity,
   - średnią lokalnego otoczenia $b(x,y)$,
   - ratio albo distance margin.
6. Odrzucić pary, których wynik jest niższy od ustalonego progu.

W uproszczonym pseudokodzie:

```python
cosine = similarity(source_embedding, target_embedding)

source_background = mean(
    similarity(source_embedding, candidate)
    for candidate in source_k_nearest_targets
)

target_background = mean(
    similarity(target_embedding, candidate)
    for candidate in target_k_nearest_sources
)

background = (source_background + target_background) / 2

ratio_margin = cosine / background
distance_margin = cosine - background
```

Dla dużych korpusów wyszukiwanie sąsiadów można wykonać za pomocą FAISS albo innej bazy/wektorowego indeksu ANN (Approximate Nearest Neighbor — przybliżone wyszukiwanie najbliższych sąsiadów). Nie trzeba tworzyć pełnej macierzy podobieństw $N \times N$.

---

## Dobór progu

Nie istnieje jeden uniwersalny próg margin, który działa poprawnie dla każdego korpusu. Wyniki zależą między innymi od:

- pary językowej,
- domeny,
- długości segmentów,
- jakości segmentacji,
- liczby duplikatów i parafraz,
- charakterystyki modelu embeddingowego,
- wartości $k$.

Najlepszym rozwiązaniem jest ręczne oznaczenie próbki par jako:

- `1` — poprawna para tłumaczeniowa
- `0` — błędna para

Następnie próg należy dobrać na tej próbce, zależnie od oczekiwanego kompromisu między precision (precyzja — jaki odsetek zaakceptowanych par jest faktycznie poprawny) a recall (pełność — jaki odsetek wszystkich poprawnych par udało się zachować). W oryginalnej metodzie próg był optymalizowany na danych deweloperskich albo dobierany tak, aby uzyskać korpus o określonej wielkości.

W przypadku filtrowania danych treningowych zwykle bardziej zależy nam na wysokiej precyzji niż na zachowaniu wszystkich poprawnych par.

---

## Dodatkowe warunki

Margin można łączyć z dodatkowymi heurystykami, na przykład:

```python
accept = (
    ratio_margin >= margin_threshold
    and cosine >= cosine_threshold
    and length_ratio <= maximum_length_ratio
)
```

Można też wymagać, aby zdania były wzajemnymi najbliższymi sąsiadami:

- najlepszym kandydatem dla $x$ jest $y$
- oraz najlepszym kandydatem dla $y$ jest $x$

Warunek mutual nearest neighbour (wzajemny najbliższy sąsiad) nie jest częścią samej definicji margin, ale może dodatkowo ograniczyć liczbę błędnych i przesuniętych par.

---

## Ograniczenia

Margin może zaniżać wynik poprawnych par, gdy w korpusie znajduje się wiele:

- duplikatów,
- bardzo podobnych wariantów tego samego zdania,
- równoważnych parafraz,
- powtarzalnych komunikatów lub szablonów.

W takim przypadku poprawna para nie wyróżnia się mocno na tle sąsiadów, ponieważ wiele innych zdań jest niemal równie podobnych. Oryginalna praca również wskazuje karanie zdań posiadających wiele parafraz jako potencjalną wadę metody.

---

## Ważne rozróżnienie

Margin-based scoring opisany powyżej **nie jest tym samym** co additive margin softmax, używany podczas trenowania LaBSE.

- **Additive margin softmax** — element funkcji straty (loss function) podczas treningu modelu.
- **Margin-based scoring** — metoda oceniania i filtrowania par przy użyciu gotowych embeddingów.

LaBSE dostarcza embeddingi, natomiast margin-based scoring określa, jak porównywać parę z jej lokalnym otoczeniem.

---

## TL;DR

Zamiast pytać wyłącznie:

> «Czy cosine similarity tej pary jest wystarczająco wysokie?»

margin-based scoring pyta:

> «Czy similarity tej pary jest wysokie w porównaniu z najlepszymi alternatywnymi dopasowaniami dla obu zdań?»

Najczęściej używana wersja:

$$
score(x,y) = \frac{\cos(x,y)}{\dfrac{1}{2k}\left(\displaystyle\sum_{z \in NN_k(x)}\cos(x,z) + \sum_{z \in NN_k(y)}\cos(y,z)\right)}
$$

Rozsądny punkt wyjścia:

```
model: LaBSE
margin: ratio
k: 4
threshold: dobrany na ręcznie oznaczonej próbce
```

---

## Źródła

1. Mikel Artetxe, Holger Schwenk, *"Margin-based Parallel Corpus Mining with Multilingual Sentence Embeddings"*, ACL 2019, DOI: `10.18653/v1/P19-1309`. Podstawowa publikacja opisująca metodę margin-based scoring, warianty "distance" i "ratio" oraz dwukierunkowe sąsiedztwo.
2. Fangxiaoyu Feng et al., *"Language-agnostic BERT Sentence Embedding"*, ACL 2022, DOI: `10.18653/v1/2022.acl-long.62`. Publikacja opisująca model LaBSE i sposób uczenia wielojęzycznych embeddingów zdań.