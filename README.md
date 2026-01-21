# CNN klasifikator odeće (Fashion‑MNIST)

Ovaj projekat implementira **konvolucionu** neuronsku mrežu za klasifikaciju slika odeće iz Fashion‑MNIST skupa podataka, u potpunosti od nule u NumPy‑u (bez PyTorch/TensorFlow). Fokus je na ručnoj implementaciji slojeva, backpropagation‑a i trening petlje.

## Funkcionalnosti

- CNN arhitekture: `CNNSmall`, `CNNMedium`, `CNNHigh`, plus generički `CNN` model.
- Slojevi: `Conv2D`, `ReLU`, `MaxPool2D`, `AvgPool2D`, `GlobalAvgPool2D`, `BatchNorm2D`, `LayerNorm1D`, `Dense`, `Flatten`.
- Loss: softmax + unakrsna entropija sa stabilnom numeričkom implementacijom.
- Optimizatori: `SGD`, `Momentum`, `NAG`, `RMSProp`, `Adam`.
- Trening: mini‑batch, validacioni set, praćenje metrika, early stopping.

## Struktura projekta

- `src/ccc/models/layers/` – implementacija svih slojeva i pomoćnih funkcija (`conv2d.py`, `relu.py`, `maxpool2d.py`, `avg_pool2d.py`, `global_avg_pool2d.py`, `batch_norm.py`, `layer_norm.py`, `dense.py`, `loss.py`, `utils.py`).  
- `src/ccc/models/` – različite CNN arhitekture (`cnn_small.py`, `cnn_medium.py`, `cnn_high.py`, `cnn.py`).  
- `src/ccc/optim/` – optimizatori (`sgd.py`, `momentum.py`, `nag.py`, `rmsprop.py`, `adam.py`).  
- `src/ccc/engine/` – trening i metrike (`trainer.py`, `metrics.py`).  
- `src/ccc/data/fmnist.py` – učitavanje Fashion‑MNIST podataka, normalizacija i podela na train/val/test.

## Podaci: Fashion‑MNIST

- Ulaz: slike dimenzija `28×28×1`, normalizovane na opseg `[0, 1]`.  
- Ciljevi: 10 klasa odeće, one‑hot vektori dužine 10.  
- U `fmnist.py` se radi podela na trening, validacioni i test skup.

## Trening i hiperparametri

Trening se tipično pokreće kroz funkcije u `trainer.py`, uz izbor:

- modela (`CNNSmall`, `CNNMedium`, `CNNHigh`, `CNN`),  
- optimizatora i learning rate‑a,  
- veličine batch‑a, broja epoha i parametara za early stopping.

Za sistematsko poređenje kombinacija koristi se skripta `sweep.py` (vidi ispod).

### `sweep.py`

- Radi **grid search** preko kombinacija:
  - modela (`MODELGRID`, npr. `cnn`, `small`, `medium`, `high`),
  - optimizatora (`OPTIMIZERGRID`, npr. `sgd`, `momentum`, `nag`, `adam`, `rmsprop`),
  - learning rate‑ova (`OPTIMIZER_LRS`) i batch size‑a (`BATCHGRID`).[file:14]
- Učitava Fashion‑MNIST (`get_mnist_data`), trenira modele pozivom `fit(...)` iz `trainer.py` i meri:
  - najbolju validacionu tačnost,
  - test tačnost i macro F1,
  - vreme treniranja.[file:14]
- Za svaki run:
  - pravi poseban folder u `runs/`,
  - snima konfiguraciju (`config.json`), istoriju treninga (`history.csv`),
  - najbolji model (`best_model.pickle`),
  - konfuzionu matricu i grafike (loss/accuracy krive).[file:14]
- Upisuje sumarizaciju svih run‑ova u zajednički `mastersummary.csv` (putanja je u promenljivoj `MASTER_CSV`).[file:14]

### `conclusion.ipynb`

- Notebook za **analizu rezultata eksperimenata** nakon izvršavanja `sweep.py`.[file:13]
- Automatski učitava sve `mastersummary.csv` fajlove iz `runs/` direktorijuma i spaja ih u jedan DataFrame.[file:13]
- Računa i prikazuje:
  - top run‑ove po test accuracy i macro F1,
  - najbolje kombinacije po modelu i po optimizatoru,
  - statistiku (mean/std/max) za test accuracy po optimizatoru i batch size‑u.[file:13]
- Pravi grafike:  
  - accuracy vs. vreme po epohi,  
  - uporedne bar grafike za različite modele i optimizatore.[file:13]

## Zašto je projekat zanimljiv

- CNN je implementiran u čistom NumPy‑u, uključujući `im2col` konvolucije i ručni backpropagation.  
- Postoji kompletan eksperimentalni pipeline: od trening skripte (`sweep.py`) do analitičkog notebook‑a (`conclusion.ipynb`), koji pokazuje kako se sistematski porede modeli i optimizatori na Fashion‑MNIST‑u.
