# Wireless Traffic Modeling: Cellular Load Forecasting

## Problem

Cellular base stations run at close to full power all day, while the traffic they carry
drops sharply at night and peaks in the evening. Capacity is wasted in quiet hours and
stretched in busy ones.

## Approach

We forecast the load of each cell for the next hour, then decide: if the forecast load
is below a threshold, the cell can be switched off for that hour.

Four models are compared:

| Model | Idea |
|---|---|
| Naive | Load = same hour yesterday (baseline) |
| SARIMA | Statistical model with a 24-hour season |
| LSTM | Neural network that reads the last 48 hours |
| Hybrid | SARIMA for the seasonal part + an LSTM that predicts SARIMA's errors |

## Metrics

- **MAE / RMSE / MAPE**: forecast error.
- **hours_asleep_%**: share of hours the cell is switched off (energy saving).
- **oracle_asleep_%**: what a perfect forecast would achieve.
- **missed_%**: share of switch-off hours that were wrong (real load was above the threshold).

## Results (Milan, 3 cells, test week 22-28 Nov 2013)

| Model | MAE | RMSE | MAPE | Hours asleep | Wrong sleep decisions |
|---|---|---|---|---|---|
| SARIMA | 285.6 | 400.7 | 10.3% | 15.5% | 6.1% |
| LSTM | 290.9 | 411.8 | 10.0% | 16.7% | 13.0% |
| Hybrid SARIMA+LSTM | 302.6 | 421.3 | 11.2% | 13.9% | 11.5% |
| Naive (yesterday) | 585.4 | 908.5 | 21.9% | 19.3% | 20.6% |

A perfect forecast would switch the cells off for 18.45% of hours.

## Running

Requires Python 3.12.

```bash
python -m venv .venv
```

Windows:

```bash
.venv\Scripts\python -m pip install -r requirements.txt
```

Linux / macOS:

```bash
.venv/bin/python -m pip install -r requirements.txt
```

Run on the included Milan data (reproduces the results above):

```bash
python run.py --milan data/milan
```

Run on synthetic data (no dataset needed):

```bash
python run.py
```

Note: both commands write to `results/`, so the synthetic run overwrites the Milan results.

Outputs in `results/`: `summary.csv` (average over cells), `per_cell.csv`, and one forecast
plot per cell.

## Milan data

Source: Telecom Italia Big Data Challenge on Harvard Dataverse, dataset
"Telecommunications - SMS, Call, Internet - MI" (DOI: `10.7910/DVN/EGZHFV`).

`data/milan/` already contains the 28 daily files for 1-28 November 2013, filtered to the
three cells used (about 6 MB). To rebuild it from the raw data:

1. Download `sms-call-internet-mi-2013-11-01.txt` to `2013-11-28` from the dataset page.
   The site asks you to fill in a guestbook form first. Each file is about 350 MB
   (about 10 GB in total).
2. Filter them to the chosen cells:

```bash
python scripts/filter_milan.py path/to/downloaded/files
```

Cells used: 5060 (city centre, Duomo), 4259 (Bocconi), 4456 (Navigli).

## Project structure

```
run.py                     runs the full experiment
src/data.py                Milan data loader + synthetic data generator
src/models.py              the four models
src/evaluate.py            error metrics and the sleep policy
scripts/filter_milan.py    shrinks the raw Milan files to the chosen cells
data/milan/                filtered Milan data
results/                   tables and plots
```
