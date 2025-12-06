# TimeGAN for Synthetic Financial Time-Series Generation

**Generate realistic synthetic S&P 500 market data using a TensorFlow implementation of TimeGAN.**

## Overview
This project implements TimeGAN (Time-series Generative Adversarial Network). It trains on 12 years of S&P 500 historical data to learn market patterns. The model then generates new, synthetic stock market sequences that statistically resemble the real data.

## Key Features
* **Automatic Data Pipeline:** Downloads real-time S&P 500 data.
* **Full Architecture:** Implements Embedder, Recovery, Generator, Supervisor, and Discriminator networks using TensorFlow.
* **3-Stage Training:** Trains Autoencoder, Supervisor, and then Joint Adversarial networks.
* **Analytics:** Generates t-SNE, KDE, and Overlay plots to verify data quality.

## Installation

Ensure you have Python 3.7+ installed. Run the following command:

pip install yfinance tensorflow pandas numpy matplotlib scikit-learn seaborn

**Main Dependencies:**
* tensorflow
* yfinance
* scikit-learn
* seaborn
* pandas & numpy

## Configuration
You can adjust these settings in the script:

* **SEQ_LEN:** 24 (Window size for time series)
* **BATCH_SIZE:** 128
* **ITERATIONS:** 2000 (Training steps)
* **TICKER:** ^GSPC (S&P 500 symbol)

## Visualizations
The script generates three graphs to evaluate performance:

1. **t-SNE Visualization (Figure 5):** Shows if real (Red) and synthetic (Blue) data points overlap.
2. **Feature Distribution (Figure 6):** Compares the statistical spread (density) of real vs. synthetic prices.
3. **Sequence Overlay (Figure 7):** Plots random real and synthetic sequences to compare temporal trends.

## How to Run

1. Clone the repository:
   git clone <your_repo_url>
   cd <repo_name>

2. Run the script:
   python timegan_sp500.py

## Architecture Details
The model uses four networks:
1. **Embedder:** Compresses data into latent space.
2. **Recovery:** Reconstructs data from latent space.
3. **Generator:** Creates synthetic data from random noise.
4. **Supervisor:** Ensures the model learns time-based rules.
5. **Discriminator:** Distinguishes between real and fake data.

## Authors
* Muhammad Ahmed Siddiq
* Muhammad Abubakar
