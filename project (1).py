# ==========================================
# 1. INSTALL DEPENDENCIES
# ==========================================
# !pip install yfinance tensorflow pandas numpy matplotlib scikit-learn seaborn

import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from sklearn.manifold import TSNE
from tensorflow.keras import layers, models, Input

# ==========================================
# 2. CONFIGURATION
# ==========================================
SEQ_LEN = 24       # Sequence length (window size)
HIDDEN_DIM = 24    # Hidden dimension for GRU
BATCH_SIZE = 128   # Batch size
ITERATIONS = 500  # Increased from 500 to 2000 for better results
TICKER = "^GSPC"   # S&P 500

# ==========================================
# 3. DATA LOADING & PREPROCESSING
# ==========================================
print("Downloading S&P 500 Data...")
data = yf.download(TICKER, start="2010-01-01", end="2023-01-01", auto_adjust=False)

# Handle MultiIndex columns if necessary
if isinstance(data.columns, pd.MultiIndex):
    data.columns = data.columns.get_level_values(0)

# Select features
features = ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']
data = data[features]

# Normalize Data (0 to 1)
scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(data)

# Create Sliding Windows
def make_windows(data, seq_len):
    windows = []
    for i in range(len(data) - seq_len):
        windows.append(data[i:i+seq_len])
    return np.array(windows)

real_data = make_windows(data_scaled, SEQ_LEN)

# Shuffle data for training
np.random.shuffle(real_data)
real_data = real_data.astype(np.float32)

print(f"Data Shape: {real_data.shape}")

# ==========================================
# 4. TIMEGAN ARCHITECTURE
# ==========================================
feature_dim = len(features)

def make_net(input_shape, output_dim, name="network"):
    inputs = Input(shape=input_shape)
    x = layers.GRU(HIDDEN_DIM, return_sequences=True, activation='tanh')(inputs)
    x = layers.GRU(HIDDEN_DIM, return_sequences=True, activation='tanh')(x)
    outputs = layers.Dense(output_dim, activation='sigmoid')(x)
    return models.Model(inputs, outputs, name=name)

# Define the 4 networks
embedder = make_net((SEQ_LEN, feature_dim), HIDDEN_DIM, "Embedder")
recovery = make_net((SEQ_LEN, HIDDEN_DIM), feature_dim, "Recovery")
generator = make_net((SEQ_LEN, feature_dim), HIDDEN_DIM, "Generator")
supervisor = make_net((SEQ_LEN, HIDDEN_DIM), HIDDEN_DIM, "Supervisor")
discriminator = make_net((SEQ_LEN, HIDDEN_DIM), 1, "Discriminator")

# Optimizers
g_opt = tf.keras.optimizers.Adam(learning_rate=0.001)
d_opt = tf.keras.optimizers.Adam(learning_rate=0.001)
e_opt = tf.keras.optimizers.Adam(learning_rate=0.001)

# Loss functions
mse = tf.keras.losses.MeanSquaredError()
bce = tf.keras.losses.BinaryCrossentropy()

# ==========================================
# 5. TRAINING FUNCTIONS (Autoencoder, Supervisor, Joint)
# ==========================================

@tf.function
def train_autoencoder(x):
    with tf.GradientTape() as tape:
        h = embedder(x)
        x_tilde = recovery(h)
        loss = mse(x, x_tilde)
    grads = tape.gradient(loss, embedder.trainable_variables + recovery.trainable_variables)
    e_opt.apply_gradients(zip(grads, embedder.trainable_variables + recovery.trainable_variables))
    return loss

@tf.function
def train_supervisor(x):
    with tf.GradientTape() as tape:
        h = embedder(x)
        h_hat = supervisor(h)
        loss = mse(h[:, 1:, :], h_hat[:, :-1, :])
    grads = tape.gradient(loss, supervisor.trainable_variables + embedder.trainable_variables)
    g_opt.apply_gradients(zip(grads, supervisor.trainable_variables + embedder.trainable_variables))
    return loss

@tf.function
def train_joint(x, z):
    # --- Generator Training ---
    with tf.GradientTape() as tape_g:
        # 1. Adversarial Loss (Fool the discriminator)
        e_hat = generator(z)
        h_hat = supervisor(e_hat)
        y_fake = discriminator(h_hat)
        g_loss_u = bce(tf.ones_like(y_fake), y_fake)
        
        # 2. Supervised Loss (Capture temporal dynamics)
        h = embedder(x)
        h_hat_super = supervisor(h)
        g_loss_s = mse(h[:, 1:, :], h_hat_super[:, :-1, :])
        
        # 3. Moment Losses (Mean & Std Dev matching)
        x_hat = recovery(h_hat)
        g_loss_v = tf.reduce_mean(tf.abs(tf.math.reduce_std(x_hat, axis=0) - tf.math.reduce_std(x, axis=0)))
        g_loss_m = tf.reduce_mean(tf.abs(tf.math.reduce_mean(x_hat, axis=0) - tf.math.reduce_mean(x, axis=0)))
        
        # Total Generator Loss
        g_loss = g_loss_u + 10 * tf.sqrt(g_loss_s) + 100 * g_loss_v + 100 * g_loss_m

    grads_g = tape_g.gradient(g_loss, generator.trainable_variables + supervisor.trainable_variables)
    g_opt.apply_gradients(zip(grads_g, generator.trainable_variables + supervisor.trainable_variables))

    # --- Embedder Training ---
    with tf.GradientTape() as tape_e:
        h = embedder(x)
        x_tilde = recovery(h)
        e_loss_t0 = mse(x, x_tilde)
        h_hat_super = supervisor(h)
        e_loss_s = mse(h[:, 1:, :], h_hat_super[:, :-1, :])
        e_loss = e_loss_t0 + 0.1 * tf.sqrt(e_loss_s)

    grads_e = tape_e.gradient(e_loss, embedder.trainable_variables + recovery.trainable_variables)
    e_opt.apply_gradients(zip(grads_e, embedder.trainable_variables + recovery.trainable_variables))

    # --- Discriminator Training ---
    with tf.GradientTape() as tape_d:
        h = embedder(x)
        e_hat = generator(z)
        h_hat = supervisor(e_hat)
        y_real = discriminator(h)
        y_fake = discriminator(h_hat)
        d_loss = bce(tf.ones_like(y_real), y_real) + bce(tf.zeros_like(y_fake), y_fake)

    grads_d = tape_d.gradient(d_loss, discriminator.trainable_variables)
    d_opt.apply_gradients(zip(grads_d, discriminator.trainable_variables))

    return d_loss, g_loss

# ==========================================
# 6. TRAINING LOOP
# ==========================================
print("Starting Training (This may take a few minutes)...")
dataset = tf.data.Dataset.from_tensor_slices(real_data).batch(BATCH_SIZE).shuffle(1000)

for step in range(ITERATIONS):
    for x_batch in dataset:
        z_batch = np.random.uniform(0, 1, (len(x_batch), SEQ_LEN, feature_dim)).astype(np.float32)
        
        if step < 500: # Train Autoencoder only
            l = train_autoencoder(x_batch)
        elif step < 1000: # Train Supervisor only
            l = train_supervisor(x_batch)
        else: # Joint Training
            d_l, g_l = train_joint(x_batch, z_batch)
    
    if step % 100 == 0:
        print(f"Step: {step}/{ITERATIONS}")

print("Training Complete!")

# ==========================================
# 7. GENERATING SYNTHETIC DATA
# ==========================================
print("Generating Synthetic Data...")
# Generate as many samples as we have in real data
z_test = np.random.uniform(0, 1, (len(real_data), SEQ_LEN, feature_dim)).astype(np.float32)
e_hat = generator(z_test)
h_hat = supervisor(e_hat)
synthetic_data = recovery(h_hat).numpy()

print(f"Synthetic Data Shape: {synthetic_data.shape}")

# ==========================================
# 8. VISUALIZATION & ANALYTICS (Full Requirements)
# ==========================================

# --- GRAPH 1: t-SNE Visualization (Overall Distribution) ---
print("Generating t-SNE Plot...")
sample_size = 1000
real_sample = real_data[:sample_size].reshape(sample_size, -1)
syn_sample = synthetic_data[:sample_size].reshape(sample_size, -1)
combined_data = np.concatenate((real_sample, syn_sample), axis=0)

tsne = TSNE(n_components=2, random_state=42)
tsne_results = tsne.fit_transform(combined_data)

plt.figure(figsize=(8, 6))
sns.scatterplot(x=tsne_results[:sample_size, 0], y=tsne_results[:sample_size, 1], color='red', alpha=0.4, label='Real Data')
sns.scatterplot(x=tsne_results[sample_size:, 0], y=tsne_results[sample_size:, 1], color='blue', alpha=0.4, label='Synthetic Data')
plt.title('t-SNE: Real vs. Synthetic Data Distribution')
plt.legend()
plt.show()

# --- GRAPH 2: Distribution Plots (KDE) for Key Features ---
# This checks if the model captured the statistical spread (e.g., Close Price, Volume)
print("Generating Distribution Plots...")
feat_idx = 3  # Index 3 is 'Close' price
plt.figure(figsize=(8, 6))
sns.kdeplot(real_data[:, :, feat_idx].flatten(), label='Real Close Price', color='red', fill=True, alpha=0.2)
sns.kdeplot(synthetic_data[:, :, feat_idx].flatten(), label='Synthetic Close Price', color='blue', fill=True, alpha=0.2)
plt.title('Distribution Comparison: Close Price (Real vs Synthetic)')
plt.xlabel('Normalized Price')
plt.ylabel('Density')
plt.legend()
plt.show()

# --- GRAPH 3: Overlay Sequence Comparison (Temporal Dynamics) ---
# This checks if the model learned the "movement" over time (up/down trends)
print("Generating Sequence Overlay Plots...")
plt.figure(figsize=(12, 6))
for i in range(3): # Plot 3 random sequences
    plt.subplot(1, 3, i+1)
    idx = np.random.randint(0, len(real_data))
    plt.plot(real_data[idx, :, 3], label='Real', color='red')
    plt.plot(synthetic_data[idx, :, 3], label='Synthetic', color='blue', linestyle='--')
    plt.title(f'Random Sequence {i+1}')
    if i == 0: plt.legend()

plt.suptitle('Temporal Dynamics: Real vs Synthetic Sequences (Close Price)')
plt.show()
