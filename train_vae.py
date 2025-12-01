from cnn_vae import build_vae
from dataset_loader import load_grayscale_norm_images_from_dir
from sklearn.model_selection import train_test_split

# Load only non-rotten fruit
clean_dir = "data/non_rotten_all"
X_clean = load_grayscale_norm_images_from_dir(clean_dir)

X_train, X_val = train_test_split(X_clean, test_size=0.1, random_state=42)

vae, encoder, decoder = build_vae()

history = vae.fit(
    X_train,
    X_train,
    epochs=50,
    batch_size=64,
    validation_data=(X_val, X_val),
)
