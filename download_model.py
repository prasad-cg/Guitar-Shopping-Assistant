"""
Download the embedding model for offline/local use.

Run this script to download the all-MiniLM-L6-v2 model (~80 MB) into the
models/ directory. After downloading, the RAG pipeline works fully offline.

Usage:
    python download_model.py

If you're behind a corporate proxy and this fails, run it on a personal machine
and copy the models/all-MiniLM-L6-v2/ folder to this project manually.
"""
import os


def download_model():
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    save_path = os.path.join(os.path.dirname(__file__), "models", "all-MiniLM-L6-v2")

    if os.path.exists(save_path):
        print(f"Model already exists at: {save_path}")
        print("Delete the folder and re-run if you want to re-download.")
        return

    print(f"Downloading model: {model_name}")
    print(f"Saving to: {save_path}")
    print("-" * 50)

    try:
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer(model_name)
        os.makedirs(save_path, exist_ok=True)
        model.save(save_path)

        print("-" * 50)
        print(f"SUCCESS! Model saved to: {save_path}")
        print(f"Size: ~80 MB")
        print(f"\nYou can now run the application — RAG will work offline.")

    except ImportError:
        print("ERROR: sentence-transformers not installed.")
        print("Run: pip install sentence-transformers")
        print("Then re-run this script.")

    except Exception as e:
        print(f"ERROR: {e}")
        print("\nIf behind a proxy, run this on a machine with internet access,")
        print("then copy the models/all-MiniLM-L6-v2/ folder here.")


if __name__ == "__main__":
    download_model()
