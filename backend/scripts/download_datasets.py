import os

def download_kaggle_datasets():
    print("=== Kaggle Dataset Setup ===")
    print("To improve model accuracy, we will use large-scale datasets from Kaggle.")
    print("Prerequisites:")
    print("1. Create a Kaggle Account")
    print("2. Create a specific API Token (kaggle.json) from settings")
    print("3. Place 'kaggle.json' in C:\\Users\\rasib\\.kaggle\\ or configure env vars.")
    print("====================================")
    
    # 1. Text: DAIGT V2 (LLM Detection)
    # This is better than the Pile for modern LLMs
    print("\n[TEXT] Downloading 'DAIGT V2 Train Dataset'...")
    print("Command: kaggle datasets download -d thedevastator/daigt-v2-train-dataset")
    os.system("kaggle datasets download -d thedevastator/daigt-v2-train-dataset -p datasets/text/daigt_v2 --unzip")
    
    # 2. Image: CIFAKE (Real vs AI)
    print("\n[IMAGE] Downloading 'CIFAKE'...")
    print("Command: kaggle datasets download -d birdy654/cifake-real-and-ai-generated-synthetic-images")
    os.system("kaggle datasets download -d birdy654/cifake-real-and-ai-generated-synthetic-images -p datasets/image/cifake --unzip")
    
    # 3. Gen AI Misinformation (New 2024-2025) - Good for modern context
    print("\n[TEXT-MISINFO] Downloading 'Gen AI Misinformation 2024-2025'...")
    os.system("kaggle datasets download -d mfaaris/gen-ai-misinformation-detection-data-2024-2025 -p datasets/text/misinfo_2025 --unzip")

    # 4. Text: AI vs Human Text
    print("\n[TEXT] Downloading 'AI vs Human Text'...")
    os.system("kaggle datasets download -d shanegerami/ai-vs-human-text -p datasets/text/ai_vs_human --unzip")

    # 5. Image: 140k Real and Fake Faces
    print("\n[IMAGE] Downloading '140k Real and Fake Faces'...")
    os.system("kaggle datasets download -d xhlulu/140k-real-and-fake-faces -p datasets/image/140k_faces --unzip")
    
    # 6. Text: LLM Detect AI Generated Text (Competition Data)
    print("\n[TEXT] Downloading 'LLM Detect AI Generated Text'...")
    os.system("kaggle datasets download -d star-blinders/llm-detect-ai-generated-text-dataset -p datasets/text/llm_detect_ai --unzip")

    print("\nDownloads attempted. If commands failed, please ensure 'kaggle' (pip install kaggle) is installed and authenticated.")

if __name__ == "__main__":
    download_kaggle_datasets()
