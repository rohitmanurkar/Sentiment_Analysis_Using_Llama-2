#Sentiment_Analysis_Using_Llama-2
A project demonstrating how to efficiently fine-tune the meta-llama/Llama-2-7b-chat-hf model for financial news sentiment analysis using QLoRA (4-bit quantization and LoRA) with the Hugging Face TRL library.

# Fine-Tuning Llama 2 for Financial Sentiment Analysis using QLoRA

📖 Overview

This project provides a complete walkthrough of fine-tuning the `meta-llama/Llama-2-7b-chat-hf` model for sentiment analysis on financial news headlines. It leverages state-of-the-art techniques for parameter-efficient fine-tuning (PEFT), specifically QLoRA, to train the large language model on a consumer-grade GPU (like the T4 available in Google Colab).

The goal is to instruct-tune the model to classify a given news headline into one of three categories: *positive*, *neutral*, or *negative*.

The core workflow involves:
1.  Loading the 7-billion parameter Llama 2 model in 4-bit precision using `bitsandbytes`.
2.  Using the `peft` library to configure Low-Rank Adaptation (LoRA) modules for training.
3.  Formatting the dataset into instruction prompts.
4.  Running the supervised fine-tuning job using the `SFTTrainer` from the `trl` library.
5.  Evaluating the model's performance on a test set.
6.  Merging the trained LoRA adapters with the base model to create a standalone, fine-tuned model for inference.

---

✨ Key Features

* Model: `meta-llama/Llama-2-7b-chat-hf`
* Technique: **QLoRA** for memory-efficient fine-tuning.
* Libraries: Hugging Face `transformers`, `peft`, `trl`, `accelerate`, and `bitsandbytes`.
* Dataset: A CSV file (`all-data_fin.csv`) containing financial news headlines and their corresponding sentiment labels.
* Task: 3-class sentiment classification (Positive, Neutral, Negative).

---

📂 File Structure

├── Sentiment_Analysis.ipynb    # The main Jupyter Notebook with all the code.
├── all-data_fin.csv            # The dataset for training and evaluation.
└── README.md                   # You are reading it!


---

 🚀 Getting Started

# Prerequisites

* Python 3.8+
* A Hugging Face account with access granted to the Llama 2 model. You can request access [here](https://huggingface.co/meta-llama/Llama-2-7b-chat-hf).
* A GPU is highly recommended for running the notebook (e.g., NVIDIA T4, A100).

# Installation

1.  Clone the repository:
    ```bash
    git clone [https://github.com/YOUR_USERNAME/YOUR_REPOSITORY_NAME.git](https://github.com/YOUR_USERNAME/YOUR_REPOSITORY_NAME.git)
    cd YOUR_REPOSITORY_NAME
    ```

2.  Create and activate a virtual environment (recommended):
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  Install the required libraries:
    The notebook installs all necessary packages. The key dependencies are listed below and can be installed via pip:
    ```bash
    pip install -q -U torch transformers datasets peft trl accelerate bitsandbytes tensorboard scikit-learn pandas
    ```

# Usage

1.  Hugging Face Login:
    To download the Llama 2 model, you need to log in with your Hugging Face account token. Run the following command in your terminal or a code cell and enter your token when prompted:
    ```bash
    huggingface-cli login
    ```

2.  Open and Run the Notebook:
    Launch Jupyter Notebook and open `Sentiment_Analysis.ipynb`.
    ```bash
    jupyter notebook Sentiment_Analysis.ipynb
    ```

3.  Update File Paths (if necessary):
    The notebook is configured to read the dataset from a specific path. Ensure that `all-data_fin.csv` is in the same directory as the notebook, and update the filename path in the notebook if you place it elsewhere.

4.  Execute the Cells:
    Run the cells in the notebook sequentially to:
    * Load and preprocess the data.
    * Load the quantized Llama 2 model and tokenizer.
    * Run the SFTTrainer to fine-tune the model.
    * Evaluate the performance of the final merged model.

---

🔧 Code Explanation

The `Sentiment_Analysis.ipynb` notebook is structured as follows:

1.  Setup:Installs all required Python packages and imports libraries.
2.  Data Loading and Preprocessing:
    * Loads the `all-data_fin.csv` dataset.
    * Splits the data into training, testing, and evaluation sets.
    * Defines a function `generate_prompt` to format each data sample into a clear instruction for the model.
3.  Model Loading:
    * A `BitsAndBytesConfig` is created to load the Llama 2 model in 4-bit precision (`nf4` type).
    * `AutoModelForCausalLM.from_pretrained` loads the quantized model onto the available GPU device.
4.  Baseline Evaluation (Zero-Shot):
    * Before fine-tuning, the script evaluates the base Llama 2 model's ability to perform sentiment analysis in a zero-shot setting.
5.  **Fine-Tuning:**
    * A `LoraConfig` from the `peft` library is defined to specify which layers of the model to adapt.
    * `TrainingArguments` are set to configure the training process (e.g., learning rate, number of epochs, batch size).
    * The `SFTTrainer` is initialized with the model, datasets, and configurations, and the `trainer.train()` command starts the fine-tuning process.
6.  Saving and Merging:
    * The trained LoRA adapters are saved.
    * The script then demonstrates how to merge these adapters back into the original model's weights to create a self-contained, fine-tuned model ready for deployment.
7.  Final Evaluation:
    The performance of the merged model is evaluated on the test set, with accuracy, a classification report, and a confusion matrix printed to show the results.
---
