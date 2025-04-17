# LSTM-based Language Model for Text Generation

This project implements an **LSTM-based neural language model** to generate text in the style of a mystery novel. The model is trained on the *Mystery of the Yellow Room* and compared to outputs from a pre-trained **GPT-2** model.

---

## Project Details
 
- **Notebook**: `DL_assignment_task_3.ipynb`  
- **Dataset**: `61262-0.txt` (The Mystery of the Yellow Room)

---

## Objective

- Clean and preprocess text data from a literary source.
- Train an LSTM-based model for next-word prediction.
- Evaluate and compare the model's output against GPT-2.
- Explore advanced strategies to improve language modeling.

---

## Implementation Summary

### Data Preparation

- Mounted Google Drive and loaded a `.txt` file.
- Cleaned and tokenized the text (removed punctuation, lowercased, removed non-ASCII).
- Created sequences of 41 words (first 40 as input, last as target).
- Tokenized words using Keras `Tokenizer`.

### Model Training

- Built a deep LSTM model:
  - Embedding Layer (size=50)
  - LSTM(64) → LSTM(128)
  - Dense(128, relu) → Dense(output vocab size, softmax)
- Trained for 500 epochs (accuracy: **97.32%**, loss: **0.1012**).
- Saved model and tokenizer to Google Drive.

### Text Generation

- Defined `text_gen()` function to generate next words using LSTM.
- Compared LSTM output with GPT-2 using the same prompt.
- Evaluated fluency, grammar, and long-range coherence.

---

## Evaluation & Comparison

| Aspect               | LSTM Model                   | GPT-2 Model                  |
|----------------------|------------------------------|------------------------------|
| Training Corpus      | Mystery novel (single book)  | General internet texts       |
| Output Fluency       | Structured, slightly rigid   | More fluent and natural      |
| Long-term Coherence  | Moderate                     | Superior                     |
| Vocabulary Diversity | Limited                      | Rich                         |

---

## Technologies Used

- Python 3.x  
- TensorFlow / Keras  
- PyTorch (`transformers`)  
- Google Colab  
- GPT-2 (via HuggingFace)  
- LucidChart (for diagrams)

---

## Suggested Improvements

The notebook explores improvements to text generation using:

- Bidirectional LSTM (BiLSTM)
- Transformer / Self-Attention architectures
- Pre-trained embeddings (GloVe, Word2Vec)
- Subword tokenization (Byte Pair Encoding)
- Curriculum learning and dynamic batching
- Weight tying and neural cache models

---

## Example Prompt & Output

**Prompt:**  
`"The gentleman was"`

**LSTM Output:**  
Generated using `text_gen(model, tokenizer, ...)`

**GPT-2 Output:**  
Generated using HuggingFace's pre-trained GPT-2 model

---

## Author

- **Pranav Sunil Raja**  
- Newcastle University  
- GitHub: [@pranavsraja](https://github.com/pranavsraja)

---

## License

This project is for educational use only. Literary data belongs to the public domain (Project Gutenberg).
