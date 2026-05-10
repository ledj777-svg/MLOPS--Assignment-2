# MLOps Assignment 2: Hugging Face Fine-Tuning, Experiment Tracking & Model Deployment
This project demonstrates a complete **MLOps workflow** — converting a Jupyter notebook into production scripts, tracking experiments with Weights & Biases, and deploying the model to Hugging Face Hub.

=================================================================================================



**Project Description**
This project implements a complete MLOps pipeline for fine-tuning a DistilBERT model on the UCSD Goodreads book reviews dataset to classify book reviews into 8 different genres (Poetry, Children, Comics & Graphic, Fantasy & Paranormal, History & Biography, Mystery/Thriller/Crime, Romance, and Young Adult). 
The workflow includes converting a Jupyter notebook into clean, modular Python scripts (data.py, train.py, eval.py, utils.py), proper data preprocessing and tokenization, model training using Hugging Face Transformers with experiment tracking via Weights & Biases (W&B), detailed evaluation with classification metrics, 
and finally deploying the trained model along with the tokenizer to the Hugging Face Hub for public access. The entire process emphasizes reproducibility, production-ready code practices, and end-to-end MLOps principles.

=================================================================================================

**Model Selection**
DistilBERT-base-cased was selected because it is 40% smaller and 60% faster than BERT-base while retaining ~97% of its language understanding capability. 
This made it perfect for this MLOps assignment focused on workflow efficiency rather than maximum accuracy.

=================================================================================================

### Setup Instructions

**Clone Repository**

bash
git clone https://github.com/ledj777-svg/MLOPS--Assignment-2.git

cd mlops-assignment2

=================================================================================================

**Create & Activate Environment**

uv venv

.venv\Scripts\activate    

=================================================================================================

**Install Dependencies**

uv pip install -r requirements.txt

=================================================================================================

**Logins**

wandb login

huggingface-cli login

=================================================================================================

**Project Strcutre**

1-data.py

2-train.py

3-eval.py

4-utils.py

5-requirements.txt

6-README.md

7-results              

8-eval_report.json

=================================================================================================

**Training with W&B tracking**

python train.py

=================================================================================================

**Final Evaluation**

python eval.py

=================================================================================================

**Result**

Accuracy-0.782

F1 Score- 0.779

Eval Loss-0.612

=================================================================================================

**Hugging Face Model Link:**
https://huggingface.co/JD45-d/distilbert-goodreads-genre-classifier/tree/main


**W&B Dashboard Link:**
https://wandb.ai/ledj777-deloitte/mlops-assignment2/runs/50uvtmb5?nw=nwuserledj777



