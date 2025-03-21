# Knowledge Distillation for Customer Support LLMs

## Project Overview  
This project addresses the challenge of improving the accuracy and speed of a customer support LLM while adhering to data privacy constraints. By leveraging synthetic data generation and fine-tuning techniques, we demonstrate how to train a smaller, faster LLM (Meta-Llama-3.1-8B-Instruct) for real-time analysis of customer support requests and compare the results to a base model. The workflow is divided into four core steps:  

1. **Setup & Model Initialization**  
2. **Data Preparation**  
3. **Fine-Tuning with LoRA**  
4. **Inference, Evaluation, and Benchmarking**  

---

### Step 1: Environment Setup & Model Initialization  
**Overview**
- This is the starting notebook of the project. In this step, we download all the required models and install the needed libraries.

**Purpose:**  
- Install required libraries (transformers, vllm)  
- Download models required for this project  

**Key Components:**  
- Installs HuggingFace Transformers and vLLM for distributed inference  
- Initializes two foundational models:  
  - `Meta-Llama-3.1-8B-Instruct` (target for fine-tuning)  
  - `Microsoft Phi-4` (used later as an evaluation judge)  

**Output:**  
- Ready-to-use models and libraries for subsequent steps  

---

### Step 2: Data Preparation  
**Overview**
- In this notebook, we use the output data from Synthetic Data Studio (SDS) and process it for finetuning and evaluation. Cloudera customer support separates and processes  customer and Cloudera comments using two different output formats. Thus, we use different SDS generated data for each comment type. In addition, the SDS outputs is a list of topics and each topic contains the relevant prompt, completion, and evaluation. We use the evaluation score to filter low-quality data  and combine the prompt with the expected completion to teach the LLM using finetuning. For LLM finetuning, we combine the customer and Cloudera comments into one LLM for efficiency. We also leave 1000 samples out for processing Cloudera and customer comments. 

**Purpose:**  
- Generate structured training data from raw customer support comments  
- Split data into training/validation sets  

**Process:**  
1. Loads raw data from `ClouderaComments.json` and `CustomerComments.json`  
2. Filters high-quality entries (score >4.9)  
3. Formats entries into prompt-answer pairs:  
   - **Prompt:** Customer support comment + structured questions  
   - **Completion:** Model answers to the questions (e.g., scores, summaries)  
4. Splits data into `Train_Clean` (3500 samples) and `Evaluation_Clean` (500 samples for Cloudera comments and 500 samples for customer comments)  

**Output:**  
- Cleaned datasets in `AllComments_Clean_Train.json` and evaluation files  

---

### Step 3: Fine-Tuning with LoRA  
**Overview**
- In this notebook, we finetune the LLM using distilled knowledge from SDS. At a highlevel, we add the special tokens before finetuning, split the data into training and dev sets, finetune lora adapters, and merge and store the model.
**Purpose:**  
- Adapts the Meta-Llama-3.1-8B-Instruct model to the customer support domain  
- Uses LoRA (Low-Rank Adaptation) for efficient parameter updates  

**Key Configurations:**  
- **LoRA Parameters:**  
  - Rank (`lora_r`): 128  
  - Alpha (`lora_alpha`): 64  
  - Dropout: 0.05  
- **Training:**  
  - Dataset from Step 2 formatted into chat templates  
  - Trained for 1 epoch with gradient accumulation  
  - Saves fine-tuned model to `/tmp/merged_...`  

**Output:**  
- A domain-specific model optimized for customer support tasks  

---

### Step 4: Inference, Evaluation, and Benchmarking 
**Overview**
- In this final notebook, we infer the output (completion) for each Cloudera and customer comments separately. Using the generated answers, we parse the output, extract the releveant information and instruct an LLM-as-a-judge to compare the outputs of the two LLMs (score if A or B are best or if it is a tie). Here, we evaluate only on answers that there is no tie between the models and compute the Winrate and the percentage of ties. Also, this step shows example outputs from each LLM.



**Purpose:**  
- Compare the fine-tuned model against the baseline (Meta-Llama-3.1-8B-Instruct)  
- Use an external judge (Phi-4 14B) to evaluate output quality  

**Process:**  
1. Generate outputs for both models on evaluation data  
2. Format outputs into structured comparisons for the judge  
3. Judge evaluates pairs of answers and selects the better-performing model  


---

### Setup & Installation  
1. **Dependencies:**  
   ```bash  
   pip install transformers vllm  
   ```  
2. **GPU Requirements:**  
   - Step 2 and 3 require a GPU with ~48GB VRAM for 8B model training

3. **Cleaning resources**
   - After finishing step 2, we need to reset the kernel of the notebook to release the GPU.

---

### Usage Guide  
1. **Run all steps in sequence in all notebooks steps 0-3**  

