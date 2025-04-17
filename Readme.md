IMPORTANT: Please read the following before proceeding. This AMP includes or otherwise depends on certain third party software packages. Information about such third party software packages are made available in the notice file associated with this AMP. By configuring and launching this AMP, you will cause such third party software packages to be downloaded and installed into your environment, in some instances, from third parties' websites. For each third party software package, please see the notice file and the applicable websites for more information, including the applicable license terms.

If you do not wish to download and install the third party software packages, do not configure, launch or otherwise use this AMP. By configuring, launching or otherwise using the AMP, you acknowledge the foregoing statement and agree that Cloudera is not responsible or liable in any way for the third party software packages.

Copyright (c) 2025 - Cloudera, Inc. All rights reserved.


# Knowledge Distillation for Customer Support LLMs

## Project Overview  
This project addresses the challenge of improving the accuracy and speed of a customer support LLM while adhering to data privacy constraints. By leveraging synthetic data generation and fine-tuning techniques, we demonstrate how to train a smaller, faster LLM (Meta-Llama-3.1-8B-Instruct) for real-time analysis of customer support requests and compare the results to a base model. The workflow is divided into four core steps:  

0. **Setup & Model Initialization**  
1. **Data Preparation**  
2. **Fine-Tuning with LoRA**  
3. **Inference, Evaluation, and Benchmarking**  

---

### Step 0: Environment Setup & Model Initialization  
**Overview**
- This is the starting notebook of the project. In this step, we download all the required models and install the needed libraries.

**Purpose:**  
- Download models required for this project  

**Key Components:**  
- Initializes two foundational models:  
  - `Meta-Llama-3.1-8B-Instruct` (target for fine-tuning)  
  - `Microsoft Phi-4` (used later as an evaluation judge)  

**Output:**  
- Ready-to-use models and libraries for subsequent steps  

---

### Step 1: Data Preparation  
**Overview**
- In this notebook, we use the output data from Synthetic Data Studio (SDS) and process it for finetuning and evaluation. Cloudera's customer support team separates and processes  customer and Cloudera comments using two different output formats. Thus, we use different SDS generated data for each comment type. In addition, the SDS output is a list of topics and each topic contains the relevant prompt, completion, and evaluation. We use the evaluation score to filter low-quality data  and combine the prompt with the expected completion to teach the LLM using finetuning. For LLM finetuning, we combine the customer and Cloudera comments into one LLM for efficiency. We also leave 1000 samples out for processing Cloudera and customer comments. 

**Purpose:**  
- Generate structured training data from raw customer support comments  
- Split data into training/evaluation sets  

**Process:**  
1. Loads raw data from `ClouderaComments.json` and `CustomerComments.json`  
2. Filters high-quality entries (score >4.9)  
3. Formats entries into prompt-answer pairs:  
   - **Prompt:** Customer support comment + structured questions  
   - **Completion:** Model answers to the questions (e.g., scores, summaries)  
4. Splits data into `Train_Clean` (3500 samples) and `Evaluation_Clean` (500 samples for Cloudera comments and 500 samples for customer comments)  

**Variables to Customize** 
- Filenames for input data (e.g., `filename='Data/ClouderaComments'`).  
- Data split sizes (e.g., `3500` for training vs `500` for evaluation).  

**Output:**  
- Cleaned datasets in `AllComments_Clean_Train.json` and evaluation files  

---

### Step 2: Fine-Tuning with LoRA  
**Overview**
- In this notebook, we finetune the LLM using distilled knowledge from SDS. At a high-level, we add the special tokens before fine-tuning, split the data into training and dev sets, finetune lora adapters, and merge and store the model.
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
  - Saves fine-tuned model to `./tmp/merged_...`  

**Variables to Customize**
- **LoRA Parameters**
  - `lora_r`: Rank (default `128`).  
  - `lora_alpha`: Scaling factor (default `64`).  
  - `lora_dropout`: Dropout rate (default `0.05`).  
- `Data_Size`: Number of samples used for training (default `5000`). if the number of samples is more than data samples available, it uses the maximum available.  
- `FT-num_train_epochs`: Number of training epochs (default `1`).


**Output:**  
- A domain-specific model optimized for customer support tasks  

---

### Step 3: Inference, Evaluation, and Benchmarking 
**Overview**
- In this final notebook, we infer the output (completion) for each Cloudera and customer comments separately. Using the generated answers, we parse the output, extract the relevant information and instruct an LLM-as-a-judge to compare the outputs of the two LLMs (score if A or B model is best or if it is a tie). Here, we evaluate only on answers that there is no tie between the models and compute the winrate and the percentage of ties. Also, this step shows example outputs from each LLM.

**Purpose:**  
- Compare the fine-tuned model against the baseline (Meta-Llama-3.1-8B-Instruct)  
- Use an external judge (Phi-4 14B) to evaluate output quality  

**Process:**  
1. Generate outputs for both models on evaluation data  
2. Format outputs into structured comparisons for the judge  
3. Judge evaluates pairs of answers and selects the better-performing model
4. Compute the winrate of the Finetuned model compared to baseline for each question and average.

**Variables to Customize**
- `Customer`: `0` for Cloudera comments, `1` for customer comments.  
- `EvalLLM`: Path to the evaluation model (e.g., `microsoft/phi-4`).  


---

### Setup & Installation  
1. **Environment requirements:**
   - Python 3.11+

2. **Tested models:**
   - Tested for Finetuning: unsloth/Meta-Llama-3.1-8B-Instruct and unsloth/gemma-2-2b-it
   - Tested for LLM-as-a-judge: microsoft/phi-4


3. **Memory and GPU Requirements:**  
   - Step 2 and 3 require a GPU with ~48GB VRAM for 8B model training
   - 48GB of cpu memory

4. **Cleaning resources**
   - After finishing step 3, we need to reset the kernel of the notebook to release the GPU.
     

---

## Usage Guide  
1. **Run Step 0 first** to download models.  
2. **Execute Step 1** to process raw data into training/evaluation sets.  
3. **Proceed to Step 2** to fine-tune the LLM.  
4. **Run Step 3** to evaluate performance against the baseline.  

---

## Advanced Usage Guide  
The AMP enables three main categories of customizations: custom input data, custom model choice, and custom LLM-as-a-judge evaluation
1. Custom input data: To modify the input data with your own custom data you need to update the input files used in Step1 (such as ClouderaComments.json). The data need to follow the format exported by [SDS](https://github.com/cloudera/CAI_AMP_Synthetic_Data_Studio). The format also expects data quality evaluation scores to be used for filtering as exported by SDS. Note that if you need to update the prompts, we need to update the CommentText and PreppendClouderaQuestions variables to reflect the new prompts.
2. Custom model choice: To use your own model for finetuning, you need to download the model in step0 and set the ModelFT variable to the new model (step2 and step3). Note that the second cell and the Config/Config.py provides variables finetuning parameters to choose from, for example, the learning rate etc. Finally, here we can modify the target output path by setting the TargetDir variable.
3. Custom LLM-as-a-judge evaluation: To select your own LLM-as-a-judge, you need to download the model in step0 and set the variable EvalLLM in step3. In addition, you can load a new evaluation set by replacing the files evaluation files (such as Data/CustomerComments_Evaluation_Clean.json). If you need to ignore the first lines of the finetuned LLM or the base LLM you can use the variables: StartLineFT=1, StartLineBase=1 to ignore 1 line for example. Finally, you can modify the LLM-as-a-judge instructions by changing cells 18 and 19.
---

## Expected Outputs  
- **Step 2**: A fine-tuned model saved to `./tmp/merged_*`.  
- **Step 3**: Evaluation metrics (win rate, tie percentage) printed in the notebook.  

---

## Testing correct execution
- Run all steps with unsloth/Meta-Llama-3.1-8B-Instruct for finetuning
- Evaluate using LLM-as-a-judge microsoft/phi-4
- The final cell in step 3 tests if the average winrate is 82%


## Known issues
- Both the AMP deployment and session need to run the same python version
