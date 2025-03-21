Config=dict()
State=dict()
StateTest = dict()
StateTest["TestSamples"] = []
State["Test"]=StateTest
UseCuda='False'
ConfigQuestion=dict()
ConfigQuestion["type"]="LLM"
ConfigQuestion["llm-type"]="VLLMInferenceService"
ConfigQuestion["llm-type"]="VLLMInference"
ConfigQuestion["llm-path"]="Qwen/CodeQwen1.5-7B-Chat"
ConfigQuestion["llm-path"]="iqbalamo93/Meta-Llama-3.1-8B-Instruct-GPTQ-Q_8"
ConfigQuestion["tokenizer-path"]="Qwen/CodeQwen1.5-7B-Chat"
ConfigQuestion["tokenizer-path"]="bartowski/Meta-Llama-3.1-8B-Instruct-GGUF"
ConfigQuestion["tokenizer-file-gguf"]="Meta-Llama-3.1-8B-Instruct-Q6_K.gguf"
ConfigQuestion["llm-path"]="Qwen/Qwen2.5-Coder-7B-Instruct"
ConfigQuestion["tokenizer-path"]="Qwen/Qwen2.5-Coder-7B-Instruct"
ConfigQuestion["tokenizer-file-gguf"]=""
ConfigQuestion["llm-path"]="Qwen/CodeQwen1.5-7B-Chat"
ConfigQuestion["tokenizer-path"]="Qwen/Qwen2.5-Coder-7B-Instruct"
ConfigQuestion["max_num_batched_tokens"]="2048"
ConfigQuestion["Filename"]="Questions.pkl"
ConfigQuestion["LoadFlag"]="True"
ConfigQuestion["seed"]="None"
ConfigQuestion["gpu_memory_utilization"]="0.49"
ConfigQuestion["gpu_memory_utilization"]="0.99"
ConfigQuestion["gpu-id"]="0"
ConfigQuestion["enforce_eager"]="True"
ConfigQuestion["cpu_offload_gb"]="0"
ConfigQuestion["quantization"]=None

ConfigQuestion["swap_space"]="0"
ConfigQuestion["dataset_size"]="1000"
ConfigQuestion["llm-engine"]="GenerateVLLMSingle"
ConfigQuestion["LLM_Address_1"]="ipc:///tmp/pipeline2.ipc"
ConfigQuestion["GenerateChatTemplate"]="GenerateChatWithSystem"
ConfigQuestion["GenerationMode"]="InferenceProcess"
ConfigQuestion["GenerateChatTemplate"]="GenerateChatWithSystem"
ConfigQuestion["GenerateChatTemplate"]="GenerateChatWithSystem"
ConfigQuestion["GenerateChatTemplate"]="GenerateChatWithSystem" #{GenerateChatWithSystem,GenerateChatWithoutSystem,GenerateChatWithSystemWithAssistant,GenerateChatWithoutSystemWithAssistant}
ConfigQuestion["PreTopic"]=" The topic is about "
ConfigQuestion["Topics"]=['trees', 'graphs', 'lists', 'strings', 'heaps', 'linked lists', 'matrices', 'intervals', 'dynamic programming', 'binary numbers', 'data structures', 'backtracking','base conversion','bfs','binary search','binaty trees','bit manipulation','bitmasks','count pairs in array','custom comparator','greedy scheduling','heaps','intervals','line sweep','sliding windows','string stream','rolling hash','trie','topological sort','two pointers','','','','','','','','','','','','','','','','','','','','','','','','','','','','','','','','','','']
ConfigQuestion["Topics"]=['']



#ConfigQuestion["llm-tokenizer_model_id"]="Qwen/CodeQwen1.5-7B-Chat"

ConfigQuestion["max_tokens"]="2000"
ConfigQuestion["top_p"]="0.85"
ConfigQuestion["top_k"]="35"
ConfigQuestion["min_p"]="0.15"
ConfigQuestion["max_tokens"]="2000"
ConfigQuestion["tensor_parallel_size"]="1"


ConfigQuestion["temperature"]="0.9"
ConfigQuestion["prompt"]="You are helpful coding assistant."
ConfigQuestion["user"]="Please generate one python programming question."
ConfigQuestion["use-grammar"]="False"
ConfigQuestion["grammar"]="Yes|No"
ConfigQuestion["grammar-analyze"]="True"
ConfigQuestion["max_num_seqs"]="16"
ConfigQuestion["max_model_len"]="2048"
ConfigQuestion["SystemRole"] = "True"
ConfigQuestion["UseOutlines"]="False"

ConfigTest=dict()
ConfigTest["type"]="LLM"
ConfigTest["llm-type"]="VLLM"
ConfigTest["llm-path"]="Qwen/CodeQwen1.5-7B-Chat"
ConfigTest["tokenizer-path"]="Qwen/CodeQwen1.5-7B-Chat"
ConfigTest["tokenizer-file-gguf"]=""
ConfigTest["max_tokens"]="2000"
ConfigTest["temperature"]="0.7"
ConfigTest["max_num_seqs"]="256"
ConfigTest["max_model_len"]="2048"
ConfigTest["max_num_batched_tokens"]="8192"

ConfigTest["input"]=""
ConfigTest["prompt"]="You are helpful coding assistant"
ConfigTest["user"]="Please generate one python programming question."

ConfigTest["grammar"]="False"
ConfigTest["NumberofTests"]="100"
ConfigTest["LoadTest"]='False'
ConfigTest["LoadTestFile"]='TestData2.json'
ConfigTest["LoadEval"]='False'
ConfigTest["LoadEvalFile"]='EvalRes.json'
ConfigTest["use-grammar"]="False"
ConfigTest["enforce_eager"]="True"
ConfigTest["llm-engine"]="GenerateVLLMSingle"
ConfigTest["LLM_Address_1"]="ipc:///tmp/pipeline2.ipc"
ConfigTest["gpu-id"]="0"





ConfigSolution=dict()
if UseCuda=='True':
  ConfigSolution["llm-device_map"]='cuda'
else:
  ConfigSolution["llm-device_map"]='cpu'

ConfigSolution["type"]="LLM"
ConfigSolution["Filename"]="Solutions.pkl"
ConfigSolution["LoadFlag"]="True"


ConfigSolution["use-grammar"]="False"
ConfigSolution["grammar"]="Yes|No"
ConfigSolution["grammar-analyze"]="True"

ConfigSolution["llm-type"]="VLLMInferenceService"
ConfigSolution["llm-type"]="VLLMInference"
ConfigSolution["GenerationMode"]="InferenceProcess"

#ConfigSolution["llm-path"]="Qwen/CodeQwen1.5-7B-Chat"
#ConfigSolution["llm-ref-path"]="Qwen/CodeQwen1.5-7B-Chat"
ConfigSolution["llm-path"]="microsoft/Phi-3.5-mini-instruct"
ConfigSolution["llm-ref-path"]="microsoft/Phi-3.5-mini-instruct"
ConfigSolution["llm-path"]="Qwen/CodeQwen1.5-7B-Chat"
ConfigSolution["llm-ref-path"]="Qwen/CodeQwen1.5-7B-Chat"
ConfigSolution["tokenizer-path"]="Qwen/CodeQwen1.5-7B-Chat"
ConfigSolution["tokenizer-file-gguf"]=""
ConfigSolution["quantization"]=None

ConfigSolution["max_tokens"]="3000"
ConfigSolution["temperature"]="0.1"
ConfigSolution["max_num_seqs"]="512"
ConfigSolution["max_model_len"]="16000"
ConfigSolution["max_num_batched_tokens"]="16000"
ConfigSolution["use-grammar"]="False"
ConfigSolution["grammar"]="Yes|No"
ConfigSolution["grammar-analyze"]="True"
ConfigSolution["seed"]="None"
ConfigSolution["enforce_eager"]="True"
ConfigSolution["cpu_offload_gb"]="0"
ConfigSolution["swap_space"]="0"
ConfigSolution["gpu-id"]="0"
ConfigSolution["top_p"]="0.85"
ConfigSolution["top_k"]="35"
ConfigSolution["min_p"]="0.15"
ConfigSolution["UseOutlines"]="True"


ConfigSolution["input"]=""
ConfigSolution["system"]="Solve the following question and make sure you write production code with exception handling, checking for all edge cases, and writing inline comments:"
ConfigSolution["llm-torch_dtype"]="torch.float16"
ConfigSolution["llm-ref-≈"]="torch.float16"
ConfigSolution["llm-path-bos_token"]="True"
ConfigSolution["FT-output-dir"]='./results'
ConfigSolution["FT-overwrite_output_dir"]="True"
ConfigSolution["FT-do_train"]="True"
ConfigSolution["FT-do_eval"]="True"
ConfigSolution["FT-per_device_train_batch_size"]="1"
ConfigSolution["FT-per_device_eval_batch_size"]="1"
ConfigSolution["FT-learning_rate"]="5e-5"
ConfigSolution["FT-num_train_epochs"]="1"
ConfigSolution["FT-logging_dir"]='./logs'
ConfigSolution["FT-logging_steps"]="10"
ConfigSolution["FT-save_steps"]="100"
ConfigSolution["FT-optim"]="adamw_bnb_8bit"
#ConfigSolution["FT-optim"]="sgd"
ConfigSolution["FT-tokenizer-truncation"]="True"
ConfigSolution["FT-tokenizer-padding"]='max_length'
ConfigSolution["FT-tokenizer-max_length"]="2000"
ConfigSolution["FT-SaveModel"]='a.model'
ConfigSolution["LoadData"]='False'
ConfigSolution["DataFileName"]='DataFile'
ConfigSolution["gradient_accumulation_steps"]="8"
ConfigSolution["gpu_memory_utilization"]="0.99"
ConfigQuestion["gpu_memory_utilization"]="0.99"

ConfigSolution["llm-engine"]="GenerateVLLMSingle"
ConfigSolution["LLM_Address_1"]="ipc:///tmp/pipeline2.ipc"
ConfigSolution["SystemRole"] = "True"
ConfigSolution["GenerateChatTemplate"]="GenerateChatWithSystem" #{GenerateChatWithSystem,GenerateChatWithoutSystem,GenerateChatWithSystemWithAssistant,GenerateChatWithoutSystemWithAssistant}





ConfigEmbedding=dict()
if UseCuda=='True':
  ConfigEmbedding["llm-device_map"]='cuda'
else:
  ConfigEmbedding["llm-device_map"]='cpu'

ConfigEmbedding["Filename"]="Solutions.pkl"
ConfigEmbedding["LoadFlag"]="True"
ConfigEmbedding["type"]="LLM"
ConfigEmbedding["llm-engine"]="GenerateVLLMSingleEmbeddings"
ConfigEmbedding["LLM_Address_1"]="ipc:///tmp/pipeline2.ipc"
ConfigEmbedding["quantization"]=None
ConfigEmbedding["llm-type"]="VLLMInference"
ConfigEmbedding["GenerationMode"]="InferenceProcess"
ConfigEmbedding["use-grammar"]="False"
ConfigEmbedding["grammar"]="Yes|No"
ConfigEmbedding["grammar-analyze"]="True"
ConfigEmbedding["llm-type"]="VLLMEmbeddings"
ConfigEmbedding["llm-path"]="intfloat/e5-mistral-7b-instruct"
ConfigEmbedding["tokenizer-path"]="intfloat/e5-mistral-7b-instruct"
ConfigEmbedding["tokenizer-file-gguf"]=""
ConfigEmbedding["max_tokens"]="2000"
ConfigEmbedding["temperature"]="0.9"
ConfigEmbedding["max_num_seqs"]="2048"
ConfigEmbedding["max_model_len"]="2048"
ConfigEmbedding["max_num_batched_tokens"]="2048"
ConfigEmbedding["use-grammar"]="False"
ConfigEmbedding["grammar"]="Yes|No"
ConfigEmbedding["grammar-analyze"]="True"
ConfigEmbedding["seed"]="1"
ConfigEmbedding["MaxSimilarity"]="0.95"
ConfigEmbedding["input"]=""
ConfigEmbedding["system"]="Solve the following question and make sure you write production code with exception handling, checking for all edge cases, and writing inline comments:"
ConfigEmbedding["llm-torch_dtype"]="torch.float16"
ConfigEmbedding["llm-ref-torch_dtype"]="torch.float16"
ConfigEmbedding["llm-path-bos_token"]="True"
ConfigEmbedding["LoadData"]='False'
ConfigEmbedding["DataFileName"]='DataFile'
ConfigEmbedding["gradient_accumulation_steps"]="2"
ConfigEmbedding["TSNE-n_components"]="2"
ConfigEmbedding["gpu_memory_utilization"]="0.4"

ConfigEmbedding["enforce_eager"]="True"
ConfigEmbedding["cpu_offload_gb"]="0"
ConfigEmbedding["swap_space"]="0"
ConfigEmbedding["gpu-id"]="0"
ConfigEmbedding["top_p"]="0.85"
ConfigEmbedding["top_k"]="35"
ConfigEmbedding["min_p"]="0.15"
ConfigEmbedding["SystemRole"] = "True"
ConfigEmbedding["tensor_parallel_size"]="1"





ConfigPreference=dict()
ConfigPreference["Filename"]="Preferences.pkl"
ConfigPreference["LoadFlag"]="True"
ConfigPreference["type"]="LLM"
ConfigPreference["quantization"]=None

ConfigPreference["llm-type"]="VLLMInferenceService"
ConfigPreference["llm-type"]="VLLMInference"
ConfigPreference["LLM_Address_1"]="ipc:///tmp/pipeline2.ipc"

ConfigPreference["llm-path"]="Qwen/CodeQwen1.5-7B-Chat"
ConfigPreference["tokenizer-path"]="Qwen/CodeQwen1.5-7B-Chat"

#ConfigPreference["llm-path"]="UCLA-AGI/Gemma-2-9B-It-SPPO-Iter3"
#ConfigPreference["tokenizer-path"]="UCLA-AGI/Gemma-2-9B-It-SPPO-Iter3"
#ConfigPreference["llm-path"]="unsloth/Meta-Llama-3.1-8B"
#ConfigPreference["tokenizer-path"]="unsloth/Meta-Llama-3.1-8B"


ConfigPreference["tokenizer-file-gguf"]=""
ConfigPreference["max_tokens"]="2000"
ConfigPreference["temperature"]="0.1"
ConfigPreference["max_num_seqs"]="2100"
ConfigPreference["max_model_len"]="2100"
ConfigPreference["max_num_batched_tokens"]="16384"
ConfigPreference["use-grammar"]="True"
ConfigPreference["grammar"]="Yes|No"
ConfigPreference["grammar-dual"]="A|B"
ConfigPreference["grammar-analyze"]="True"
ConfigPreference["seed"]="1"
ConfigPreference["cpu_offload_gb"]="0"
ConfigPreference["swap_space"]="0"
ConfigPreference["top_p"]="0.85"
ConfigPreference["top_k"]="35"
ConfigPreference["min_p"]="0.15"
ConfigPreference["gpu_memory_utilization"]="0.49"
ConfigPreference["gpu_memory_utilization"]="0.99"
ConfigPreference["SystemRole"] = "False"
ConfigPreference["tensor_parallel_size"]="1"

ConfigPreference["max_tokens"]="2000"
ConfigPreference["temperature"]="0"
ConfigPreference["input"]=""
ConfigPreference["system"]="You are given a programming questions and a solution below:\n\n"
ConfigPreference["system-dual"]="You are given a programming question and two solutions below:\n\nProgramming question:\n\n"
ConfigPreference["use-grammar"]="True"
ConfigPreference["user"]=[" Does this code tests all edge cases and is production code? (Yes or No): ",
                              " Does this code tests all edge cases? (Yes or No): ",
                              " Is this code production code? (Yes or No): "]
ConfigPreference["user-dual"]=[" Which solution best tests all edge cases and is production code? (A or B): ",
                              " Which solution best tests all edge cases? (A or B): ",
                              " Which solution best uses production code? (A or B): "]
ConfigPreference["enforce_eager"]="True"
ConfigPreference["llm-engine"]="GenerateVLLMSingle"
ConfigPreference["GenerationMode"]="InferenceProcess"

ConfigPreference["gpu-id"]="0"
ConfigPreference["UseOutlines"]="False"
ConfigPreference["GenerateChatTemplate"]="GenerateChatWithSystem" #{GenerateChatWithSystem,GenerateChatWithoutSystem,GenerateChatWithSystemWithAssistant,GenerateChatWithoutSystemWithAssistant}


ConfigBroker=dict()
ConfigBroker["type"]="Broker"
ConfigBroker["port"]="8000"
ConfigBroker["LLM_Address_1"]="ipc:///tmp/pipeline2.ipc"
ConfigBroker["ReadyAddress"]="ipc:///tmp/pipeline.ipc"
#tokenizer.add_special_tokens({"bos_token": tokenizer.eos_token})
#tokenizer.bos_token_id = tokenizer.eos_token_id
MyDataset={
         "prompt": [],
         "completion": [],
         "label": []
        }


Config['ConfigQuestion'] = ConfigQuestion
Config['ConfigSolution'] = ConfigSolution
Config['ConfigPreference'] = ConfigPreference
Config['ConfigTest']  = ConfigTest
Config['ConfigEmbedding'] = ConfigEmbedding
Config['ConfigBroker'] = ConfigBroker


MyDatasets=[]