import pynng
import os
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"]="python"

from lmformatenforcer.regexparser import RegexParser
from lmformatenforcer.integrations.vllm import build_vllm_logits_processor, build_vllm_token_enforcer_tokenizer_data
from transformers import AutoTokenizer, AutoModelForCausalLM
import time
from vllm import LLM, SamplingParams
import vllm
import pickle
import outlines 
from vllm.distributed.parallel_state import destroy_model_parallel


def GetInferenceClass():
  class Inference:
      def Destruct(self):
        #from vllm.distributed.parallel_state import destroy_model_parallel
        destroy_model_parallel()
        #del self.llm.llm_engine.model_executor

        #del self.llm
      def __init__(self, Config):
        self.Config=Config.copy()
        self.State={}
        import os
        ##TODO find number of GPUs
        os.environ["CUDA_VISIBLE_DEVICES"] = self.Config["gpu-id"]
        os.environ["NCCL_SHM_DISABLE"]="1"
        os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
        #os.environ["NCCL_P2P_DISABLE"]="1"


        from lmformatenforcer.regexparser import RegexParser
        from lmformatenforcer.integrations.vllm import build_vllm_logits_processor, build_vllm_token_enforcer_tokenizer_data
        from transformers import AutoTokenizer, AutoModelForCausalLM
        import time
        from vllm import LLM, SamplingParams
        import vllm
        import outlines
        print(self.Config["llm-path"])
        self.llm = LLM(model=self.Config["llm-path"],max_num_seqs=int(self.Config["max_num_seqs"]),max_model_len=int(self.Config["max_model_len"]),max_num_batched_tokens=int(Config["max_num_batched_tokens"]), gpu_memory_utilization=float(self.Config["gpu_memory_utilization"]),enforce_eager=bool(self.Config["enforce_eager"]), cpu_offload_gb=float(self.Config["cpu_offload_gb"]), swap_space=float(self.Config["swap_space"]),tensor_parallel_size=int(self.Config["tensor_parallel_size"]),quantization=self.Config["quantization"])#,distributed_executor_backend="ray")

        if self.Config["UseOutlines"] == "True":
          self.llm=outlines.models.VLLM(self.llm)
        if "adapter-path" in self.Config and self.Config["adapter-path"]!="":
          self.llm.load_adapter(self.Config["adapter-path"], self.Config["adapter-name"])

        if self.Config["tokenizer-file-gguf"]!="":
            self.tokenizer = AutoTokenizer.from_pretrained(self.Config["tokenizer-path"], gguf_file=self.Config["tokenizer-file-gguf"])
            if 'chat_template' in self.Config:
                self.tokenizer.chat_template=self.Config['chat_template']
        else: 
            self.tokenizer = AutoTokenizer.from_pretrained(self.Config["tokenizer-path"])
            if 'chat_template' in self.Config:
                self.tokenizer.chat_template=self.Config['chat_template']


      def SetSamplingParams(self, Config):
        from lmformatenforcer.regexparser import RegexParser
        from lmformatenforcer.integrations.vllm import build_vllm_logits_processor, build_vllm_token_enforcer_tokenizer_data
        from transformers import AutoTokenizer, AutoModelForCausalLM
        import time
        from vllm import LLM, SamplingParams
        import vllm
        if  self.Config["use-grammar"]=="False":
            if Config["seed"]=='None':
                self.sampling_params = vllm.SamplingParams(seed=None,max_tokens=int(self.Config["max_tokens"]),temperature=float(self.Config["temperature"]),top_p=float(self.Config["top_p"]),min_p=float(self.Config["min_p"]),top_k=int(self.Config["top_k"]))
            else:
                self.sampling_params = vllm.SamplingParams(seed=int(Config["seed"]),max_tokens=int(self.Config["max_tokens"]),temperature=float(self.Config["temperature"]),top_p=float(self.Config["top_p"]),min_p=float(self.Config["min_p"]),top_k=int(self.Config["top_k"]))

        else:
            self.parser = RegexParser(self.Config["grammar"])
            if self.Config["UseOutlines"] == "True":
                self.generator = outlines.generate.regex(self.llm,r".*```python.*")#self.Config["grammar"])


        #if self.Config["tokenizer-file-gguf"]!="":
        #    self.tokenizer = AutoTokenizer.from_pretrained(self.Config["tokenizer-path"], gguf_file=self.Config["tokenizer-file-gguf"])
        #else: 
        #    self.tokenizer = AutoTokenizer.from_pretrained(self.Config["tokenizer-path"])

        if  Config["use-grammar"]=="True":
            if  Config["UseOutlines"]=="False":
              tokenizer_data = build_vllm_token_enforcer_tokenizer_data(self.llm)
              logits_processor = build_vllm_logits_processor(tokenizer_data, self.parser, analyze=bool(self.Config["grammar-analyze"]))

            if Config["seed"]=='None':
                if  Config["UseOutlines"]=="False":
                  self.sampling_params = vllm.SamplingParams(seed=None,max_tokens=int(self.Config["max_tokens"]),temperature=float(self.Config["temperature"]),top_p=float(self.Config["top_p"]),min_p=float(self.Config["min_p"]),top_k=int(self.Config["top_k"]), logits_processors=[logits_processor])
                else:
                  self.sampling_params = vllm.SamplingParams(seed=None,max_tokens=int(self.Config["max_tokens"]),temperature=float(self.Config["temperature"]),top_p=float(self.Config["top_p"]),min_p=float(self.Config["min_p"]),top_k=int(self.Config["top_k"]))

                    
            else:
                self.sampling_params = vllm.SamplingParams(seed=int(Config["seed"]),max_tokens=int(self.Config["max_tokens"]),temperature=float(self.Config["temperature"]),top_p=float(self.Config["top_p"]),min_p=float(self.Config["min_p"]),top_k=int(self.Config["top_k"]), logits_processors=[logits_processor])

      def GenerateChatWithSystem(self,System,User):
        self.chat = [self.tokenizer.decode(self.tokenizer.apply_chat_template([{"role": "system", "content": System},{"role": "user", "content": j}], tokenize=True)) for j in User]
        #print(self.chat[0])
        tmp = [self.tokenizer.apply_chat_template([{"role": "system", "content": System},{"role": "user", "content": j}], tokenize=True) for j in User]
        #print(tmp[0])
        #print(len(tmp[0]))

#      def GenerateChatWithoutSystem(self,System,User):
#        self.chat = [self.tokenizer.decode(self.tokenizer.apply_chat_template([{"role": "user", "content": i+"\n"+j}], tokenize=True)) for i,j in zip(System,User)]

      def GenerateChatWithSystemWithAssistant(self,System,User):
        #print()
        self.chat = [self.tokenizer.decode(self.tokenizer.apply_chat_template([{"role": "system", "content": System},{"role": "user", "content": j},{"role": "assistant", "content": self.Config["AssistantPrompt"]}], tokenize=True)[:-int(self.Config["AssistantPromptElementsRemoval"])or None]) for j in User]
        

      def GenerateChatWithoutSystemWithAssistant(self,System,User):
        self.chat = [self.tokenizer.decode(self.tokenizer.apply_chat_template([{"role": "user", "content":i},{"role": "assistant", "content":self.Config["AssistantPrompt"]}], tokenize=True)[:-int(self.Config["AssistantPromptElementsRemoval"])or None]) for i in User]

      def GenerateChatWithoutSystemWithAssistantStart(self,System,User):
        self.chat = [self.tokenizer.decode(self.tokenizer.apply_chat_template([{"role": "user", "content":i}], tokenize=True, add_generation_prompt=True)) for i in User]

      def GenerateChatWithSystemWithAssistantStart(self,System,User):
        self.chat = [self.tokenizer.apply_chat_template([{"role": "system", "content": System},{"role": "user", "content": j}], tokenize=False, add_generation_prompt=True) for j in User]

      def GenerateChatSetSystem(self,System,User):
        self.chat = [self.tokenizer.decode(
                         self.tokenizer.apply_chat_template(
                             System + [{"role": "user", "content": j},{"role": "assistant", "content": self.Config["AssistantPrompt"]}],
                             tokenize=True
                         )[:-int(self.Config["AssistantPromptElementsRemoval"]) or None]
                     )
                     for j in User
                    ]


      def GenerateVLLMSingle(self,System,User):
        func = getattr(locals()["self"], self.Config["GenerateChatTemplate"])
        func(System,User)
        msg=""
        #print(self.chat[0])
        #print(self.chat[1])
          
        #return self.chat
        if self.Config["UseOutlines"] == "False":
          Out=self.llm.generate(self.chat, self.sampling_params)

        else:
          Out=self.generator(self.chat)
        #print(Out)
        return Out
      #def GenerateVLLMSingleEmbeddings(self,System,User):
      #  self.chat = [self.tokenizer.decode(self.tokenizer.apply_chat_template([{"role": "system", "content": i},{"role": "user", "content": j}], tokenize=True)) for i,j in zip(System,User)]
      #  Out=self.llm.encode(self.chat)
      #  return Out

      def GenerateVLLMSingleEmbeddings(self,System,User):
        Out=self.llm.encode(User)
        return Out


  return Inference


def VLLMInferenceService(Config):
    LLMsRunning={}
    import pynng
    import pickle
    for Keys,Conf in Config.items():
        if Conf["type"]=="LLM" and Conf["llm-type"]=="VLLMInferenceService":
            if Conf["llm-path"] not in LLMsRunning:
                LLMsRunning[Conf["llm-path"]] = GetInferenceClass()(Conf)
    with pynng.Req0() as sockLoaded:
        sockLoaded.dial(Config["ConfigBroker"]["ReadyAddress"])
        sockLoaded.send(b'Done.')
    with pynng.Rep0() as sock:
        sock.listen(Config["ConfigBroker"]["LLM_Address_1"])
        
        while True:
            msg = sock.recv()
            (Conf,System,User)=pickle.loads(msg)
            if Conf["type"]=="LLM" and Conf["llm-type"]=="VLLMInferenceService":
                Infer=LLMsRunning[Conf["llm-path"]]
            else:
                continue
            Infer.SetSamplingParams(Conf)
            func=getattr(Infer,Conf["llm-engine"])
            #print(Conf)

            Out=func(System,User)
            #print(Conf)

            stat = sock.send(pickle.dumps(Out))
    return Out
    
def Inference(Config, System, User):
    Infer=GetInferenceClass()(Config)
    Infer.SetSamplingParams(Config)
    Out=Infer.GenerateVLLMSingle(System,User)
    return Out

  
def InferenceProcess(Config,State,Input):
    System=Input["System"]
    User=Input["User"]
    Infer=GetInferenceClass()(Config)
    Infer.SetSamplingParams(Config)
    Out=Infer.GenerateVLLMSingle(System,User)
    State["q"].put(Out)

def InferenceProcessBlocked(Config,State,Input):
    System=Input["System"]
    User=Input["User"]
    Infer=GetInferenceClass()(Config)
    Infer.SetSamplingParams(Config)
    Out=[]
    BlockSize=int(Config["BlockSize"])
    for i in range(0,len(User),BlockSize):
      OutTemp=Infer.GenerateVLLMSingle(System[i:min(i+BlockSize,len(User))],User[i:min(i+BlockSize,len(User))])
      Out.extend(OutTemp)
    State["q"].put(Out)


def InitAndRunEmbeddings(Config, System, User):
    Infer=GetInferenceClass()(Config)
    Infer.SetSamplingParams(Config)
    Out=Infer.GenerateVLLMSingleEmbeddings(System,User)
    return Out

def VLLMInferenceServiceClient(Config, System, User):
    with pynng.Req0() as sock:
        sock.dial(Config["LLM_Address_1"])
        msg=pickle.dumps((Config, System, User))
        out=sock.send(msg)
        results=pickle.loads(sock.recv())
    return results

def VLLMInferenceClient(Config, System, User):
    import time
    import os
    #from torch.multiprocessing import Pool, Process, set_start_method
    import torch.multiprocessing  as mp

    #with Pool(processes=1,) as pool:
    #    results = pool.starmap(InitAndRunInference, [(Config, System, User)])
    os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
    func=globals()[Config['GenerationMode']]

    Input={}
    State={}
    Input["System"]=System
    Input["User"]=User
    State["q"]=mp.Queue()

    #results=[InitAndRunInferenceProcess(Config, System, User,q)]
    p=mp.Process(target=func,args=(Config,State,Input))
    p.start()
    results=[State["q"].get(block=True)]
    p.join()
    return results[0]

def VLLMEmbeddingsClient(Config, System, User):
    import time
    import os
    from torch.multiprocessing import Pool, Process, set_start_method
    #InitAndRunInference(Config, System, User)
    #multiprocessing.set_start_method('spawn')
    with Pool(processes=1,) as pool:
        results = pool.starmap(InitAndRunEmbeddings, [(Config, System, User)])
    return results[0]


def GenerateChat(Config,System, User):
    func=globals()[Config['llm-type']+'Client']
    #print(func)
    chat=func(Config, System, User)

    #print(chat)
    Results=[]
    ResultsLogging=[]
    for i in range(0,len(chat)):
        Results.append(chat[i].outputs[0].text)
        ResultsLogging.append(chat[i])
    return (Results,ResultsLogging)

def GenerateEmbeddings(Config,System, User,func=VLLMEmbeddingsClient):
    func=globals()[Config['llm-type']+'Client']

    chat=func(Config, System, User)
    Results=[]
    ResultsLogging=[]
    for i in range(0,len(chat)):
        Results.append(chat[i].outputs.embedding)
        ResultsLogging.append(chat[i])
    return (Results,ResultsLogging)

def GetQuestions(ConfigQ, NumberOfSamples):
    User=[ConfigQ["user"]]*int(NumberOfSamples)
    System=ConfigQ["prompt"]#]*int(NumberOfSamples )
    (Questions,QuestionsLogging)=GenerateChat(ConfigQ,System,User)
    return (Questions,QuestionsLogging)

def GetEmbeddings(ConfigS, Questions, System):
    User=Questions
    System = [System]*len(Questions)
    (Embeddings,EmbeddingsLogging)=GenerateEmbeddings(ConfigS,System,User)
    return (Embeddings,EmbeddingsLogging)


def GetSolutions(ConfigS, Questions, System):
    User=Questions
    (SolutionsBase,SolutionsBaseLogging)=GenerateChat(ConfigS,System,User)
    return (SolutionsBase,SolutionsBaseLogging)


def GetPreferences(ConfigP, Questions,Solutions,SystemExample,UserList):
    System = [SystemExample]*len(Questions*len(UserList))
    User=[]
    for i in range(0,len(UserList)):
        User_=[i+j+k for i,j,k in zip(Questions,Solutions,[UserList[i]]*len(Questions))]
        User.extend(User_)
    (PreferencesBaseSingle,PreferencesBaseSingleLogging)=GenerateChat(ConfigP,System,User)
    return (PreferencesBaseSingle,PreferencesBaseSingleLogging)

def GetEmbeddingsMaxSimilarity(ConfigE, Questions):
    from sklearn.metrics.pairwise import cosine_similarity
    import numpy as np


    (QuestionsEmbeddings,QuestionsEmbeddingsLogging)=GetEmbeddings(ConfigE, Questions,'')
    pairwise_cos=cosine_similarity(QuestionsEmbeddings, QuestionsEmbeddings)
    maxSimilarity=[]
    for i in range(0,len(pairwise_cos)-1):
        maxSimilarity.append(max(pairwise_cos[i][i+1:]))
    MyInd=[ind for ind,val in enumerate(maxSimilarity) if val<float(ConfigE['MaxSimilarity']) ]
    return MyInd


def GetQuestion(ConfigQ):
    NumOfQuestions=int(ConfigQ["dataset_size"])
    (Questions,QuestionsLogging)=GetQuestions(ConfigQ,NumOfQuestions)
    return Questions

def GetSolutionSingle(ConfigS, Questions):
    System=ConfigS["system"]
    (SolutionsBase,SolutionsBaseLogging)=GetSolutions(ConfigS, Questions, System)
    return SolutionsBase

def GetSolutionDual(ConfigS, Questions):
    System=ConfigS["system"]
    (SolutionsBase,SolutionsBaseLogging)=GetSolutions(ConfigS, Questions, System)

    ConfigS['seed']="2"
    FileName="Dual"+ConfigS["Filename"]
    (SolutionsSecond,SolutionsSecondLogging)=GetSolutions,(ConfigS, Questions, System)
    return (SolutionsBase,SolutionsSecond)

def GetPreferenceSingle(ConfigP,Questions,SolutionsBase):
    SolutionsSingle=[j+i for i,j in zip(SolutionsBase,["\n\nSolution:\n"]*len(Questions))]
    SystemExample=ConfigP["system"]
    UserList=ConfigP["user"]
    (PreferencesFirst,PreferencesFirstLogging)=GetPreferences(ConfigP,Questions,SolutionsSingle,SystemExample,UserList)
    return PreferencesFirst

def GetEmbeddingsSolutions(ConfigE, Questions, System_=""):
    User=Questions
    System = [System_]*len(Questions)

    (EmbeddingsSolutions,EmbeddingsSolutionsLogging)=GenerateEmbeddings(ConfigE,System,User)
    return (EmbeddingsSolutions,EmbeddingsSolutionsLogging)

def GetTSNESolutions(ConfigE, SolutionsBase, System_=""):
    (EmbeddingsSolutions,EmbeddingsSolutionsLogging)=GetEmbeddingsSolutions(ConfigE, SolutionsBase)
    import numpy as np
    from sklearn.manifold import TSNE

    n_components=ConfigE["TSNE-n_components"]
    np.array(EmbeddingsSolutions).shape
    tsne = TSNE(int(n_components))
    tsne_result = tsne.fit_transform(np.array(EmbeddingsSolutions))
    return tsne_result

def GetConfig(ConfigName):
    ConfigNameSpace={}
    Config=[]
    State=[]
    with open(ConfigName,'r') as file:
        ConfigContent=file.read()
        exec(ConfigContent, ConfigNameSpace)
        Config=ConfigNameSpace['Config']
        State=ConfigNameSpace['State']
        return Config


def UnitTest(SolutionsCleaned,UnitTestsClean,RemoveKeywords=[],Timeout=5):
    import multiprocessing
    import concurrent.futures

    import re
    import sys
    from io import StringIO
    def TestFun(Statement1,Statement2,queue):
      ret=queue.get()
      value=False
      old_stdout = sys.stdout
      sys.stdout = StringIO()
      redirected_output = sys.stdout
      scope={}
      try:
        exec(Statement1,scope)
        exec(Statement2,scope)
        Text=redirected_output.getvalue()
        sys.stdout = old_stdout
        value=True
      except Exception as e:
        Text=redirected_output.getvalue()
        Text=str(e)+"\n\n"+Text
        sys.stdout = old_stdout
        value=False

      sys.stdout = old_stdout
      ret["ret"]=(value,Text)
      queue.put(ret)

    UnitTestsResults=[]
    UnitTestsPass=[]
    UnitTestsString=[]
    counter=-1;
    for i,j in zip(SolutionsCleaned,UnitTestsClean):
      counter+=1
      loc={}
      UnitTestsString.append([i,j])
      RemoveFound=any([k in i+"\n"+j  for k in RemoveKeywords])

      if RemoveFound:
        UnitTestsPass.append(False)
        UnitTestsResults.append("RemoveKeywords")
        continue

      ret = {'ret': None}
      queue = multiprocessing.Queue()
      queue.put(ret)

      print(counter)
      process = multiprocessing.Process(target=TestFun,args=(i,j,queue))
      process.start()
      process.join(Timeout)
      if process.is_alive():
        UnitTestsPass.append(False)
        UnitTestsResults.append("Timeout")
        continue
      Ret=queue.get(ret)["ret"]
      UnitTestsPass.append(Ret[0])
      UnitTestsResults.append(Ret[1])

    return (UnitTestsResults, UnitTestsPass, UnitTestsString)


import sys
from io import StringIO
import multiprocessing

def RunUnitTest(i,j,q,RemoveKeywords=[],Timeout=5):
      #print("dsfsdfsad")
      TestString=[i,j]
      loc={}
      RemoveFound=any([k in i+"\n"+j  for k in RemoveKeywords])

      if RemoveFound:
        q.put(("RemoveKeywords",False,TestString))
        return

      ret = {'ret': None}
      queue = multiprocessing.Queue()
      queue.put(ret)

      process = multiprocessing.Process(target=TestFun,args=(i,j,queue))
      process.start()
      process.join(Timeout)
      if process.is_alive():
        q.put(("Timeout",False,TestString))
        process.terminate()
        return None
   
      Ret=queue.get(ret)["ret"]
      q.put((Ret[1],Ret[0],TestString))
      process.close()


def TestFun(Statement1,Statement2,queue):

      ret=queue.get()
      value=False
      old_stdout = sys.stdout
      sys.stdout = StringIO()
      redirected_output = sys.stdout
      scope={}
      try:
        exec(Statement1,scope)
        exec(Statement2,scope)
        Text=redirected_output.getvalue()
        sys.stdout = old_stdout
        value=True
      except Exception as e:
        Text=redirected_output.getvalue()
        Text=str(e)+"\n\n"+Text
        sys.stdout = old_stdout
        value=False

      sys.stdout = old_stdout
      ret["ret"]=(value,Text)
      queue.put(ret)


def UnitTestParallel(SolutionsCleaned,UnitTestsClean,RemoveKeywords=[],Timeout=5,BlockSize=64):
    import multiprocessing

    import re
    import sys
    from io import StringIO
    lists=([],[],[])

    for counter in range(0,len(SolutionsCleaned),BlockSize):
      Queues=[]
      Processes=[]
      print(counter)

      for i,j in zip(SolutionsCleaned[counter:counter+min(BlockSize,len(SolutionsCleaned))],UnitTestsClean[counter:counter+min(BlockSize,len(SolutionsCleaned))]):

        Queues.append(multiprocessing.Queue())

        Processes.append(multiprocessing.Process(target=RunUnitTest,args=(i,j,Queues[-1])))
        Processes[-1].start()

      
      for i in range(len(Queues)):
        Out=Queues[i].get(block=True)

        lists[0].append(Out[0])
        lists[1].append(Out[1])
        lists[2].append(Out[2])
        Processes[i].join()
        Processes[i].close()



    return lists



