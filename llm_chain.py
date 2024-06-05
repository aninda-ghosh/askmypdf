from transformers import AutoConfig, AutoTokenizer, BitsAndBytesConfig, AutoModelForCausalLM, pipeline
from langchain_community.llms import HuggingFacePipeline
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
import torch

class LLM_Chain:
    """
    A class to create a Language Model Chain (LLMChain) using the HuggingFace pipeline and LangChain.

    Attributes:
        llm_model_path (str): Path to the pre-trained language model.
        llm_config (transformers.configuration_utils.PretrainedConfig): Configuration of the pre-trained model.
        llm_tokenizer (transformers.tokenization_utils_base.PreTrainedTokenizerBase): Tokenizer for the pre-trained model.
        quantization_config (BitsAndBytesConfig or None): Configuration for model quantization.
        llm_model (transformers.modeling_utils.PreTrainedModel): The loaded pre-trained model.
        chain (LLMChain): The LangChain chain object for handling prompts and generating responses.

    Methods:
        __init__(llm_model_path='mistralai/Mistral-7B-Instruct-v0.3', enable_quantization=True):
            Initializes the LLM_Chain with the specified model path and quantization settings.
        
        get_quantization_config() -> BitsAndBytesConfig:
            Generates the configuration for quantization of the model.
    """

    def __init__(self, llm_model_path='mistralai/Mistral-7B-Instruct-v0.3', enable_quantization=True):
        """
        Initializes the LLM_Chain with the specified model path and quantization settings.

        Args:
            llm_model_path (str): Path to the pre-trained language model. Default is 'mistralai/Mistral-7B-Instruct-v0.3'.
            enable_quantization (bool): Flag to enable model quantization. Default is True.
        """
        self.llm_model_path = llm_model_path
        self.llm_config = AutoConfig.from_pretrained(self.llm_model_path)
        self.llm_tokenizer = AutoTokenizer.from_pretrained(self.llm_model_path, trust_remote_code=True)
        self.llm_tokenizer.pad_token = self.llm_tokenizer.eos_token
        self.llm_tokenizer.padding_side = 'right'

        if enable_quantization:
            self.quantization_config = self.get_quantization_config()
        else:
            self.quantization_config = None
        
        self.llm_model = AutoModelForCausalLM.from_pretrained(
            self.llm_model_path, 
            quantization_config=self.quantization_config
        )

        llm_pipeline = pipeline(            
            model=self.llm_model,
            tokenizer=self.llm_tokenizer, 
            task='text-generation', 
            temperature=0.005,
            repetition_penalty=1.1,
            return_full_text=True, 
            max_new_tokens=300
        )

        prompt_template = """
        ### [INST] 
        Instruction: Answer the question based on the following context, be grounded to the context and provide a detailed answer.

        ### CONTEXT:

        {context}

        ### QUESTION:
        {question} 

        [/INST]
        """

        llm = HuggingFacePipeline(pipeline=llm_pipeline)

        prompt = PromptTemplate(
            input_variables=['context', 'page_number', 'question'],
            template=prompt_template
        )

        self.chain = LLMChain(llm=llm, prompt=prompt)

    def get_quantization_config(self):
        """
        Generates the configuration for quantization of the model.

        Returns:
            BitsAndBytesConfig: Configuration for 4-bit quantization of the model.
        """
        # Activate 4-bit precision base model loading
        use_4bit = True
        # Compute dtype for 4-bit base models
        bnb_4bit_compute_dtype = "float16"
        # Quantization type (fp4 or nf4)
        bnb_4bit_quant_type = "nf4"
        # Activate nested quantization for 4-bit base models (double quantization)
        use_nested_quant = False

        compute_dtype = getattr(torch, bnb_4bit_compute_dtype)
        
        # Check GPU compatibility with bfloat16
        if compute_dtype == torch.float16 and use_4bit:
            major, _ = torch.cuda.get_device_capability()
            if major >= 8:
                print("=" * 80)
                print("Your GPU supports bfloat16: accelerate training with bf16=True")
                print("=" * 80)
        
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=use_4bit,
            bnb_4bit_quant_type=bnb_4bit_quant_type,
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=use_nested_quant,
        )

        return bnb_config
