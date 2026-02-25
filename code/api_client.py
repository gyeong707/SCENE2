import os
import json
import torch
from openai import OpenAI
from dotenv import load_dotenv
from transformers import AutoModelForCausalLM, AutoTokenizer

# 환경 변수 로드
load_dotenv()

class BaseHandler:
    def generate(self, prompt, system_content="You are a helpful assistant.", temperature=0.0, seed=None):
        raise NotImplementedError("Error")
    
    def prepare_batch_line(self, custom_id, prompt, system_content="You are a helpful assistant.", temperature=0.0, seed=None):
        raise NotImplementedError("Not Supporting Batch API.")

    def submit_batch_job(self, jsonl_path):
        raise NotImplementedError("Not Supporting Batch API.")

    def check_batch_status(self, batch_id):
        raise NotImplementedError("Not Supporting Batch API.")

    def retrieve_batch_result(self, file_id):
        raise NotImplementedError("Not Supporting Batch API.")


class OpenAIHandler(BaseHandler):
    def __init__(self, model_name):
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.model_name = model_name

    def _build_messages(self, prompt, system_content="You are a helpful assistant."):
        return [
            {"role": "system", "content": system_content},
            {"role": "user", "content": prompt}
        ]

    def generate(self, prompt, system_content="You are a helpful assistant.", temperature=0.0, max_token=512, seed=None):
        try:
            # print(self._build_messages(prompt, system_content))
            params = {
                "model": self.model_name,
                "messages": self._build_messages(prompt, system_content),
                "temperature": temperature,
            }
            if seed is not None:
                params['seed'] = seed
            
            response = self.client.chat.completions.create(**params)
            return response.choices[0].message.content.strip()
        
        except Exception as e:
            print(f"OpenAI API Error: {e}")
            return "Error"

    def prepare_batch_line(self, custom_id, prompt, system_content="You are a helpful assistant.", temperature=0.0, seed=None):
        body = {
            "model": self.model_name,
            "messages": self._build_messages(prompt, system_content),
            "temperature": temperature
        }
        if seed is not None:
            body['seed'] = seed

        request_obj = {
            "custom_id": str(custom_id),
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": body
        }
        # print(self._build_messages(prompt, system_content))
        return json.dumps(request_obj)
    

    def submit_batch_job(self, jsonl_path):
        """파일 업로드 및 배치 작업 생성"""
        try:
            print(f"파일 업로드 중: {jsonl_path}")
            
            with open(jsonl_path, "rb") as f:
                batch_input_file = self.client.files.create(
                    file=f,
                    purpose="batch"
                )
            
            print(f"배치 작업 요청 중... (File ID: {batch_input_file.id})")
            batch_job = self.client.batches.create(
                input_file_id=batch_input_file.id,
                endpoint="/v1/chat/completions",
                completion_window="24h",
                metadata={"description": f"bias_eval_{self.model_name}"}
            )
            return batch_job.id
        except Exception as e:
            print(f"배치 제출 실패: {e}")
            return None

    def check_batch_status(self, batch_id):
        try:
            return self.client.batches.retrieve(batch_id)
        except Exception as e:
            print(f"상태 확인 실패: {e}")
            return None

    def retrieve_batch_result(self, file_id):
        try:
            return self.client.files.content(file_id).text
        except Exception as e:
            print(f"결과 다운로드 실패: {e}")
            return None
        
    
class TransformersHandler(BaseHandler):

    def __init__(self, model_path):
        print(f":: Loading Local Model from: {model_path} ...")
        self.model_path = model_path
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # 모델별 thinking 억제 플래그
        self.is_qwen = "Qwen" in model_path
        self.is_exaone = "EXAONE" in model_path
        self.is_gpt_oss = "gpt-oss" in model_path.lower()

        # gpt-oss 판별 추가
        try:
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_path, 
                trust_remote_code=True
            )
            
            # Load model (GPU 자동 할당, FP16/BF16 자동 선택)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                trust_remote_code=True,
                device_map="auto",
                torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
            )
            self.model.eval()
            
            # Pad Token이 없으면 Eos Token으로 대체 (생성 에러 방지)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                
            print(f":: Model Loaded on {self.device}")
            if self.is_qwen:
                print(f":: Qwen3 detected - thinking mode disabled")
            elif self.is_exaone:
                print(f":: EXAONE-Deep detected - thinking mode disabled")
            elif self.is_gpt_oss:
                print(f":: GPT-OSS detected - Reasoning suppression active")            


        except Exception as e:
            print(f":: Model Load Failed: {e}")
            raise e

    def generate(self, prompt, system_content="You are a helpful assistant.", temperature=0.0, max_token=512, seed=None):
        try:
            # Message construction
            messages = [
                {"role": "system", "content": system_content},
                {"role": "user", "content": prompt}
            ]
            
            # Apply chat template (모델별 thinking 억제)
            try:
                template_kwargs = {
                    "tokenize": False,
                    "add_generation_prompt": True,
                }

                if self.is_qwen or self.is_exaone or self.is_gpt_oss:
                    template_kwargs["enable_thinking"] = False
                    template_kwargs["include_reasoning"] = False
                
                if self.is_gpt_oss:
                    template_kwargs["reasoning_effort"] = 'low'

                text_input = self.tokenizer.apply_chat_template(messages, **template_kwargs)
                # print(text_input)
            
            except Exception as e:
                print(f"Chat template warning: {e}. Using raw concatenation.")
                text_input = f"{system_content}\n\n{prompt}"

            # Tokenizing
            inputs = self.tokenizer(text_input, return_tensors="pt").to(self.model.device)

            # Setting
            do_sample = temperature > 0.0
            gen_kwargs = {
                "max_new_tokens": max_token,
                "do_sample": do_sample,
                "pad_token_id": self.tokenizer.pad_token_id,
                "eos_token_id": self.tokenizer.eos_token_id
                }
            if do_sample:
                gen_kwargs["temperature"] = temperature
                gen_kwargs["top_p"] = 0.9

            if seed is not None:
                torch.manual_seed(seed)

            # Inferencing
            with torch.no_grad():
                outputs = self.model.generate(**inputs, **gen_kwargs)
            
            # Decoding
            input_len = inputs.input_ids.shape[1]
            generated_ids = outputs[0][input_len:]
            decoded = self.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

            if getattr(self, 'is_gpt_oss', False):
                current_max_token = max_token
                
                if 'assistantfinal' in decoded and 'A' in decoded and 'B' in decoded: 
                    is_finished = True
                else: is_finished = False

                while not is_finished:
                    # 상한선 체크 (1024)
                    if current_max_token >= 1024:
                        print(">>> GPT-OSS: Reached 1024 tokens limit. Giving up.")
                        break
                    
                    # 토큰 2배 증가
                    current_max_token *= 2
                    if current_max_token > 1024: 
                        current_max_token = 1024
                    
                    print(f">>> GPT-OSS: Output truncated. Retrying with {current_max_token} tokens...")
                    
                    # 설정 업데이트 및 재생성
                    gen_kwargs["max_new_tokens"] = current_max_token
                    
                    with torch.no_grad():
                        outputs = self.model.generate(**inputs, **gen_kwargs)
                    
                    # 결과 확인
                    generated_ids = outputs[0][input_len:]
                    decoded = self.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
                    
                    if 'assistantfinal' in decoded and 'A' in decoded and 'B' in decoded: 
                        is_finished = True
                        break

            elif getattr(self, 'is_exaone', False):
                # 시도할 온도 리스트
                retry_temps = [0.0, 0.2, 0.4, 0.6, 0.8]
                
                print(f">>> Exaone: Output incomplete/looped. Starting Temperature Scaling Strategy...")
                suffix_prompt = "</thought>"
                
                retry_text = text_input + suffix_prompt
                retry_inputs = self.tokenizer(retry_text, return_tensors="pt").to(self.model.device)
                retry_input_len = retry_inputs.input_ids.shape[1]

                print(f">>> Exaone: Loop suspected. Retrying with instruction reinforcement...")

                for i, temp_val in enumerate(retry_temps):
                    print(f">>> Exaone: Retry {i+1}/5 with Temperature={temp_val}, Penalty=1.1  ...")
                    if temp_val >= 0.1:
                        gen_kwargs["do_sample"] = True
                        gen_kwargs["temperature"] = temp_val 
                        gen_kwargs["top_p"] = 0.9
                        gen_kwargs["repetition_penalty"] = 1.1

                    with torch.no_grad():
                        outputs = self.model.generate(**retry_inputs, **gen_kwargs)
                    
                    decoded = self.tokenizer.decode(outputs[0][retry_input_len:], skip_special_tokens=True).strip()

                    if "A: [" in decoded or "A: " in decoded or "</thought>" in decoded:
                        is_finished = True
                        print(f">>> Exaone: Success at Temperature={temp_val}")
                        break
                
                if not is_finished:
                     print(">>> Exaone: Failed all temperature retries.")

            return decoded

        except Exception as e:
            print(f"Local Inference Error: {e}")
            return "Error"


def get_model_handler(model_name):
    """
    main.py의 args.model_name을 실제 핸들러로 매핑
    """
    model_map = {
        # GPT (frontier)
        'gpt-5.1': 'gpt-5.1', 
        'gpt-5.2': 'gpt-5.2',

        # GPT-OSS
        'gpt-oss-20b': 'openai/gpt-oss-20b',

        # Exaone
        'exaone-7b': 'LGAI-EXAONE/EXAONE-Deep-7.8B',
        'exaone-32b': 'LGAI-EXAONE/EXAONE-Deep-32B',
        
        # Llama
        'llama-7b': 'meta-llama/Llama-3.1-8B-Instruct',
        'llama-70b': 'meta-llama/Llama-3.1-70B-Instruct',
        
        # Qwen
        'qwen-8b': 'Qwen/Qwen3-8B',
        'qwen-32b': 'Qwen/Qwen3-32B',

        # 
    }
    
    real_path = model_map.get(model_name, model_name)
    name_lower = model_name.lower()
    
    if name_lower.startswith("gpt-5"):
        return OpenAIHandler(real_path)
    else:
        return TransformersHandler(real_path)