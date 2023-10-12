import torch
from transformers import (AutoTokenizer, 
                          LlamaForCausalLM, 
                          AutoModelForCausalLM, 
                          )
from flask import Flask, jsonify  


# 定义api app, 模型路径，以及设备(CUDA or CPU)
app = Flask(__name__)  
MODEL_PATH = "microsoft/phi-1_5"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  

# 定义推理函数  
def inference(prompt):
    # 加载模型  
    model = AutoModelForCausalLM.from_pretrained(MODEL_PATH,trust_remote_code=True,device_map=device) 
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH,trust_remote_code=True) 

    model = torch.compile(model)
    model.eval()
    
    # 处理输入，生成输出
    inputs = tokenizer(prompt,return_tensors="pt").to(device)
    generate_ids = model.generate(**inputs, 
                                    # num_beams=5,
                                    max_new_tokens=50,
                                    early_stopping=True,
                                    ) 
    with torch.no_grad():
        result = tokenizer.decode(generate_ids[0], skip_special_tokens=True)
    
    return result  

# 定义推理api接口  
# 通过访问 <LINK>/prompt/<STRING> 来获取推理结果
@app.route('/prompt/<string:prompt>', methods=['GET'])
def recommend(prompt):
    print("Prompt:",prompt)
    mock_recommendation = {
        "prompt": prompt,
        "response": inference(prompt)
    }

    return jsonify(mock_recommendation)  

if __name__ == "__main__":
    # 启动api服务
    app.run(host='0.0.0.0', port=8082)
