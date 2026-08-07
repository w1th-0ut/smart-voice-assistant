import time
import ollama

def ask_llm_stream(prompt):
    start_time = time.time()
    first_token_time = None
    
    # 启用 stream=True 开启流式响应
    stream = ollama.chat(
        model='qwen2.5:1.5b',
        messages=[
            {
                'role': 'system',
                'content': '你是一个语音助手。请用极简短、自然的一句话回答（20字以内），绝不要换行，不要Markdown符号。'
            },
            {'role': 'user', 'content': prompt}
        ],
        stream=True,
        options={
            'num_predict': 50,  # 严格限制最大输出 Token 数量，大幅提升速度
            'temperature': 0.3  # 较低的随机性，生成更快
        }
    )

    print("AI 响应: ", end="", flush=True)
    full_response = ""
    
    for chunk in stream:
        if first_token_time is None:
            first_token_time = time.time() - start_time
        content = chunk['message']['content']
        print(content, end="", flush=True)
        full_response += content
        
    print(f"\n\n[性能统计] 首字延迟: {first_token_time:.2f}秒 | 总耗时: {time.time() - start_time:.2f}秒")
    return full_response

if __name__ == '__main__':
    ask_llm_stream("今天天气不错，你能做些什么？")
