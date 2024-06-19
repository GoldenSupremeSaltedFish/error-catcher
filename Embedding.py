import requests  # 导入用于发送HTTP请求的库
import json  # 导入用于处理JSON数据的库
import nltk  # 导入自然语言处理库
import re  # 导入正则表达式库
from nltk.corpus import stopwords  # 从nltk库中导入停用词模块
from nltk.tokenize import word_tokenize  # 从nltk库中导入分词模块
from collections import Counter  # 导入计数器模块，用于词频统计
from concurrent.futures import ThreadPoolExecutor  # 导入线程池执行器模块，用于并发处理

# 下载nltk相关资源
nltk.download('punkt')  # 下载分词器数据
nltk.download('stopwords')  # 下载停用词数据

# 设置API密钥
api_key = ''  # 这里填写你的API密钥
custom_keywords = ["error", "aaa"]  # 自定义关键词列表

# 设置请求头
headers = {
    'Authorization': f'Bearer {api_key}',  # 设置授权头，使用Bearer Token
    'Content-Type': 'application/json'  # 设置内容类型为JSON
}

def read_log_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:  # 打开日志文件，使用UTF-8编码
        return file.readlines()  # 读取文件的所有行并返回

def process_text(input_text):
    max_tokens = 8191  # 定义每个文本块的最大标记数
    input_tokens = nltk.word_tokenize(input_text)  # 对输入文本进行分词
    chunks = [input_tokens[i:i + max_tokens] for i in range(0, len(input_tokens), max_tokens)]  # 将文本分块
    for chunk in chunks:
        chunk_text = ' '.join(chunk)  # 将分块后的标记重新组合成文本
        data = {
            "model": "text-embedding-ada-002",  # 指定使用的模型
            "input": chunk_text,  # 输入文本
        }
        response = requests.post('https://api.openai.com/v1/embeddings', headers=headers, data=json.dumps(data))  # 发送POST请求
        if response.status_code == 200:  # 检查请求是否成功
            response_data = response.json()  # 获取响应数据
            words = word_tokenize(data["input"])  # 对输入文本再次分词
            stop_words = set(stopwords.words('english'))  # 获取英文停用词
            filtered_words = [word for word in words if word.lower() not in stop_words]  # 过滤掉停用词
            word_counts = Counter(filtered_words)  # 统计词频
            custom_keywords_counts = {word: word_counts[word] for word in custom_keywords if word in word_counts}  # 统计自定义关键词的词频
            # 检查是否包含自定义关键词
            if any(word in filtered_words for word in custom_keywords):
                print("嵌入向量：", response_data['data'][0]['embedding'])  # 打印嵌入向量
                keyword_positions = {word: [i for i, token in enumerate(words) if token.lower() == word.lower()] for word in custom_keywords if word in words}  # 获取关键词的位置
                print("关键词及其位置信息：", keyword_positions)  # 打印关键词及其位置信息
                pattern = r'error \d{4}'  # 定义正则表达式模式
                matches = re.finditer(pattern, data["input"])  # 查找匹配的模式
                match_positions = {match.group(): match.start() for match in matches}  # 获取匹配的模式及其位置
                print("自定义关键词的词频：", custom_keywords_counts)  # 打印自定义关键词的词频
                print("匹配到的固定格式关键词及其位置信息：", match_positions)  # 打印匹配到的固定格式关键词及其位置信息
        else:
            print("请求失败：", response.status_code, response.text)  # 打印请求失败的信息

def handle_line(line):
    if 'assertion failed' in line.lower() or 'compiler error' in line.lower():  # 检查行中是否包含特定的错误信息
        print("Special handling for line:", line.strip())  # 对特定错误信息进行特殊处理
    else:
        process_text(line)  # 处理其他行

def main():
    log_file_path = "C:\\Users\\Administrator\\Desktop\\错误处理\\txt\\error1.log"  # 这里填写你的.log文件路径
    log_lines = read_log_file(log_file_path)  # 读取日志文件的所有行
    with ThreadPoolExecutor(max_workers=5) as executor:  # 创建一个有5个工作线程的线程池
        executor.map(handle_line, log_lines)  # 并发处理每一行

if __name__ == "__main__":
    main()  # 运行主函数
    #返回embedding，和特殊错误行，报错信息合并未实现，等待具体数据
