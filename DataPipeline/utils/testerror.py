import json
import ast

# 打开输出文件，以追加模式写入
with open('DataPipeline/output/test/test3_1.json', 'w') as g:
    dst = []

    # 逐行读取源文件内容
    with open('DataPipeline/output/message/error_answer_store.json', 'r') as f:
        for line in f:
            try:
                # 使用 ast.literal_eval 尝试解析每一行的内容
                items = ast.literal_eval(line.strip())
                for item in items:
                    if item['风险点'] == '无':
                        item['风险类别'] = '无风险'
                    else:
                        item['风险类别'] = '有风险'
                    dst.append(item)
                
                # 写入当前解析的结果到输出文件
                json.dump(dst, g, indent=4, ensure_ascii=False)
                g.write('\n')  # 写入换行符，以便下一行内容

            except Exception as e:
                print(f'以下文本无法解析: {line}')
                print(f'错误信息: {str(e)}')
