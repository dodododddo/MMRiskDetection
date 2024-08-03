import re
from openai import OpenAI

def insert_stars(original_str, positions):
    str_list = list(original_str)
    
    sorted_positions = sorted(positions, key=lambda x: x['start_position'], reverse=True)
    # print(sorted_positions)
    
    for pos in sorted_positions:
        start = pos['start_position']
        end = pos['end_position']
        str_list.insert(end, '</span>')
        str_list.insert(start, '<span style="color: red;">')
    
    modified_str = ''.join(str_list)
    return modified_str

def Ext(content):

    openai_api_key = "EMPTY"
    openai_api_base = "http://localhost:8000/v1"

    client = OpenAI(
        api_key=openai_api_key,
        base_url=openai_api_base,
    )

    prompt = (
        f"从以下文本中提取风险点"
        "风险点应以逗号分隔"
        "必须从原文中提取"
    )

    text1 = "您好，欢迎加入百花齐放群，这里有很多赚钱的机会。请点击链接下载STQ软件，按照提示购买春、夏、秋、东即可获得收益。记得先充值哦，这样才能开始赚钱。"
    text2 = "您好，我是金源贷款的客服，请问您是否有贷款需求？如果有，可以加我的QQ（QQ号：123456789）详细了解。我们提供快速便捷的贷款服务，您只需下载我们的APP并填写相关信息即可。"
    text3 = "您好，请问您是否开通了京东金条？为了您的账户安全，我们建议您尽快注销该服务。请您下载云视讯APP，并在APP内按照指示操作，将您建设银行账户中的资金转至我们提供的建行账户，以完成注销流程。"

    response = client.chat.completions.create(
        model="llama3",
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": text1},
            {"role": "assistant", "content": "点击链接下载,先充值"},
            {"role": "user", "content": text2},
            {"role": "assistant", "content": "下载我们的APP,填写相关信息"},
            {"role": "user", "content": text3},
            {"role": "assistant", "content": "下载云视讯APP,资金转至我们提供的建行账户"},
            {"role": "user", "content": content}
        ],
        stream=False
    )
    keywords = response.choices[0].message.content.strip()
    # 将关键字转换为列表
    keywords_list = [keyword.strip() for keyword in keywords.split(',')]

    # 查找每个关键字在原文本中的位置
    result = []
    for keyword in keywords_list:
        matches = [match.start() for match in re.finditer(re.escape(keyword), content)]
        if matches:  # 只有在找到匹配时才加入结果
            for match in matches:
                result.append({
                    "keyword": keyword, 
                    "start_position": match, 
                    "end_position": match + len(keyword),
                    "length": len(keyword)
                })

    return insert_stars(content, result)

if __name__ == '__main__':
    # 输入文本
    content = input("请输入需要分析的文本：")

    # 使用OpenAI API提取关键字
    result = Ext(content)

    # 输出关键字
    print(result)