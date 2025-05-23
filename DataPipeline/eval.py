import os
from openai import OpenAI
from utils import pipeline


if __name__ == '__main__':
    api_key = os.getenv('DEEPSEEK_API_KEY')
    client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")

    seed_template = [
        {
            "role": "user",
            "content": 
            ''' <start_csv>
            文本,风险点,标签
            "您好，这里是中国建设银行，我们怀疑您的银行账户出现异常活动，请提供您的个人信息以验证身份。",["提供您的个人信息"],1
            "尊敬的客户，您已被选中参加我们的奖品抽奖活动，只需支付一小笔费用即可获得高额奖金。",["支付一小笔费用"，"高额奖金"],1
            "您好，我是申通快递，您买的一个包裹，公司给您弄丢了，这里需要加我们的理赔客服对您快递进行理赔200元。",["加我们的理赔客服"],1
            "您好，这是一条关于您的快递的通知，由于地址错误，需要支付额外的费用进行重新寄送。",[],0
            "您好，我是您的移动运营商客服，您的账户余额已不足，请及时充值以避免影响正常使用。",[],0
            "尊敬的客户，您的手机尾号2345的机主，目前已经欠费10元，将会影响您的宽带使用。",[],0
            <end_csv>
            以上是一个csv格式的诈骗文本数据集，标签为1代表有很高诈骗风险的样本，0代表几乎没有诈骗风险的样本。风险点表示文本被判定为高风险的证据。从文本中抽取得到，可以有0-5个风险点。请按此格式生成更多数据集，
            要求如下：
            1. 必须符合人类可读性。
            2.文本长度不超过100个字,数据总量不限制。
            3. 数据之间不要重复，尽可能在发送方身份，接收方身份，事件背景，事件内容和要求上都做到多样
            4. 数据格式必须为csv格式。
            5. 句式本身尽可能多样。
            6. 必须包含一定比例的标签为1的文本和标签为0的文本
            7. 给我的回复里不要包含数据集以外的任何内容
            '''
        }
    ]
    
    pipe = pipeline(client)
    answer = pipe(seed_template)
    print('\n' + answer)



