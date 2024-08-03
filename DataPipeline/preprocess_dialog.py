import pandas as pd


# print(delete_rows['text'])

def deleteDuplicates(df):
    print(len(df))
    duplicates = df.duplicated(subset=['text'])                 # 保留
    # duplicates = df.duplicated(subset=['text'], keep=False)     # 全部去除
    delete_rows = df[duplicates]
    print(delete_rows)
    df.drop(delete_rows.index, inplace=True)
    # df.to_csv("DataPipeline/output/dialog/prompt1_unrepeated.csv",encoding="utf-8",index=False)

    

def delete0(df):
    df_filtered = df[df['f_index'] != '0']
    df_filtered.to_csv("DataPipeline/output/dialog/prompt1_test1.csv",encoding="utf-8",index=False)
    return df_filtered

def typeCount(df):
    value_counts = df['riskType'].value_counts()
    value_counts_df = value_counts.reset_index()
    value_counts_df.columns = ['Type', 'Count']
    value_counts_df['Proportion'] = value_counts_df['Count']/len(df)
    # value_counts_df.to_csv("account.csv",encoding="utf-8")
    print(value_counts_df)
    for value, count in value_counts.items():
        print(f"Value: {value}, Count: {count}, Proportion: {count/len(df)*100:.2f}%")

if __name__ == '__main__':
    df1 = pd.read_csv("DataPipeline/output/dialog/prompt1_unrepeated.csv",encoding="utf-8")
    df2 = pd.read_csv("DataPipeline/output/dialog/prompt2_unrepeated.csv",encoding="utf-8")
    
    
    # print("---------------")
    # typeCount(df2)
    # print(len(delete0(df1)))
    # print(len(df2))
    # deleteDuplicates(df1)
    typeCount(df2)
    print("---------------------------------")
    # typeCount(df2)
    # print(len(df1))
