import pandas as pd
import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))



if __name__ == '__main__':
    datapath = os.path.join(BASE_DIR,"dataset/PII43k.csv")
    df = pd.read_csv(
        datapath,
        on_bad_lines='skip',
        engine='python'
    )

    print(df.columns)

    filled_column = df['Filled Template'].dropna().astype(str)


    cleaned_text = filled_column.apply(lambda x: ' '.join(x.strip().split()))
    output_path = os.path.join(BASE_DIR,"cleaned_dataset/cleaned_pii.txt")

    with open(output_path, 'w', encoding='utf-8') as f:
        for line in cleaned_text:
            f.write(line + '\n')