import pandas as pd


df = pd.read_csv("paragraphs.csv")

print(df.iloc[150].Paragraphs)
print(df.iloc[150].Questions)

# # make a new dataset with only paragraphs and questions
# paragraphs = df['Paragraph']
# questions = df['Question']

# # save the new dataset
# new_df = pd.DataFrame({'Paragraphs': paragraphs, 'Questions': questions})
# new_df.to_csv("paragraphs.csv", index=False)
