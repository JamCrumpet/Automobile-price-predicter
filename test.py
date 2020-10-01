import pandas as pd

df = pd.read_csv("Automobile price data _Raw_.csv")

with open(df,"r") as file_object:
    """ String remover """
    contents = file_object.read()
    print(contents.replace("?",""))
