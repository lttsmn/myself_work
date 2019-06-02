import pandas as pd
df = pd.DataFrame({"key":['green','red', 'blue'],
            "data1":['a','b','c'],"sorce": [33,61,99]})
data1 = pd.concat([df,df],ignore_index=True)
print("---------------")
print(data1)
print("---------------")
data2=data1[-data1.sorce.isin([61])]
print(data2)
