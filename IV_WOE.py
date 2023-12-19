dataset = pd.read_parquet('s3://lending-data-science/Vishal/Nonstarter/Version2/Finalrawdata_ForModelv3_step2/')
dataset.index=dataset['loan_account_number']
dataset =dataset.drop(['customer_id','asof','loan_account_number'],axis=1)
​
flag_list = [0,1]
dataset = dataset[dataset["NS_Flag3"].isin(flag_list)]
​
train_dataset = dataset[dataset["Sample"] == "train"]
test_dataset = dataset[dataset["Sample"] == "test"]
oot_dataset = dataset[dataset["Sample"] == "oot"]
​
train_dataset = train_dataset.drop("Sample",axis =1)
test_dataset = test_dataset.drop("Sample",axis =1)
oot_dataset = oot_dataset.drop("Sample",axis =1)
​
final_features=train_dataset.columns
​
​
# calculate iv value using optbinning library:
# print tab to see the table.
# use special_codes=[-9999.0] for handle missing value. For this check the next cell.
#!pip install optbinning
​
from optbinning import OptimalBinning
​
ivs = dict(np.complex64(x) for x in range(0))
​
details_binning_table = pd.DataFrame(columns = ['Variable_Name','Bin', 'Count', 'Count (%)', 'Non-event', 'Event', 'Event rate', 'WoE','IV', 'JS'])
len(details_binning_table.columns)
​
​
features = final_features
​
​
j=0
for variable in features:
    x = train_dataset[variable].values
    y = train_dataset['NS_Flag3']
    
    optb = OptimalBinning(name=variable, dtype="numerical", solver="cp",special_codes=[-999999.0])
    optb.fit(x, y)
​
    binning_table = optb.binning_table
    tab = binning_table.build()
​
    tab['Variable_Name'] = variable
    details_binning_table.loc[len(details_binning_table.index)] = tab.groupby("Variable_Name").agg(list).reset_index().loc[0]
    iv = tab.IV.Totals.round(3)
    print(iv)
    ivs[variable] = round(iv,2)
    print(ivs)
    j=j+1
    print('####################',j,'#################')
​
ivs_dataframe = pd.DataFrame(ivs.items(), columns = ["Variable_Name", "IV"]).sort_values(by="IV", ascending = False)
​
# # print("Writing to dataframe")
iv_summary_path = f"IV_all_Feature_special_9999_571feat.csv"
ivs_dataframe.to_csv(iv_summary_path, index = False)
​
details_binning_table.to_csv('IV_NS_Features_60feat.csv',index = False)
