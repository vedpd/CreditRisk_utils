
import numpy as np
import pandas as pd
from pandas._libs.lib import is_integer

def weighted_qcut(values, weights, q, **kwargs):
    'Return weighted quantile cuts from a given series, values.'
    if is_integer(q):
        quantiles = np.linspace(0, 1, q + 1)
    else:
        quantiles = q
    order = weights.iloc[values.argsort()].cumsum()
#     print("order", order)
    bins = pd.cut(order / order.iloc[-1], quantiles, **kwargs)
    return bins.sort_index()




def gains_table(pipeline1_data):
    
    final_prod_pdf = pipeline1_data[["customer_id", "prob", "target",'Sampling_Weights_ttd']]
#     final_prod_pdf['model_tiers'] = pd.qcut(final_prod_pdf['prob'], 20, labels = False) 
    final_prod_pdf['model_tiers'] = weighted_qcut(final_prod_pdf['prob'], final_prod_pdf['Sampling_Weights_ttd'], 20, labels=False)
    final_prod_pdf['model_tiers'] = 20 - final_prod_pdf['model_tiers'] #+ 1
    
    count_bad_pord = final_prod_pdf.query("target == 0").groupby(["model_tiers"])["Sampling_Weights_ttd"].sum().reset_index(name="count_bad")
    count_good_pord = final_prod_pdf.query("target == 1").groupby(["model_tiers"])["Sampling_Weights_ttd"].sum().reset_index(name="count_good")
    count_indeterminate_pord = final_prod_pdf.query("target == 2").groupby(["model_tiers"])["Sampling_Weights_ttd"].sum().reset_index(name="count_indeterminate")
    
    score_max_prod = final_prod_pdf.groupby(["model_tiers"])["prob"].max().reset_index(name="score_max")
    score_median_prod = final_prod_pdf.groupby(["model_tiers"])["prob"].median().reset_index(name="score_median")
    score_min_prod = final_prod_pdf.groupby(["model_tiers"])["prob"].min().reset_index(name="score_min")

    population_tier_prod = final_prod_pdf.groupby(["model_tiers"])["Sampling_Weights_ttd"].sum().reset_index(name="tier_count")
    
    final_df_prod = pd.merge(population_tier_prod, count_bad_pord, how='left', on='model_tiers')
    final_df_prod = pd.merge(final_df_prod, count_good_pord, how='left', on='model_tiers')
    final_df_prod = pd.merge(final_df_prod, count_indeterminate_pord, how='left', on='model_tiers')

    final_df_1_prod = pd.merge(final_df_prod, score_min_prod, how='left', on='model_tiers')
    final_df_1_prod = pd.merge(final_df_1_prod, score_median_prod, how='left', on='model_tiers')
    final_df_2_prod = pd.merge(final_df_1_prod, score_max_prod, how='left', on='model_tiers')
    
    final_df_2_prod['total_popu_percent'] = round((final_df_2_prod.tier_count / final_df_2_prod.tier_count.sum())*100,2)
    
    final_df_2_prod = final_df_2_prod.sort_values(by=['model_tiers'],ascending = True).reset_index(drop=True)
    
    final_df_2_prod['total_popu_cum'] = final_df_2_prod['total_popu_percent'].cumsum()

    # final_df_2_prod['prod_count_good'] = final_df_2_prod.prod_tier_count - final_df_2_prod.prod_count_bad
    final_df_2_prod['bad_rate'] = round((final_df_2_prod.count_bad / final_df_2_prod.count_bad.sum())*100,5)
    final_df_2_prod['good_rate'] = round((final_df_2_prod.count_good / final_df_2_prod.count_good.sum())*100,5)
    final_df_2_prod['indeterminate_rate'] = round((final_df_2_prod.count_indeterminate / final_df_2_prod.count_indeterminate.sum())*100,5)
    
    final_df_2_prod['decile_good_rate'] = round((final_df_2_prod.count_good / final_df_2_prod.tier_count)*100,5)
    final_df_2_prod['decile_bad_rate'] =  round((final_df_2_prod.count_bad / final_df_2_prod.tier_count)*100,5)
    final_df_2_prod['decile_indeterminate_rate'] =  round((final_df_2_prod.count_indeterminate / final_df_2_prod.tier_count)*100,5)

    final_df_2_prod['cumulative_bad_rate'] = final_df_2_prod.loc[::-1, 'bad_rate'].cumsum()[::-1]
    final_df_2_prod['cumulative_good_rate'] = final_df_2_prod.loc[::-1, 'good_rate'].cumsum()[::-1]
    final_df_2_prod['cumulative_indeterminate_rate'] = final_df_2_prod.loc[::-1, 'indeterminate_rate'].cumsum()[::-1]
    
    final_df_2_prod['cumulative_decile_good_rate'] = round((final_df_2_prod.count_good.cumsum() / final_df_2_prod.tier_count.cumsum())*100,5)
    final_df_2_prod['cumulative_decile_bad_rate'] =  round((final_df_2_prod.count_bad.cumsum() / final_df_2_prod.tier_count.cumsum())*100,5)
     

    final_df_2_prod['KS_prod'] = final_df_2_prod['cumulative_good_rate']-final_df_2_prod['cumulative_bad_rate']
    final_df_2_prod["KS_prod"] = final_df_2_prod["KS_prod"].abs()
    
    
#     final_df_2_prod_new = final_df_2_prod.loc[:,['model_tiers', 'prod_tier_count','score_max','score_min',
#        'prod_count_good','prod_count_bad', 'prod_count_indeterminate', 
#        'good_distribution','bad_distribution','indeterminate_distribution',
#        'cumulative_good_rate_prod', 'cumulative_bad_rate_prod', 
#        'cumulative_indeterminate_rate_prod', 'KS_prod']]
    
    
    
    final_df_2_prod_new = final_df_2_prod.loc[:,['model_tiers','tier_count','total_popu_cum',
                                                 'score_min','score_max','count_good','count_bad','count_indeterminate',
                                                 'bad_rate','indeterminate_rate',
                                                 'decile_bad_rate','decile_good_rate','decile_indeterminate_rate',
                                                 'cumulative_good_rate','cumulative_bad_rate','cumulative_indeterminate_rate',
                                                 'cumulative_decile_good_rate','cumulative_decile_bad_rate','KS_prod']]
    
    
    final_df_2_prod_new.columns = ['model_tiers','# accounts','cumulative accounts',
                                   'score_min','score_max','# good','# bad','# indeterminate','bad_rate',
                                   'indeterminate_rate','tier_bad_rate','tier_good_rate','tier_indeterminate_rate',
                                   'cumulative_good_captured','cumulative_bad_captured',
                                'cumulative_indeterminant_captured','tier_cumulative_good_rate','tier_cumulative_bad_rate','KS']
    
    print(pipeline1_data['Sampling_Weights_ttd'].sum())
    print(final_df_2_prod_new['# accounts'].sum())
    
    return final_df_2_prod_new.round(2)


def gains_table_pd_qcut(pipeline1_data):
    
    final_prod_pdf = pipeline1_data[["loan_account_number", "prob", "target",'Sampling_Weights_ttd']]
    final_prod_pdf['model_tiers'] = pd.qcut(final_prod_pdf['prob'], 10, labels=False)
    final_prod_pdf['model_tiers'] = 10 - final_prod_pdf['model_tiers'] #+ 1
    # print(final_prod_pdf.head())
    
    count_bad_pord = final_prod_pdf.query("target == 0").groupby(["model_tiers"])["Sampling_Weights_ttd"].sum().reset_index(name="count_bad")
    count_good_pord = final_prod_pdf.query("target == 1").groupby(["model_tiers"])["Sampling_Weights_ttd"].sum().reset_index(name="count_good")
    #count_indeterminate_pord = final_prod_pdf.query("target == 2").groupby(["model_tiers"])["Sampling_Weights_ttd"].sum().reset_index(name="count_indeterminate")
    score_max_prod = final_prod_pdf.groupby(["model_tiers"])["prob"].max().reset_index(name="score_max")
    score_median_prod = final_prod_pdf.groupby(["model_tiers"])["prob"].median().reset_index(name="score_median")
    score_min_prod = final_prod_pdf.groupby(["model_tiers"])["prob"].min().reset_index(name="score_min")
    population_tier_prod = final_prod_pdf.groupby(["model_tiers"])["Sampling_Weights_ttd"].sum().reset_index(name="tier_count")
    
    final_df_prod = pd.merge(population_tier_prod, count_bad_pord, how='inner', on='model_tiers')
    final_df_prod = pd.merge(final_df_prod, count_good_pord, how='inner', on='model_tiers')
    #final_df_prod = pd.merge(final_df_prod, count_indeterminate_pord, how='inner', on='model_tiers')
    final_df_1_prod = pd.merge(final_df_prod, score_min_prod, how='inner', on='model_tiers')
    final_df_1_prod = pd.merge(final_df_1_prod, score_median_prod, how='inner', on='model_tiers')
    final_df_2_prod = pd.merge(final_df_1_prod, score_max_prod, how='inner', on='model_tiers')
    
    final_df_2_prod['total_popu_percent'] = round((final_df_2_prod.tier_count / final_df_2_prod.tier_count.sum())*100,2)
    
    final_df_2_prod = final_df_2_prod.sort_values(by=['model_tiers'],ascending = True).reset_index(drop=True)
    
    final_df_2_prod['total_popu_cum'] = final_df_2_prod['total_popu_percent'].cumsum()
#     final_df_2_prod['prod_count_good'] = final_df_2_prod.prod_tier_count - final_df_2_prod.prod_count_bad
    final_df_2_prod['bad_rate'] = round((final_df_2_prod.count_bad / final_df_2_prod.count_bad.sum())*100,5)
    final_df_2_prod['good_rate'] = round((final_df_2_prod.count_good / final_df_2_prod.count_good.sum())*100,5)
   # final_df_2_prod['indeterminate_rate'] = round((final_df_2_prod.count_indeterminate / final_df_2_prod.count_indeterminate.sum())*100,5)
    
    final_df_2_prod['decile_good_rate'] = round((final_df_2_prod.count_good / final_df_2_prod.tier_count)*100,5)
    final_df_2_prod['decile_bad_rate'] =  round((final_df_2_prod.count_bad / final_df_2_prod.tier_count)*100,5)
    #final_df_2_prod['decile_indeterminate_rate'] =  round((final_df_2_prod.count_indeterminate / final_df_2_prod.tier_count)*100,5)
    final_df_2_prod['cumulative_bad_rate'] = final_df_2_prod.loc[::-1, 'bad_rate'].cumsum()[::-1]
    final_df_2_prod['cumulative_good_rate'] = final_df_2_prod.loc[::-1, 'good_rate'].cumsum()[::-1]
    #final_df_2_prod['cumulative_indeterminate_rate'] = final_df_2_prod.loc[::-1, 'indeterminate_rate'].cumsum()[::-1]
    
    final_df_2_prod['cumulative_decile_good_rate'] = round((final_df_2_prod.count_good.cumsum() / final_df_2_prod.tier_count.cumsum())*100,5)
    final_df_2_prod['cumulative_decile_bad_rate'] =  round((final_df_2_prod.count_bad.cumsum() / final_df_2_prod.tier_count.cumsum())*100,5)
     
    final_df_2_prod['KS_prod'] = final_df_2_prod['cumulative_good_rate']-final_df_2_prod['cumulative_bad_rate']
    final_df_2_prod["KS_prod"] = final_df_2_prod["KS_prod"].abs()
    
    
    
    # final_df_2_prod_new = final_df_2_prod.loc[:,['model_tiers','tier_count','total_popu_cum',
    #                                              'score_min','score_max','count_good','count_bad',
    #                                              'count_indeterminate',
    #                                              'bad_rate',
    #                                              'indeterminate_rate',
    #                                              'decile_bad_rate','decile_good_rate',
    #                                              'decile_indeterminate_rate',
    #                                              'cumulative_good_rate','cumulative_bad_rate',
    #                                              'cumulative_indeterminate_rate',
    #                                              'cumulative_decile_good_rate','cumulative_decile_bad_rate','KS_prod']]
    
    
    final_df_2_prod_new = final_df_2_prod.loc[:,['model_tiers','tier_count','total_popu_cum',
                                                 'score_min','score_max','count_good','count_bad',
                                                 'bad_rate',
                                                 'decile_bad_rate','decile_good_rate',
                                                 'cumulative_good_rate','cumulative_bad_rate',
                                                 'cumulative_decile_good_rate','cumulative_decile_bad_rate','KS_prod']]
    
    
    
    final_df_2_prod_new.columns = ['model_tiers','# accounts','cumulative accounts',
                                   'score_min','score_max','# good','# bad',
                                   'bad_rate',
                                   'tier_bad_rate','tier_good_rate',
                                   'cumulative_good_captured','cumulative_bad_captured',
                                   'tier_cumulative_good_rate','tier_cumulative_bad_rate','KS']
    
    # print(pipeline1_data['Sampling_Weights_ttd'].sum())
    # print(final_df_2_prod_new['# accounts'].sum())
    
    return final_df_2_prod_new



