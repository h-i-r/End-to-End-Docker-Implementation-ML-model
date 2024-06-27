import pandas as pd


def compare_rankings(X_sample, clf, file_path, number_of_potential_customers, model_type):
    rankings = rank_potential_customers(number_of_potential_customers, X_sample, clf)
    ranks_ids = rankings.id.values
    df = pd.read_csv(file_path)
    ranked = rankings.reset_index(drop=True)
    original = df.loc[(df.id.isin(ranks_ids)), ['id', 'state']].reset_index(drop=True)
    results = pd.merge(original, ranked, on='id').sort_values(by='rank', ascending=True).rename(
        columns={'id': "new_customer_id", 'state': "true_state"})
    results.to_csv(f'results/rankings_{model_type}.csv', index=False)
    print('\nR A N K I N G S', '\n\n', results)


def rank_potential_customers(numberOfPotentialCustomers, X_Sample, clf):
    '''
    This function performs the following tasks:
    - predict probability of lost & completed on sample data
    - rank completed_probability based on highest to lowest probability score
    '''
    sample = X_Sample.sample(numberOfPotentialCustomers, random_state=42)
    id_column = sample.id
    sample = sample.drop(columns=['id'])
    probabilities = clf.predict_proba(sample)
    probabilities = pd.DataFrame(probabilities, columns=clf.classes_)
    predictions = pd.concat([id_column.reset_index(drop=True), probabilities], axis=1)
    predictions = predictions.rename(columns={0: "lost_probability", 1: "success_probability"})
    predictions = predictions.sort_values(by='success_probability', ascending=False)

    if numberOfPotentialCustomers > 1:
        ranking = predictions.head(numberOfPotentialCustomers)
    else:
        top_potential_customer = predictions.head(1)
        top_potential_customer['rank'] = 1
        ranking = top_potential_customer

    ranking['rank'] = range(1, len(ranking) + 1)
    return ranking
