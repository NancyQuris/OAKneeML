import argparse
import pickle
import shap

import pandas as pd

import util_fn


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Get the Shapley value of clinical features using XGB model.')
    parser.add_argument('--keys', type=list, default=util_fn.keys_of_interest)
    parser.add_argument('--label', type=str)
    parser.add_argument('--mcid', type=float)
    parser.add_argument('--model_path', type=str)
    args = parser.parse_args()

    train_df = util_fn.preprocess_dataframe(pd.read_csv(util_fn.train_file))
    test_df = util_fn.preprocess_dataframe(pd.read_csv(util_fn.test_file))

    preprocessor = util_fn.encode_category_columns(util_fn.categorical_cols, util_fn.get_feature(train_df, args.keys))

    test_features = preprocessor.transform(util_fn.get_feature(test_df, args.keys))
    test_labels = util_fn.get_label(test_df, args.label, args.mcid)

    model = pickle.load(open(args.model_path, 'rb'))
    explainer = shap.Explainer(model)
    shap_values = explainer(test_features)
    shap.summary_plot(shap_values, test_features, plot_type = 'bar')