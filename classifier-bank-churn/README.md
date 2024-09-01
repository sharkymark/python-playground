# Predict bank churn

If a model file is not present, train an ML classifier model from a train dataset.

Then prompt the user for inputs to predict if a specific customer will churn.

The preprocessor deletes irrelevant columns like gender, age, surname, row number and customer id.

## Run the app

An example is:

```sh
python3 bank-churn.py --train_dataset ./sets/train_bank_churn.csv --model_file bank_churn.pkl
```

## Resources

[One Hot Encoding](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OneHotEncoder.html)

[scikit-learn](https://scikit-learn.org/stable/index.html)

[pandas](https://pandas.pydata.org/)

## License

This project is licensed under the [MIT License](LICENSE)

## Contributing

### Disclaimer: Unmaintained and Untested Code

Please note that this program is not actively maintained or tested. While it may work as intended, it's possible that it will break or behave unexpectedly due to changes in dependencies, environments, or other factors.

Use this program at your own risk, and be aware that:
1. Bugs may not be fixed
1. Compatibility issues may arise
1. Security vulnerabilities may exist

If you encounter any issues or have concerns, feel free to open an issue or submit a pull request.