# Predict wine quality

A dataset for red and white [Vinho Verde](http://www.vinhoverde.pt/en/) wine in Portugal. 

The output is `quality`, a score between 0 and 10 with 10 being best.

If a model file is not present, train an ML classifier model from a train dataset.

Then prompt the user for inputs to predict if a specific customer will churn.

## A Jupyter notebook

`wine-analysis.ipynb` analyzes the columns/features/input parameters. It also splits out train and test datasets.

## Run the app

An example is:

```sh
python3 wine-quality.py --train_dataset ./sets/train_wine_quality_red.csv --model_file wine_quality_red.pkl
python3 wine-quality.py --train_dataset ./sets/train_wine_quality_white.csv --model_file wine_quality_white.pkl
```

If you want to use the neutral network program:

```sh
python3 wine-quality-neural-net.py --train_dataset ./sets/train_wine_quality_red.csv --model_file wine_quality_red-neural-net.keras
python3 wine-quality-neural-net.py --train_dataset ./sets/train_wine_quality_white.csv --model_file wine_quality_white_neural_net.keras
```

## Resources

[UCI dataset repository](https://archive.ics.uci.edu/dataset/186/wine+quality)

[Keras](https://keras.io/)

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